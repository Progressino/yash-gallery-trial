"""Production Job Order MIN control — revise / approve / confirm / correct / additional / print.

Mirrors the Purchase JWO MIN controls (``jwo_issue_variance`` + ``update_min_status``):

* Draft: issue qty is editable per line (reason required when ≠ BOM); every change is audited
  in ``jo_min_qty_adjustments``. Outside the shared tolerance → ``Pending Approval``.
* Confirm is the stock event: Item Master stock is reduced by the **actual issue qty** and, for
  printed (checked) fabric, ``jo_fabric_issues`` + Checked stock are posted via ``issue_fabric``.
* Confirmed notes are read-only; Manager+ may post an authorized correction (stock delta + audit).
* Additional MINs (partial / extra issues) are separate notes against the same JO; their tolerance is
  judged on the JO cumulative (prior confirmed + this issue vs BOM).

INTERIM APPROVAL (temporary — see ``approve`` / ``reject`` / ``send_back`` / ``resubmit``)
    Over-tolerance approval lives here only until the centralized ERP Approval Workflow exists.
    It is deliberately shaped like that workflow so it can be swapped without a second system:
    status ``Pending Approval`` → approve | reject | send back (remarks), approver != issuer
    (maker-checker), ``approval_status / approval_remarks / approval_actor / approval_actor_at`` on the
    note, and one audit row per decision in ``jo_min_qty_adjustments``. Later, replace the role gate
    (``min_tolerance.may_approve``) and the four action functions with calls into the central engine
    (entity ``APPROVAL_ENTITY_TYPE``); stock is still posted only by ``confirm`` after approval.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

from . import min_tolerance
from .jo_issue_notes import (
    _next_in_number,
    _prod_connect,
    get_issue_note_by_id,
    init_issue_note_tables,
    jo_cumulative_issue,
)
from .jo_issue_notes import cumulative_context
from .min_tolerance import EPS, r3

REASONS = min_tolerance.PRODUCTION_VARIANCE_REASONS
PHASE_PRE = "PRE_ISSUE"
PHASE_POST = "POST_ISSUE"
PHASE_APPROVED = "APPROVED"
PHASE_REJECTED = "REJECTED"
PHASE_SENT_BACK = "SENT_BACK"
PHASE_RESUBMITTED = "RESUBMITTED"

# Key a centralized Approval Workflow will use for this document type.
APPROVAL_ENTITY_TYPE = "JO_MIN"

_OPEN_STATUSES = (min_tolerance.STATUS_DRAFT, min_tolerance.STATUS_PENDING)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ── loading helpers ───────────────────────────────────────────────────────────

def _load(conn, in_id: int, line_id: int | None = None) -> tuple[dict, dict | None]:
    row = conn.execute("SELECT * FROM jo_issue_notes WHERE id=?", (in_id,)).fetchone()
    if not row:
        raise ValueError("Material issue note not found")
    note = dict(row)
    line = None
    if line_id is not None:
        lr = conn.execute(
            "SELECT * FROM jo_issue_note_lines WHERE id=? AND issue_note_id=?", (line_id, in_id)
        ).fetchone()
        if not lr:
            raise ValueError("Issue line not found on this note")
        line = dict(lr)
    return note, line


def _lines(conn, in_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM jo_issue_note_lines WHERE issue_note_id=? ORDER BY line_no, id", (in_id,)
        ).fetchall()
    ]


def _bom_qty(line: dict) -> float:
    v = line.get("bom_required_qty")
    return float(v if v is not None else (line.get("required_qty") or 0))


def _issue_qty(line: dict) -> float:
    v = line.get("issue_qty")
    return float(v if v is not None else (line.get("required_qty") or 0))


def _assert_doc_editable(joid: int) -> None:
    try:
        from ..db.document_audit_db import assert_doc_editable

        assert_doc_editable("JO", int(joid))
    except ValueError:
        raise
    except Exception:
        pass


def _basis(conn, note: dict, line: dict, issue: float | None = None) -> tuple[float, float, bool]:
    """(BOM basis, issue basis, is_cumulative) used for variance / tolerance.

    Primary note: this note's BOM vs its issue qty. Additional MIN: the JO's BOM total vs
    prior confirmed issue + this issue (cumulative — 525 + 20 = 545 vs 525 → +3.81%).
    """
    issue = _issue_qty(line) if issue is None else float(issue)
    if not int(note.get("parent_in_id") or 0):
        return _bom_qty(line), issue, False
    prior, bom_total = cumulative_context(conn, int(note["jo_id"]), int(note["id"]), line.get("material_code"))
    return bom_total, round(prior + issue, 3), True


def _exceeds(conn, note: dict, line: dict, issue: float | None = None) -> bool:
    bom, total, cumulative = _basis(conn, note, line, issue)
    if cumulative:
        own = _issue_qty(line) if issue is None else float(issue)
        # Additional notes only breach on over-issue; a short additional can be followed by another.
        if own <= EPS or total <= bom + EPS:
            return False
    return min_tolerance.exceeds_tolerance(bom, total)


def _log(conn, note: dict, line: dict, *, phase: str, old_issue: float, new_issue: float,
         reason: str, remarks: str, user: str, stock_posted: bool, bom: float | None = None) -> None:
    basis_bom, basis_new, _ = _basis(conn, note, line, new_issue)
    bom = (basis_bom if bom is None else bom)
    conn.execute(
        """INSERT INTO jo_min_qty_adjustments(
            in_id, in_number, jo_id, jo_number, line_id, material_code, phase,
            old_issue_qty, new_issue_qty, delta_qty, bom_required_qty, variance_pct,
            reason, remarks, stock_posted, user_name, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            note["id"], note.get("in_number") or "", note.get("jo_id"), note.get("jo_number") or "",
            line.get("id"), line.get("material_code") or "", phase,
            r3(old_issue), r3(new_issue), r3(new_issue - old_issue), r3(bom),
            min_tolerance.variance_pct(bom, basis_new), reason, remarks, 1 if stock_posted else 0,
            user, _now(),
        ),
    )


def _refresh_state(conn, in_id: int) -> dict[str, Any]:
    """Draft ⇄ Pending Approval from line variances; any qty change clears a prior approval."""
    note, _ = _load(conn, in_id)
    status = str(note.get("status") or "")
    if status not in _OPEN_STATUSES:
        return {"status": status, "approval_required": bool(note.get("approval_required"))}
    over = any(_exceeds(conn, note, ln) for ln in _lines(conn, in_id))
    conn.execute(
        """UPDATE jo_issue_notes SET status=?, approval_required=?, approved_by='', approved_at='',
           approval_status='', approval_remarks='', approval_actor='', approval_actor_at=''
           WHERE id=?""",
        (min_tolerance.STATUS_PENDING if over else min_tolerance.STATUS_DRAFT, 1 if over else 0, in_id),
    )
    return {
        "status": min_tolerance.STATUS_PENDING if over else min_tolerance.STATUS_DRAFT,
        "approval_required": over,
    }


def _result(in_id: int, **extra: Any) -> dict[str, Any]:
    out = {"ok": True, "tolerance_pct": min_tolerance.get_tolerance_pct(),
           "issue_note": get_issue_note_by_id(in_id)}
    out.update(extra)
    return out


# ── stock helpers ─────────────────────────────────────────────────────────────

def _grey_path() -> str:
    from ..db import production_db

    return production_db._grey_db_path()


def _checked_stock(code: str) -> dict | None:
    """Printed (checked) fabric balance row for a material code, if it is a printed fabric."""
    if not code or not os.path.exists(_grey_path()):
        return None
    try:
        gc = sqlite3.connect(_grey_path())
        gc.row_factory = sqlite3.Row
        row = gc.execute(
            "SELECT fabric_code, fabric_name, available_qty FROM printed_fabric_checked_stock WHERE fabric_code=?",
            (code,),
        ).fetchone()
        gc.close()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def _jo_so_sku(jo_id: int) -> tuple[str, str]:
    """SO number + SKU of the JO — the key of the reservation this MIN may consume."""
    from ..db import production_db

    conn = production_db._connect()
    try:
        r = conn.execute("SELECT so_number, sku FROM job_orders WHERE id=?", (jo_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        return "", ""
    return str(r["so_number"] or "").strip(), str(r["sku"] or "").strip()


def _printed_issuable(code: str, so_number: str, sku: str) -> dict[str, float]:
    """What a JO for (SO, SKU) may issue of a printed fabric on Confirm.

    SO allocation moves metres ``available_qty → reserved_qty``, so reserved fabric shows free = 0.
    Issuable = this JO's SO reservation (consumed by ``lock_printed_on_cutting_issue``)
             + free checked stock + eligible Item Master opening stock not yet checked.
    """
    from . import fabric_allocation_engine as fae

    chk = _checked_stock(code) or {}
    free = float(chk.get("available_qty") or 0)
    reserved = 0.0
    opening = 0.0
    try:
        reserved = fae.reserved_consumable_qty(code, so_number, sku)
    except Exception:
        reserved = 0.0
    try:
        opening = float(fae.printed_opening_allocatable(code).get("allocatable") or 0)
    except Exception:
        opening = 0.0
    return {"reserved": r3(reserved), "free": r3(free), "opening": r3(opening),
            "total": r3(reserved + free + opening)}


def _check_stock_available(qty_by_code: dict[str, float], *, so_number: str = "", sku: str = "") -> None:
    """Block negative stock — Item Master and, for printed fabric, Checked stock.

    Printed fabric: the JO's own SO reservation counts as issuable (it is consumed on Confirm via
    ``issue_fabric`` → ``lock_printed_on_cutting_issue``); anything beyond it must fit in free
    checked + eligible opening stock, so another SO's reservation can never be taken.
    """
    from ..db.item_db import get_item_by_code

    for code, qty in qty_by_code.items():
        if qty <= EPS:
            continue
        item = get_item_by_code(code)
        if item is not None:
            stock = float(item.get("stock") or 0)
            if stock + 1e-9 < qty:
                raise ValueError(
                    f"Insufficient stock for {code} — available {stock:.3f}, issue qty {qty:.3f}"
                )
        chk = _checked_stock(code)
        if chk is not None:
            pool = _printed_issuable(code, so_number, sku)
            if pool["total"] + 0.001 < qty:
                raise ValueError(
                    f"Insufficient checked printed fabric for {code} — available {pool['total']:.3f} "
                    f"(SO-reserved {pool['reserved']:.3f} + free checked {pool['free']:.3f} + opening "
                    f"{pool['opening']:.3f}), issue qty {qty:.3f}"
                )


def _apply_item_stock(deltas: list[tuple[str, float, str]]) -> list[tuple[int, float, str]]:
    """Apply (code, qty, OUT|IN) to Item Master; returns applied entries for rollback."""
    from ..db.item_db import apply_document_stock_delta, get_item_by_code

    applied: list[tuple[int, float, str]] = []
    try:
        for code, qty, direction in deltas:
            if qty <= EPS:
                continue
            item = get_item_by_code(code)
            if not item:
                continue  # non-Item-Master material (same skip as Purchase MIN)
            apply_document_stock_delta(int(item["id"]), qty, direction)
            applied.append((int(item["id"]), qty, direction))
    except Exception:
        _rollback_item_stock(applied)
        raise
    return applied


def _rollback_item_stock(applied: list[tuple[int, float, str]]) -> None:
    from ..db.item_db import apply_document_stock_delta

    for item_id, qty, direction in reversed(applied):
        try:
            apply_document_stock_delta(item_id, qty, "IN" if direction == "OUT" else "OUT")
        except Exception:
            pass


def _post_fabric_issue(note: dict, line: dict, qty: float, user: str, remarks: str) -> None:
    """Printed fabric OUT → jo_fabric_issues + Checked stock (reuses ``issue_fabric``)."""
    from ..db import production_db

    production_db.issue_fabric(
        int(note["jo_id"]),
        {
            "issue_date": _today(),
            "fabric_code": line.get("material_code") or "",
            "fabric_name": line.get("material_name") or "",
            "issued_qty": qty,
            "unit": line.get("unit") or "MTR",
            "issued_by": user,
            "remarks": remarks,
            "so_number": note.get("so_number") or "",
        },
    )


def _post_fabric_return(note: dict, line: dict, qty: float, user: str, remarks: str) -> None:
    """Authorized correction (issue reduced): negative jo_fabric_issues row + Checked stock back IN."""
    from ..db import production_db

    conn = production_db._connect()
    try:
        conn.execute(
            """INSERT INTO jo_fabric_issues(jo_id, jo_line_id, issue_date, fabric_code, fabric_name,
                issued_qty, unit, issued_by, remarks) VALUES(?,?,?,?,?,?,?,?,?)""",
            (int(note["jo_id"]), None, _today(), line.get("material_code") or "",
             line.get("material_name") or "", -abs(qty), line.get("unit") or "MTR", user, remarks),
        )
        conn.execute(
            """UPDATE job_orders SET fabric_issued_qty = MAX(0, COALESCE(fabric_issued_qty,0) - ?),
               updated_at = datetime('now') WHERE id=?""",
            (abs(qty), int(note["jo_id"])),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        gc = sqlite3.connect(_grey_path())
        gc.execute(
            "UPDATE printed_fabric_checked_stock SET available_qty = available_qty + ? WHERE fabric_code=?",
            (abs(qty), line.get("material_code") or ""),
        )
        gc.commit()
        gc.close()
    except sqlite3.Error:
        pass


def _assert_jo_ok(jo_id: int, action: str) -> None:
    from ..db import production_db

    conn = production_db._connect()
    try:
        jo = conn.execute("SELECT * FROM job_orders WHERE id=?", (jo_id,)).fetchone()
    finally:
        conn.close()
    if not jo:
        raise ValueError("JO not found")
    production_db._assert_jo_mutable(dict(jo), action=action)


# ── revise (Draft / Pending Approval) ─────────────────────────────────────────

def revise_line(in_id: int, line_id: int, new_issue_qty: float, *, reason: str = "",
                remarks: str = "", user_name: str = "") -> dict[str, Any]:
    """Edit issue qty on an unconfirmed note; reason required when ≠ BOM; tolerance gate."""
    init_issue_note_tables()
    new_issue = r3(new_issue_qty)
    if new_issue < 0:
        raise ValueError("Issue qty cannot be negative")
    conn = _prod_connect()
    try:
        note, line = _load(conn, in_id, line_id)
        assert line is not None
        _assert_doc_editable(int(note["jo_id"]))
        status = str(note.get("status") or "")
        if status == min_tolerance.STATUS_CONFIRMED:
            raise ValueError("Material Issue is confirmed — use Issue Qty Correction (authorized) instead")
        if status == min_tolerance.STATUS_CANCELLED:
            raise ValueError("Material Issue is cancelled")
        bom = _bom_qty(line)
        old_issue = _issue_qty(line)
        if abs(new_issue - old_issue) <= EPS:
            raise ValueError("Issue qty is unchanged")
        b_bom, b_total, is_cum = _basis(conn, note, line, new_issue)
        # Additional MIN: reason is needed when over the JO BOM cumulatively or short of the remaining BOM.
        varies = (min_tolerance.differs_from_bom(bom, new_issue)
                  or (is_cum and b_total > b_bom + EPS))
        reason, remarks = min_tolerance.validate_reason(reason, remarks, required=varies, allowed=REASONS)
        if not varies:
            reason, remarks = "", ""
        conn.execute(
            """UPDATE jo_issue_note_lines SET issue_qty=?, variance_qty=?, variance_pct=?,
               variance_reason=?, variance_remarks=?, revised_by=?, revised_at=? WHERE id=?""",
            (new_issue, min_tolerance.variance_qty(b_bom, b_total),
             min_tolerance.variance_pct(b_bom, b_total), reason, remarks, user_name, _now(), line_id),
        )
        _log(conn, note, line, phase=PHASE_PRE, old_issue=old_issue, new_issue=new_issue,
             reason=reason, remarks=remarks, user=user_name, stock_posted=False)
        state = _refresh_state(conn, in_id)
        conn.commit()
    finally:
        conn.close()
    return _result(in_id, approval_required=state["approval_required"], status=state["status"],
                   over_tolerance=state["approval_required"])


# ── interim approval: approve / reject / send back / resubmit ─────────────────
# TEMPORARY — role gate is enforced by the router (``min_tolerance.may_approve``: Manager / Admin /
# Super Admin / Sir). A centralized ERP Approval Workflow will replace these four functions and the
# role gate; the stored shape (status, approval_* columns, audit rows) is the contract it plugs into.

def _actors(conn, note: dict) -> set[str]:
    """Everyone who prepared / revised / confirmed this note (maker side of maker-checker)."""
    out: set[str] = set()
    for ln in _lines(conn, int(note["id"])):
        out.add(str(ln.get("revised_by") or "").strip().lower())
    for r in conn.execute(
        "SELECT user_name FROM jo_min_qty_adjustments WHERE in_id=? AND phase=?", (note["id"], PHASE_PRE)
    ).fetchall():
        out.add(str(r["user_name"] or "").strip().lower())
    out.add(str(note.get("confirmed_by") or "").strip().lower())
    out.add(str(note.get("created_by") or "").strip().lower())
    out.discard("")
    return out


def _approval_event(conn, note: dict, *, phase: str, actor: str, remarks: str,
                    approval_status: str, only_over: bool = True) -> None:
    """Single write point for an approval decision (audit rows + note trail).

    Hook: a central Approval Workflow would be notified from here (entity ``APPROVAL_ENTITY_TYPE``,
    id ``note['id']``, decision ``approval_status``, actor, remarks) instead of / in addition to this.
    """
    for ln in _lines(conn, int(note["id"])):
        if only_over and not _exceeds(conn, note, ln):
            continue
        _log(conn, note, ln, phase=phase, old_issue=_issue_qty(ln), new_issue=_issue_qty(ln),
             reason=ln.get("variance_reason") or "", remarks=remarks, user=actor, stock_posted=False)
    conn.execute(
        "UPDATE jo_issue_notes SET approval_status=?, approval_remarks=?, approval_actor=?, approval_actor_at=? "
        "WHERE id=?",
        (approval_status, remarks, actor, _now(), note["id"]),
    )


def _require_pending(note: dict, what: str) -> None:
    if str(note.get("status") or "") != min_tolerance.STATUS_PENDING:
        raise ValueError(f"Only a Pending Approval Material Issue can be {what}")


def _require_remarks(remarks: str, what: str) -> str:
    remarks = str(remarks or "").strip()
    if not remarks:
        raise ValueError(f"Remarks are required to {what} a Material Issue")
    return remarks


def approve(in_id: int, *, user_name: str = "", remarks: str = "") -> dict[str, Any]:
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        _require_pending(note, "approved")
        me = str(user_name or "").strip().lower()
        if me and me in _actors(conn, note):
            raise ValueError(
                "You prepared / revised this Material Issue — a different Manager / Admin must approve it"
            )
        conn.execute(
            "UPDATE jo_issue_notes SET status=?, approved_by=?, approved_at=? WHERE id=?",
            (min_tolerance.STATUS_DRAFT, user_name, _now(), in_id),
        )
        _approval_event(conn, note, phase=PHASE_APPROVED, actor=user_name,
                        remarks=str(remarks or "").strip(), approval_status="Approved")
        conn.commit()
    finally:
        conn.close()
    return _result(in_id)


def send_back(in_id: int, *, remarks: str, user_name: str = "") -> dict[str, Any]:
    """Approver returns the note to the issuer (Draft) with remarks; qty is kept for the issuer to fix."""
    init_issue_note_tables()
    remarks = _require_remarks(remarks, "send back")
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        _require_pending(note, "sent back")
        conn.execute(
            "UPDATE jo_issue_notes SET status=?, approval_required=1, approved_by='', approved_at='' WHERE id=?",
            (min_tolerance.STATUS_DRAFT, in_id),
        )
        _approval_event(conn, note, phase=PHASE_SENT_BACK, actor=user_name, remarks=remarks,
                        approval_status="Sent Back")
        conn.commit()
    finally:
        conn.close()
    return _result(in_id)


def resubmit(in_id: int, *, user_name: str = "", remarks: str = "") -> dict[str, Any]:
    """Issuer sends a Sent-Back note for approval again (without changing qty)."""
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        if str(note.get("status") or "") != min_tolerance.STATUS_DRAFT or str(
            note.get("approval_status") or ""
        ) != "Sent Back":
            raise ValueError("Only a Sent Back Material Issue can be resubmitted for approval")
        over = any(_exceeds(conn, note, ln) for ln in _lines(conn, in_id))
        if not over:
            raise ValueError("Issue qty is within tolerance — confirm it directly")
        _approval_event(conn, note, phase=PHASE_RESUBMITTED, actor=user_name,
                        remarks=str(remarks or "").strip(), approval_status="")
        conn.execute(
            "UPDATE jo_issue_notes SET status=?, approval_required=1, approved_by='', approved_at='' WHERE id=?",
            (min_tolerance.STATUS_PENDING, in_id),
        )
        conn.commit()
    finally:
        conn.close()
    return _result(in_id)


def reject(in_id: int, *, remarks: str, user_name: str = "") -> dict[str, Any]:
    """Approver rejects the over-tolerance issue. No stock moves.

    Additional MIN → ``Rejected`` (closed). Primary MIN cannot be closed, so lines are reset to the
    BOM qty and the note returns to Draft (the rejection stays in the audit trail).
    """
    init_issue_note_tables()
    remarks = _require_remarks(remarks, "reject")
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        _require_pending(note, "rejected")
        _approval_event(conn, note, phase=PHASE_REJECTED, actor=user_name, remarks=remarks,
                        approval_status="Rejected")
        if int(note.get("parent_in_id") or 0):
            conn.execute(
                "UPDATE jo_issue_notes SET status=?, approval_required=0, approved_by='', approved_at='' WHERE id=?",
                (min_tolerance.STATUS_REJECTED, in_id),
            )
        else:
            for ln in _lines(conn, in_id):
                bom = _bom_qty(ln)
                conn.execute(
                    """UPDATE jo_issue_note_lines SET issue_qty=?, variance_qty=0, variance_pct=0,
                       variance_reason='', variance_remarks='' WHERE id=?""",
                    (bom, ln["id"]),
                )
            conn.execute(
                "UPDATE jo_issue_notes SET status=?, approval_required=0, approved_by='', approved_at='' WHERE id=?",
                (min_tolerance.STATUS_DRAFT, in_id),
            )
        conn.commit()
    finally:
        conn.close()
    return _result(in_id)


# ── confirm (the stock event) ─────────────────────────────────────────────────

def confirm(in_id: int, *, user_name: str = "", issued_by: str = "", received_by: str = "") -> dict[str, Any]:
    """Post stock for the actual issue qty: Item Master OUT + jo_fabric_issues for printed fabric."""
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        status = str(note.get("status") or "")
        if status == min_tolerance.STATUS_PENDING:
            raise ValueError(
                "Issue qty is outside the BOM tolerance — a Manager / Admin must approve before this can be confirmed"
            )
        if status != min_tolerance.STATUS_DRAFT:
            raise ValueError("Only Draft issue notes can be confirmed")
        if int(note.get("stock_posted") or 0):
            raise ValueError("Stock already posted for this issue note")
        lines = _lines(conn, in_id)
    finally:
        conn.close()

    # Defence in depth: unapproved over-tolerance lines never post stock.
    if not str(note.get("approved_by") or "").strip():
        chk_conn = _prod_connect()
        try:
            over_lines = [ln for ln in lines if _exceeds(chk_conn, note, ln)]
        finally:
            chk_conn.close()
        if over_lines:
            ln = over_lines[0]
            sent_back = " (sent back — revise the qty or resubmit)" if str(note.get("approval_status") or "") == "Sent Back" else ""
            raise ValueError(
                f"Issue qty for {ln.get('material_code')} is outside the "
                f"{min_tolerance.get_tolerance_pct():g}% BOM tolerance — Manager approval required{sent_back}"
            )

    issue_lines = [ln for ln in lines if _issue_qty(ln) > EPS and str(ln.get("material_code") or "").strip()]
    if not issue_lines:
        raise ValueError("Nothing to issue — set an issue qty greater than zero on at least one line")

    _assert_doc_editable(int(note["jo_id"]))
    _assert_jo_ok(int(note["jo_id"]), "issue fabric")

    qty_by_code: dict[str, float] = {}
    for ln in issue_lines:
        code = str(ln["material_code"]).strip()
        qty_by_code[code] = r3(qty_by_code.get(code, 0.0) + _issue_qty(ln))
    so_number, sku = _jo_so_sku(int(note["jo_id"]))
    so_number = so_number or str(note.get("so_number") or "").strip()
    _check_stock_available(qty_by_code, so_number=so_number, sku=sku)

    applied = _apply_item_stock([(c, q, "OUT") for c, q in qty_by_code.items()])
    try:
        for ln in issue_lines:
            if _checked_stock(str(ln["material_code"]).strip()) is not None:
                _post_fabric_issue(
                    note, ln, _issue_qty(ln), user_name or issued_by,
                    f"{note['in_number']} confirmed (MIN)",
                )
        conn = _prod_connect()
        try:
            cur = conn.execute(
                """UPDATE jo_issue_notes SET status=?, stock_posted=1, issued_by=?, received_by=?,
                   confirmed_at=?, confirmed_by=? WHERE id=? AND status=? AND COALESCE(stock_posted,0)=0""",
                (min_tolerance.STATUS_CONFIRMED, issued_by or user_name, received_by, _now(), user_name,
                 in_id, min_tolerance.STATUS_DRAFT),
            )
            if cur.rowcount != 1:
                raise ValueError("Issue note changed while confirming — refresh and retry")
            conn.execute(
                "UPDATE jo_issue_note_lines SET issued_qty = COALESCE(issue_qty, required_qty) WHERE issue_note_id=?",
                (in_id,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        _rollback_item_stock(applied)
        raise
    return _result(in_id)


# ── correct (after confirm, authorized) ───────────────────────────────────────

def correct(in_id: int, line_id: int, new_issue_qty: float, *, reason: str, remarks: str = "",
            user_name: str = "") -> dict[str, Any]:
    init_issue_note_tables()
    new_issue = r3(new_issue_qty)
    if new_issue < 0:
        raise ValueError("Issue qty cannot be negative")
    reason, remarks = min_tolerance.validate_reason(reason, remarks, required=True, allowed=REASONS)
    conn = _prod_connect()
    try:
        note, line = _load(conn, in_id, line_id)
        assert line is not None
    finally:
        conn.close()
    if str(note.get("status") or "") != min_tolerance.STATUS_CONFIRMED or not int(note.get("stock_posted") or 0):
        raise ValueError("Only a confirmed (stock posted) Material Issue can be corrected — edit the Draft instead")
    _assert_doc_editable(int(note["jo_id"]))
    _assert_jo_ok(int(note["jo_id"]), "issue fabric")
    old_issue = _issue_qty(line)
    delta = r3(new_issue - old_issue)
    if abs(delta) <= EPS:
        raise ValueError("Issue qty is unchanged")
    code = str(line.get("material_code") or "").strip()
    direction = "OUT" if delta > 0 else "IN"
    if delta > 0:
        so_number, sku = _jo_so_sku(int(note["jo_id"]))
        _check_stock_available({code: delta}, so_number=so_number or str(note.get("so_number") or ""), sku=sku)
    applied = _apply_item_stock([(code, abs(delta), direction)])
    try:
        if _checked_stock(code) is not None:
            if delta > 0:
                _post_fabric_issue(note, line, delta, user_name, f"{note['in_number']} correction: {reason}")
            else:
                _post_fabric_return(note, line, -delta, user_name, f"{note['in_number']} correction: {reason}")
        bom = _bom_qty(line)
        conn = _prod_connect()
        try:
            b_bom, b_total, _ = _basis(conn, note, line, new_issue)
            conn.execute(
                """UPDATE jo_issue_note_lines SET issue_qty=?, issued_qty=?, variance_qty=?, variance_pct=?,
                   variance_reason=?, variance_remarks=?, revised_by=?, revised_at=? WHERE id=?""",
                (new_issue, new_issue, min_tolerance.variance_qty(b_bom, b_total),
                 min_tolerance.variance_pct(b_bom, b_total), reason, remarks, user_name, _now(), line_id),
            )
            _log(conn, note, line, phase=PHASE_POST, old_issue=old_issue, new_issue=new_issue,
                 reason=reason, remarks=remarks, user=user_name, stock_posted=True)
            conn.commit()
        finally:
            conn.close()
    except Exception:
        _rollback_item_stock(applied)
        raise
    return _result(in_id, delta_qty=delta, stock_direction=direction)


# ── additional / partial MIN ──────────────────────────────────────────────────

def create_additional(joid: int, *, lines: list[dict] | None = None, remarks: str = "",
                      user_name: str = "", extra: bool = False) -> dict[str, Any]:
    """Another Draft MIN for the remaining (or explicit extra) qty against the same JO."""
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        primary = conn.execute(
            "SELECT * FROM jo_issue_notes WHERE jo_id=? AND COALESCE(parent_in_id,0)=0 ORDER BY id LIMIT 1",
            (joid,),
        ).fetchone()
        if not primary:
            raise ValueError("No issue note for this job order — generate it from BOM first")
        primary = dict(primary)
        if str(primary["status"] or "") != min_tolerance.STATUS_CONFIRMED:
            raise ValueError(
                "Confirm (or edit) the primary issue note first — additional issue is for qty beyond a confirmed MIN"
            )
        open_extra = conn.execute(
            "SELECT in_number FROM jo_issue_notes WHERE jo_id=? AND parent_in_id=? AND status IN (?,?)",
            (joid, primary["id"], *_OPEN_STATUSES),
        ).fetchone()
        if open_extra:
            raise ValueError(f"Confirm or cancel {open_extra['in_number']} before creating another additional issue")

        cumulative = {c["material_code"]: c for c in jo_cumulative_issue(conn, joid)}
        primary_lines = {str(l["material_code"]).strip(): l for l in _lines(conn, int(primary["id"]))}

        wanted: list[dict] = []
        if lines:
            for req in lines:
                code = str(req.get("material_code") or "").strip()
                qty = r3(req.get("issue_qty"))
                if not code or qty <= EPS:
                    continue
                if code not in primary_lines:
                    raise ValueError(f"{code} is not a material on {primary['in_number']}")
                wanted.append({"code": code, "qty": qty, "reason": req.get("variance_reason") or req.get("reason") or "",
                               "remarks": req.get("variance_remarks") or req.get("remarks") or ""})
        else:
            for code, cum in cumulative.items():
                if cum["remaining_qty"] > EPS and code in primary_lines:
                    wanted.append({"code": code, "qty": cum["remaining_qty"], "reason": "", "remarks": ""})
        if not wanted and extra and not lines:
            # Extra issue beyond BOM: zero-qty lines the issuer edits (qty > 0 vs zero BOM needs approval).
            wanted = [{"code": c, "qty": 0.0, "reason": "", "remarks": ""} for c in primary_lines]
        if not wanted:
            raise ValueError("Nothing remaining to issue against the BOM — pass explicit lines for an extra issue")

        in_num = _next_in_number(conn)
        conn.execute(
            """INSERT INTO jo_issue_notes(
                in_number, in_date, jo_id, jo_number, jo_date, so_number, process,
                finished_item_code, finished_item_name, planned_qty, status, remarks, parent_in_id, created_by)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (in_num, _today(), joid, primary["jo_number"], primary.get("jo_date") or "",
             primary.get("so_number") or "", primary.get("process") or "",
             primary.get("finished_item_code") or "", primary.get("finished_item_name") or "",
             primary.get("planned_qty") or 0, min_tolerance.STATUS_DRAFT,
             remarks or f"Additional issue against {primary['in_number']}", primary["id"], user_name),
        )
        new_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        for i, w in enumerate(wanted, start=1):
            pl = primary_lines[w["code"]]
            remaining = cumulative.get(w["code"], {}).get("remaining_qty", 0.0)
            # Cumulative basis: prior confirmed + this issue vs the JO BOM (525 + 20 vs 525 → +3.81%).
            bom_total = float(cumulative.get(w["code"], {}).get("bom_required_qty", 0.0))
            prior = float(cumulative.get(w["code"], {}).get("confirmed_issued_qty", 0.0))
            cum_total = r3(prior + w["qty"])
            varies = min_tolerance.differs_from_bom(remaining, w["qty"]) or cum_total > bom_total + EPS
            reason, rem = min_tolerance.validate_reason(w["reason"], w["remarks"], required=varies, allowed=REASONS)
            if not varies:
                reason, rem = "", ""
            conn.execute(
                """INSERT INTO jo_issue_note_lines(
                    issue_note_id, line_no, finished_item_code, finished_item_name, finished_planned_qty,
                    material_code, material_name, material_type, bom_qty_per_unit, required_qty, unit, remarks,
                    bom_required_qty, issue_qty, variance_qty, variance_pct, variance_reason, variance_remarks)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (new_id, i, pl.get("finished_item_code") or "", pl.get("finished_item_name") or "",
                 pl.get("finished_planned_qty") or 0, pl["material_code"], pl.get("material_name") or "",
                 pl.get("material_type") or "", pl.get("bom_qty_per_unit") or 0, remaining,
                 pl.get("unit") or "", f"Additional issue vs remaining BOM {remaining}",
                 remaining, w["qty"], min_tolerance.variance_qty(bom_total, cum_total),
                 min_tolerance.variance_pct(bom_total, cum_total), reason, rem),
            )
        state = _refresh_state(conn, new_id)
        conn.commit()
    finally:
        conn.close()
    return _result(new_id, status=state["status"], approval_required=state["approval_required"],
                   in_number=in_num)


# ── cancel / list / print ─────────────────────────────────────────────────────

def cancel(in_id: int, *, user_name: str = "") -> dict[str, Any]:
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        if str(note.get("status") or "") not in _OPEN_STATUSES:
            raise ValueError("Only an unconfirmed (Draft / Pending Approval) issue note can be cancelled")
        if not int(note.get("parent_in_id") or 0):
            raise ValueError("The primary BOM issue note cannot be cancelled — edit or refresh it instead")
        conn.execute("UPDATE jo_issue_notes SET status=? WHERE id=?", (min_tolerance.STATUS_CANCELLED, in_id))
        conn.commit()
    finally:
        conn.close()
    return _result(in_id)


def list_adjustments(in_id: int) -> list[dict]:
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM jo_min_qty_adjustments WHERE in_id=? ORDER BY id DESC", (in_id,)
            ).fetchall()
        ]
    finally:
        conn.close()


def print_payload(in_id: int) -> dict[str, Any]:
    """JSON for the print HTML: header, per-line BOM / issue / diff / % / reason, JO cumulative."""
    note = get_issue_note_by_id(in_id)
    if not note:
        raise ValueError("Material issue note not found")
    lines = []
    for ln in note["lines"]:
        bom, issue = _bom_qty(ln), _issue_qty(ln)
        cum = ln.get("cum_bom_qty") is not None  # additional MIN → cumulative variance basis
        v_bom = float(ln["cum_bom_qty"]) if cum else bom
        v_issue = float(ln["cum_total_qty"]) if cum else issue
        lines.append({
            "line_no": ln.get("line_no"),
            "material_code": ln.get("material_code"),
            "material_name": ln.get("material_name"),
            "material_type": ln.get("material_type"),
            "finished_item_code": ln.get("finished_item_code"),
            "finished_item_name": ln.get("finished_item_name"),
            "finished_planned_qty": ln.get("finished_planned_qty"),
            "unit": ln.get("unit"),
            "bom_required_qty": bom,
            "issue_qty": issue,
            "diff_qty": min_tolerance.variance_qty(v_bom, v_issue),
            "variance_pct": min_tolerance.variance_pct(v_bom, v_issue),
            "over_tolerance": min_tolerance.exceeds_tolerance(v_bom, v_issue) and v_issue > v_bom + EPS if cum
                              else min_tolerance.exceeds_tolerance(bom, issue),
            "cumulative_bom_qty": ln.get("cum_bom_qty"),
            "cumulative_issue_qty": ln.get("cum_total_qty"),
            "variance_reason": ln.get("variance_reason") or "",
            "variance_remarks": ln.get("variance_remarks") or "",
            "remarks": ln.get("remarks") or "",
        })
    keys = ("id", "in_number", "in_date", "jo_id", "jo_number", "jo_date", "so_number", "process",
            "finished_item_code", "finished_item_name", "planned_qty", "status", "remarks", "issued_by",
            "received_by", "confirmed_at", "confirmed_by", "approved_by", "approved_at", "parent_in_id",
            "approval_status", "approval_remarks", "approval_actor", "approval_actor_at", "created_by")
    out = {k: note.get(k) for k in keys}
    out.update(lines=lines, cumulative=note.get("cumulative") or [],
               tolerance_pct=min_tolerance.get_tolerance_pct(), generated_at=_now())
    return out

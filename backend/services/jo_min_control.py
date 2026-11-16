"""Production Job Order MIN control — revise / approve / confirm / correct / additional / print.

Mirrors the Purchase JWO MIN controls (``jwo_issue_variance`` + ``update_min_status``):

* Draft: issue qty is editable per line (reason required when ≠ BOM); every change is audited
  in ``jo_min_qty_adjustments``. Outside the shared tolerance → ``Pending Approval``.
* Confirm is the stock event: Item Master stock is reduced by the **actual issue qty** and, for
  printed (checked) fabric, ``jo_fabric_issues`` + Checked stock are posted via ``issue_fabric``.
* Confirmed notes are read-only; Manager+ may post an authorized correction (stock delta + audit).
* Additional MINs (partial / extra issues) are separate notes against the same JO.
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
from .min_tolerance import EPS, r3

REASONS = min_tolerance.PRODUCTION_VARIANCE_REASONS
PHASE_PRE = "PRE_ISSUE"
PHASE_POST = "POST_ISSUE"
PHASE_APPROVED = "APPROVED"

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


def _log(conn, note: dict, line: dict, *, phase: str, old_issue: float, new_issue: float,
         reason: str, remarks: str, user: str, stock_posted: bool, bom: float | None = None) -> None:
    bom = _bom_qty(line) if bom is None else bom
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
            min_tolerance.variance_pct(bom, new_issue), reason, remarks, 1 if stock_posted else 0,
            user, _now(),
        ),
    )


def _refresh_state(conn, in_id: int) -> dict[str, Any]:
    """Draft ⇄ Pending Approval from line variances; any qty change clears a prior approval."""
    note, _ = _load(conn, in_id)
    status = str(note.get("status") or "")
    if status not in _OPEN_STATUSES:
        return {"status": status, "approval_required": bool(note.get("approval_required"))}
    over = any(
        min_tolerance.exceeds_tolerance(_bom_qty(ln), _issue_qty(ln)) for ln in _lines(conn, in_id)
    )
    conn.execute(
        """UPDATE jo_issue_notes SET status=?, approval_required=?, approved_by='', approved_at=''
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


def _check_stock_available(qty_by_code: dict[str, float]) -> None:
    """Block negative stock — Item Master and, for printed fabric, Checked stock."""
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
        if chk is not None and float(chk.get("available_qty") or 0) + 0.001 < qty:
            raise ValueError(
                f"Insufficient checked printed fabric for {code} — "
                f"available {float(chk.get('available_qty') or 0):.3f}, issue qty {qty:.3f}"
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
        varies = min_tolerance.differs_from_bom(bom, new_issue)
        reason, remarks = min_tolerance.validate_reason(reason, remarks, required=varies, allowed=REASONS)
        if not varies:
            reason, remarks = "", ""
        conn.execute(
            """UPDATE jo_issue_note_lines SET issue_qty=?, variance_qty=?, variance_pct=?,
               variance_reason=?, variance_remarks=?, revised_by=?, revised_at=? WHERE id=?""",
            (new_issue, min_tolerance.variance_qty(bom, new_issue),
             min_tolerance.variance_pct(bom, new_issue), reason, remarks, user_name, _now(), line_id),
        )
        _log(conn, note, line, phase=PHASE_PRE, old_issue=old_issue, new_issue=new_issue,
             reason=reason, remarks=remarks, user=user_name, stock_posted=False)
        state = _refresh_state(conn, in_id)
        conn.commit()
    finally:
        conn.close()
    return _result(in_id, approval_required=state["approval_required"], status=state["status"],
                   over_tolerance=state["approval_required"])


# ── approve (Manager+, role enforced by router) ───────────────────────────────

def approve(in_id: int, *, user_name: str = "") -> dict[str, Any]:
    init_issue_note_tables()
    conn = _prod_connect()
    try:
        note, _ = _load(conn, in_id)
        if str(note.get("status") or "") != min_tolerance.STATUS_PENDING:
            raise ValueError("Only a Pending Approval Material Issue can be approved")
        conn.execute(
            "UPDATE jo_issue_notes SET status=?, approved_by=?, approved_at=? WHERE id=?",
            (min_tolerance.STATUS_DRAFT, user_name, _now(), in_id),
        )
        for ln in _lines(conn, in_id):
            if min_tolerance.exceeds_tolerance(_bom_qty(ln), _issue_qty(ln)):
                _log(conn, note, ln, phase=PHASE_APPROVED, old_issue=_issue_qty(ln),
                     new_issue=_issue_qty(ln), reason=ln.get("variance_reason") or "",
                     remarks=ln.get("variance_remarks") or "", user=user_name, stock_posted=False)
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
        for ln in lines:
            if min_tolerance.exceeds_tolerance(_bom_qty(ln), _issue_qty(ln)):
                raise ValueError(
                    f"Issue qty for {ln.get('material_code')} is outside the "
                    f"{min_tolerance.get_tolerance_pct():g}% BOM tolerance — Manager approval required"
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
    _check_stock_available(qty_by_code)

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
        _check_stock_available({code: delta})
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
            conn.execute(
                """UPDATE jo_issue_note_lines SET issue_qty=?, issued_qty=?, variance_qty=?, variance_pct=?,
                   variance_reason=?, variance_remarks=?, revised_by=?, revised_at=? WHERE id=?""",
                (new_issue, new_issue, min_tolerance.variance_qty(bom, new_issue),
                 min_tolerance.variance_pct(bom, new_issue), reason, remarks, user_name, _now(), line_id),
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
                finished_item_code, finished_item_name, planned_qty, status, remarks, parent_in_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (in_num, _today(), joid, primary["jo_number"], primary.get("jo_date") or "",
             primary.get("so_number") or "", primary.get("process") or "",
             primary.get("finished_item_code") or "", primary.get("finished_item_name") or "",
             primary.get("planned_qty") or 0, min_tolerance.STATUS_DRAFT,
             remarks or f"Additional issue against {primary['in_number']}", primary["id"]),
        )
        new_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        for i, w in enumerate(wanted, start=1):
            pl = primary_lines[w["code"]]
            remaining = cumulative.get(w["code"], {}).get("remaining_qty", 0.0)
            varies = min_tolerance.differs_from_bom(remaining, w["qty"])
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
                 remaining, w["qty"], min_tolerance.variance_qty(remaining, w["qty"]),
                 min_tolerance.variance_pct(remaining, w["qty"]), reason, rem),
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
            "diff_qty": min_tolerance.variance_qty(bom, issue),
            "variance_pct": min_tolerance.variance_pct(bom, issue),
            "over_tolerance": min_tolerance.exceeds_tolerance(bom, issue),
            "variance_reason": ln.get("variance_reason") or "",
            "variance_remarks": ln.get("variance_remarks") or "",
            "remarks": ln.get("remarks") or "",
        })
    keys = ("id", "in_number", "in_date", "jo_id", "jo_number", "jo_date", "so_number", "process",
            "finished_item_code", "finished_item_name", "planned_qty", "status", "remarks", "issued_by",
            "received_by", "confirmed_at", "confirmed_by", "approved_by", "approved_at", "parent_in_id")
    out = {k: note.get(k) for k in keys}
    out.update(lines=lines, cumulative=note.get("cumulative") or [],
               tolerance_pct=min_tolerance.get_tolerance_pct(), generated_at=_now())
    return out

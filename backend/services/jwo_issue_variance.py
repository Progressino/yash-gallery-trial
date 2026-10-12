"""Printing JWO — grey issue variance vs BOM (reverse output calc, controlled corrections, report).

Original plan (BOM default) is frozen on ``jwo_lines.original_*`` / ``min_lines.bom_issue_qty``;
revisions change the live qty and expected printed output by the BOM average
(grey issue ÷ planned output), and every change lands in ``min_qty_adjustments``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..db import purchase_db
from .jwo_min_notes import _ensure_min_schema, create_min_for_jwo, get_min_by_jwo_id

VARIANCE_REASONS = (
    "Bundle Size",
    "Vendor MOQ",
    "Roll / Than Length",
    "Wastage Allowance",
    "Planning Change",
    "Stock Shortage",
    "Other",
)

PHASE_PRE = "PRE_ISSUE"
PHASE_POST = "POST_ISSUE"
_EPS = 0.0005


def _r3(x: float) -> float:
    return round(float(x or 0), 3)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_reason(reason: str, remarks: str, *, required: bool) -> tuple[str, str]:
    reason = str(reason or "").strip()
    remarks = str(remarks or "").strip()
    if not required:
        return reason, remarks
    if not reason:
        raise ValueError("Reason is required when Grey Issue Qty differs from the BOM requirement")
    if reason not in VARIANCE_REASONS:
        raise ValueError(f"Reason must be one of: {', '.join(VARIANCE_REASONS)}")
    if reason == "Other" and not remarks:
        raise ValueError("Remarks are required when reason is Other")
    return reason, remarks


def _line_basis(line: dict) -> tuple[float, float, float]:
    """(BOM grey requirement, original planned output, current expected output)."""
    issue = float(line.get("issue_qty") or 0)
    out = float(line.get("output_qty") or 0)
    bom_issue = line.get("bom_issue_qty")
    orig_out = line.get("original_output_qty")
    expected = line.get("expected_output_qty")
    return (
        float(bom_issue) if bom_issue is not None else issue,
        float(orig_out) if orig_out is not None else out,
        float(expected) if expected is not None else out,
    )


def expected_output_for_issue(line: dict, new_issue: float) -> float | None:
    """Reverse calc: printed output = grey issue ÷ BOM average (grey per unit output)."""
    bom_issue, orig_out, _ = _line_basis(line)
    if bom_issue <= 0 or orig_out <= 0 or not str(line.get("output_material") or "").strip():
        return None
    return _r3(float(new_issue) * orig_out / bom_issue)


def _load(conn, min_id: int, line_id: int) -> tuple[dict, dict]:
    note = conn.execute("SELECT * FROM material_issue_notes WHERE id=?", (min_id,)).fetchone()
    if not note:
        raise ValueError("Material issue note not found")
    line = conn.execute(
        "SELECT * FROM min_lines WHERE id=? AND min_id=?", (line_id, min_id)
    ).fetchone()
    if not line:
        raise ValueError("Issue line not found on this MIN")
    return dict(note), dict(line)


def _assert_editable(min_id: int) -> None:
    try:
        from ..db import document_audit_db as _audit

        _audit.assert_doc_editable("MIN", int(min_id))
    except ValueError:
        raise
    except Exception:
        pass


def _audit_event(min_id: int, actor: str, detail: str) -> None:
    try:
        from ..db import document_audit_db as _audit

        _audit.record_edit_event("MIN", int(min_id), actor=actor, detail=detail)
    except Exception:
        pass


def _log(conn, note: dict, line: dict, *, phase: str, old_issue: float, new_issue: float,
         old_out: float, new_out: float, reason: str, remarks: str, user: str, stock_posted: bool) -> None:
    jwo_number = str(note.get("jwo_reference") or "")
    conn.execute(
        """INSERT INTO min_qty_adjustments(
            min_id, min_number, jwo_id, jwo_number, material_code, output_material, phase,
            old_issue_qty, new_issue_qty, delta_qty, old_output_qty, new_output_qty,
            reason, remarks, stock_posted, user_name, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            note["id"], note.get("min_number") or "", note.get("jwo_id"), jwo_number,
            line.get("material_code") or "", line.get("output_material") or "", phase,
            _r3(old_issue), _r3(new_issue), _r3(new_issue - old_issue),
            _r3(old_out), _r3(new_out), reason, remarks, 1 if stock_posted else 0, user, _now(),
        ),
    )


def _apply_to_jwo_lines(conn, jwo_id: int, line: dict, new_issue: float, new_out: float,
                        reason: str, remarks: str) -> None:
    """Scale JWO lines for this output to the revised expected output (originals frozen first)."""
    out_code = str(line.get("output_material") or "").strip()
    jls = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM jwo_lines WHERE jwo_id=? AND TRIM(output_material)=?", (jwo_id, out_code)
        ).fetchall()
    ]
    if not jls:
        return
    old_total = sum(float(j["output_qty"] or 0) for j in jls)
    mat = str(line.get("material_code") or "").strip()
    for j in jls:
        cur_out = float(j["output_qty"] or 0)
        scale = (new_out / old_total) if old_total > 0 else 0.0
        j_out = new_out if len(jls) == 1 else _r3(cur_out * scale)
        if len(jls) == 1 and str(j["input_material"] or "").strip() == mat:
            j_in = _r3(new_issue)
        else:
            j_in = _r3(float(j["input_qty"] or 0) * scale) if old_total > 0 else float(j["input_qty"] or 0)
        orig_out = j["original_output_qty"] if j["original_output_qty"] is not None else cur_out
        revised = abs(j_out - float(orig_out)) > _EPS
        conn.execute(
            """UPDATE jwo_lines SET
                original_input_qty = COALESCE(original_input_qty, input_qty),
                original_output_qty = COALESCE(original_output_qty, output_qty),
                input_qty=?, output_qty=?, amount=? * COALESCE(rate,0),
                variance_reason=?, variance_remarks=?
               WHERE id=?""",
            (j_in, j_out, j_out, reason if revised else "", remarks if revised else "", j["id"]),
        )
    conn.execute(
        "UPDATE jwo_headers SET total=(SELECT COALESCE(SUM(amount),0) FROM jwo_lines WHERE jwo_id=?) WHERE id=?",
        (jwo_id, jwo_id),
    )


def revise_issue_qty(min_id: int, line_id: int, new_issue_qty: float, *, reason: str = "",
                     remarks: str = "", user_name: str = "") -> dict[str, Any]:
    """Before confirmation: change grey issue qty; expected output + JWO + MIN follow."""
    new_issue = _r3(new_issue_qty)
    if new_issue <= 0:
        raise ValueError("Issue qty must be greater than zero")
    _assert_editable(min_id)
    conn = purchase_db._connect()
    _ensure_min_schema(conn)
    try:
        note, line = _load(conn, min_id, line_id)
        status = str(note.get("status") or "")
        if status == "Confirmed":
            raise ValueError("Material Issue is confirmed — use Issue Qty Correction (authorized) instead")
        if status == "Cancelled":
            raise ValueError("Material Issue is cancelled")
        bom_issue, orig_out, old_out = _line_basis(line)
        old_issue = float(line.get("issue_qty") or 0)
        if abs(new_issue - old_issue) <= _EPS:
            raise ValueError("Issue qty is unchanged")
        varies = abs(new_issue - bom_issue) > _EPS
        reason, remarks = validate_reason(reason, remarks, required=varies)
        if not varies:
            reason, remarks = "", ""
        new_out = expected_output_for_issue(line, new_issue)
        jwo_id = note.get("jwo_id")
        if jwo_id and new_out is not None:
            _apply_to_jwo_lines(conn, int(jwo_id), line, new_issue, new_out, reason, remarks)
        else:
            new_out = old_out
            rate = float(line.get("rate") or 0)
            conn.execute(
                """UPDATE min_lines SET bom_issue_qty=COALESCE(bom_issue_qty, issue_qty),
                    issue_qty=?, amount=?, variance_reason=?, variance_remarks=? WHERE id=?""",
                (new_issue, new_issue * rate, reason, remarks, line_id),
            )
        _log(conn, note, line, phase=PHASE_PRE, old_issue=old_issue, new_issue=new_issue,
             old_out=old_out, new_out=new_out, reason=reason, remarks=remarks,
             user=user_name, stock_posted=False)
        conn.commit()
    finally:
        conn.close()

    result = None
    if jwo_id and new_out is not None:
        jwo = purchase_db.get_jwo_by_id(int(jwo_id))
        if jwo:
            result = create_min_for_jwo(int(jwo_id), jwo["jwo_number"], jwo, jwo.get("lines") or [])
            c2 = purchase_db._connect()
            c2.execute(
                """UPDATE min_lines SET revised_by=?, revised_at=?
                   WHERE min_id=? AND TRIM(output_material)=?""",
                (user_name, _now(), min_id, str(line.get("output_material") or "").strip()),
            )
            c2.commit()
            c2.close()
            result = get_min_by_jwo_id(int(jwo_id))
    _audit_event(
        min_id, user_name,
        f"Issue qty {line.get('material_code')}: {_r3(old_issue)} → {new_issue}"
        + (f" ({reason})" if reason else "") + f"; expected output → {new_out}",
    )
    return {"ok": True, "expected_output_qty": new_out, "issue_note": result or purchase_db.get_min_by_number(note["min_number"])}


def correct_confirmed_issue(min_id: int, line_id: int, new_issue_qty: float, *, reason: str,
                            remarks: str = "", user_name: str = "") -> dict[str, Any]:
    """After confirmation: authorized correction with stock posting + audit (no silent overwrite)."""
    from ..db.item_db import apply_document_stock_delta, get_item_by_code

    new_issue = _r3(new_issue_qty)
    if new_issue < 0:
        raise ValueError("Issue qty cannot be negative")
    reason, remarks = validate_reason(reason, remarks, required=True)
    _assert_editable(min_id)
    conn = purchase_db._connect()
    _ensure_min_schema(conn)
    try:
        note, line = _load(conn, min_id, line_id)
        if str(note.get("status") or "") != "Confirmed" or not int(note.get("stock_posted") or 0):
            raise ValueError("Only a confirmed (stock posted) Material Issue can be corrected — edit the Draft instead")
        old_issue = float(line.get("issue_qty") or 0)
        delta = _r3(new_issue - old_issue)
        if abs(delta) <= _EPS:
            raise ValueError("Issue qty is unchanged")
        bom_issue, orig_out, old_out = _line_basis(line)
        new_out = expected_output_for_issue(line, new_issue)
        if new_out is None:
            new_out = old_out

        code = str(line.get("material_code") or "").strip()
        item = get_item_by_code(code) if code else None
        direction = "OUT" if delta > 0 else "IN"
        if item:
            apply_document_stock_delta(int(item["id"]), abs(delta), direction)
        try:
            rate = float(line.get("rate") or 0)
            conn.execute(
                """UPDATE min_lines SET
                    bom_issue_qty=?, original_output_qty=?,
                    issue_qty=?, amount=?, output_qty=?, expected_output_qty=?,
                    variance_reason=?, variance_remarks=?, revised_by=?, revised_at=?
                   WHERE id=?""",
                (bom_issue, orig_out, new_issue, new_issue * rate, new_out, new_out,
                 reason, remarks, user_name, _now(), line_id),
            )
            if note.get("jwo_id") and str(line.get("output_material") or "").strip():
                _apply_to_jwo_lines(conn, int(note["jwo_id"]), line, new_issue, new_out, reason, remarks)
            _log(conn, note, line, phase=PHASE_POST, old_issue=old_issue, new_issue=new_issue,
                 old_out=old_out, new_out=new_out, reason=reason, remarks=remarks,
                 user=user_name, stock_posted=bool(item))
            conn.commit()
        except Exception:
            conn.rollback()
            if item:
                apply_document_stock_delta(int(item["id"]), abs(delta), "IN" if direction == "OUT" else "OUT")
            raise
    finally:
        conn.close()

    purchase_db._post_min_grey_ledger(
        note,
        [{"material_code": code, "issue_qty": abs(delta), "unit": line.get("unit") or "MTR"}],
        txn="MIN Adjustment - Extra Issue" if delta > 0 else "MIN Adjustment - Returned",
    )
    _audit_event(
        min_id, user_name,
        f"Post-issue correction {code}: {_r3(old_issue)} → {new_issue} ({reason}); "
        f"stock {direction} {abs(delta)}; expected output → {new_out}",
    )
    return {
        "ok": True,
        "delta_qty": delta,
        "stock_direction": direction,
        "stock_posted": bool(item),
        "expected_output_qty": new_out,
        "issue_note": purchase_db.get_min_by_number(note["min_number"]),
    }


def carry_jwo_line_variance(ln: dict, old: dict | None) -> dict:
    """JWO edit form: keep originals; grey Input Qty change → reverse-calc output by BOM average."""
    ln = dict(ln)
    new_in = float(ln.get("input_qty") or 0)
    new_out = float(ln.get("output_qty") or 0)
    if old is None:
        ln.update(original_input_qty=new_in, original_output_qty=new_out,
                  variance_reason="", variance_remarks="")
        return ln
    old_in = float(old.get("input_qty") or 0)
    old_out = float(old.get("output_qty") or 0)
    o_in = float(old["original_input_qty"]) if old.get("original_input_qty") is not None else old_in
    o_out = float(old["original_output_qty"]) if old.get("original_output_qty") is not None else old_out
    same_out = abs(new_out - old_out) <= _EPS
    same_in = abs(new_in - old_in) <= _EPS
    if same_out and same_in:
        ln.update(original_input_qty=o_in, original_output_qty=o_out,
                  variance_reason=old.get("variance_reason") or "",
                  variance_remarks=old.get("variance_remarks") or "")
        return ln
    if same_out and o_in > 0 and o_out > 0:
        varies = abs(new_in - o_in) > _EPS
        reason, remarks = validate_reason(
            ln.get("variance_reason") or "", ln.get("variance_remarks") or "", required=varies
        )
        ln.update(
            output_qty=_r3(new_in * o_out / o_in) if varies else o_out,
            original_input_qty=o_in, original_output_qty=o_out,
            variance_reason=reason if varies else "", variance_remarks=remarks if varies else "",
            _revised=True, _old_in=old_in, _old_out=old_out,
        )
        return ln
    ln.update(original_input_qty=new_in, original_output_qty=new_out,
              variance_reason="", variance_remarks="")
    return ln


def log_jwo_form_revisions(jwo_id: int, jwo_number: str, revisions: list[dict], *, user_name: str = "") -> None:
    note = get_min_by_jwo_id(int(jwo_id))
    if not note:
        return
    conn = purchase_db._connect()
    try:
        for ln in revisions:
            out_code = str(ln.get("output_material") or "").strip()
            mat = str(ln.get("input_material") or "").strip()
            line = next(
                (l for l in note["lines"]
                 if str(l.get("output_material") or "").strip() == out_code
                 and str(l.get("material_code") or "").strip() == mat),
                {"material_code": mat, "output_material": out_code},
            )
            _log(conn, note, line, phase=PHASE_PRE, old_issue=float(ln.get("_old_in") or 0),
                 new_issue=float(ln.get("input_qty") or 0), old_out=float(ln.get("_old_out") or 0),
                 new_out=float(ln.get("output_qty") or 0), reason=ln.get("variance_reason") or "",
                 remarks=ln.get("variance_remarks") or "", user=user_name, stock_posted=False)
        conn.commit()
    finally:
        conn.close()


def list_adjustments(min_id: int) -> list[dict]:
    conn = purchase_db._connect()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM min_qty_adjustments WHERE min_id=? ORDER BY id DESC", (min_id,)
            ).fetchall()
        ]
    finally:
        conn.close()


def _pct(part: float, base: float) -> float | None:
    return round(part / base * 100, 2) if base else None


def printing_variance_report(*, from_date: str = "", to_date: str = "", processor: str = "",
                             reason: str = "", only_variance: bool = False) -> dict[str, Any]:
    """Original MRP requirement vs revised issue/output vs actual received, per JWO MIN line."""
    conn = purchase_db._connect()
    _ensure_min_schema(conn)
    try:
        q = """SELECT ml.*, mn.min_number, mn.status AS min_status, mn.jwo_id,
                      jh.jwo_number, jh.jwo_date, jh.processor_name, jh.so_reference, jh.pr_reference,
                      jh.status AS jwo_status
               FROM min_lines ml
               JOIN material_issue_notes mn ON mn.id = ml.min_id
               JOIN jwo_headers jh ON jh.id = mn.jwo_id
               WHERE COALESCE(mn.status,'') != 'Cancelled' AND COALESCE(jh.status,'') != 'Cancelled'"""
        params: list = []
        if from_date:
            q += " AND jh.jwo_date >= ?"
            params.append(from_date)
        if to_date:
            q += " AND jh.jwo_date <= ?"
            params.append(to_date)
        if processor:
            q += " AND jh.processor_name LIKE ?"
            params.append(f"%{processor}%")
        q += " ORDER BY jh.jwo_date DESC, jh.id DESC, ml.line_no, ml.id"
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]

        pr_cache: dict[tuple[str, str], float | None] = {}
        adj_counts = {
            int(r[0]): int(r[1])
            for r in conn.execute(
                "SELECT min_id, COUNT(*) FROM min_qty_adjustments GROUP BY min_id"
            ).fetchall()
        }

        def mrp_req(pr_ref: str, code: str) -> float | None:
            key = (pr_ref, code)
            if key not in pr_cache:
                r = conn.execute(
                    """SELECT SUM(pl.required_qty) FROM pr_lines pl JOIN pr_headers ph ON ph.id=pl.pr_id
                       WHERE ph.pr_number=? AND TRIM(pl.material_code)=?""",
                    key,
                ).fetchone()
                pr_cache[key] = float(r[0]) if r and r[0] is not None else None
            return pr_cache[key]

        out_rows = []
        seen_outputs: set[tuple[str, str]] = set()
        totals = {"bom_grey": 0.0, "issued_grey": 0.0, "original_output": 0.0,
                  "expected_output": 0.0, "received": 0.0, "mrp_requirement": 0.0,
                  "expected_output_with_mrp": 0.0}
        by_reason: dict[str, dict[str, float]] = {}
        for r in rows:
            bom_issue, orig_out, expected = _line_basis(r)
            issue = float(r.get("issue_qty") or 0)
            out_code = str(r.get("output_material") or "").strip()
            excess_grey = _r3(issue - bom_issue)
            excess_out = _r3(expected - orig_out)
            rsn = str(r.get("variance_reason") or "")
            if abs(excess_grey) <= _EPS:
                rsn = ""
            req = mrp_req(str(r.get("pr_reference") or ""), out_code) if r.get("pr_reference") else None
            over_mrp = req is not None and expected - req > _EPS
            if only_variance and abs(excess_grey) <= _EPS and not over_mrp:
                continue
            if reason and rsn != reason:
                continue
            received = (
                purchase_db._grn_accepted_for_jwo_line(conn, r["jwo_number"], out_code) if out_code else 0.0
            )
            out_rows.append({
                "jwo_number": r["jwo_number"],
                "jwo_date": r.get("jwo_date") or "",
                "jwo_status": r.get("jwo_status") or "",
                "processor_name": r.get("processor_name") or "",
                "so_reference": r.get("so_reference") or "",
                "min_id": r["min_id"],
                "min_number": r["min_number"],
                "min_status": r["min_status"],
                "grey_code": r.get("material_code") or "",
                "printed_code": out_code,
                "unit": r.get("unit") or "MTR",
                "mrp_requirement_qty": _r3(req) if req is not None else None,
                "original_planned_output": _r3(orig_out),
                "bom_grey_requirement": _r3(bom_issue),
                "revised_grey_issue": _r3(issue),
                "revised_expected_output": _r3(expected),
                "actual_received_qty": _r3(received),
                "excess_grey_qty": excess_grey,
                "excess_grey_pct": _pct(excess_grey, bom_issue),
                "excess_output_qty": excess_out,
                "excess_output_pct": _pct(excess_out, orig_out),
                "output_vs_mrp_qty": _r3(expected - req) if req is not None else None,
                "output_vs_mrp_pct": _pct(expected - req, req) if req else None,
                "received_vs_expected_qty": _r3(received - expected),
                "received_vs_original_qty": _r3(received - orig_out),
                "reason": rsn,
                "remarks": r.get("variance_remarks") or "",
                "revised_by": r.get("revised_by") or "",
                "adjustments": adj_counts.get(int(r["min_id"]), 0),
            })
            totals["bom_grey"] += bom_issue
            totals["issued_grey"] += issue
            okey = (r["jwo_number"], out_code)
            if okey not in seen_outputs:
                seen_outputs.add(okey)
                totals["original_output"] += orig_out
                totals["expected_output"] += expected
                totals["received"] += received
                if req is not None:
                    totals["mrp_requirement"] += req
                    totals["expected_output_with_mrp"] += expected
            if rsn:
                b = by_reason.setdefault(rsn, {"reason": rsn, "lines": 0, "excess_grey_qty": 0.0, "excess_output_qty": 0.0})
                b["lines"] += 1
                b["excess_grey_qty"] = _r3(b["excess_grey_qty"] + excess_grey)
                b["excess_output_qty"] = _r3(b["excess_output_qty"] + excess_out)
        summary = {k: _r3(v) for k, v in totals.items()}
        summary["excess_grey_qty"] = _r3(summary["issued_grey"] - summary["bom_grey"])
        summary["excess_grey_pct"] = _pct(summary["excess_grey_qty"], summary["bom_grey"])
        summary["excess_output_qty"] = _r3(summary["expected_output"] - summary["original_output"])
        summary["excess_output_pct"] = _pct(summary["excess_output_qty"], summary["original_output"])
        summary["output_vs_mrp_qty"] = _r3(summary.pop("expected_output_with_mrp") - summary["mrp_requirement"])
        summary["output_vs_mrp_pct"] = _pct(summary["output_vs_mrp_qty"], summary["mrp_requirement"])
        summary["lines"] = len(out_rows)
        summary["lines_with_variance"] = sum(1 for x in out_rows if abs(x["excess_grey_qty"]) > _EPS)
        return {
            "rows": out_rows,
            "summary": summary,
            "by_reason": sorted(by_reason.values(), key=lambda b: -abs(b["excess_grey_qty"])),
            "reasons": list(VARIANCE_REASONS),
        }
    finally:
        conn.close()

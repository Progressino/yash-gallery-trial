"""Bulk Opening Stock import — set Item Master physical stock to a target from Excel.

Admin-only migration helper. Does not create items. Does not auto-drop reservations;
rows with active reservations are flagged and skipped unless force_ignore_reservations.
Repeated uploads with the same target are no-ops (idempotent via target vs current stock).
"""
from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any

_log = logging.getLogger(__name__)

TEMPLATE_HEADERS = [
    "item_code",
    "target_qty",
    "unit",
    "reason",
    "reference_no",
    "entry_date",
]


def template_xlsx_bytes() -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Opening Stock"
    ws.append(TEMPLATE_HEADERS)
    ws.append(["P356Blue Mosscreape Chundi Print RGP", 1562, "MTR", "Opening Stock — physical count", "PHYS-2026-11", date.today().isoformat()])
    ws.append(["EXAMPLE-ZERO-FABRIC", 0, "MTR", "Opening Stock — zero on floor", "", date.today().isoformat()])
    note = wb.create_sheet("Notes")
    note.append(["item_code must already exist in Item Master (case-insensitive match)."])
    note.append(["target_qty is the desired physical stock AFTER import (not a delta)."])
    note.append(["IN/OUT adjustment is calculated automatically: target − current stock."])
    note.append(["Active SO / Advance reservations are flagged; those rows are skipped."])
    note.append(["Re-uploading the same target for an item creates no extra adjustment."])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _parse_rows(file_bytes: bytes) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c or "").strip().lower() for c in next(rows_iter, [])]
    col = {h: i for i, h in enumerate(header) if h}

    def cell(row, *names, default=""):
        for n in names:
            if n in col and col[n] < len(row):
                v = row[col[n]]
                if v is None:
                    return default
                return v
        return default

    out: list[dict[str, Any]] = []
    for raw in rows_iter:
        if not raw or all(v is None or str(v).strip() == "" for v in raw):
            continue
        code = str(cell(raw, "item_code", "pcode", "fabric_code", "code") or "").strip()
        if not code:
            continue
        try:
            target = float(cell(raw, "target_qty", "qty", "quantity", "stock", default=0) or 0)
        except (TypeError, ValueError):
            out.append({"item_code": code, "ok": False, "error": "Invalid target_qty"})
            continue
        unit = str(cell(raw, "unit", "uom", default="MTR") or "MTR").strip() or "MTR"
        reason = str(cell(raw, "reason", default="Opening Stock — bulk import") or "").strip()
        if not reason:
            reason = "Opening Stock — bulk import"
        ref = str(cell(raw, "reference_no", "reference", "ref", default="") or "").strip()
        ed = cell(raw, "entry_date", "date", default="")
        if hasattr(ed, "isoformat"):
            entry_date = ed.isoformat()[:10]
        else:
            entry_date = str(ed or "").strip()[:10] or date.today().isoformat()
        out.append({
            "item_code": code,
            "target_qty": round(target, 3),
            "unit": unit,
            "reason": reason,
            "reference_no": ref,
            "entry_date": entry_date,
        })
    return out


def _active_reservation_qty(fabric_code: str) -> float:
    try:
        from ..db import grey_db as gdb

        conn = gdb._connect()
        try:
            row = conn.execute(
                """SELECT COALESCE(SUM(qty),0) q FROM printed_fabric_reservations
                   WHERE UPPER(TRIM(fabric_code))=UPPER(TRIM(?))
                     AND status IN ('Active','JO Created')""",
                (fabric_code,),
            ).fetchone()
            return float(row["q"] or 0)
        finally:
            conn.close()
    except Exception:
        return 0.0


def preview_opening_import(file_bytes: bytes) -> dict[str, Any]:
    from ..db.item_db import get_item_by_code_ci, opening_stock_for_code

    parsed = _parse_rows(file_bytes)
    rows: list[dict[str, Any]] = []
    for p in parsed:
        if p.get("ok") is False:
            rows.append(p)
            continue
        code = p["item_code"]
        im = opening_stock_for_code(code)
        item = None
        try:
            item = get_item_by_code_ci(code)
        except Exception:
            item = None
        if not im.get("found") and not item:
            rows.append({**p, "ok": False, "error": "Item not found in Item Master"})
            continue
        canonical = str(im.get("item_code") or (item or {}).get("item_code") or code)
        current = float(im.get("stock") if im.get("found") else (item or {}).get("stock") or 0)
        target = float(p["target_qty"])
        delta = round(target - current, 3)
        rsv = _active_reservation_qty(canonical)
        conflict = rsv > 0.001 and abs(delta) > 0.001
        rows.append({
            **p,
            "item_code": canonical,
            "current_qty": round(current, 3),
            "delta": delta,
            "direction": "IN" if delta > 0 else ("OUT" if delta < 0 else "NONE"),
            "reserved_qty": round(rsv, 3),
            "reservation_conflict": conflict,
            "noop": abs(delta) <= 0.001,
            "ok": True,
        })
    return {
        "ok": True,
        "rows": rows,
        "total": len(rows),
        "conflicts": sum(1 for r in rows if r.get("reservation_conflict")),
        "actions": sum(1 for r in rows if r.get("ok") and not r.get("noop") and not r.get("reservation_conflict")),
    }


def apply_opening_import(
    file_bytes: bytes,
    *,
    created_by: str = "",
    skip_conflicts: bool = True,
) -> dict[str, Any]:
    """Apply target-stock adjustments. Conflicts are skipped when skip_conflicts=True."""
    from ..db.item_db import adjust_item_stock, get_item_by_code_ci

    preview = preview_opening_import(file_bytes)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for r in preview["rows"]:
        if not r.get("ok"):
            errors.append(r)
            continue
        if r.get("noop"):
            skipped.append({**r, "skip_reason": "Already at target"})
            continue
        if r.get("reservation_conflict") and skip_conflicts:
            skipped.append({**r, "skip_reason": f"Active reservations {r.get('reserved_qty')} m — release first"})
            continue
        try:
            item = get_item_by_code_ci(r["item_code"])
            if not item:
                errors.append({**r, "error": "Item not found"})
                continue
            delta = abs(float(r["delta"]))
            direction = r["direction"]
            if direction == "NONE" or delta <= 0:
                skipped.append({**r, "skip_reason": "No change"})
                continue
            ref = r.get("reference_no") or f"BULK-OPEN-{date.today().isoformat()}"
            adj = adjust_item_stock(
                int(item["id"]),
                delta,
                direction,
                entry_date=r.get("entry_date") or date.today().isoformat(),
                reason=r.get("reason") or "Opening Stock — bulk import",
                reference_no=ref,
                unit=r.get("unit") or item.get("uom") or "MTR",
                created_by=created_by or "",
            )
            applied.append({**r, "adjustment": adj})
        except Exception as e:
            _log.exception("opening import failed for %s", r.get("item_code"))
            errors.append({**r, "error": str(e)})

    return {
        "ok": len(errors) == 0,
        "applied": applied,
        "skipped": skipped,
        "errors": errors,
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "error_count": len(errors),
    }

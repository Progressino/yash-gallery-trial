"""Unit tests for Manual SO multi-line JO draft helper (JS logic mirrored in Python)."""
from __future__ import annotations


def append_manual_jo_line(lines: list[dict], next_line: dict) -> list[dict]:
    """Mirror of frontend/src/pages/joLineHelpers.ts appendManualJoLine."""
    sku = (next_line.get("sku") or "").strip()
    qty = int(next_line.get("planned_qty") or 0)
    if not sku or qty <= 0:
        return lines
    key = sku.upper()
    for i, row in enumerate(lines):
        if (row.get("sku") or "").strip().upper() == key:
            out = list(lines)
            out[i] = {
                **row,
                "planned_qty": qty,
                "so_qty": next_line.get("so_qty", qty),
                "sku_name": next_line.get("sku_name") or row.get("sku_name") or "",
                "vendor_rate": next_line.get("vendor_rate") or row.get("vendor_rate") or 0,
            }
            return out
    return [*lines, {**next_line, "sku": sku}]


def test_append_keeps_previous_lines():
    lines: list[dict] = []
    lines = append_manual_jo_line(
        lines,
        {"sku": "1294YKGREEN-S", "planned_qty": 10, "sku_name": "S", "so_number": "M1", "style": "", "vendor_rate": 0, "remarks": ""},
    )
    lines = append_manual_jo_line(
        lines,
        {"sku": "1294YKGREEN-M", "planned_qty": 12, "sku_name": "M", "so_number": "M1", "style": "", "vendor_rate": 0, "remarks": ""},
    )
    lines = append_manual_jo_line(
        lines,
        {"sku": "1294YKGREEN-L", "planned_qty": 5, "sku_name": "L", "so_number": "M1", "style": "", "vendor_rate": 0, "remarks": ""},
    )
    assert len(lines) == 3
    assert [l["sku"] for l in lines] == ["1294YKGREEN-S", "1294YKGREEN-M", "1294YKGREEN-L"]
    assert sum(l["planned_qty"] for l in lines) == 27


def test_append_same_sku_updates_qty_not_duplicate():
    lines = append_manual_jo_line([], {"sku": "S", "planned_qty": 3, "sku_name": "", "so_number": "", "style": "", "vendor_rate": 0, "remarks": ""})
    lines = append_manual_jo_line(lines, {"sku": "s", "planned_qty": 8, "sku_name": "n", "so_number": "", "style": "", "vendor_rate": 1, "remarks": ""})
    assert len(lines) == 1
    assert lines[0]["planned_qty"] == 8

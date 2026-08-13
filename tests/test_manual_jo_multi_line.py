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


def size_from_sku(sku: str, style: str = "") -> str:
    st = (style or "").strip()
    if st:
        return st
    s = (sku or "").strip()
    i = s.rfind("-")
    if i > 0 and i < len(s) - 1:
        return s[i + 1 :]
    return ""


def format_jo_header_sku(jo: dict) -> str:
    lines = [l for l in (jo.get("lines") or []) if str(l.get("sku") or "").strip()]
    unique = []
    seen = set()
    for l in lines:
        sku = str(l.get("sku") or "").strip()
        k = sku.upper()
        if k in seen:
            continue
        seen.add(k)
        unique.append(l)
    if len(unique) <= 1:
        sku = (unique[0].get("sku") if unique else jo.get("sku")) or ""
        name = (unique[0].get("sku_name") if unique else jo.get("sku_name")) or ""
        if not sku:
            return name
        return f"{sku} — {name}" if name else sku
    parts = []
    for l in unique:
        sku = str(l.get("sku") or "").strip()
        qty = int(l.get("planned_qty") or 0)
        parts.append(f"{sku} ({qty})" if qty else sku)
    return " · ".join(parts)


def expand_jo_export_rows(jo: dict) -> list[list]:
    lines = [l for l in (jo.get("lines") or []) if str(l.get("sku") or "").strip()]
    sources = lines or [
        {
            "sku": jo.get("sku"),
            "sku_name": jo.get("sku_name"),
            "style": "",
            "planned_qty": jo.get("planned_qty"),
            "received_qty": jo.get("received_qty"),
            "issued_qty": jo.get("issued_qty"),
            "balance_qty": jo.get("balance_qty"),
            "remarks": jo.get("remarks"),
        }
    ]
    rows = []
    for s in sources:
        sku = str(s.get("sku") or jo.get("sku") or "").strip()
        planned = int(s.get("planned_qty") or jo.get("planned_qty") or 0)
        received = int(s.get("received_qty") or 0)
        style = str(s.get("style") or "").strip()
        size = size_from_sku(sku, style)
        rows.append(
            [
                jo.get("jo_number"),
                sku,
                str(s.get("sku_name") or jo.get("sku_name") or ""),
                style,
                size,
                planned,
                received,
            ]
        )
    return rows


def test_export_multi_size_jo_emits_one_row_per_line():
    jo = {
        "jo_number": "PJO-0775",
        "sku": "1303YKBLACK-L",
        "sku_name": "1303YKBLACK - L",
        "planned_qty": 100,
        "lines": [
            {"sku": "1303YKBLACK-L", "sku_name": "L", "style": "L", "planned_qty": 70, "received_qty": 0},
            {"sku": "1303YKBLACK-4XL", "sku_name": "4XL", "style": "4XL", "planned_qty": 20, "received_qty": 0},
            {"sku": "1303YKBLACK-8XL", "sku_name": "8XL", "style": "8XL", "planned_qty": 10, "received_qty": 0},
        ],
    }
    rows = expand_jo_export_rows(jo)
    assert len(rows) == 3
    skus = [r[1] for r in rows]
    assert skus == ["1303YKBLACK-L", "1303YKBLACK-4XL", "1303YKBLACK-8XL"]
    assert [r[5] for r in rows] == [70, 20, 10]
    assert [r[4] for r in rows] == ["L", "4XL", "8XL"]
    header = format_jo_header_sku(jo)
    assert "1303YKBLACK-4XL" in header
    assert "1303YKBLACK-8XL" in header
    assert "1303YKBLACK-L" in header


def test_export_single_line_jo_one_row():
    jo = {
        "jo_number": "PJO-1",
        "sku": "STYLE-M",
        "sku_name": "Style M",
        "planned_qty": 12,
        "received_qty": 0,
        "lines": [{"sku": "STYLE-M", "sku_name": "Style M", "style": "M", "planned_qty": 12, "received_qty": 0}],
    }
    rows = expand_jo_export_rows(jo)
    assert len(rows) == 1
    assert rows[0][1] == "STYLE-M"
    assert rows[0][5] == 12
    assert format_jo_header_sku(jo).startswith("STYLE-M")


def test_export_header_only_jo_without_lines():
    jo = {"jo_number": "PJO-2", "sku": "SOLO-S", "sku_name": "Solo", "planned_qty": 5, "received_qty": 0, "lines": []}
    rows = expand_jo_export_rows(jo)
    assert len(rows) == 1
    assert rows[0][1] == "SOLO-S"


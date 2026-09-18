"""Bulk Demand / Sales Order line import — CSV/XLSX templates + parsing."""
from __future__ import annotations

import math
from typing import Any


DEMAND_IMPORT_COLUMNS = [
    "sku",
    "sku_name",
    "demand_qty",
]

SO_IMPORT_COLUMNS = [
    "sku",
    "sku_name",
    "qty",
    "unit",
    "rate",
    "hsn_code",
    "gst_pct",
    "merchant_code",
    "priority",
    "line_delivery_date",
    "remarks",
]


def demand_import_template_csv() -> str:
    header = ",".join(DEMAND_IMPORT_COLUMNS)
    examples = [
        "TEST-SKU-S,Test Style S,50",
        "TEST-SKU-M,Test Style M,80",
        "TEST-SKU-L,Test Style L,40",
    ]
    return "\n".join([header, *examples]) + "\n"


def so_import_template_csv() -> str:
    header = ",".join(SO_IMPORT_COLUMNS)
    examples = [
        "TEST-SKU-S,Test Style S,50,PCS,199,6109,5,,Normal,2026-10-15,",
        "TEST-SKU-M,Test Style M,80,PCS,199,6109,5,,Normal,2026-10-15,",
        "TEST-SKU-L,Test Style L,40,PCS,199,6109,5,,High,2026-10-20,Rush",
    ]
    return "\n".join([header, *examples]) + "\n"


def _cell(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    s = str(raw).strip()
    if s.lower() in {"", "nan", "none", "nat"}:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _num(raw: Any, default: float = 0.0) -> float:
    s = _cell(raw)
    if not s:
        return float(default)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float(default)


def _int(raw: Any, default: int = 0) -> int:
    return int(_num(raw, float(default)))


def parse_demand_import_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (lines, errors). Merges duplicate SKUs by summing qty."""
    errors: list[str] = []
    by_sku: dict[str, dict] = {}
    for i, row in enumerate(rows):
        row_n = i + 2
        norm = {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}
        sku = _cell(norm.get("sku") or norm.get("item_code") or norm.get("oms_sku"))
        if not sku:
            if any(_cell(v) for v in norm.values()):
                errors.append(f"Row {row_n}: sku is required")
            continue
        qty = _int(norm.get("demand_qty") or norm.get("qty") or norm.get("quantity"), 0)
        if qty <= 0:
            errors.append(f"Row {row_n} ({sku}): demand_qty must be > 0")
            continue
        name = _cell(norm.get("sku_name") or norm.get("item_name") or norm.get("name"))
        key = sku.upper()
        if key in by_sku:
            by_sku[key]["demand_qty"] += qty
            if name and not by_sku[key].get("sku_name"):
                by_sku[key]["sku_name"] = name
        else:
            by_sku[key] = {"sku": sku, "sku_name": name, "demand_qty": qty}
    return list(by_sku.values()), errors


def parse_so_import_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (lines, errors). Merges duplicate SKUs by summing qty (keeps first rate/HSN)."""
    errors: list[str] = []
    by_sku: dict[str, dict] = {}
    for i, row in enumerate(rows):
        row_n = i + 2
        norm = {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}
        sku = _cell(norm.get("sku") or norm.get("item_code") or norm.get("oms_sku"))
        if not sku:
            if any(_cell(v) for v in norm.values()):
                errors.append(f"Row {row_n}: sku is required")
            continue
        qty = _int(norm.get("qty") or norm.get("quantity") or norm.get("so_qty"), 0)
        if qty <= 0:
            errors.append(f"Row {row_n} ({sku}): qty must be > 0")
            continue
        name = _cell(norm.get("sku_name") or norm.get("item_name") or norm.get("name"))
        line = {
            "sku": sku,
            "sku_name": name,
            "qty": qty,
            "unit": _cell(norm.get("unit")) or "PCS",
            "rate": _num(norm.get("rate"), 0),
            "hsn_code": _cell(norm.get("hsn_code") or norm.get("hsn")),
            "gst_pct": _num(norm.get("gst_pct") or norm.get("gst"), 0),
            "merchant_code": _cell(norm.get("merchant_code") or norm.get("merchant")),
            "priority": _cell(norm.get("priority")) or "Normal",
            "line_delivery_date": _cell(
                norm.get("line_delivery_date") or norm.get("delivery_date")
            ),
            "remarks": _cell(norm.get("remarks") or norm.get("remark")),
        }
        key = sku.upper()
        if key in by_sku:
            by_sku[key]["qty"] += qty
        else:
            by_sku[key] = line
    return list(by_sku.values()), errors

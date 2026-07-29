"""Embroidery measurement units (Border meters, Yog count, etc.) for Set BOM + JOs."""
from __future__ import annotations

from typing import Any, Optional

EMBROIDERY_TYPES: tuple[str, ...] = ("Border", "Yog", "Boota", "Other")

_UNIT_BY_TYPE: dict[str, str] = {
    "Border": "MTR",
    "Yog": "YOG",
    "Boota": "BOOTA",
    "Other": "PCS",
}

_TYPE_LABELS: dict[str, str] = {
    "Border": "meters of border per piece",
    "Yog": "Yog per piece",
    "Boota": "Boota per piece",
    "Other": "units per piece",
}


def embroidery_unit_for_type(embroidery_type: object) -> str:
    key = str(embroidery_type or "").strip().title()
    if key == "Yog":
        return "YOG"
    if key == "Boota":
        return "BOOTA"
    if key == "Border":
        return "MTR"
    return _UNIT_BY_TYPE.get(key, "PCS")


def embroidery_qty_label(embroidery_type: object) -> str:
    key = str(embroidery_type or "").strip().title()
    return _TYPE_LABELS.get(key, "measurement per piece")


def normalize_embroidery_measurement_fields(line: dict[str, Any] | None) -> dict[str, Any]:
    """Align embroidery type / per-piece qty with timing flags."""
    ln = dict(line or {})
    emb_type = str(ln.get("embroidery_type") or "").strip().title()
    if emb_type not in EMBROIDERY_TYPES:
        emb_type = ""
    try:
        per_piece = float(ln.get("embroidery_qty_per_piece") or 0)
    except (TypeError, ValueError):
        per_piece = 0.0
    per_piece = max(0.0, per_piece)
    requires = bool(ln.get("requires_embroidery")) or "Embroidery" in str(ln.get("routing") or "")
    before = bool(ln.get("embroidery_before_cutting"))
    if not requires:
        ln["embroidery_type"] = ""
        ln["embroidery_qty_per_piece"] = 0.0
        ln["embroidery_unit"] = ""
        return ln
    if emb_type:
        ln["embroidery_type"] = emb_type
        ln["embroidery_unit"] = embroidery_unit_for_type(emb_type)
    else:
        ln["embroidery_type"] = ""
        ln["embroidery_unit"] = str(ln.get("embroidery_unit") or "").strip().upper() or ""
    ln["embroidery_qty_per_piece"] = per_piece
    if before and requires:
        if not emb_type:
            raise ValueError(
                "Embroidery Type is required when Embroidery Timing = Before cutting (fabric)"
            )
        if per_piece <= 0:
            raise ValueError(
                f"{embroidery_qty_label(emb_type)} must be greater than 0 for before-cutting embroidery"
            )
    return ln


def find_embroidery_line_for_sku(bom: dict[str, Any] | None, sku: str) -> Optional[dict[str, Any]]:
    """Match Set BOM line for a component/panel SKU."""
    if not bom:
        return None
    from .set_components import parse_component_sku

    raw = str(sku or "").strip().upper()
    _main, comp = parse_component_sku(raw)
    comp_code = str(comp or "").strip().upper()
    lines = bom.get("lines") or []
    if comp_code:
        for ln in lines:
            if str(ln.get("component_code") or "").strip().upper() == comp_code:
                return dict(ln)
    # Set-component SKU (e.g. STYLE-M-TOP) or main size SKU — match first embroidered line.
    for ln in lines:
        if bool(ln.get("requires_embroidery")) or "Embroidery" in str(ln.get("routing") or ""):
            return dict(ln)
    return None


def compute_embroidery_measurement(
    *,
    garment_pieces: int,
    qty_per_piece: float,
    stock_available: float = 0.0,
) -> dict[str, Any]:
    """Gross requirement minus in-stock embroidery material → vendor JO qty."""
    pieces = max(int(garment_pieces or 0), 0)
    per = max(float(qty_per_piece or 0), 0.0)
    gross = round(pieces * per, 4)
    stock = max(float(stock_available or 0), 0.0)
    stock_used = round(min(stock, gross), 4)
    net = round(max(0.0, gross - stock_used), 4)
    return {
        "garment_pieces": pieces,
        "qty_per_piece": per,
        "gross_measurement": gross,
        "stock_available": stock,
        "stock_used": stock_used,
        "net_measurement": net,
    }


def format_embroidery_jo_qty(
    *,
    measurement: float,
    unit: str,
    embroidery_type: str,
    garment_pieces: int,
    stock_used: float = 0.0,
) -> str:
    u = str(unit or "PCS").strip().upper()
    t = str(embroidery_type or "Embroidery").strip()
    base = f"{measurement:g} {u} {t}".strip()
    if garment_pieces > 0:
        base += f" ({garment_pieces} pcs"
        if stock_used > 0:
            base += f", {stock_used:g} {u} from stock"
        base += ")"
    return base

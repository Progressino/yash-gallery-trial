"""Quantity control, tolerance, and MRP commitment helpers for PO / GRN / JO flows."""
from __future__ import annotations

from typing import Optional

# Receive tolerance by material type (fraction, e.g. 0.05 = +5%)
TOLERANCE_BY_MATERIAL_TYPE: dict[str, float] = {
    "GF": 0.05,
    "GREY": 0.05,
    "GREIGE": 0.05,
    "Grey Fabric": 0.05,
    "RM": 0.05,
    "SFG": 0.05,
    "Semi Finished": 0.05,
    "Printed": 0.05,
    "Fabric": 0.05,
    "Accessories": 0.0,
    "ACC": 0.0,
    "Button": 0.0,
    "BTN": 0.0,
    "Trim": 0.0,
}

DEFAULT_GREY_TOLERANCE = 0.05
DEFAULT_ACCESSORY_TOLERANCE = 0.0
DEFAULT_PIECE_TOLERANCE = 0.0


def tolerance_pct(material_type: str = "", material_code: str = "") -> float:
    mt = (material_type or "").strip()
    if mt in TOLERANCE_BY_MATERIAL_TYPE:
        return TOLERANCE_BY_MATERIAL_TYPE[mt]
    code = (material_code or "").upper()
    if code.startswith(("GF-", "GREY", "PF-", "PRINT")):
        return DEFAULT_GREY_TOLERANCE
    if any(tok in code for tok in ("BTN", "BUTTON", "ZIP", "LABEL", "TAG")):
        return DEFAULT_ACCESSORY_TOLERANCE
    return DEFAULT_GREY_TOLERANCE if mt.upper() in ("RM", "SFG", "") else 0.0


def max_allowed_receive(ordered_qty: float, tolerance_pct: float) -> float:
    ordered = float(ordered_qty or 0)
    if ordered <= 0:
        return 0.0
    return round(ordered * (1.0 + float(tolerance_pct or 0)), 6)


def validate_receive_qty(
    ordered_qty: float,
    already_received: float,
    incoming_qty: float,
    tolerance_pct: float,
    *,
    doc_label: str = "document",
) -> None:
    """Raise ValueError if incoming would exceed ordered + tolerance."""
    incoming = float(incoming_qty or 0)
    if incoming <= 0:
        raise ValueError("Receive quantity must be greater than zero")
    ordered = float(ordered_qty or 0)
    already = float(already_received or 0)
    cap = max_allowed_receive(ordered, tolerance_pct)
    remaining = max(0.0, cap - already)
    if already + incoming > cap + 1e-9:
        raise ValueError(
            f"{doc_label}: cannot receive {incoming} — max allowed {cap:.3f} "
            f"(ordered {ordered:.3f} + {tolerance_pct * 100:.1f}% tolerance), "
            f"already received {already:.3f}, remaining {remaining:.3f}"
        )


def po_should_auto_close(ordered_qty: float, received_qty: float, tolerance_pct: float) -> bool:
    ordered = float(ordered_qty or 0)
    if ordered <= 0:
        return False
    received = float(received_qty or 0)
    return received >= ordered and received <= max_allowed_receive(ordered, tolerance_pct)


def jo_should_auto_close(planned_qty: int, received_qty: int, tolerance_pct: float = 0.0) -> bool:
    planned = int(planned_qty or 0)
    if planned <= 0:
        return False
    received = int(received_qty or 0)
    cap = int(max_allowed_receive(planned, tolerance_pct) + 0.999999)
    return received >= planned and received <= cap

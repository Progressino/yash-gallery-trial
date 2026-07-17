"""Component BOM — per-component material consumption under Set BOM.

Main SKU stays the business identity (sales, planning, FG). Set BOM defines
production components (TOP/PANT/DUPATTA) each with their own material lines.
Cutting JOs are created per component; issue notes explode only that component's
materials.
"""
from __future__ import annotations

from typing import Any


def get_component_material_lines(main_sku: str, component_code: str) -> list[dict[str, Any]]:
    """Return material rows for one component from the Set BOM."""
    from ..db.production_db import get_set_bom_for_sku

    code = str(component_code or "").strip().upper()
    if not code:
        return []
    bom = get_set_bom_for_sku(main_sku)
    if not bom:
        return []
    for ln in bom.get("lines") or []:
        if str(ln.get("component_code") or "").upper() == code:
            return list(ln.get("materials") or [])
    return []


def explode_component_materials(
    main_sku: str,
    component_code: str,
    component_sku: str,
    component_name: str,
    finished_qty: float,
) -> list[dict[str, Any]]:
    """Material requirements for one component Cutting JO (Set BOM materials)."""
    finished_qty = float(finished_qty or 0)
    if finished_qty <= 0:
        return []
    materials = get_component_material_lines(main_sku, component_code)
    if not materials:
        return []
    out: list[dict[str, Any]] = []
    for m in materials:
        mat_code = str(m.get("material_code") or "").strip()
        if not mat_code:
            continue
        cons = float(m.get("quantity") or 0)
        if cons <= 0:
            continue
        required = round(cons * finished_qty, 3)
        if required <= 0:
            continue
        out.append(
            {
                "finished_item_code": component_sku,
                "finished_item_name": component_name or component_sku,
                "finished_planned_qty": finished_qty,
                "material_code": mat_code,
                "material_name": str(m.get("material_name") or mat_code),
                "material_type": "RM",
                "bom_qty_per_unit": cons,
                "required_qty": required,
                "unit": str(m.get("unit") or "MTR"),
                "bom_anchor_code": main_sku,
                "component_code": str(component_code or "").upper(),
            }
        )
    return out


def should_auto_create_component_jos(data: dict) -> bool:
    """True when a Cutting JO for a main SKU should explode into component JOs."""
    process = str(data.get("process") or data.get("stage") or "Cutting").strip()
    if process != "Cutting":
        return False
    if data.get("create_component_jos") is False:
        return False
    from .set_components import parse_component_sku

    sku = str(data.get("sku") or "").strip().upper()
    if not sku:
        return False
    if parse_component_sku(sku)[1]:
        return False
    from ..db.production_db import get_set_bom_for_sku

    bom = get_set_bom_for_sku(sku)
    return bool(bom and bom.get("lines"))

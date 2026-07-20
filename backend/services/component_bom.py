"""Component BOM — per-component material consumption under Set BOM.

Main SKU stays the business identity (sales, planning, FG). Set BOM defines
production components (TOP/PANT/DUPATTA) each with their own material lines.
Cutting JOs are created per component; issue notes explode only that component's
materials.
"""
from __future__ import annotations

from typing import Any, Optional

_GARMENT_COMPONENT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DUPATTA", ("DUPATTA", "CHUNNI", "STOLE", "ODHNI")),
    ("PANT", ("PANT", "PANTS", "PALAZZO", "SALWAR", "BOTTOM", "TROUSER", "LEGGING")),
    ("TOP", ("TOP", "KURTA", "KURTI", "SHIRT", "BLOUSE", "TUNIC", "KAMEEZ", "CHOLI")),
)


def resolve_cutting_main_sku(data: dict) -> str:
    """Main size SKU for a Cutting JO payload (header or first line)."""
    from .set_components import parse_component_sku

    sku = str(data.get("sku") or "").strip().upper()
    if sku:
        main, comp = parse_component_sku(sku)
        if comp:
            return main or sku
        return sku
    for ln in data.get("lines") or []:
        s = str(ln.get("sku") or "").strip().upper()
        if not s:
            continue
        main, comp = parse_component_sku(s)
        if comp:
            return main or s
        return s
    return ""


def infer_garment_component_code(name: str, item_code: str = "") -> Optional[str]:
    blob = f"{name} {item_code}".upper()
    for code, hints in _GARMENT_COMPONENT_HINTS:
        if any(h in blob for h in hints):
            return code
    return None


def _set_bom_from_fg_item_bom(main_sku: str) -> Optional[dict]:
    """When Production Set BOM is missing, derive components from FG BOM SFG lines."""
    from .jo_issue_notes import (
        _get_default_bom,
        _get_item_by_code,
        _get_item_by_id,
        _item_connect,
        _resolve_bom_item,
        explode_bom_materials,
    )

    main_sku = str(main_sku or "").strip().upper()
    if not main_sku:
        return None
    try:
        iconn = _item_connect()
    except FileNotFoundError:
        return None
    try:
        _bom_code, item = _resolve_bom_item(iconn, main_sku)
        if not item:
            return None
        bom = _get_default_bom(iconn, int(item["id"]))
        if not bom:
            return None
        lines_out: list[dict[str, Any]] = []
        seen: set[str] = set()
        rows = iconn.execute(
            """SELECT bl.*, rs.name AS process_name
               FROM bom_lines bl
               LEFT JOIN routing_steps rs ON rs.id = bl.process_id
               WHERE bl.bom_id=?""",
            (int(bom["id"]),),
        ).fetchall()
        for ln in rows:
            ln = dict(ln)
            ctype = str(ln.get("component_type") or "RM").upper()
            if ctype not in ("SFG", "FG"):
                continue
            comp = None
            if ln.get("component_item_id"):
                comp = _get_item_by_id(iconn, int(ln["component_item_id"]))
            raw = str(ln.get("component_name") or "").strip()
            code_guess = raw.split(" — ")[0].strip() if " — " in raw else raw
            if not comp and code_guess:
                comp = _get_item_by_code(iconn, code_guess)
            comp_code_item = (comp or {}).get("item_code") or code_guess
            comp_name = (comp or {}).get("item_name") or raw or comp_code_item
            comp_token = infer_garment_component_code(comp_name, str(comp_code_item or ""))
            if not comp_token:
                continue
            if comp_token in seen:
                continue
            seen.add(comp_token)
            ratio = max(int(round(float(ln.get("quantity") or 1))), 1)
            materials: list[dict[str, Any]] = []
            if comp_code_item:
                for m in explode_bom_materials(
                    str(comp_code_item), str(comp_name), 1.0, process="Cutting"
                ):
                    if str(m.get("material_type") or "RM").upper() in (
                        "SVC",
                        "SERVICE",
                        "PROCESS",
                        "SFG",
                        "FG",
                    ):
                        continue
                    materials.append(
                        {
                            "material_code": m["material_code"],
                            "material_name": m.get("material_name") or m["material_code"],
                            "quantity": float(m.get("bom_qty_per_unit") or 0),
                            "unit": m.get("unit") or "MTR",
                        }
                    )
            lines_out.append(
                {
                    "component_code": comp_token,
                    "component_name": comp_token.title(),
                    "qty_per_set": ratio,
                    "materials": materials,
                }
            )
        if len(lines_out) < 2:
            return None
        return {
            "style_key": main_sku,
            "style_name": item.get("item_name") or main_sku,
            "lines": lines_out,
            "source": "item_master_fg_bom",
        }
    finally:
        iconn.close()


def effective_set_bom_for_cutting(sku: str) -> Optional[dict]:
    """Production Set BOM, or FG Item Master SFG children (Top/Pant/Dupatta)."""
    from ..db.production_db import get_set_bom_for_sku

    raw = str(sku or "").strip().upper()
    if not raw:
        return None
    bom = get_set_bom_for_sku(raw)
    if bom and bom.get("lines"):
        return bom
    return _set_bom_from_fg_item_bom(raw)


def get_component_material_lines(main_sku: str, component_code: str) -> list[dict[str, Any]]:
    """Return material rows for one component from the Set BOM."""
    code = str(component_code or "").strip().upper()
    if not code:
        return []
    bom = effective_set_bom_for_cutting(main_sku)
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

    main_sku = resolve_cutting_main_sku(data)
    if not main_sku:
        return False
    if parse_component_sku(main_sku)[1]:
        return False
    bom = effective_set_bom_for_cutting(main_sku)
    return bool(bom and bom.get("lines"))

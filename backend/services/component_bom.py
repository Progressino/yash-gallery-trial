"""Component BOM — per-component material consumption under Set BOM.

Main SKU stays the business identity (sales, planning, FG). Set BOM defines
production components (TOP/PANT/DUPATTA) each with their own material lines.
Cutting JOs are created per **set component** only.

Panels (Front / Back / Sleeve) may live on the same Set BOM for embroidery WIP,
but they are **not** independent Cutting Job Orders — they belong under a parent
set component (usually TOP).
"""
from __future__ import annotations

from typing import Any, Optional

ROLE_SET_COMPONENT = "SET_COMPONENT"
ROLE_PANEL = "PANEL"

_GARMENT_COMPONENT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DUPATTA", ("DUPATTA", "CHUNNI", "STOLE", "ODHNI")),
    ("PANT", ("PANT", "PANTS", "PALAZZO", "SALWAR", "BOTTOM", "TROUSER", "LEGGING")),
    ("TOP", ("TOP", "KURTA", "KURTI", "SHIRT", "BLOUSE", "TUNIC", "KAMEEZ", "CHOLI")),
)

# Codes that are panels of a garment piece, not set-level Cutting JO components.
_PANEL_CODE_TOKENS = (
    "FRONT",
    "BACK",
    "SLEEVE",
    "SLEEVES",
    "NECK",
    "COLLAR",
    "YOKE",
    "POCKET",
    "PANEL",
    "PLACKET",
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


def _looks_like_panel_code(code: str, name: str = "") -> bool:
    blob = f"{code} {name}".upper().replace("-", " ").replace("_", " ")
    tokens = [t for t in blob.split() if t]
    if any(t in _PANEL_CODE_TOKENS for t in tokens):
        return True
    # TOPFRONT / TOP_BACK style codes
    compact = "".join(tokens)
    return any(tok in compact for tok in ("FRONT", "BACK", "SLEEVE", "PANEL"))


def _infer_panel_parent(code: str, set_codes: set[str]) -> str:
    """Best-effort parent for legacy panel rows (TOP_FRONT → TOP)."""
    raw = str(code or "").strip().upper().replace("-", "_")
    for parent in ("TOP", "PANT", "BOTTOM", "DUPATTA"):
        if parent in set_codes and (raw.startswith(parent + "_") or raw.startswith(parent)):
            rest = raw[len(parent) :].lstrip("_")
            if rest and _looks_like_panel_code(rest):
                return parent
    if "TOP" in set_codes:
        return "TOP"
    if "PANT" in set_codes:
        return "PANT"
    if "BOTTOM" in set_codes:
        return "BOTTOM"
    return ""


def normalize_line_role(line: dict[str, Any], *, sibling_codes: set[str] | None = None) -> str:
    """Return SET_COMPONENT or PANEL for a Set BOM line."""
    role = str(line.get("component_role") or line.get("role") or "").strip().upper()
    if role in (ROLE_PANEL, "PANEL_PART", "PART"):
        return ROLE_PANEL
    if role in (ROLE_SET_COMPONENT, "COMPONENT", "SET"):
        return ROLE_SET_COMPONENT
    if line.get("creates_cutting_jo") is False or str(line.get("creates_cutting_jo") or "").strip() in (
        "0",
        "false",
        "False",
    ):
        return ROLE_PANEL
    # Explicit True must not override panel-code heuristics when role was omitted.
    parent = str(line.get("parent_component_code") or "").strip().upper()
    if parent:
        return ROLE_PANEL
    code = str(line.get("component_code") or "").strip().upper()
    name = str(line.get("component_name") or "").strip()
    if _looks_like_panel_code(code, name):
        # Exact garment set tokens stay components even if name mentions front/back.
        if code in {"TOP", "PANT", "BOTTOM", "DUPATTA", "PANTS"}:
            return ROLE_SET_COMPONENT
        return ROLE_PANEL
    return ROLE_SET_COMPONENT


def annotate_set_bom_roles(bom: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fill component_role / parent_component_code on every line (in place)."""
    if not bom or not bom.get("lines"):
        return bom
    codes = {
        str(ln.get("component_code") or "").strip().upper()
        for ln in bom["lines"]
        if str(ln.get("component_code") or "").strip()
    }
    set_codes = {
        c
        for c in codes
        if c in {"TOP", "PANT", "BOTTOM", "DUPATTA", "PANTS"}
        or not _looks_like_panel_code(c)
    }
    for ln in bom["lines"]:
        role = normalize_line_role(ln, sibling_codes=codes)
        ln["component_role"] = role
        if role == ROLE_PANEL:
            parent = str(ln.get("parent_component_code") or "").strip().upper()
            if not parent:
                parent = _infer_panel_parent(str(ln.get("component_code") or ""), set_codes)
            ln["parent_component_code"] = parent
            ln["creates_cutting_jo"] = False
        else:
            ln["parent_component_code"] = str(ln.get("parent_component_code") or "").strip().upper()
            ln["creates_cutting_jo"] = True
    return bom


def set_component_lines(bom: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Lines that create Cutting Job Orders / set-match components."""
    annotate_set_bom_roles(bom)
    if not bom:
        return []
    return [
        ln
        for ln in (bom.get("lines") or [])
        if normalize_line_role(ln) == ROLE_SET_COMPONENT
    ]


def panel_lines(
    bom: dict[str, Any] | None,
    *,
    parent_component_code: str | None = None,
) -> list[dict[str, Any]]:
    """Panel rows (Front/Back/…) managed inside a parent Cutting JO."""
    annotate_set_bom_roles(bom)
    if not bom:
        return []
    parent = str(parent_component_code or "").strip().upper()
    out = []
    for ln in bom.get("lines") or []:
        if normalize_line_role(ln) != ROLE_PANEL:
            continue
        if parent and str(ln.get("parent_component_code") or "").strip().upper() != parent:
            continue
        out.append(ln)
    return out


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
        return annotate_set_bom_roles(bom)
    derived = _set_bom_from_fg_item_bom(raw)
    return annotate_set_bom_roles(derived) if derived else None


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
    return bool(set_component_lines(bom))

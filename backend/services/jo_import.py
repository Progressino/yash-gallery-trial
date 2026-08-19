"""Bulk Job Order import — template + row parsing for component / panel Set BOM flow.

Recommended Cutting upload (one row per size SKU):
  so_number + main size sku → auto-creates TOP / PANT / DUPATTA Cutting JOs.
  FRONT / BACK are Set BOM *panels* under TOP — they are NOT separate JO rows.
  Panel stock appears on the TOP Cutting JO after Receive (split).

Optional: set component_code=TOP|PANT|DUPATTA to create only that Cutting JO.
"""
from __future__ import annotations

import math
from typing import Any

from .component_bom import panel_lines, set_component_lines
from .set_components import component_sku, normalize_component_code, parse_component_sku

JO_IMPORT_COLUMNS = [
    "so_number",
    "sku",
    "component_code",
    "planned_qty",
    "process",
    "production_mode",
    "create_component_jos",
    "exec_type",
    "vendor_name",
    "vendor_rate",
    "expected_completion",
    "fabric_code",
    "fabric_qty",
    "fabric_unit",
    "sku_name",
    "remarks",
]

_PANEL_CODES = frozenset({"FRONT", "BACK", "SLEEVE", "SLEEVES", "NECK", "COLLAR", "YOKE", "POCKET", "PANEL", "PLACKET"})


def jo_import_template_csv() -> str:
    """CSV template with examples for main-SKU and optional per-component upload."""
    header = ",".join(JO_IMPORT_COLUMNS)
    # Example 1: one main size row → TOP + PANT + DUPATTA JOs (panels live under TOP).
    row_main = (
        "SO-0001,TEST SKU-M,,10,Cutting,,yes,Inhouse,,0,2026-08-15,"
        ",,,Test Style M,"
        "\"Recommended: one row per size. Creates TOP/PANT/DUPATTA Cutting JOs from Set BOM. "
        "Do NOT add FRONT/BACK rows — panels appear inside TOP JO after Receive.\""
    )
    # Example 2: Cut-to-Pack row (vendor path — skips in-house processes).
    row_top = (
        "SO-0001,TEST SKU-M,TOP,10,Cutting,cut_to_pack,no,Outsource,Vendor XYZ,0,2026-08-15,"
        "FABRIC-TOP,20,MTR,Top only,"
        "\"production_mode: inhouse | cut_to_pack | stitch_to_pack. "
        "Blank = inherit from SO default. Determines downstream workflow.\""
    )
    # Example 3: Stitch-to-Pack row.
    row_pant = (
        "SO-0001,TEST SKU-M,PANT,10,Cutting,stitch_to_pack,no,Outsource,Vendor ABC,0,2026-08-15,"
        "FABRIC-PANT,15,MTR,Pant only,"
        "\"stitch_to_pack route: Cutting → Stitching → Finishing (skips Embroidery/Handwork/Kaj).\""
    )
    # Example 4: Stitching (no component explode).
    row_stitch = (
        "SO-0001,TEST SKU-M-TOP,,10,Stitching,,no,Outsource,Vendor ABC,25,2026-08-25,"
        ",,,,"
        "\"Later processes: use the component JO SKU (…-TOP) or main SKU as needed. "
        "create_component_jos is ignored outside Cutting.\""
    )
    return "\n".join([header, row_main, row_top, row_pant, row_stitch]) + "\n"


def _import_cell_str(raw: Any) -> str:
    """Normalize CSV/XLSX cell values; treat pandas NaN / blank as empty."""
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    s = str(raw).strip()
    if s.lower() in {"", "nan", "none", "nat"}:
        return ""
    return s


def _truthy_yes(raw: Any) -> bool | None:
    """Parse create_component_jos: None = auto, True/False = forced."""
    s = str(raw if raw is not None else "").strip().lower()
    if not s or s in {"auto", "default", "nan", "none"}:
        return None
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _looks_like_panel_code(code: str) -> bool:
    c = str(code or "").strip().upper()
    if not c:
        return False
    if c in _PANEL_CODES:
        return True
    return any(tok in c for tok in ("FRONT", "BACK", "SLEEVE", "PANEL"))


def looks_like_ready_to_wip_columns(columns: list[Any] | tuple[Any, ...] | None) -> bool:
    """True when file is Ready-To WIP migration template, not Job Order import."""
    cols = {str(c or "").strip().lower().replace(" ", "_") for c in (columns or [])}
    if "ready_to_stage" in cols and ("oms_sku" in cols or "sku" in cols):
        return True
    # ready_to_*_wip_template.csv: OMS_SKU + Quantity, no JO planned_qty/sku
    if "oms_sku" in cols and "quantity" in cols and "sku" not in cols and "planned_qty" not in cols:
        return True
    return False


def build_jo_payload_from_import_row(row: dict[str, Any], *, default_process: str = "Cutting") -> dict[str, Any]:
    """Map one CSV/XLSX row into create_jo payload. Raises ValueError on bad panel rows."""
    so_number = _import_cell_str(row.get("so_number") or row.get("so"))
    sku = _import_cell_str(row.get("sku") or row.get("oms_sku")).upper()
    if not so_number or not sku:
        raise ValueError("so_number and sku required (use production JO template, not Ready-To WIP template)")

    planned = float(row.get("planned_qty") or row.get("qty") or row.get("quantity") or 0)
    process = str(
        row.get("process") or row.get("ready_to_stage") or default_process or "Cutting"
    ).strip() or "Cutting"
    delivery = str(
        row.get("expected_completion")
        or row.get("delivery_date")
        or row.get("delivery")
        or ""
    ).strip()
    create_flag = _truthy_yes(row.get("create_component_jos"))

    raw_comp = _import_cell_str(row.get("component_code") or row.get("component")).upper()
    parsed_main, parsed_comp = parse_component_sku(sku)

    # Reject panel SKUs / panel component codes for Cutting JO creation.
    if process == "Cutting":
        if parsed_comp and _looks_like_panel_code(parsed_comp):
            raise ValueError(
                f"{sku} is a panel SKU — do not import FRONT/BACK as Job Orders. "
                "Define panels on Set BOM under TOP; they appear on the TOP Cutting JO after Receive."
            )
        if raw_comp and _looks_like_panel_code(raw_comp):
            raise ValueError(
                f"component_code={raw_comp} is a panel — not a Cutting JO. "
                "Upload the main size SKU (or component_code=TOP/PANT/DUPATTA). "
                "FRONT/BACK stay inside the parent Cutting JO."
            )

    main_sku = parsed_main or sku
    component_code = ""
    create_component_jos: bool | None = create_flag

    if raw_comp:
        component_code = normalize_component_code(raw_comp)
    elif parsed_comp and process != "Cutting":
        component_code = parsed_comp
        create_component_jos = False
    elif parsed_comp and process == "Cutting":
        raise ValueError(
            f"Cannot import Cutting JO on component SKU {sku}; "
            f"use main size SKU {parsed_main} with optional component_code={parsed_comp}."
        )
    elif process != "Cutting":
        create_component_jos = False if create_component_jos is None else create_component_jos

    raw_mode = _import_cell_str(
        row.get("production_mode") or row.get("production_path") or row.get("path")
    )

    data: dict[str, Any] = {
        "so_number": so_number,
        "sku": main_sku,
        "sku_name": _import_cell_str(row.get("sku_name")),
        "process": process,
        "exec_type": str(row.get("exec_type") or "Inhouse").strip() or "Inhouse",
        "vendor_name": str(row.get("vendor_name") or "").strip(),
        "vendor_rate": float(row.get("vendor_rate") or row.get("rate") or 0),
        "planned_qty": planned,
        "expected_completion": delivery[:10] if delivery else "",
        "remarks": _import_cell_str(row.get("remarks")),
        "fabric_code": _import_cell_str(row.get("fabric_code")),
        "fabric_qty": float(row.get("fabric_qty") or 0),
        "fabric_unit": str(row.get("fabric_unit") or "MTR").strip() or "MTR",
        "create_component_jos": create_component_jos,
    }
    if raw_mode:
        data["production_mode"] = raw_mode

    if raw_comp:
        # Direct component Cutting JO (TOP/PANT/DUPATTA only — never FRONT/BACK).
        from ..db.production_db import get_set_bom_for_sku

        _KNOWN_SET = frozenset({
            "TOP", "PANT", "DUPATTA", "KURTA", "BOTTOM", "SKIRT", "BLOUSE",
            "JACKET", "SHIRT", "SHARARA", "GHAGRA", "LEHENGA", "PALAZZO",
            "TROUSER", "TROUSERS",
        })
        bom = get_set_bom_for_sku(main_sku)
        if bom:
            set_codes = {str(ln.get("component_code") or "").upper() for ln in set_component_lines(bom)}
            panel_codes = {str(ln.get("component_code") or "").upper() for ln in panel_lines(bom)}
            if component_code in panel_codes:
                raise ValueError(
                    f"{component_code} is a Set BOM panel — managed inside parent Cutting JO, not imported."
                )
            if set_codes and component_code not in set_codes:
                raise ValueError(
                    f"component_code={component_code} not in Set BOM set-components "
                    f"({', '.join(sorted(set_codes))}). Panels: {', '.join(sorted(panel_codes)) or 'none'}."
                )
        elif component_code not in _KNOWN_SET:
            raise ValueError(
                f"component_code={component_code} is not a known set component "
                f"({', '.join(sorted(_KNOWN_SET))}) and no Set BOM exists for {main_sku}."
            )
        csku = component_sku(main_sku, component_code)
        data.update(
            {
                "sku": csku,
                "main_sku": main_sku,
                "component_code": component_code,
                "sku_role": "COMPONENT",
                "create_component_jos": False,
                "sku_name": data["sku_name"] or f"{main_sku} {component_code}".strip(),
                "lines": [
                    {
                        "so_number": so_number,
                        "sku": csku,
                        "sku_name": data["sku_name"] or component_code,
                        "planned_qty": planned,
                        "parent_sku": main_sku,
                        "component_code": component_code,
                        "sku_role": "COMPONENT",
                        "vendor_rate": data["vendor_rate"],
                        "remarks": data["remarks"],
                    }
                ],
            }
        )
        return data

    if parsed_comp and process != "Cutting":
        data["sku"] = sku
        data["main_sku"] = main_sku
        data["component_code"] = component_code
        data["sku_role"] = "COMPONENT"
        data["create_component_jos"] = False

    data["lines"] = [
        {
            "so_number": so_number,
            "sku": data["sku"],
            "sku_name": data["sku_name"],
            "planned_qty": planned,
            "vendor_rate": data["vendor_rate"],
            "remarks": data["remarks"],
            "component_code": data.get("component_code") or "",
            "sku_role": data.get("sku_role") or "MAIN",
            "parent_sku": data.get("main_sku") or "",
        }
    ]
    return data

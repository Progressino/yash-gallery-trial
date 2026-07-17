"""Multi-component set production — Component BOM, Cutting JOs, Finishing set-match.

Flow (component-wise Cutting JOs — default when Set BOM exists)
---------------------------------------------------------------
1. Main size SKU (e.g. ``1001YKBEIGE-XS``) is used for sales / planning / FG.
2. Set BOM defines components (TOP/PANT/DUPATTA) each with material consumption.
3. Creating a Cutting JO for the main SKU auto-creates one JO per component
   (``1001YKBEIGE-XS-TOP``, etc.) with issue notes from that component's materials.
4. Components move independently through Issue / Receive / Pending / Reject.
5. At Finishing, ``complete_sets = min(component_avail / ratio)``.
6. ``commit_set_match`` moves matched sets onto the main SKU at Packing.

Legacy flow (``create_component_jos=false``): single main Cutting JO, split on receive.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .helpers import get_parent_sku

_COMP_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,24}$")


def normalize_component_code(raw: str) -> str:
    code = re.sub(r"[^A-Za-z0-9_]+", "", str(raw or "").strip().upper().replace(" ", "_"))
    if not code or not _COMP_TOKEN_RE.match(code):
        raise ValueError(f"Invalid component code: {raw!r}")
    # Avoid colliding with size tokens that get_parent_sku strips.
    if code in {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"}:
        raise ValueError(f"Component code must not be a size token: {code}")
    return code


def component_sku(main_sku: str, component_code: str) -> str:
    main = str(main_sku or "").strip().upper()
    code = normalize_component_code(component_code)
    if not main:
        raise ValueError("main_sku is required")
    return f"{main}-{code}"


def parse_component_sku(sku: str) -> tuple[str, str] | tuple[None, None]:
    """Return (main_sku, component_code) when ``sku`` looks like a set component."""
    s = str(sku or "").strip().upper()
    if "-" not in s:
        return None, None
    main, code = s.rsplit("-", 1)
    if not main or not code or not _COMP_TOKEN_RE.match(code):
        return None, None
    if code in {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"}:
        return None, None
    return main, code


def style_key_for_set_bom(sku: str) -> str:
    """Parent style used to look up Set BOM (size stripped when possible)."""
    s = str(sku or "").strip().upper()
    main, _comp = parse_component_sku(s)
    if main:
        s = main
    parent = get_parent_sku(s)
    return str(parent or s).strip().upper()


def compute_complete_sets(component_avails: list[dict[str, Any]]) -> dict[str, Any]:
    """
    ``component_avails`` items: component_code, qty_per_set, available_qty.

    Returns complete_sets, per-component shortfall/extra, and matched consumption.
    """
    if not component_avails:
        return {
            "complete_sets": 0,
            "components": [],
            "ok": False,
            "message": "No components defined",
        }
    floors: list[int] = []
    enriched: list[dict[str, Any]] = []
    for row in component_avails:
        ratio = max(int(row.get("qty_per_set") or 1), 1)
        avail = max(int(row.get("available_qty") or 0), 0)
        floors.append(avail // ratio)
        enriched.append({**row, "qty_per_set": ratio, "available_qty": avail})
    complete = int(min(floors)) if floors else 0
    components_out: list[dict[str, Any]] = []
    for row in enriched:
        ratio = int(row["qty_per_set"])
        avail = int(row["available_qty"])
        needed = complete * ratio
        components_out.append(
            {
                "component_code": row.get("component_code"),
                "component_name": row.get("component_name") or row.get("component_code"),
                "component_sku": row.get("component_sku"),
                "qty_per_set": ratio,
                "available_qty": avail,
                "matched_qty": needed,
                "extra_qty": max(avail - needed, 0),
                "shortfall_qty": max(needed - avail, 0) if complete == 0 else 0,
                # Pending to reach next complete set beyond current match:
                "pending_to_next_set": max(ratio - (avail % ratio), 0) % ratio
                if avail % ratio
                else 0,
            }
        )
    # Shortfall relative to max component availability (what blocks more sets)
    max_possible = max(floors) if floors else 0
    for row, floor in zip(components_out, floors):
        row["shortfall_to_max_peer"] = max(0, (max_possible - floor) * int(row["qty_per_set"]))
    return {
        "complete_sets": complete,
        "components": components_out,
        "ok": True,
        "message": f"{complete} complete set(s) available",
    }

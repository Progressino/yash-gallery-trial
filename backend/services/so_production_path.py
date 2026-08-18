"""SO-level production path — Cut-to-Pack / Stitch-to-Pack / In-house.

Style/BOM routing stays the full in-house catalog. The *sales order* chooses how
that style is executed so SO-01 can be Cut-to-Pack while SO-02 of the same style
runs in-house at the same time.
"""
from __future__ import annotations

from typing import Optional

PRODUCTION_MODES = ("inhouse", "cut_to_pack", "stitch_to_pack")

PRODUCTION_MODE_LABELS = {
    "inhouse": "In-house",
    "cut_to_pack": "Cut-to-Pack (vendor)",
    "stitch_to_pack": "Stitch-to-Pack (vendor)",
}

# In-house hops skipped at our factory; vendor returns FG at Finishing.
_MODE_PATHS: dict[str, list[str]] = {
    "cut_to_pack": ["Cutting", "Finishing"],
    "stitch_to_pack": ["Cutting", "Stitching", "Finishing"],
}

_OUTSOURCE_AT = {
    "cut_to_pack": "Cutting",
    "stitch_to_pack": "Stitching",
}


def normalize_production_mode(raw: str | None) -> str:
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "in_house": "inhouse",
        "internal": "inhouse",
        "c2p": "cut_to_pack",
        "cuttopack": "cut_to_pack",
        "cut_pack": "cut_to_pack",
        "cutpack": "cut_to_pack",
        "s2p": "stitch_to_pack",
        "stitchtopack": "stitch_to_pack",
        "stich_to_pack": "stitch_to_pack",
        "stichtopack": "stitch_to_pack",
        "stich_pack": "stitch_to_pack",
        "stichpack": "stitch_to_pack",
        "stitch_pack": "stitch_to_pack",
        "stitchpack": "stitch_to_pack",
    }
    s = aliases.get(s, s)
    return s if s in PRODUCTION_MODES else "inhouse"


def get_so_production_mode(so_number: str | None) -> str:
    so = str(so_number or "").strip()
    if not so:
        return "inhouse"
    try:
        from ..db.sales_db import _connect

        conn = _connect()
        row = conn.execute(
            "SELECT production_mode FROM sales_orders WHERE so_number=?",
            (so,),
        ).fetchone()
        conn.close()
        if row:
            return normalize_production_mode(dict(row).get("production_mode"))
    except Exception:
        pass
    return "inhouse"


def production_path_for(
    sku: str,
    *,
    so_number: str | None = None,
    production_mode: str | None = None,
    item_path: list[str] | None = None,
) -> list[str]:
    """Process hops for this SKU on this SO (or explicit mode)."""
    mode = normalize_production_mode(production_mode) if production_mode else get_so_production_mode(so_number)
    if mode in _MODE_PATHS:
        return list(_MODE_PATHS[mode])
    if item_path is not None:
        return list(item_path)
    from ..db.production_db import get_component_routing

    return get_component_routing(sku)


def suggested_exec_type(production_mode: str | None, process: str) -> str:
    mode = normalize_production_mode(production_mode)
    at = _OUTSOURCE_AT.get(mode)
    if at and str(process or "").strip() == at:
        return "Outsource"
    return "Inhouse"

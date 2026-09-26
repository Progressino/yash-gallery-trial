"""SO-level production path — Cut-to-Pack / Stitch-to-Pack / In-house.

Style/BOM routing stays the full in-house catalog. The *sales order* chooses how
that style is executed so SO-01 can be Cut-to-Pack while SO-02 of the same style
runs in-house at the same time.
"""
from __future__ import annotations

import threading
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


_SO_MODE_CACHE: dict = {"fp": None, "modes": {}}
_SO_MODE_LOCK = threading.Lock()


def _so_mode_map() -> dict[str, str]:
    """All SO production modes, reloaded only when sales.db changes.

    Ready-To / JO routing asks for the mode of every stock row several times;
    one connection per call made a single Ready-To request open ~15k connections.
    """
    from ..db import sales_db
    from .db_fingerprint import db_fingerprint

    fp = db_fingerprint(sales_db._DB)
    cached = _SO_MODE_CACHE
    if cached["fp"] == fp:
        return cached["modes"]
    with _SO_MODE_LOCK:
        if _SO_MODE_CACHE["fp"] == fp:
            return _SO_MODE_CACHE["modes"]
        modes: dict[str, str] = {}
        conn = sales_db._connect()
        try:
            for row in conn.execute("SELECT so_number, production_mode FROM sales_orders"):
                key = str(row["so_number"] or "").strip()
                if key and key not in modes:
                    modes[key] = normalize_production_mode(row["production_mode"])
        finally:
            conn.close()
        _SO_MODE_CACHE["modes"] = modes
        _SO_MODE_CACHE["fp"] = fp
        return modes


def get_so_production_mode(so_number: str | None) -> str:
    so = str(so_number or "").strip()
    if not so:
        return "inhouse"
    try:
        return _so_mode_map().get(so, "inhouse")
    except Exception:
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

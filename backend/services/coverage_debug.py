"""Structured coverage / data-source diagnostics for admin UI (no SSH)."""
from __future__ import annotations

import os
from typing import Any

from .shared_frames import (
    frame_row_count,
    session_uses_shared_frames,
    shared_frames_enabled,
)


def _platform_flags(sess) -> dict[str, bool]:
    keys = {
        "amazon": "mtr_df",
        "myntra": "myntra_df",
        "meesho": "meesho_df",
        "flipkart": "flipkart_df",
        "snapdeal": "snapdeal_df",
    }
    return {name: frame_row_count(attr, sess) > 0 for name, attr in keys.items()}


def _resolve_sales_source(sess) -> str:
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled

        if not normalized_tables_enabled():
            return "warm_cache"
        from ..db.forecast_ops_pg import _require_conn, ops_pg_enabled

        if not ops_pg_enabled():
            return "warm_cache"
        conn = _require_conn()
        if conn is None:
            return "warm_cache"
        with conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM forecast_sales_transactions WHERE platform = %s",
                ("unified",),
            ).fetchone()
            if n and int(n[0] or 0) > 0:
                return "postgres_normalized"
            n2 = conn.execute("SELECT COUNT(*) FROM forecast_sales_transactions").fetchone()
            if n2 and int(n2[0] or 0) > 0:
                return "postgres_normalized"
    except Exception:
        pass
    try:
        import backend.main as _main

        disk_dir = getattr(_main, "_DISK_CACHE_DIR", "") or ""
        if disk_dir and os.path.exists(os.path.join(disk_dir, "sales_df.parquet")):
            return "disk_parquet"
    except Exception:
        pass
    return "warm_cache"


def _resolve_inventory_source(sess) -> str:
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled, load_inventory_dataframe

        if normalized_tables_enabled():
            inv = load_inventory_dataframe()
            if inv is not None and not inv.empty:
                return "postgres_normalized"
    except Exception:
        pass
    try:
        import backend.main as _main

        wc_dir = getattr(_main, "_DISK_CACHE_DIR", "") or ""
        if wc_dir and os.path.exists(os.path.join(wc_dir, "inventory_df_variant.parquet")):
            return "disk_parquet"
    except Exception:
        pass
    return "warm_cache"


def _resolve_snapshot_source() -> str:
    try:
        from ..db.forecast_ops_pg import ops_pg_enabled, shared_snapshot_status

        if not ops_pg_enabled():
            return "none"
        st = shared_snapshot_status()
        if st.get("tables", {}).get("sales_by_platform"):
            return "postgres_normalized"
        if st.get("present"):
            return "postgres"
        return "none"
    except Exception:
        return "unknown"


def build_coverage_debug(sess) -> dict[str, Any]:
    import backend.main as _main
    from ..services.session_hydrate import session_hydrate_inflight, session_warm_hydration_complete

    hydrated = False
    try:
        hydrated = session_warm_hydration_complete(sess)
    except Exception:
        hydrated = frame_row_count("sales_df", sess) > 0 and frame_row_count(
            "inventory_df_variant", sess
        ) > 0

    mat_status: dict[str, Any] = {}
    try:
        from ..db.forecast_sales_materializations import materialization_status

        mat_status = materialization_status()
    except Exception:
        pass

    pg_tables: dict[str, Any] = {}
    try:
        from ..db.forecast_ops_tables import tables_status

        pg_tables = tables_status()
    except Exception:
        pass

    session_id = getattr(sess, "_session_id", "") or ""
    loaded_at = getattr(_main, "_warm_cache_loaded_at", None)

    return {
        "source": {
            "sales": _resolve_sales_source(sess),
            "inventory": _resolve_inventory_source(sess),
            "snapshot": _resolve_snapshot_source(),
            "shared_frames": shared_frames_enabled(),
        },
        "session": {
            "hydrated": hydrated,
            "shared_frames": session_uses_shared_frames(sess),
            "sales_rows": frame_row_count("sales_df", sess),
            "inventory_rows": frame_row_count("inventory_df_variant", sess),
            "mtr_rows": frame_row_count("mtr_df", sess),
            "myntra_rows": frame_row_count("myntra_df", sess),
            "meesho_rows": frame_row_count("meesho_df", sess),
            "flipkart_rows": frame_row_count("flipkart_df", sess),
            "snapdeal_rows": frame_row_count("snapdeal_df", sess),
            "warm_cache_gen": int(getattr(sess, "_warm_cache_gen", 0) or 0),
        },
        "warm_cache": {
            "loaded": bool(_main._warm_cache),
            "generation": int(getattr(_main, "_warm_cache_generation", 0) or 0),
            "timestamp": loaded_at.isoformat() if loaded_at else None,
            "keys": list((_main._warm_cache or {}).keys()),
            "sales_rows": frame_row_count("sales_df", None),
            "inventory_rows": frame_row_count("inventory_df_variant", None),
        },
        "platforms": _platform_flags(sess),
        "postgres": {
            "tables": pg_tables,
            "materializations": mat_status,
        },
        "hydration": {
            "inflight": session_hydrate_inflight(session_id),
        },
    }

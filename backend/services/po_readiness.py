"""PO page readiness — separate from full-system background job status."""
from __future__ import annotations

from typing import Any

from ..models.schemas import CoverageResponse

PO_MIN_SALES_ROWS = 1_000_000
PO_MIN_INVENTORY_ROWS = 5_000

OPERATIONAL_DATASET_KEYS = (
    "sku_mapping",
    "mtr",
    "sales",
    "inventory",
    "myntra",
    "meesho",
    "flipkart",
    "snapdeal",
)

_BACKGROUND_JOB_ATTRS: tuple[tuple[str, str], ...] = (
    ("sales_rebuild", "sales_rebuild_status"),
    ("session_restore", "session_restore_status"),
    ("daily_auto_ingest", "daily_auto_ingest_status"),
    ("inventory_upload", "inventory_upload_status"),
    ("daily_inventory_upload", "daily_inventory_upload_status"),
    ("tier1_bulk", "tier1_bulk_status"),
    ("returns_import", "returns_import_status"),
)

_CRITICAL_RESTORE_ATTRS = (
    "session_restore_status",
    "inventory_upload_status",
)


def operational_data_complete(cov: CoverageResponse) -> bool:
    return all(bool(getattr(cov, key, False)) for key in OPERATIONAL_DATASET_KEYS)


def data_row_floors_met(cov: CoverageResponse) -> bool:
    return (
        bool(cov.sales)
        and bool(cov.inventory)
        and int(cov.sales_rows or 0) >= PO_MIN_SALES_ROWS
        and int(cov.inventory_rows or 0) >= PO_MIN_INVENTORY_ROWS
    )


def critical_restore_running(sess) -> bool:
    if sess is None:
        return False
    return any(
        str(getattr(sess, attr, "idle") or "idle") == "running"
        for attr in _CRITICAL_RESTORE_ATTRS
    )


def background_tasks_running(sess) -> dict[str, bool]:
    if sess is None:
        return {}
    out: dict[str, bool] = {}
    for name, attr in _BACKGROUND_JOB_ATTRS:
        if str(getattr(sess, attr, "idle") or "idle") == "running":
            out[name] = True
    return out


def background_job_names(sess) -> list[str]:
    return list(background_tasks_running(sess).keys())


def _pg_row_floors() -> tuple[int, int]:
    """Sales + inventory row counts from normalized PostgreSQL tables."""
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled, tables_status

        if not normalized_tables_enabled():
            return 0, 0
        st = tables_status()
        sbp = st.get("sales_by_platform") or {}
        sales = int(sbp.get("unified", 0) or 0)
        if sales <= 0:
            sales = sum(int(v or 0) for k, v in sbp.items() if k != "unified")
        inv = int(st.get("inventory_lines", 0) or 0)
        return sales, inv
    except Exception:
        return 0, 0


def _warm_row_floors() -> tuple[int, int]:
    try:
        import backend.main as _main

        wc = _main._warm_cache or {}
        sales_df = wc.get("sales_df")
        inv_df = wc.get("inventory_df_variant")
        sales = int(len(sales_df)) if sales_df is not None and hasattr(sales_df, "__len__") else 0
        inv = int(len(inv_df)) if inv_df is not None and hasattr(inv_df, "__len__") else 0
        return sales, inv
    except Exception:
        return 0, 0


def _effective_row_floors(cov: CoverageResponse, sess) -> tuple[int, int]:
    from .shared_frames import frame_row_count

    sales = int(cov.sales_rows or frame_row_count("sales_df", sess))
    inv = int(cov.inventory_rows or frame_row_count("inventory_df_variant", sess))
    pg_sales, pg_inv = _pg_row_floors()
    warm_sales, warm_inv = _warm_row_floors()
    return max(sales, pg_sales, warm_sales), max(inv, pg_inv, warm_inv)


def compute_data_ready(cov: CoverageResponse, sess=None) -> bool:
    """Session holds PO essentials — warm cache / PG row floors, not 8/8 session flags alone."""
    sales_rows, inv_rows = (
        _effective_row_floors(cov, sess)
        if sess is not None
        else (int(cov.sales_rows or 0), int(cov.inventory_rows or 0))
    )
    if sales_rows >= PO_MIN_SALES_ROWS and inv_rows >= PO_MIN_INVENTORY_ROWS:
        return True
    pg_sales, pg_inv = _pg_row_floors()
    if pg_sales >= PO_MIN_SALES_ROWS and pg_inv >= PO_MIN_INVENTORY_ROWS:
        return True
    return operational_data_complete(cov) and data_row_floors_met(cov)


def compute_po_ready(sess, cov: CoverageResponse) -> bool:
    """PO Engine may mount — data ready and no critical restore replacing inventory/session."""
    if critical_restore_running(sess):
        return False
    if compute_data_ready(cov, sess):
        return True
    sales_rows, inv_rows = _effective_row_floors(cov, sess)
    return sales_rows >= PO_MIN_SALES_ROWS and inv_rows >= PO_MIN_INVENTORY_ROWS


def _resolve_data_source(sess) -> str:
    try:
        from .coverage_debug import _resolve_sales_source

        return _resolve_sales_source(sess)
    except Exception:
        return "warm_cache"


def _hydration_label(sess, session_id: str = "") -> str:
    try:
        from .session_hydrate import session_hydrate_inflight, session_warm_hydration_complete

        sid = session_id or getattr(sess, "_session_id", "") or ""
        if session_hydrate_inflight(sid):
            return "inflight"
        if session_warm_hydration_complete(sess):
            return "complete"
        return "partial"
    except Exception:
        return "unknown"


def build_po_readiness(sess, cov: CoverageResponse, *, session_id: str = "") -> dict[str, Any]:
    pg_sales, pg_inv = _pg_row_floors()
    sales_rows, inventory_rows = _effective_row_floors(cov, sess)
    data_source = _resolve_data_source(sess)
    if pg_sales >= PO_MIN_SALES_ROWS and data_source == "warm_cache":
        data_source = "postgres_normalized"
    return {
        "po_ready": compute_po_ready(sess, cov),
        "data_ready": compute_data_ready(cov, sess),
        "sales_rows": sales_rows,
        "inventory_rows": inventory_rows,
        "data_source": data_source,
        "hydration": _hydration_label(sess, session_id=session_id),
        "background_jobs": background_job_names(sess),
        "background_tasks": background_tasks_running(sess),
        "critical_restore_running": critical_restore_running(sess),
    }


def _effective_row_floors_light(cov: CoverageResponse, sess) -> tuple[int, int]:
    """Row floors for light coverage polls — skip slow PostgreSQL GROUP BY."""
    from .shared_frames import frame_row_count

    sales = int(cov.sales_rows or frame_row_count("sales_df", sess))
    inv = int(cov.inventory_rows or frame_row_count("inventory_df_variant", sess))
    warm_sales, warm_inv = _warm_row_floors()
    return max(sales, warm_sales), max(inv, warm_inv)


def _augment_coverage_light(sess, cov: CoverageResponse) -> CoverageResponse:
    """Fast readiness for GET /coverage?light=1 — no PG scans or unified sales Source walk."""
    from .intelligence_readiness import (
        _platform_flags_ready,
        _tier3_all_platforms_have_uploads,
        hydration_complete,
        hydration_inflight,
    )

    data = cov.model_dump()
    sid = getattr(sess, "_session_id", "") or ""
    tier3 = _tier3_all_platforms_have_uploads()
    flags_ok = _platform_flags_ready(cov)
    sales_rows, inv_rows = _effective_row_floors_light(cov, sess)
    sales_ok = bool(cov.sales) and (sales_rows > 0 or tier3 or bool(cov.mtr))
    inv_ok = bool(cov.inventory) and inv_rows > 0
    platforms = flags_ok and (sales_rows > 0 or tier3)
    hydrate_inflight = hydration_inflight(sid, sess)

    data["background_tasks"] = background_tasks_running(sess)
    data["critical_restore_running"] = critical_restore_running(sess)
    data["platforms_loaded"] = platforms
    data["hydration_complete"] = hydration_complete(sess, sid)
    data["dashboard_ready"] = platforms and sales_ok
    data["intelligence_ready"] = (
        not hydrate_inflight and sales_ok and inv_ok and platforms
    )
    data["data_ready"] = (
        sales_rows >= PO_MIN_SALES_ROWS and inv_rows >= PO_MIN_INVENTORY_ROWS
    ) or (operational_data_complete(cov) and data_row_floors_met(cov))
    if critical_restore_running(sess):
        data["po_ready"] = False
    elif data["data_ready"]:
        data["po_ready"] = True
    else:
        data["po_ready"] = (
            sales_rows >= PO_MIN_SALES_ROWS and inv_rows >= PO_MIN_INVENTORY_ROWS
        )
    return CoverageResponse(**data)


def augment_coverage(sess, cov: CoverageResponse, *, light: bool = False) -> CoverageResponse:
    """Attach data_ready / po_ready / background_tasks to coverage payload."""
    if light:
        return _augment_coverage_light(sess, cov)

    from .intelligence_readiness import (
        build_intelligence_readiness,
        dashboard_gate_ready,
        hydration_complete,
        platform_frames_available,
    )

    data = cov.model_dump()
    data["data_ready"] = compute_data_ready(cov, sess)
    data["po_ready"] = compute_po_ready(sess, cov)
    data["background_tasks"] = background_tasks_running(sess)
    data["critical_restore_running"] = critical_restore_running(sess)
    sid = getattr(sess, "_session_id", "") or ""
    data["platforms_loaded"] = platform_frames_available(sess, cov)
    data["hydration_complete"] = hydration_complete(sess, sid)
    intel = build_intelligence_readiness(sess, cov, session_id=sid)
    data["intelligence_ready"] = bool(intel.get("intelligence_ready"))
    data["dashboard_ready"] = dashboard_gate_ready(sess, cov, session_id=sid)
    return CoverageResponse(**data)

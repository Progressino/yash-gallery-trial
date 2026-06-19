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


def compute_data_ready(cov: CoverageResponse) -> bool:
    """Session holds PO essentials (8/8 flags + row floors) — ignores background jobs."""
    if operational_data_complete(cov) and data_row_floors_met(cov):
        return True
    pg_sales, pg_inv = _pg_row_floors()
    if pg_sales >= PO_MIN_SALES_ROWS and pg_inv >= PO_MIN_INVENTORY_ROWS:
        return operational_data_complete(cov)
    return False


def compute_po_ready(sess, cov: CoverageResponse) -> bool:
    """PO Engine may mount — data ready and no critical restore replacing inventory/session."""
    if critical_restore_running(sess):
        return False
    if compute_data_ready(cov):
        return True
    pg_sales, pg_inv = _pg_row_floors()
    return pg_sales >= PO_MIN_SALES_ROWS and pg_inv >= PO_MIN_INVENTORY_ROWS


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
    from .shared_frames import frame_row_count

    pg_sales, pg_inv = _pg_row_floors()
    sales_rows = int(cov.sales_rows or frame_row_count("sales_df", sess))
    inventory_rows = int(cov.inventory_rows or frame_row_count("inventory_df_variant", sess))
    if sales_rows < PO_MIN_SALES_ROWS and pg_sales > sales_rows:
        sales_rows = pg_sales
    if inventory_rows < PO_MIN_INVENTORY_ROWS and pg_inv > inventory_rows:
        inventory_rows = pg_inv
    data_source = _resolve_data_source(sess)
    if pg_sales >= PO_MIN_SALES_ROWS and data_source == "warm_cache":
        data_source = "postgres_normalized"
    return {
        "po_ready": compute_po_ready(sess, cov),
        "data_ready": compute_data_ready(cov),
        "sales_rows": sales_rows,
        "inventory_rows": inventory_rows,
        "data_source": data_source,
        "hydration": _hydration_label(sess, session_id=session_id),
        "background_jobs": background_job_names(sess),
        "background_tasks": background_tasks_running(sess),
        "critical_restore_running": critical_restore_running(sess),
    }


def augment_coverage(sess, cov: CoverageResponse) -> CoverageResponse:
    """Attach data_ready / po_ready / background_tasks to coverage payload."""
    from .intelligence_readiness import (
        build_intelligence_readiness,
        dashboard_gate_ready,
        hydration_complete,
        platform_frames_available,
    )

    data = cov.model_dump()
    data["data_ready"] = compute_data_ready(cov)
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

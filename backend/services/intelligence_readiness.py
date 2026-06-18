"""Intelligence dashboard readiness — stronger than operational 8/8 alone."""
from __future__ import annotations

from typing import Any

from ..models.schemas import CoverageResponse

PLATFORM_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("amazon", "mtr_df", "mtr", "mtr_rows"),
    ("myntra", "myntra_df", "myntra", "myntra_rows"),
    ("meesho", "meesho_df", "meesho", "meesho_rows"),
    ("flipkart", "flipkart_df", "flipkart", "flipkart_rows"),
    ("snapdeal", "snapdeal_df", "snapdeal", "snapdeal_rows"),
)

_PLATFORM_PG_KEYS = tuple(p[0] for p in PLATFORM_SPECS)
_SOURCE_ALIASES: dict[str, frozenset[str]] = {
    "amazon": frozenset({"amazon", "mtr", "amz"}),
    "myntra": frozenset({"myntra", "myn"}),
    "meesho": frozenset({"meesho", "mee"}),
    "flipkart": frozenset({"flipkart", "fk"}),
    "snapdeal": frozenset({"snapdeal", "sd"}),
}


def hydration_inflight(session_id: str = "", sess=None) -> bool:
    try:
        from .session_hydrate import session_hydrate_inflight

        sid = session_id or getattr(sess, "_session_id", "") or ""
        return session_hydrate_inflight(sid)
    except Exception:
        return False


def hydration_complete(sess, session_id: str = "") -> bool:
    if hydration_inflight(session_id, sess):
        return False
    try:
        from .session_hydrate import session_warm_hydration_complete

        return session_warm_hydration_complete(sess)
    except Exception:
        return False


def sales_available(sess, cov: CoverageResponse) -> bool:
    from .shared_frames import frame_row_count, session_sales_df

    rows = int(cov.sales_rows or frame_row_count("sales_df", sess))
    if rows <= 0:
        pg = _pg_platform_sales_counts()
        if int(pg.get("unified", 0) or pg.get("amazon", 0) or 0) > 0:
            return True
        return False
    return bool(cov.sales) or not session_sales_df(sess).empty


def inventory_available(sess, cov: CoverageResponse) -> bool:
    from .shared_frames import frame_row_count, session_inventory_variant

    rows = int(cov.inventory_rows or frame_row_count("inventory_df_variant", sess))
    if rows <= 0:
        try:
            from ..db.forecast_ops_tables import load_inventory_dataframe

            inv = load_inventory_dataframe()
            if inv is not None and not inv.empty:
                return True
        except Exception:
            pass
        return False
    inv = session_inventory_variant(sess)
    return bool(cov.inventory) and (not inv.empty or rows > 0)


def _pg_platform_sales_counts() -> dict[str, int]:
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled, tables_status

        if not normalized_tables_enabled():
            return {}
        st = tables_status()
        return dict(st.get("sales_by_platform") or {})
    except Exception:
        return {}


def _tier3_all_platforms_have_uploads() -> bool:
    try:
        from .daily_store import get_summary

        summary = get_summary() or {}
        return all(
            int((summary.get(plat) or {}).get("file_count") or 0) > 0
            for plat in _PLATFORM_PG_KEYS
        )
    except Exception:
        return False


def _unified_sales_covers_platforms(sess) -> bool:
    from .shared_frames import session_sales_df

    sales = session_sales_df(sess)
    if sales is None or sales.empty or "Source" not in sales.columns:
        return False
    sources = {
        str(v).strip().lower()
        for v in sales["Source"].dropna().astype(str).unique()
    }
    found = 0
    for aliases in _SOURCE_ALIASES.values():
        if sources & aliases:
            found += 1
    return found >= 5


def platform_frames_available(sess, cov: CoverageResponse) -> bool:
    from .shared_frames import frame_row_count

    if all(frame_row_count(attr, sess) > 0 for _, attr, _, _ in PLATFORM_SPECS):
        return True

    pg = _pg_platform_sales_counts()
    if pg and all(int(pg.get(plat, 0) or 0) > 0 for plat in _PLATFORM_PG_KEYS):
        return True

    if int(pg.get("unified", 0) or 0) >= 100_000 and _unified_sales_covers_platforms(sess):
        return True

    if all(
        bool(getattr(cov, flag, False)) and int(getattr(cov, rows_key, 0) or 0) > 0
        for _, _, flag, rows_key in PLATFORM_SPECS
    ):
        return True

    if _tier3_all_platforms_have_uploads():
        return True

    return _unified_sales_covers_platforms(sess)


def intelligence_ready(sess, cov: CoverageResponse, *, session_id: str = "") -> bool:
    if hydration_inflight(session_id, sess):
        return False
    return (
        sales_available(sess, cov)
        and inventory_available(sess, cov)
        and platform_frames_available(sess, cov)
    )


def dashboard_gate_ready(sess, cov: CoverageResponse, *, session_id: str = "") -> bool:
    """Dashboard may fetch aggregates — 8/8 + platform history + hydration settled."""
    from .po_readiness import operational_data_complete

    if not operational_data_complete(cov):
        return False
    if not hydration_complete(sess, session_id):
        return False
    return platform_frames_available(sess, cov) and sales_available(sess, cov)


def build_intelligence_readiness(sess, cov: CoverageResponse, *, session_id: str = "") -> dict[str, Any]:
    from .coverage_debug import _resolve_sales_source
    from .po_readiness import background_job_names, background_tasks_running
    from .shared_frames import frame_row_count

    bg = background_tasks_running(sess)
    return {
        "intelligence_ready": intelligence_ready(sess, cov, session_id=session_id),
        "dashboard_ready": dashboard_gate_ready(sess, cov, session_id=session_id),
        "data_ready": bool(getattr(cov, "data_ready", False)),
        "platforms_loaded": platform_frames_available(sess, cov),
        "hydration_complete": hydration_complete(sess, session_id),
        "hydration_inflight": hydration_inflight(session_id, sess),
        "sales_available": sales_available(sess, cov),
        "inventory_available": inventory_available(sess, cov),
        "sales_rows": int(cov.sales_rows or frame_row_count("sales_df", sess)),
        "inventory_rows": int(cov.inventory_rows or frame_row_count("inventory_df_variant", sess)),
        "platform_rows": {
            plat: int(getattr(cov, rows_key, 0) or frame_row_count(attr, sess))
            for plat, attr, _flag, rows_key in PLATFORM_SPECS
        },
        "data_source": _resolve_sales_source(sess),
        "background_jobs": background_job_names(sess),
        "background_tasks": bg,
    }

"""Quarterly history warmup — rebuild shallow sales_df and pre-cache PO quarter columns."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Bump when quarterly payload shape / history rules change (invalidates session cache).
QUARTERLY_CACHE_SCHEMA = 3


def quarterly_cache_key(group_by_parent: bool, n_quarters: int) -> tuple:
    return (QUARTERLY_CACHE_SCHEMA, bool(group_by_parent), int(n_quarters))


# Indian FY quarters need ~2 years of shipments for 8 quarter columns.
_MIN_SPAN_DAYS_FOR_QUARTERLY = 540
_MIN_PLATFORM_ROWS_FOR_REBUILD = 5000

_PLATFORM_ATTRS = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)


def sales_df_span_days(sales_df: pd.DataFrame) -> int:
    if sales_df is None or sales_df.empty or "TxnDate" not in sales_df.columns:
        return 0
    t = pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    t = t.dropna()
    if t.empty:
        return 0
    return int((t.max() - t.min()).days)


def platform_frames_span_days(sess) -> int:
    """Max calendar span across session platform bulk frames."""
    from .platform_session_window import platform_date_column

    best = 0
    for _plat, attr in _PLATFORM_ATTRS:
        df = getattr(sess, attr, None)
        if df is None or getattr(df, "empty", True):
            continue
        col = platform_date_column(df)
        if not col:
            continue
        t = pd.to_datetime(df[col], errors="coerce").dropna()
        if t.empty:
            continue
        best = max(best, int((t.max() - t.min()).days))
    return best


def quarterly_restore_months(n_quarters: int) -> int | None:
    """
    Tier-3 months to load before building quarterly columns.
    ``None`` = full platform history (recommended for 8-quarter PO view).
    """
    raw = (os.environ.get("QUARTERLY_RESTORE_MONTHS") or "").strip()
    if raw == "0":
        return None
    if raw:
        try:
            v = int(raw)
            return None if v <= 0 else v
        except ValueError:
            pass
    # Default: full Tier-3 history (bulk Tier-1 often uses old file_date metadata).
    return None


def restore_platform_history_for_quarterly(sess, n_quarters: int = 8) -> bool:
    """
    Merge Tier-3 SQLite uploads into session platform frames deep enough for PO
    quarterly columns (default AUTO_RESTORE is only 12 months).
    """
    from ..services.daily_store import load_platform_data, merge_platform_data
    from .sales import build_sales_df

    months = quarterly_restore_months(n_quarters)
    changed = False
    for platform, attr in _PLATFORM_ATTRS:
        df = load_platform_data(platform, months=months, dedup=False, max_files=None)
        if df.empty:
            continue
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty"):
            cur = pd.DataFrame()
        merged = merge_platform_data(cur, df, platform)
        if len(merged) != len(cur) or (cur.empty and not merged.empty):
            setattr(sess, attr, merged)
            changed = True

    if not changed:
        return False

    try:
        from ..services.sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass

    sess.sales_df = build_sales_df(
        getattr(sess, "mtr_df", pd.DataFrame()),
        getattr(sess, "myntra_df", pd.DataFrame()),
        getattr(sess, "meesho_df", pd.DataFrame()),
        getattr(sess, "flipkart_df", pd.DataFrame()),
        sess.sku_mapping or {},
        snapdeal_df=getattr(sess, "snapdeal_df", pd.DataFrame()),
        return_overlay_df=getattr(sess, "po_return_overlay_df", None),
    )
    sess._quarterly_cache.clear()
    logger.info(
        "Quarterly: merged Tier-3 history (months=%s); sales_df span=%d days",
        months if months is not None else "all",
        sales_df_span_days(sess.sales_df),
    )
    return True


def ensure_sales_history_for_quarterly(sess) -> bool:
    """
    When unified sales_df only has recent daily uploads but platform bulk frames
    (MTR / Myntra / …) carry multi-year history, rebuild sales_df once per session
    so PO quarterly columns are populated automatically.
    """
    if getattr(sess, "_quarterly_sales_rebuilt", False):
        return False

    span = sales_df_span_days(getattr(sess, "sales_df", None))
    plat_span = platform_frames_span_days(sess)
    if span >= _MIN_SPAN_DAYS_FOR_QUARTERLY and plat_span <= span + 30:
        return False

    mtr = getattr(sess, "mtr_df", None)
    myntra = getattr(sess, "myntra_df", None)
    meesho = getattr(sess, "meesho_df", None)
    flipkart = getattr(sess, "flipkart_df", None)
    snapdeal = getattr(sess, "snapdeal_df", None)

    bulk_rows = sum(
        int(len(df))
        for df in (mtr, myntra, meesho, flipkart, snapdeal)
        if df is not None and not getattr(df, "empty", True)
    )
    if bulk_rows < _MIN_PLATFORM_ROWS_FOR_REBUILD and plat_span < _MIN_SPAN_DAYS_FOR_QUARTERLY:
        return False

    from .sales import build_sales_df

    logger.info(
        "Quarterly: sales_df span=%d platform_span=%d (< %d) with %s platform rows — rebuilding unified sales",
        span,
        plat_span,
        _MIN_SPAN_DAYS_FOR_QUARTERLY,
        f"{bulk_rows:,}",
    )
    rebuilt = build_sales_df(
        mtr if mtr is not None else pd.DataFrame(),
        myntra if myntra is not None else pd.DataFrame(),
        meesho if meesho is not None else pd.DataFrame(),
        flipkart if flipkart is not None else pd.DataFrame(),
        sess.sku_mapping or {},
        snapdeal_df=snapdeal if snapdeal is not None else pd.DataFrame(),
        return_overlay_df=getattr(sess, "po_return_overlay_df", None),
    )
    new_span = sales_df_span_days(rebuilt)
    if rebuilt.empty or (new_span <= span + 14 and plat_span <= span + 30):
        logger.warning(
            "Quarterly sales rebuild did not widen history (old=%d new=%d platform=%d rows=%d)",
            span,
            new_span,
            plat_span,
            len(rebuilt),
        )
        return False

    sess.sales_df = rebuilt
    sess._quarterly_sales_rebuilt = True
    sess._quarterly_cache.clear()
    logger.info(
        "Quarterly: rebuilt sales_df (%s rows, span=%d days)",
        f"{len(rebuilt):,}",
        new_span,
    )
    return True


def build_quarterly_payload(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> dict[str, Any]:
    from ..routers.data import _restore_daily_if_needed
    from .po_engine import calculate_quarterly_history

    _restore_daily_if_needed(sess)
    restore_platform_history_for_quarterly(sess, n_quarters=n_quarters)
    ensure_sales_history_for_quarterly(sess)

    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df,
        myntra_df=sess.myntra_df,
        meesho_df=sess.meesho_df,
        flipkart_df=sess.flipkart_df,
        snapdeal_df=sess.snapdeal_df,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if pivot.empty:
        return {"loaded": False, "rows": []}
    return {
        "loaded": True,
        "columns": list(pivot.columns),
        "rows": pivot.fillna(0).to_dict("records"),
    }


def warmup_quarterly_cache(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> Tuple[dict[str, Any], bool]:
    """Populate session quarterly cache; returns (payload, sales_was_rebuilt)."""
    cache_key = quarterly_cache_key(group_by_parent, n_quarters)
    if cache_key in sess._quarterly_cache and sess._quarterly_cache[cache_key].get("loaded"):
        return sess._quarterly_cache[cache_key], False

    was_rebuilt = getattr(sess, "_quarterly_sales_rebuilt", False)
    result = build_quarterly_payload(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    sess._quarterly_cache[cache_key] = result
    return result, bool(getattr(sess, "_quarterly_sales_rebuilt", False) and not was_rebuilt)

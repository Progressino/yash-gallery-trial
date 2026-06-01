"""Quarterly history warmup — rebuild shallow sales_df and pre-cache PO quarter columns."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Indian FY quarters need ~2 years of shipments for 8 quarter columns.
_MIN_SPAN_DAYS_FOR_QUARTERLY = 540
_MIN_PLATFORM_ROWS_FOR_REBUILD = 5000


def sales_df_span_days(sales_df: pd.DataFrame) -> int:
    if sales_df is None or sales_df.empty or "TxnDate" not in sales_df.columns:
        return 0
    t = pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    t = t.dropna()
    if t.empty:
        return 0
    return int((t.max() - t.min()).days)


def ensure_sales_history_for_quarterly(sess) -> bool:
    """
    When unified sales_df only has recent daily uploads but platform bulk frames
    (MTR / Myntra / …) carry multi-year history, rebuild sales_df once per session
    so PO quarterly columns are populated automatically.
    """
    if getattr(sess, "_quarterly_sales_rebuilt", False):
        return False

    span = sales_df_span_days(getattr(sess, "sales_df", None))
    if span >= _MIN_SPAN_DAYS_FOR_QUARTERLY:
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
    if bulk_rows < _MIN_PLATFORM_ROWS_FOR_REBUILD:
        return False

    from .sales import build_sales_df

    logger.info(
        "Quarterly: sales_df span=%d days (< %d) with %s platform rows — rebuilding unified sales",
        span,
        _MIN_SPAN_DAYS_FOR_QUARTERLY,
        f"{bulk_rows:,}",
    )
    rebuilt = build_sales_df(
        mtr if mtr is not None else pd.DataFrame(),
        myntra if myntra is not None else pd.DataFrame(),
        meesho if meesho is not None else pd.DataFrame(),
        flipkart if flipkart is not None else pd.DataFrame(),
        sess.sku_mapping or {},
        snapdeal=snapdeal if snapdeal is not None else pd.DataFrame(),
        return_overlay_df=getattr(sess, "po_return_overlay_df", None),
    )
    new_span = sales_df_span_days(rebuilt)
    if rebuilt.empty or new_span <= span + 14:
        logger.warning(
            "Quarterly sales rebuild did not widen history (old=%d new=%d rows=%d)",
            span,
            new_span,
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

    _boot = sess.sales_df.empty or "Sku" not in sess.sales_df.columns
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if _boot and not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if _boot and not sess.myntra_df.empty else None,
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
    rebuilt = ensure_sales_history_for_quarterly(sess)
    cache_key = (group_by_parent, n_quarters)
    if cache_key in sess._quarterly_cache and sess._quarterly_cache[cache_key].get("loaded"):
        return sess._quarterly_cache[cache_key], rebuilt

    result = build_quarterly_payload(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    sess._quarterly_cache[cache_key] = result
    return result, rebuilt

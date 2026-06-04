"""Quarterly history warmup — fast windowed Tier-3 load + session cache."""
from __future__ import annotations

import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Bump when quarterly payload shape / history rules change (invalidates session cache).
QUARTERLY_CACHE_SCHEMA = 4


def quarterly_cache_key(group_by_parent: bool, n_quarters: int) -> tuple:
    return (QUARTERLY_CACHE_SCHEMA, bool(group_by_parent), int(n_quarters))


_MIN_SPAN_DAYS_FOR_QUARTERLY = 540
_MIN_PLATFORM_ROWS_FOR_REBUILD = 5000

_PLATFORM_ATTRS = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)


def quarterly_report_window(n_quarters: int = 8) -> tuple[str, str]:
    """Calendar window covering ``n_quarters`` Indian FY columns (+ buffer)."""
    end = datetime.date.today()
    days = max(750, int(n_quarters) * 92 + 90)
    start = end - datetime.timedelta(days=days)
    return start.isoformat(), end.isoformat()


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


def _platform_frame_min_date(df: pd.DataFrame) -> str:
    from .platform_session_window import platform_date_column

    if df is None or df.empty:
        return ""
    col = platform_date_column(df)
    if not col:
        return ""
    t = pd.to_datetime(df[col], errors="coerce").dropna()
    if t.empty:
        return ""
    return str(t.min().normalize())[:10]


def _platform_frame_covers_start(df: pd.DataFrame, start_date: str) -> bool:
    if not start_date or len(start_date) != 10:
        return False
    mn = _platform_frame_min_date(df)
    if not mn:
        return False
    return mn <= start_date


def _session_platforms_need_hydrate(sess, start_date: str, n_quarters: int) -> bool:
    min_span = int(n_quarters) * 92 + 60
    if platform_frames_span_days(sess) < min_span - 45:
        return True
    for _plat, attr in _PLATFORM_ATTRS:
        df = getattr(sess, attr, None)
        if df is None or not hasattr(df, "empty"):
            df = pd.DataFrame()
        if not _platform_frame_covers_start(df, start_date):
            return True
    return False


def hydrate_platform_frames_for_quarterly(sess, n_quarters: int = 8) -> bool:
    """
    Merge only Tier-3 blobs overlapping the quarterly window (not full SQLite scan).
    Does not rebuild unified ``sales_df`` — quarterly math reads platform frames directly.
    """
    from ..services.daily_store import (
        load_platform_data,
        load_platform_data_for_report_range,
        merge_platform_data,
    )

    start, end = quarterly_report_window(n_quarters)
    tag = (QUARTERLY_CACHE_SCHEMA, start, end)
    if getattr(sess, "_quarterly_hydrate_tag", None) == tag:
        return False
    if not _session_platforms_need_hydrate(sess, start, n_quarters):
        sess._quarterly_hydrate_tag = tag
        return False

    changed = False
    for platform, attr in _PLATFORM_ATTRS:
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty"):
            cur = pd.DataFrame()
        if _platform_frame_covers_start(cur, start):
            continue
        chunk = load_platform_data_for_report_range(
            platform,
            start,
            end,
            dedup=False,
            columns_only=True,
        )
        if chunk.empty:
            months = max(27, int(n_quarters) * 3 + 2)
            chunk = load_platform_data(
                platform,
                months=months,
                dedup=False,
                max_files=80,
            )
        if chunk.empty:
            continue
        merged = merge_platform_data(cur, chunk, platform)
        if len(merged) != len(cur) or (cur.empty and not merged.empty):
            setattr(sess, attr, merged)
            changed = True

    sess._quarterly_hydrate_tag = tag
    if changed:
        sess._quarterly_cache.clear()
        logger.info(
            "Quarterly: hydrated platform frames for %s..%s (changed=%s)",
            start,
            end,
            changed,
        )
    return changed


def _ensure_session_operational_frames(sess) -> None:
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(
                sess, _main._warm_cache_generation
            )
    except Exception:
        pass
    try:
        from ..services.sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass


def ensure_sales_history_for_quarterly(sess) -> bool:
    """
    Rebuild unified sales_df once when platform frames carry deep history but sales_df does not.
    Quarterly pivot does not require this, but PO calculate and exports still use sales_df.
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
        return False

    sess.sales_df = rebuilt
    sess._quarterly_sales_rebuilt = True
    sess._quarterly_cache.clear()
    return True


def build_quarterly_payload(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> dict[str, Any]:
    from .po_engine import calculate_quarterly_history

    _ensure_session_operational_frames(sess)

    has_any = any(
        not getattr(sess, attr, pd.DataFrame()).empty for _, attr in _PLATFORM_ATTRS
    )
    if not has_any:
        from ..routers.data import _restore_daily_if_needed

        _restore_daily_if_needed(sess)

    hydrate_platform_frames_for_quarterly(sess, n_quarters=n_quarters)

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


def try_build_quarterly_payload_sync(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    timeout_sec: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Bounded-time sync build; returns None on timeout (caller should start background job)."""
    raw = (os.environ.get("QUARTERLY_SYNC_TIMEOUT_SEC") or "22").strip()
    try:
        limit = float(timeout_sec if timeout_sec is not None else raw)
    except ValueError:
        limit = 22.0
    limit = max(5.0, min(limit, 120.0))
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            build_quarterly_payload,
            sess,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
        )
        try:
            return fut.result(timeout=limit)
        except FuturesTimeout:
            return None


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

    result = build_quarterly_payload(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    sess._quarterly_cache[cache_key] = result
    return result, False


# Back-compat alias used in tests
def restore_platform_history_for_quarterly(sess, n_quarters: int = 8) -> bool:
    return hydrate_platform_frames_for_quarterly(sess, n_quarters=n_quarters)


def quarterly_restore_months(n_quarters: int) -> int | None:
    """Deprecated: windowed hydrate uses ``quarterly_report_window`` instead."""
    _ = n_quarters
    return None

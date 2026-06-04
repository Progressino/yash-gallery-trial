"""Quarterly history — session/warm-cache fast path + streaming Tier-3 aggregate."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Bump when quarterly payload shape / history rules change (invalidates caches).
QUARTERLY_CACHE_SCHEMA = 5


def quarterly_cache_key(group_by_parent: bool, n_quarters: int) -> tuple:
    return (QUARTERLY_CACHE_SCHEMA, bool(group_by_parent), int(n_quarters))


_PLATFORM_ATTRS = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)


def quarterly_report_window(n_quarters: int = 8) -> tuple[str, str]:
    import datetime

    end = datetime.date.today()
    days = max(750, int(n_quarters) * 92 + 90)
    start = end - datetime.timedelta(days=days)
    return start.isoformat(), end.isoformat()


def sales_df_span_days(sales_df: pd.DataFrame) -> int:
    if sales_df is None or sales_df.empty or "TxnDate" not in sales_df.columns:
        return 0
    t = pd.to_datetime(sales_df["TxnDate"], errors="coerce").dropna()
    if t.empty:
        return 0
    return int((t.max() - t.min()).days)


def platform_frames_span_days(sess) -> int:
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


def _session_has_platform_rows(sess) -> bool:
    return any(
        not getattr(sess, attr, pd.DataFrame()).empty for _, attr in _PLATFORM_ATTRS
    )


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
        from .sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass


def _pivot_from_session_frames(
    sess,
    *,
    group_by_parent: bool,
    n_quarters: int,
    include_sales: bool = True,
) -> pd.DataFrame:
    from .po_engine import calculate_quarterly_history

    sales = (
        getattr(sess, "sales_df", pd.DataFrame())
        if include_sales
        else pd.DataFrame()
    )
    return calculate_quarterly_history(
        sales_df=sales if sales is not None else pd.DataFrame(),
        mtr_df=getattr(sess, "mtr_df", pd.DataFrame()),
        myntra_df=getattr(sess, "myntra_df", pd.DataFrame()),
        meesho_df=getattr(sess, "meesho_df", pd.DataFrame()),
        flipkart_df=getattr(sess, "flipkart_df", pd.DataFrame()),
        snapdeal_df=getattr(sess, "snapdeal_df", pd.DataFrame()),
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )


def _pivot_to_payload(pivot: pd.DataFrame) -> dict[str, Any]:
    if pivot is None or pivot.empty:
        return {"loaded": False, "rows": []}
    return {
        "loaded": True,
        "columns": list(pivot.columns),
        "rows": pivot.fillna(0).to_dict("records"),
    }


def _build_via_streaming(
    sku_mapping: dict,
    start: str,
    end: str,
    *,
    group_by_parent: bool,
    n_quarters: int,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    acquire_memory_lock: bool = True,
) -> dict[str, Any]:
    from .po_quarterly_fast import calculate_quarterly_from_tier3_streaming

    def _run() -> dict[str, Any]:
        pivot = calculate_quarterly_from_tier3_streaming(
            sku_mapping or None,
            start,
            end,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
            progress_cb=progress_cb,
        )
        return _pivot_to_payload(pivot)

    if not acquire_memory_lock:
        return _run()

    from .concurrency import _UPLOAD_MEMORY_LOCK

    if not _UPLOAD_MEMORY_LOCK.acquire(timeout=120):
        logger.warning("Quarterly streaming skipped: memory lock busy")
        return {"loaded": False, "rows": []}
    try:
        return _run()
    finally:
        _UPLOAD_MEMORY_LOCK.release()


def build_quarterly_payload(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Never merge Tier-3 into session — warm-cache frames or streaming aggregate only."""
    _ensure_session_operational_frames(sess)

    min_span = int(n_quarters) * 92 + 45
    plat_span = platform_frames_span_days(sess)
    has_plat = _session_has_platform_rows(sess)
    sales_span = sales_df_span_days(getattr(sess, "sales_df", None))

    if has_plat and plat_span >= min_span - 60:
        if progress_cb:
            progress_cb(40, "Using saved platform history…")
        pivot = _pivot_from_session_frames(
            sess,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
            include_sales=False,
        )
        out = _pivot_to_payload(pivot)
        if out.get("loaded") and out.get("rows"):
            return out

    if has_plat or sales_span > 0:
        if progress_cb:
            progress_cb(25, "Using session sales history…")
        pivot = _pivot_from_session_frames(
            sess, group_by_parent=group_by_parent, n_quarters=n_quarters
        )
        out = _pivot_to_payload(pivot)
        if out.get("loaded") and out.get("rows"):
            return out

    start, end = quarterly_report_window(n_quarters)
    mapping = sess.sku_mapping or {}
    if progress_cb:
        progress_cb(12, "Streaming uploads (memory-safe)…")
    return _build_via_streaming(
        mapping,
        start,
        end,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
        progress_cb=progress_cb,
    )


def try_build_quarterly_payload_sync(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    timeout_sec: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    raw = (os.environ.get("QUARTERLY_SYNC_TIMEOUT_SEC") or "45").strip()
    try:
        limit = float(timeout_sec if timeout_sec is not None else raw)
    except ValueError:
        limit = 45.0
    limit = max(10.0, min(limit, 90.0))
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
    cache_key = quarterly_cache_key(group_by_parent, n_quarters)
    if cache_key in sess._quarterly_cache and sess._quarterly_cache[cache_key].get("loaded"):
        return sess._quarterly_cache[cache_key], False

    result = build_quarterly_payload(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    sess._quarterly_cache[cache_key] = result
    return result, False


def restore_platform_history_for_quarterly(sess, n_quarters: int = 8) -> bool:
    """No-op — kept for tests; hydration into session is disabled (OOM)."""
    _ = (sess, n_quarters)
    return False


def hydrate_platform_frames_for_quarterly(sess, n_quarters: int = 8) -> bool:
    _ = (sess, n_quarters)
    return False


def quarterly_restore_months(n_quarters: int) -> int | None:
    _ = n_quarters
    return None


def ensure_sales_history_for_quarterly(sess) -> bool:
    _ = sess
    return False


def schedule_shared_quarterly_prewarm() -> None:
    """Deferred pre-build — wait until warm-cache / restore memory lock is free (OOM-safe)."""
    import threading
    import time

    def _go() -> None:
        try:
            from .concurrency import upload_memory_lock_held
            from .po_quarterly_cache import get_shared_quarterly, start_shared_quarterly_build

            # Let warm-cache Phase 1+2 and session restores finish first.
            for _ in range(120):
                if not upload_memory_lock_held():
                    break
                time.sleep(5)
            time.sleep(30)
            if upload_memory_lock_held():
                logger.info("Quarterly prewarm skipped: upload memory lock still held")
                return

            key = quarterly_cache_key(False, 8)
            if get_shared_quarterly(key):
                return
            import backend.main as _main

            mapping = (_main._warm_cache or {}).get("sku_mapping") or {}
            start, end = quarterly_report_window(8)

            def _build(progress_cb):
                return _build_via_streaming(
                    mapping,
                    start,
                    end,
                    group_by_parent=False,
                    n_quarters=8,
                    progress_cb=progress_cb,
                )

            start_shared_quarterly_build(key, _build)
        except Exception:
            logger.exception("Shared quarterly prewarm failed")

    threading.Thread(target=_go, daemon=True, name="po-qtr-prewarm").start()

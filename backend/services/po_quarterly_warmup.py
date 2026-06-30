"""Quarterly history — session/warm-cache fast path + streaming Tier-3 aggregate."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Bump when quarterly payload shape / history rules change (invalidates caches).
QUARTERLY_CACHE_SCHEMA = 13


def quarterly_cache_key(group_by_parent: bool, n_quarters: int) -> tuple:
    return (QUARTERLY_CACHE_SCHEMA, bool(group_by_parent), int(n_quarters))


_QUARTER_META_COLS = frozenset(
    {"OMS_SKU", "Avg_Monthly", "Units_90d", "Units_30d", "Freq_30d", "ADS", "Status"}
)


def expected_quarter_columns(n_quarters: int = 8) -> list[str]:
    """Oldest → newest quarter labels (matches calculate_quarterly_history pivot)."""
    from .po_engine import get_indian_fy_quarter, quarter_col_name

    today = pd.Timestamp.today()
    cur_fy, cur_q = get_indian_fy_quarter(today)
    quarter_seq: list[str] = []
    fy_i, q_i = cur_fy, cur_q
    for _ in range(int(n_quarters)):
        quarter_seq.append(quarter_col_name(fy_i, q_i))
        q_i -= 1
        if q_i == 0:
            q_i = 4
            fy_i -= 1
    quarter_seq.reverse()
    return quarter_seq


def _canonical_quarter_label(label: str) -> str:
    """Normalize en/em dashes so ``Apr–Jun 2026`` maps to ``Apr-Jun 2026``."""
    import re

    s = str(label or "").strip()
    s = re.sub(r"[-–—]", "-", s)
    return s


def normalize_quarterly_payload(
    payload: dict[str, Any], *, n_quarters: int = 8
) -> dict[str, Any]:
    """Ensure every row has all ``n_quarters`` history columns (zeros if missing)."""
    if not isinstance(payload, dict) or not payload.get("loaded"):
        return payload
    rows = payload.get("rows") or []
    if not rows:
        return payload

    expected = expected_quarter_columns(n_quarters)
    existing_cols = list(payload.get("columns") or [])
    if not existing_cols and rows:
        existing_cols = list(rows[0].keys())

    # Map alternate dash spellings → canonical expected labels.
    alias: dict[str, str] = {}
    for col in existing_cols:
        canon = _canonical_quarter_label(col)
        for exp in expected:
            if _canonical_quarter_label(exp) == canon:
                alias[col] = exp
                break

    meta = [
        c
        for c in existing_cols
        if c not in alias and c not in expected and c not in _QUARTER_META_COLS
    ]
    out_cols = ["OMS_SKU"] + expected + [
        c for c in _QUARTER_META_COLS if c != "OMS_SKU" and c in existing_cols
    ]
    for m in meta:
        if m not in out_cols:
            out_cols.append(m)

    norm_rows: list[dict[str, Any]] = []
    for row in rows:
        nr: dict[str, Any] = dict(row)
        for src, dst in alias.items():
            if src in nr and dst not in nr:
                nr[dst] = nr[src]
        for col in expected:
            nr[col] = int(nr.get(col, 0) or 0)
        norm_rows.append(nr)

    return {**payload, "columns": out_cols, "rows": norm_rows}


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

        _main.try_attach_shared_frames_fast(sess)
        if not _main.session_needs_operational_data(sess):
            pass
        elif _main._warm_cache:
            _main.try_attach_shared_frames_fast(sess)
        if not _main.session_needs_operational_data(sess):
            return
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
        # Do not restrict by bundled-listing inventory — always fan out so
        # individual sizes (L, XL…) get proportional historical sales data.
    )


def _pivot_to_payload(pivot: pd.DataFrame, *, n_quarters: int = 8) -> dict[str, Any]:
    if pivot is None or pivot.empty:
        return {"loaded": False, "rows": []}
    payload = {
        "loaded": True,
        "columns": list(pivot.columns),
        "rows": pivot.fillna(0).to_dict("records"),
    }
    return normalize_quarterly_payload(payload, n_quarters=n_quarters)


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
        return _pivot_to_payload(pivot, n_quarters=n_quarters)

    if not acquire_memory_lock:
        return _run()

    from ..concurrency import _UPLOAD_MEMORY_LOCK

    if not _UPLOAD_MEMORY_LOCK.acquire(timeout=120):
        logger.warning("Quarterly streaming skipped: memory lock busy")
        return {"loaded": False, "rows": []}
    try:
        return _run()
    finally:
        _UPLOAD_MEMORY_LOCK.release()


def _tier3_covers_quarterly_window(n_quarters: int) -> bool:
    """True when Tier-3 SQLite holds platform uploads spanning the quarterly window."""
    import datetime as _dt

    from .daily_store import get_summary

    try:
        summary = get_summary()
    except Exception:
        return False
    need_start = (
        _dt.date.today() - _dt.timedelta(days=max(750, int(n_quarters) * 92 + 90))
    ).isoformat()
    for plat, _attr in _PLATFORM_ATTRS:
        plat_sum = summary.get(plat) or {}
        if int(plat_sum.get("file_count") or 0) <= 0:
            continue
        tier_min = str(plat_sum.get("min_date") or "").strip()[:10]
        if tier_min and tier_min <= need_start:
            return True
    return False


def build_quarterly_payload(
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Never merge Tier-3 into session — warm-cache frames or streaming aggregate only."""
    _ensure_session_operational_frames(sess)

    from .platform_session_window import session_platform_shorter_than_tier3

    min_span = int(n_quarters) * 92 + 45
    plat_span = platform_frames_span_days(sess)
    has_plat = _session_has_platform_rows(sess)
    sales_span = sales_df_span_days(getattr(sess, "sales_df", None))
    span_ok = plat_span >= min_span - 60 or sales_span >= min_span - 60
    tier3_deeper = session_platform_shorter_than_tier3(sess)
    tier3_deep = _tier3_covers_quarterly_window(n_quarters)

    # Tier-3 SQLite holds full FY history — always stream when available so older
    # quarters are not zero for top sellers (warm-cache session span can look wide
    # while per-SKU history is shallow).
    if tier3_deeper or tier3_deep:
        if progress_cb:
            progress_cb(
                12,
                "Session platform history is shallow — streaming full Tier-3 uploads…",
            )
        start, end = quarterly_report_window(n_quarters)
        mapping = sess.sku_mapping or {}
        return _build_via_streaming(
            mapping,
            start,
            end,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
            progress_cb=progress_cb,
        )

    # Never use session platform frames when Tier-3 exists — span can look wide while
    # per-SKU history is shallow (e.g. Amazon Tier-3 only from Apr 2026).
    if has_plat and plat_span >= min_span - 60 and not _tier3_covers_quarterly_window(n_quarters):
        if progress_cb:
            progress_cb(40, "Using saved platform history…")
        pivot = _pivot_from_session_frames(
            sess,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
            include_sales=True,
        )
        out = _pivot_to_payload(pivot, n_quarters=n_quarters)
        if out.get("loaded") and out.get("rows"):
            return out

    # Do not use a short session window — it yields empty older quarters and
    # blocks the Tier-3 streaming path that has full FY history.
    if span_ok and (has_plat or sales_span > 0):
        if progress_cb:
            progress_cb(25, "Using session sales history…")
        pivot = _pivot_from_session_frames(
            sess, group_by_parent=group_by_parent, n_quarters=n_quarters
        )
        out = _pivot_to_payload(pivot, n_quarters=n_quarters)
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


def attach_quarterly_columns_to_po_df(
    po_df: pd.DataFrame,
    sess,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> pd.DataFrame:
    """Merge quarterly shipment history columns into PO result rows (inline in Calculate PO)."""
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return po_df

    cache_key = quarterly_cache_key(group_by_parent, n_quarters)
    payload = (getattr(sess, "_quarterly_cache", None) or {}).get(cache_key)
    if not payload or not payload.get("loaded"):
        payload = try_build_quarterly_payload_sync(
            sess, group_by_parent=group_by_parent, n_quarters=n_quarters
        )
    if payload:
        payload = normalize_quarterly_payload(payload, n_quarters=n_quarters)
        if not hasattr(sess, "_quarterly_cache"):
            sess._quarterly_cache = {}
        sess._quarterly_cache[cache_key] = payload

    if not payload or not payload.get("loaded") or not payload.get("rows"):
        return po_df

    q_df = pd.DataFrame(payload["rows"])
    if q_df.empty or "OMS_SKU" not in q_df.columns:
        return po_df

    expected = expected_quarter_columns(n_quarters)
    merge_cols = [c for c in expected if c in q_df.columns and c not in po_df.columns]
    if not merge_cols:
        return po_df

    q_sub = q_df[["OMS_SKU"] + merge_cols].drop_duplicates(subset=["OMS_SKU"], keep="last")
    out = po_df.merge(q_sub, on="OMS_SKU", how="left")
    for c in merge_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def schedule_shared_quarterly_prewarm() -> None:
    """Deferred pre-build — wait until warm-cache / restore memory lock is free (OOM-safe)."""
    import os
    import threading
    import time

    if os.environ.get("PO_QUARTERLY_PREWARM", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("Quarterly prewarm disabled via PO_QUARTERLY_PREWARM=0")
        return

    def _go() -> None:
        try:
            from ..concurrency import upload_memory_lock_held
            from .po_quarterly_cache import (
                get_shared_quarterly,
                load_shared_quarterly_from_disk,
                start_shared_quarterly_build,
                store_shared_quarterly,
            )

            key = quarterly_cache_key(False, 8)
            if get_shared_quarterly(key):
                return

            # Fast path: a persisted payload from before the last restart is
            # still correct (it's invalidated whenever sales/platform data
            # actually changes) — reuse it instead of paying for a 30-180s
            # streaming rebuild on every restart.
            disk_payload = load_shared_quarterly_from_disk(key)
            if disk_payload:
                store_shared_quarterly(key, disk_payload)
                logger.info("Shared quarterly cache restored from disk (%d rows)",
                            len(disk_payload.get("rows") or []))
                return

            # Let warm-cache Phase 1+2 and session restores finish first.
            for _ in range(120):
                if not upload_memory_lock_held():
                    break
                time.sleep(5)
            time.sleep(30)
            if upload_memory_lock_held():
                logger.info("Quarterly prewarm skipped: upload memory lock still held")
                return

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

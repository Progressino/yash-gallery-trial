"""Keep Intelligence builds from stacking and answer windows past the data instantly.

One gap-fill / Tier-3 build can allocate 1–2 GB on the 8 GB VPS. The dashboard fires
summary + fast + full + extras in parallel (plus retries and background artifact
builds), so unbounded concurrency OOM-killed the backend mid-request.
"""
from __future__ import annotations

import copy
import os
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any, Callable, Hashable, Iterator, Optional, TypeVar

T = TypeVar("T")

# Re-entrant so a gated build may call helpers that are themselves gated.
_BUILD_GATE = threading.RLock()
_GATE_LOCAL = threading.local()
# User requests waiting for the gate; background builds stand aside while > 0.
_FG_WAITING = 0
_FG_COND = threading.Condition()
_INFLIGHT: dict[Hashable, Future] = {}
_INFLIGHT_LOCK = threading.Lock()

_PLATFORM_FRAMES = ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df", "sales_df")
_PARQUET_MAX_CACHE: dict[str, tuple[tuple[int, int], Optional[str]]] = {}
_FRAME_MAX_CACHE: dict[tuple[int, int], tuple[Optional[str]]] = {}
_PLATFORM_NAMES = ("Amazon", "Myntra", "Meesho", "Flipkart", "Snapdeal")


@contextmanager
def _hold_gate(foreground: bool) -> Iterator[None]:
    global _FG_WAITING
    depth = getattr(_GATE_LOCAL, "depth", 0)
    if depth:
        _BUILD_GATE.acquire()
    elif foreground:
        with _FG_COND:
            _FG_WAITING += 1
        try:
            _BUILD_GATE.acquire()
        finally:
            with _FG_COND:
                _FG_WAITING -= 1
                _FG_COND.notify_all()
    else:
        while True:
            with _FG_COND:
                while _FG_WAITING:
                    _FG_COND.wait(timeout=1.0)
            _BUILD_GATE.acquire()
            with _FG_COND:
                if not _FG_WAITING:
                    break
            _BUILD_GATE.release()
    _GATE_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _GATE_LOCAL.depth = depth
        _BUILD_GATE.release()


def gated(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run one heavy background Intelligence build at a time; user requests go first."""
    with _hold_gate(foreground=False):
        return fn(*args, **kwargs)


def single_flight(key: Hashable, fn: Callable[[], T]) -> T:
    """Concurrent callers with the same key share one gated computation."""
    with _INFLIGHT_LOCK:
        fut = _INFLIGHT.get(key)
        owner = fut is None
        if owner:
            fut = Future()
            _INFLIGHT[key] = fut
    if not owner:
        result = fut.result()
        return copy.copy(result) if isinstance(result, dict) else result
    try:
        with _hold_gate(foreground=True):
            result = fn()
    except BaseException as exc:
        fut.set_exception(exc)
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(key, None)
        raise
    fut.set_result(result)
    with _INFLIGHT_LOCK:
        _INFLIGHT.pop(key, None)
    return result


def _parquet_max_date(path: str) -> Optional[str]:
    """Max Date/TxnDate from parquet row-group statistics (no data read)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _PARQUET_MAX_CACHE.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    best: Optional[str] = None
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path).metadata
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        col = next((c for c in ("Date", "TxnDate") if c in names), None)
        if col is not None:
            idx = names.index(col)
            for rg in range(meta.num_row_groups):
                stats = meta.row_group(rg).column(idx).statistics
                if stats is None or not stats.has_min_max:
                    best = None
                    break
                mx = str(stats.max)[:10]
                if best is None or mx > best:
                    best = mx
    except Exception:
        best = None
    _PARQUET_MAX_CACHE[path] = (sig, best)
    return best


def _frame_max_date(df: Any) -> Optional[str]:
    try:
        import pandas as pd

        if df is None or not hasattr(df, "empty") or df.empty:
            return None
        sig = (id(df), len(df))
        hit = _FRAME_MAX_CACHE.get(sig)
        if hit is not None:
            return hit[0]
        col = next((c for c in ("Date", "TxnDate") if c in df.columns), None)
        if col is None:
            return None
        mx = pd.to_datetime(df[col], errors="coerce").max()
        out = None if pd.isna(mx) else str(mx)[:10]
        if len(_FRAME_MAX_CACHE) > 32:
            _FRAME_MAX_CACHE.clear()
        _FRAME_MAX_CACHE[sig] = (out,)
        return out
    except Exception:
        return None


def latest_sales_date() -> Optional[str]:
    """Latest calendar day with any marketplace sales row (Tier-3 uploads or warm cache).

    Returns None when unknown — callers must then take the normal path.
    """
    candidates: list[str] = []
    try:
        from .daily_store import get_summary

        for info in (get_summary() or {}).values():
            mx = str((info or {}).get("max_date") or "")[:10]
            if len(mx) == 10:
                candidates.append(mx)
    except Exception:
        return None
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    try:
        import backend.main as _main

        mem = _main._warm_cache or {}
    except Exception:
        mem = {}
    for attr in _PLATFORM_FRAMES:
        mx = _frame_max_date(mem.get(attr)) if attr in mem else None
        if mx is None:
            mx = _parquet_max_date(os.path.join(base, f"{attr}.parquet"))
        if mx:
            candidates.append(mx)
    return max(candidates) if candidates else None


def window_after_latest_data(start_date: str, end_date: str) -> Optional[str]:
    """Latest data day when the whole window starts after it (1-day IST slack), else None."""
    s = str(start_date or "")[:10]
    if len(s) != 10 or len(str(end_date or "")[:10]) != 10:
        return None
    latest = latest_sales_date()
    if not latest:
        return None
    try:
        slack = (date.fromisoformat(latest) + timedelta(days=1)).isoformat()
    except ValueError:
        return None
    return latest if s > slack else None


def empty_window_payload(start_date: str, end_date: str, latest: str) -> dict[str, Any]:
    """Final (not warming) Intelligence payload for a window with no uploaded sales."""
    s, e = str(start_date)[:10], str(end_date)[:10]
    zero = {"total_units": 0, "total_returns": 0, "net_units": 0, "return_rate": 0.0}
    platforms = [
        {
            "platform": name, "loaded": True, **zero,
            "top_sku": "", "trend_direction": "flat", "trend_direction_net": "flat",
            "monthly": [], "daily": [], "by_state": [],
        }
        for name in _PLATFORM_NAMES
    ]
    return {
        "status": "ready",
        "data_completeness": "full",
        "empty_window": True,
        "data_max_date": latest,
        "session_data_range": {"min": "", "max": latest},
        "message": (
            f"No marketplace sales uploaded for {s} → {e}. "
            f"Latest uploaded sales: {latest}. Upload recent daily reports or pick an earlier range."
        ),
        "sales_summary": dict(zero),
        "platform_summary": platforms,
        "top_skus": [],
        "anomalies": [],
        "dsr_brand_monthly": {"rows": [], "totals": {}, "note": ""},
    }

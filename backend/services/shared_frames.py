"""Process-wide warm-cache frame access — avoid per-session copies of huge DataFrames."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

_log = __import__("logging").getLogger(__name__)

# Frames that must not be duplicated per browser session (~300MB+ each at scale).
LARGE_FRAME_KEYS = frozenset(
    {
        "sales_df",
        "mtr_df",
        "myntra_df",
        "meesho_df",
        "flipkart_df",
        "snapdeal_df",
        "inventory_df_variant",
        "inventory_df_parent",
        "daily_inventory_history_df",
    }
)


def shared_frames_enabled() -> bool:
    raw = (os.environ.get("SESSION_SHARED_FRAMES") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled

        if normalized_tables_enabled():
            return True
    except Exception:
        pass
    try:
        import backend.main as _main

        return _main.warm_cache_po_session_only()
    except Exception:
        return False


def _warm_cache() -> dict:
    try:
        import backend.main as _main

        return _main._warm_cache or {}
    except Exception:
        return {}


def session_uses_shared_frames(sess) -> bool:
    return bool(getattr(sess, "_shared_frames", False)) and shared_frames_enabled()


def attach_shared_frames(sess, *, warm_cache_generation: int) -> None:
    """Point session at process warm cache — do not copy large frames."""
    sess._shared_frames = True
    sess._warm_cache_gen = int(warm_cache_generation or 0)
    sess._warm_cache_only = True
    wc = _warm_cache()
    if isinstance(wc.get("sku_mapping"), dict) and wc["sku_mapping"]:
        sess.sku_mapping = wc["sku_mapping"]
    for key in LARGE_FRAME_KEYS:
        val = wc.get(key)
        if val is not None and hasattr(val, "empty") and not val.empty:
            setattr(sess, key, val)
    sess._quarterly_cache.clear()


def warm_frame(key: str, sess=None) -> pd.DataFrame:
    """Unified sales / platform / inventory accessor."""
    wc = _warm_cache()
    wc_df = wc.get(key)
    if sess is not None:
        df = getattr(sess, key, None)
        if df is not None and hasattr(df, "empty") and not df.empty:
            if not session_uses_shared_frames(sess):
                return df
            # Session holds a distinct copy (upload / test seed) — prefer it over warm cache.
            if wc_df is None or df is not wc_df:
                return df
    if wc_df is not None and hasattr(wc_df, "empty"):
        return wc_df
    if sess is not None:
        df = getattr(sess, key, None)
        if df is not None and hasattr(df, "empty"):
            return df
    return pd.DataFrame()


def session_sales_df(sess) -> pd.DataFrame:
    return warm_frame("sales_df", sess)


def session_inventory_variant(sess) -> pd.DataFrame:
    return warm_frame("inventory_df_variant", sess)


def session_inventory_parent(sess) -> pd.DataFrame:
    return warm_frame("inventory_df_parent", sess)


def session_platform_df(sess, platform_key: str) -> pd.DataFrame:
    return warm_frame(platform_key, sess)


def frame_row_count(key: str, sess) -> int:
    df = warm_frame(key, sess)
    try:
        return int(len(df)) if df is not None and hasattr(df, "__len__") else 0
    except Exception:
        return 0


def should_skip_session_copy(key: str) -> bool:
    return shared_frames_enabled() and key in LARGE_FRAME_KEYS


def assign_frame_no_copy(sess, key: str, val: Any) -> None:
    """Set session attribute without copying when shared-frame mode is on."""
    if val is None:
        return
    if hasattr(val, "empty") and val.empty:
        return
    if should_skip_session_copy(key):
        setattr(sess, key, val)
        return
    if hasattr(val, "copy"):
        setattr(sess, key, val.copy())
    else:
        setattr(sess, key, val)

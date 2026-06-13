"""Platform date windows — trim copies for sales-build memory only; SQLite keeps full history."""
from __future__ import annotations

import datetime
import os
from typing import Optional

import pandas as pd

# Rolling window for build_sales_df peak memory only (never persisted to session).
SESSION_PLATFORM_MAX_DAYS = int(os.environ.get("SESSION_PLATFORM_MAX_DAYS", "730"))
AUTO_RESTORE_MONTHS_DEFAULT = int(os.environ.get("AUTO_RESTORE_MONTHS", "12"))

_PLATFORM_ATTRS: tuple[tuple[str, str], ...] = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)

_DATE_COL_CANDIDATES: tuple[str, ...] = (
    "TxnDate",
    "Date",
    "Order Date",
    "order_date",
    "Shipment Date",
    "shipment_date",
)


def platform_date_column(df: pd.DataFrame) -> str | None:
    for c in _DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def platform_df_date_bounds(df: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Min/max transaction dates in a platform frame (naive timestamps)."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return None, None
    col = platform_date_column(df)
    if not col:
        return None, None
    dt = pd.to_datetime(df[col], errors="coerce")
    lo = dt.min()
    hi = dt.max()
    if pd.isna(lo):
        lo = None
    if pd.isna(hi):
        hi = None
    return lo, hi


def trim_platform_df(
    df: pd.DataFrame,
    *,
    max_days: int | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Drop rows older than ``max_days`` (default SESSION_PLATFORM_MAX_DAYS). No-op if empty."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    days = SESSION_PLATFORM_MAX_DAYS if max_days is None else max(1, int(max_days))
    col = date_col or platform_date_column(df)
    if not col:
        return df
    cutoff = pd.Timestamp(datetime.date.today()) - pd.Timedelta(days=days)
    dt = pd.to_datetime(df[col], errors="coerce")
    mask = dt.isna() | (dt >= cutoff)
    if mask.all():
        return df
    trimmed = df.loc[mask].reset_index(drop=True)
    return trimmed


def trimmed_copy_for_sales_build(
    df: pd.DataFrame,
    *,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Windowed copy for ``build_sales_df`` memory peak — never mutates the session frame."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    return trim_platform_df(df.copy(), max_days=max_days)


def platform_frames_trimmed_for_sales_build(sess, *, max_days: int | None = None) -> dict[str, pd.DataFrame]:
    """Return trimmed copies of all platform frames on ``sess`` for sales rebuild."""
    return {
        "mtr_df": trimmed_copy_for_sales_build(getattr(sess, "mtr_df", pd.DataFrame()), max_days=max_days),
        "myntra_df": trimmed_copy_for_sales_build(getattr(sess, "myntra_df", pd.DataFrame()), max_days=max_days),
        "meesho_df": trimmed_copy_for_sales_build(getattr(sess, "meesho_df", pd.DataFrame()), max_days=max_days),
        "flipkart_df": trimmed_copy_for_sales_build(getattr(sess, "flipkart_df", pd.DataFrame()), max_days=max_days),
        "snapdeal_df": trimmed_copy_for_sales_build(getattr(sess, "snapdeal_df", pd.DataFrame()), max_days=max_days),
    }


def trim_session_platform_frames(sess, *, max_days: int | None = None) -> bool:
    """Deprecated: trimming session frames drops dashboard history. Use trimmed_copy_for_sales_build."""
    changed = False
    for _, attr in _PLATFORM_ATTRS:
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            continue
        trimmed = trim_platform_df(cur, max_days=max_days)
        if len(trimmed) < len(cur):
            setattr(sess, attr, trimmed)
            changed = True
    return changed


def session_platform_shorter_than_tier3(sess) -> bool:
    """
    True when Tier-3 SQLite has older or newer order dates than the in-memory platform frame.
    Used to avoid marking ``daily_restored`` after a partial warm-cache copy.
    """
    try:
        from .daily_store import get_summary

        summary = get_summary()
    except Exception:
        return False
    for plat, attr in _PLATFORM_ATTRS:
        plat_sum = summary.get(plat) or {}
        if int(plat_sum.get("file_count") or 0) <= 0:
            continue
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            return True
        sess_min, sess_max = platform_df_date_bounds(cur)
        tier_min = str(plat_sum.get("min_date") or "")[:10]
        tier_max = str(plat_sum.get("max_date") or "")[:10]
        if tier_min and sess_min is not None and tier_min < str(sess_min.date())[:10]:
            return True
        if tier_max and sess_max is not None and tier_max > str(sess_max.date())[:10]:
            return True
    return False

"""Limit in-memory platform order DataFrames to a rolling date window (default 2 years)."""
from __future__ import annotations

import datetime
import os
from typing import Iterable

import pandas as pd

# Max calendar days kept in browser session / warm-cache copy (SQLite keeps full history).
SESSION_PLATFORM_MAX_DAYS = int(os.environ.get("SESSION_PLATFORM_MAX_DAYS", "730"))
AUTO_RESTORE_MONTHS_DEFAULT = int(os.environ.get("AUTO_RESTORE_MONTHS", "24"))

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


def trim_session_platform_frames(sess, *, max_days: int | None = None) -> bool:
    """Trim all platform order frames on ``sess``. Returns True if any frame shrank."""
    changed = False
    for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            continue
        trimmed = trim_platform_df(cur, max_days=max_days)
        if len(trimmed) < len(cur):
            setattr(sess, attr, trimmed)
            changed = True
    return changed

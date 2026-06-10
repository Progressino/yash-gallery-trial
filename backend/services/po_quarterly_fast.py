"""
Fast PO quarterly history — stream Tier-3 parquet blobs and aggregate in memory
without merging multi-million-row frames into the user session (OOM-safe).
"""
from __future__ import annotations

import io
import logging
from collections import defaultdict
from datetime import timedelta
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .po_engine import (
    canonical_oms_key,
    get_indian_fy_quarter,
    get_parent_sku,
    quarter_col_name,
)

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[int, str], None]]

_PLATFORM_SPECS = (
    ("amazon", True, False),
    ("myntra", False, True),
    ("meesho", False, True),
    ("flipkart", False, True),
    ("snapdeal", False, True),
)


def _quarter_seq(n_quarters: int) -> list[tuple[int, int]]:
    today = pd.Timestamp.today()
    cur_fy, cur_q = get_indian_fy_quarter(today)
    seq: list[tuple[int, int]] = []
    fy_i, q_i = cur_fy, cur_q
    for _ in range(n_quarters):
        seq.append((fy_i, q_i))
        q_i -= 1
        if q_i == 0:
            q_i = 4
            fy_i -= 1
    return list(reversed(seq))


def _ordered_q_cols(n_quarters: int) -> list[str]:
    return [quarter_col_name(fy, qn) for fy, qn in _quarter_seq(n_quarters)]


def _accumulate_shipment_frame(
    df: pd.DataFrame,
    platform: str,
    sku_mapping: Optional[Dict[str, str]],
    *,
    strip_pl: bool,
    canonical_oms: bool,
    group_by_parent: bool,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cutoff_90: pd.Timestamp,
    cutoff_30: pd.Timestamp,
    q_label_map: dict[tuple[int, int], str],
    quarter_sums: dict[tuple[str, str], int],
    units_90: dict[str, int],
    units_30: dict[str, int],
    days_30: dict[str, Set[pd.Timestamp]],
) -> int:
    """Add one blob's shipment rows into aggregate dicts. Returns rows processed."""
    from .daily_store import _PLATFORM_METRICS_COLUMNS
    from .po_engine import _PL_RE, _strip_pl

    if df is None or df.empty:
        return 0
    cols = _PLATFORM_METRICS_COLUMNS.get(platform)
    if not cols:
        return 0

    sku_col = next(
        (c for c in cols if c in df.columns and c in ("SKU", "OMS_SKU")),
        None,
    )
    date_col = "Date" if "Date" in df.columns else None
    qty_col = "Quantity" if "Quantity" in df.columns else None
    txn_col = next(
        (c for c in df.columns if c in ("Transaction_Type", "TxnType")),
        None,
    )
    if not sku_col or not date_col or not qty_col:
        return 0

    ship_mask = pd.Series(True, index=df.index)
    if txn_col:
        ship_mask &= (
            df[txn_col].astype(str).str.strip().str.lower().eq("shipment")
        )
    work = df.loc[ship_mask, [sku_col, date_col, qty_col]].copy()
    work.columns = ["SKU", "Date", "Qty"]
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0)
    work = work.dropna(subset=["Date"])
    work = work[(work["Date"] >= start_ts) & (work["Date"] <= end_ts)]
    work = work[work["Qty"] > 0]
    if work.empty:
        return 0

    raw_skus = work["SKU"].astype(str).unique()
    canon: dict[str, str] = {}
    for s in raw_skus:
        if strip_pl:
            if sku_mapping:
                k = _strip_pl(s, sku_mapping)
            else:
                k = _PL_RE.sub(r"\1\2", str(s).strip().upper())
        elif canonical_oms:
            k = canonical_oms_key(s, sku_mapping)
        else:
            k = canonical_oms_key(s, sku_mapping) if sku_mapping else str(s).strip().upper()
        if k:
            canon[s] = get_parent_sku(k) if group_by_parent else k
    work["SKU"] = work["SKU"].map(canon)
    work = work[work["SKU"].astype(str).str.len() > 0]
    if work.empty:
        return 0

    month = work["Date"].dt.month
    year = work["Date"].dt.year
    fy = np.where(month >= 4, year + 1, year)
    qn = np.select(
        [(month >= 4) & (month <= 6), (month >= 7) & (month <= 9), month >= 10],
        [1, 2, 3],
        default=4,
    )
    # Only aggregate quarters in the requested window (n_quarters). Older Tier-3
    # rows outside that window must not KeyError the label map.
    valid = np.array(
        [q_label_map.get((int(f), int(q))) is not None for f, q in zip(fy, qn)],
        dtype=bool,
    )
    if not valid.any():
        return 0
    if not valid.all():
        work = work.loc[valid].reset_index(drop=True)
        fy = fy[valid]
        qn = qn[valid]
    cols_lbl = [q_label_map[(int(f), int(q))] for f, q in zip(fy, qn)]
    skus = work["SKU"].astype(str).values
    qtys = work["Qty"].astype(int).values
    for sku, col, q in zip(skus, cols_lbl, qtys):
        quarter_sums[(sku, col)] += int(q)

    recent = work[work["Date"] >= cutoff_90]
    if not recent.empty:
        for sku, q in recent.groupby("SKU")["Qty"].sum().items():
            units_90[str(sku)] += int(q)

    recent30 = work[work["Date"] >= cutoff_30]
    if not recent30.empty:
        for sku, q in recent30.groupby("SKU")["Qty"].sum().items():
            units_30[str(sku)] += int(q)
        for sku, grp in recent30.groupby("SKU"):
            days = days_30[str(sku)]
            for d in grp["Date"].dt.normalize().unique():
                days.add(d)

    return len(work)


def calculate_quarterly_from_tier3_streaming(
    sku_mapping: Optional[Dict[str, str]],
    start_date: str,
    end_date: str,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
    progress_cb: ProgressCb = None,
) -> pd.DataFrame:
    """Aggregate quarterly pivot by reading one Tier-3 blob at a time."""
    from .daily_store import (
        _PLATFORM_METRICS_COLUMNS,
        _get_conn,
        _tier3_window_sql_clause,
    )

    s0 = str(start_date)[:10]
    s1 = str(end_date)[:10]
    if len(s0) != 10 or len(s1) != 10:
        return pd.DataFrame()

    start_ts = pd.Timestamp(s0).normalize()
    end_ts = pd.Timestamp(s1).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    today = pd.Timestamp.today()
    cutoff_90 = today - timedelta(days=90)
    cutoff_30 = today - timedelta(days=30)

    q_label_map = {(fy, qn): quarter_col_name(fy, qn) for fy, qn in _quarter_seq(n_quarters)}
    ordered_q_cols = _ordered_q_cols(n_quarters)

    quarter_sums: dict[tuple[str, str], int] = defaultdict(int)
    units_90: dict[str, int] = defaultdict(int)
    units_30: dict[str, int] = defaultdict(int)
    days_30: dict[str, Set[pd.Timestamp]] = defaultdict(set)

    conn = _get_conn()
    clause = _tier3_window_sql_clause()
    count_sql = f"""
        SELECT COUNT(*)
        FROM daily_uploads
        WHERE platform = ?
          AND ({clause})
    """
    select_sql = f"""
        SELECT filename, data_parquet
        FROM daily_uploads
        WHERE platform = ?
          AND ({clause})
        ORDER BY file_date ASC
    """
    total = 0
    for plat, _sp, _ca in _PLATFORM_SPECS:
        row = conn.execute(count_sql, (plat, s1, s0)).fetchone()
        total += int(row[0] if row else 0)

    if progress_cb:
        progress_cb(8, f"Scanning {total} upload file(s)…")

    done = 0
    for plat, _sp, _ca in _PLATFORM_SPECS:
        rows = conn.execute(select_sql, (plat, s1, s0))
        for _fn, blob in rows:
            done += 1
            if progress_cb and total:
                pct = 10 + int(85 * done / total)
                progress_cb(pct, f"Aggregating {plat} ({done}/{total})…")
            try:
                want = _PLATFORM_METRICS_COLUMNS.get(plat)
                d = pd.read_parquet(
                    io.BytesIO(blob),
                    engine="pyarrow",
                    columns=want,
                )
            except Exception:
                try:
                    d = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
                except Exception:
                    continue
            strip_pl, canon = next(
                (sp, ca) for pn, sp, ca in _PLATFORM_SPECS if pn == plat
            )
            _accumulate_shipment_frame(
                d,
                plat,
                sku_mapping,
                strip_pl=strip_pl,
                canonical_oms=canon,
                group_by_parent=group_by_parent,
                start_ts=start_ts,
                end_ts=end_ts,
                cutoff_90=cutoff_90,
                cutoff_30=cutoff_30,
                q_label_map=q_label_map,
                quarter_sums=quarter_sums,
                units_90=units_90,
                units_30=units_30,
                days_30=days_30,
            )
            del d, blob
    conn.close()

    if not quarter_sums:
        return pd.DataFrame()

    if progress_cb:
        progress_cb(96, "Building SKU table…")

    skus = sorted({k[0] for k in quarter_sums})
    rows = []
    for sku in skus:
        row = {"OMS_SKU": sku}
        for col in ordered_q_cols:
            row[col] = quarter_sums.get((sku, col), 0)
        rows.append(row)
    pivot = pd.DataFrame(rows)

    last4 = ordered_q_cols[-4:]
    pivot["Avg_Monthly"] = (pivot[last4].mean(axis=1) / 3).round(1)
    pivot["Units_90d"] = pivot["OMS_SKU"].map(lambda s: units_90.get(s, 0)).astype(int)
    pivot["ADS"] = (pivot["Units_90d"] / 90).round(3)
    pivot["Units_30d"] = pivot["OMS_SKU"].map(lambda s: units_30.get(s, 0)).astype(int)
    pivot["Freq_30d"] = pivot["OMS_SKU"].map(lambda s: len(days_30.get(s, set()))).astype(int)

    ads = pivot["ADS"]
    pivot["Status"] = np.select(
        [ads >= 1.0, ads >= 0.33, ads >= 0.10],
        ["Fast Moving", "Moderate", "Slow Selling"],
        default="Not Moving",
    )
    return pivot

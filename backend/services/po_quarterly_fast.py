"""
Fast PO quarterly history — stream Tier-3 parquet blobs and aggregate in memory
without merging multi-million-row frames into the user session (OOM-safe).

Tier-3 daily uploads are the primary source; warm-cache platform frames and unified
sales_df fill SKU-days missing from Tier-3 (e.g. Amazon bulk MTR before daily sync).
"""
from __future__ import annotations

import io
import logging
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .po_engine import (
    apply_quarterly_bundled_fan_out,
    canonical_oms_key,
    get_indian_fy_quarter,
    get_parent_sku,
    quarter_col_name,
)

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[int, str], None]]

DayKey = Tuple[str, pd.Timestamp]
PlatformDayKey = Tuple[str, str, pd.Timestamp]

_PLATFORM_SPECS = (
    ("amazon", True, False),
    ("myntra", False, True),
    ("meesho", False, True),
    ("flipkart", False, True),
    ("snapdeal", False, True),
)

_WARM_FRAME_ATTRS = (
    ("amazon", "mtr_df", True, False),
    ("myntra", "myntra_df", False, True),
    ("meesho", "meesho_df", False, True),
    ("flipkart", "flipkart_df", False, True),
    ("snapdeal", "snapdeal_df", False, True),
)

# Unified sales_df Source labels → platform keys used in platform_day_keys.
_SALES_SOURCE_PLATFORM = (
    ("amazon", "amazon"),
    ("myntra", "myntra"),
    ("meesho", "meesho"),
    ("flipkart", "flipkart"),
    ("snapdeal", "snapdeal"),
)


def _sales_source_to_platform(src) -> str:
    s = str(src or "").strip().lower()
    if not s or s in ("nan", "none", "nat"):
        return ""
    for needle, plat in _SALES_SOURCE_PLATFORM:
        if needle in s:
            return plat
    return s

_SALES_READ_COLS = ["Sku", "TxnDate", "Quantity", "Transaction Type", "Source"]

# Transaction types that contribute positively (net sold) vs negatively (returns/cancels).
# "ReturnCancel" = customer cancelled their return → unit stays with customer → positive.
_POSITIVE_TXN_TYPES = frozenset({"shipment", "returncancel"})
_NEGATIVE_TXN_TYPES = frozenset({"refund", "cancel"})
_NET_TXN_TYPES = _POSITIVE_TXN_TYPES | _NEGATIVE_TXN_TYPES


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


def _filter_new_platform_days(
    work: pd.DataFrame,
    *,
    platform: str,
    skip_days: Optional[Set[PlatformDayKey]],
    record_days: bool,
    platform_day_keys: Set[PlatformDayKey],
) -> pd.DataFrame:
    """Keep only rows whose (platform, SKU, day) is not already recorded for that platform."""
    if work.empty:
        return work
    work = work.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_sku"] = work["SKU"].astype(str)
    if skip_days:
        keys = [(platform, sku, day) for sku, day in zip(work["_sku"], work["_day"])]
        keep = np.array([k not in skip_days for k in keys], dtype=bool)
        work = work.loc[keep]
        if work.empty:
            return work
    if record_days:
        for sku, day in zip(work["_sku"], work["_day"]):
            platform_day_keys.add((platform, str(sku), pd.Timestamp(day)))
    return work.drop(columns=["_day", "_sku"], errors="ignore")


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
    platform_day_keys: Optional[Set[PlatformDayKey]] = None,
    skip_days: Optional[Set[PlatformDayKey]] = None,
) -> int:
    """Add one blob's shipment rows into aggregate dicts. Returns rows processed."""
    from .daily_store import _PLATFORM_METRICS_COLUMNS

    if df is None or df.empty:
        return 0
    cols = _PLATFORM_METRICS_COLUMNS.get(platform)
    if not cols:
        return 0

    # Strip FBA shadow rows from Amazon MTR blobs before aggregating — prevents
    # an unkeyed aggregate row (no Order_Id) duplicating a keyed row (with Order_Id)
    # for the same shipment. This is the same dedup applied on upload; re-running
    # here ensures historical Tier-3 blobs uploaded before the fix are also clean.
    if (
        platform == "amazon"
        and len(df) > 1
        and "Invoice_Number" in df.columns
        and "Order_Id" in df.columns
    ):
        try:
            from .mtr import dedup_amazon_mtr_dataframe
            df = dedup_amazon_mtr_dataframe(df)
        except Exception:
            pass
    if df is None or df.empty:
        return 0

    if platform == "amazon":
        from .sales import apply_amazon_free_replacement_txn

        df = apply_amazon_free_replacement_txn(df, txn_col="Transaction_Type")

    if platform == "meesho":
        from .meesho import backfill_meesho_sku_from_suborder_inplace

        df = df.copy()
        backfill_meesho_sku_from_suborder_inplace(df)
        if "SKU" in df.columns and "OMS_SKU" in df.columns:
            from .meesho import _meesho_pick_sku_series

            oms_pick = _meesho_pick_sku_series(df, "OMS_SKU")
            sku_pick = _meesho_pick_sku_series(df, "SKU")
            df["OMS_SKU"] = oms_pick.where(oms_pick.ne(""), sku_pick)

    sku_col = next(
        (c for c in cols if c in df.columns and c in ("SKU", "OMS_SKU")),
        None,
    )
    if platform == "amazon":
        date_col = next(
            (c for c in ("Reporting_Date", "Date", "TxnDate") if c in df.columns),
            None,
        )
    else:
        # Meesho / Myntra / Flipkart warm frames may use TxnDate after metrics rebuild.
        date_col = next(
            (c for c in ("Date", "TxnDate") if c in df.columns),
            None,
        )
    qty_col = "Quantity" if "Quantity" in df.columns else None
    txn_col = next(
        (c for c in df.columns if c in ("Transaction_Type", "TxnType")),
        None,
    )
    if not sku_col or not date_col or not qty_col:
        return 0

    # Net sold = Sale + ReturnCancel (positive) − Return − Cancellation (negative).
    # Prior versions only counted "Shipment" (gross); now all four event types are
    # included with the correct sign so the quarterly reflects true net sold units.
    ship_mask = pd.Series(True, index=df.index)
    if txn_col:
        _txn_str = df[txn_col].astype(str).str.strip().str.lower()
        ship_mask &= _txn_str.isin(_NET_TXN_TYPES)
        _extract_cols = [sku_col, date_col, qty_col, txn_col]
    else:
        _extract_cols = [sku_col, date_col, qty_col]
    asin_col = "ASIN" if "ASIN" in df.columns else None
    if asin_col:
        _extract_cols = list(_extract_cols) + [asin_col]
    work = df.loc[ship_mask, _extract_cols].copy()
    if txn_col:
        rename = {sku_col: "SKU", date_col: "Date", qty_col: "Qty", txn_col: "_Txn"}
        if asin_col:
            rename[asin_col] = "ASIN"
        work = work.rename(columns=rename)
        _txn_lower = work["_Txn"].astype(str).str.strip().str.lower()
        if platform == "amazon":
            # Amazon MTR net: Shipment − Refund (cancels / free replacements not demand).
            _neg = _txn_lower == "refund"
            _zero = _txn_lower.isin(["cancel", "freereplacement"])
        else:
            _neg = _txn_lower.isin(_NEGATIVE_TXN_TYPES)
            _zero = pd.Series(False, index=work.index)
        work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0)
        work["Qty"] = np.where(
            _zero,
            0,
            np.where(_neg, -work["Qty"].abs(), work["Qty"].abs()),
        )
        work = work.drop(columns=["_Txn"])
    else:
        rename = {sku_col: "SKU", date_col: "Date", qty_col: "Qty"}
        if asin_col:
            rename[asin_col] = "ASIN"
        work = work.rename(columns=rename)
        work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0)
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work = work[(work["Date"] >= start_ts) & (work["Date"] <= end_ts)]
    work = work[work["Qty"] != 0]
    if work.empty:
        return 0

    from .combo_sku_map import explode_sku_qty_dataframe, resolve_active_combo_sku_map

    # Combo / DPT listings → component OMS SKUs; non-combo rows canonicalize 1:1.
    # Must run on raw listing keys (before last-wins 1:1 map drops siblings).
    _ = canonical_oms  # call-site flag retained; explode uses strip_pl + mapping
    work = explode_sku_qty_dataframe(
        work,
        sku_col="SKU",
        qty_col="Qty",
        sku_mapping=sku_mapping,
        combo_map=resolve_active_combo_sku_map(),
        strip_pl=bool(strip_pl),
        retain_combo_listings=True,
    )
    if work.empty:
        return 0
    if group_by_parent:
        _u = work["SKU"].astype(str).unique()
        _p = {s: get_parent_sku(s) for s in _u}
        work = work.copy()
        work["SKU"] = work["SKU"].map(_p)
    tentative = work["SKU"].fillna("").astype(str).str.strip()
    work["SKU"] = tentative
    if asin_col and asin_col in work.columns:
        work = work.drop(columns=[asin_col], errors="ignore")
    work = work[
        work["SKU"].ne("")
        & ~work["SKU"].str.lower().isin(["nan", "none", "nat"])
    ]
    if work.empty:
        return 0

    record_days = platform_day_keys is not None
    work = _filter_new_platform_days(
        work,
        platform=platform,
        skip_days=skip_days,
        record_days=record_days,
        # Pass the caller's set BY REFERENCE even when it is currently empty. Using
        # ``platform_day_keys or set()`` handed a throwaway set to the recorder whenever
        # the accumulator was empty (a falsy set) — so Tier-1 (SKU, day) keys were never
        # recorded, and the Tier-3 gap-fill + sales_df supplement then re-counted days
        # already present in Tier-1, inflating every SKU's quarterly units.
        platform_day_keys=platform_day_keys if platform_day_keys is not None else set(),
    )
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
            # Only count days with positive net contribution as "selling days"
            # so that return-only days do not inflate Freq_30d / reduce ADS.
            day_net = grp.groupby(grp["Date"].dt.normalize())["Qty"].sum()
            for d, net in day_net.items():
                if net > 0:
                    days.add(d)

    return len(work)


def _load_unified_sales_df() -> pd.DataFrame:
    """Warm-cache unified sales (session-free) for quarterly supplement."""
    try:
        import backend.main as _main

        wc = (_main._warm_cache or {}).get("sales_df")
        if wc is not None and not wc.empty:
            return wc
    except Exception:
        pass
    path = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")) / "sales_df.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=_SALES_READ_COLS)
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()


def _load_platform_frame_from_disk(attr: str) -> pd.DataFrame:
    """Full Tier-1 platform parquet from ``WARM_CACHE_DIR`` (no rolling window clip)."""
    path = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")) / f"{attr}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _sync_load_bulk_mtr_if_needed() -> None:
    """Ensure deferred Amazon MTR bulk is in warm cache before Tier-1 quarterly."""
    import time

    from .shared_frames import warm_frame

    for _ in range(60):
        if len(warm_frame("mtr_df")) >= 50_000:
            return
        time.sleep(2.0)
    try:
        import backend.main as _main
        from pathlib import Path

        if len(warm_frame("mtr_df")) < 1000 and _main._warm_cache is not None:
            p = Path(getattr(_main, "_DISK_CACHE_DIR", "/data/warm_cache")) / "mtr_df.parquet"
            if p.is_file():
                _main._warm_cache["mtr_df"] = pd.read_parquet(p)
    except Exception:
        pass


def _warm_cache_platform_frames() -> dict[str, pd.DataFrame]:
    """Tier-1 / Tier-2 bulk platform history — warm RAM, then disk parquets."""
    from .shared_frames import warm_frame

    try:
        import backend.main as _main

        if not _main._warm_cache:
            _main.bootstrap_warm_cache_if_empty()
            _main._warm_cache_ready.wait(timeout=90.0)
        _sync_load_bulk_mtr_if_needed()
    except Exception:
        pass

    out: dict[str, pd.DataFrame] = {}
    for _plat, attr, _sp, _ca in _WARM_FRAME_ATTRS:
        if attr == "meesho_df":
            # Deepdive prefers OMS-filled disk when RAM is blank-heavy — quarterly must too.
            from .shared_frames import resolve_meesho_frame

            df = resolve_meesho_frame(warm_frame(attr))
        else:
            df = warm_frame(attr)
            if df is None or getattr(df, "empty", True):
                df = _load_platform_frame_from_disk(attr)
        if df is not None and not getattr(df, "empty", True):
            out[attr] = df
    if out:
        return out
    try:
        import backend.main as _main

        stub = type("_QuarterlyWarmStub", (), {})()
        _main.try_attach_shared_frames_fast(stub)
        for _plat, attr, _sp, _ca in _WARM_FRAME_ATTRS:
            df = getattr(stub, attr, None)
            if df is not None and not getattr(df, "empty", True):
                out[attr] = df
    except Exception:
        logger.debug("Warm-cache platform frames unavailable for quarterly", exc_info=True)
    return out


def _tier1_platform_frame(
    plat: str,
    attr: str,
    *,
    warm_frames: Optional[dict[str, pd.DataFrame]] = None,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """
    Best available Tier-1 / Tier-2 bulk frame for one platform (memory-safe).

    Order: warm RAM → disk parquet → windowed SQLite uploads for the quarterly range.
    Never loads unbounded ``months=None`` history (OOM on large MTR archives).
    """
    from .daily_store import load_platform_data_for_report_range

    frames = warm_frames if warm_frames is not None else _warm_cache_platform_frames()
    df = frames.get(attr, pd.DataFrame())
    if attr == "meesho_df":
        from .shared_frames import resolve_meesho_frame

        df = resolve_meesho_frame(df if df is not None and not getattr(df, "empty", True) else None)
    elif df is None or getattr(df, "empty", True):
        df = _load_platform_frame_from_disk(attr)
    s0 = str(start_date)[:10]
    s1 = str(end_date)[:10]
    if (df is None or getattr(df, "empty", True)) and len(s0) == 10 and len(s1) == 10:
        try:
            df = load_platform_data_for_report_range(plat, s0, s1, dedup=True)
        except Exception:
            df = pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _accumulate_tier1_platform_history(
    sku_mapping: Optional[Dict[str, str]],
    *,
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
    platform_day_keys: Set[PlatformDayKey],
    progress_cb: ProgressCb = None,
) -> None:
    """Primary quarterly source — Tier-1 bulk + Tier-2 uploads (warm cache / disk / SQLite)."""
    if progress_cb:
        progress_cb(10, "Loading Tier-1 bulk platform history…")
    warm_frames = _warm_cache_platform_frames()
    for plat, attr, strip_pl, canon in _WARM_FRAME_ATTRS:
        df = _tier1_platform_frame(
            plat,
            attr,
            warm_frames=warm_frames,
            start_date=str(start_ts.date()),
            end_date=str(end_ts.date()),
        )
        if df is None or df.empty:
            continue
        date_col = next((c for c in ("Date", "TxnDate") if c in df.columns), None)
        if date_col:
            d = pd.to_datetime(df[date_col], errors="coerce")
            df = df[(d >= start_ts) & (d <= end_ts)]
        if df.empty:
            continue
        _accumulate_shipment_frame(
            df,
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
            platform_day_keys=platform_day_keys,
        )


def _accumulate_sales_df_shipments(
    sales_df: pd.DataFrame,
    sku_mapping: Optional[Dict[str, str]],
    *,
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
    platform_day_keys: Set[PlatformDayKey],
) -> int:
    """Append unified sales shipment rows for SKU-days not already on platform side."""
    from .po_engine import _sales_shipment_history_part

    part = _sales_shipment_history_part(sales_df)
    if part.empty:
        return 0
    work = part.copy()
    # Include all net-relevant TxnTypes with correct signs (mirrors _accumulate_shipment_frame).
    _txn_lower = work["TxnType"].astype(str).str.strip().str.lower()
    work = work[_txn_lower.isin(_NET_TXN_TYPES)].copy()
    if work.empty:
        return 0
    _txn_lower2 = work["TxnType"].astype(str).str.strip().str.lower()
    _src = (
        work["_Source"].astype(str).str.strip().str.lower()
        if "_Source" in work.columns
        else pd.Series("", index=work.index)
    )
    _is_amazon = _src.eq("amazon")
    _neg = _txn_lower2.eq("refund") | (~_is_amazon & _txn_lower2.eq("cancel"))
    _zero = _is_amazon & _txn_lower2.isin(["cancel", "freereplacement"])
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0)
    work["Qty"] = np.where(
        _zero,
        0,
        np.where(_neg, -work["Qty"].abs(), work["Qty"].abs()),
    )
    # Keep _Source through explode + gap-fill so Meesho sales are not suppressed
    # merely because Amazon already recorded the same (SKU, day).
    work = work.dropna(subset=["Date"])
    work = work[(work["Date"] >= start_ts) & (work["Date"] <= end_ts)]
    work = work[work["Qty"] != 0]
    if work.empty:
        return 0

    from .combo_sku_map import explode_sku_qty_dataframe, resolve_active_combo_sku_map

    work = explode_sku_qty_dataframe(
        work,
        sku_col="SKU",
        qty_col="Qty",
        sku_mapping=sku_mapping,
        combo_map=resolve_active_combo_sku_map(),
        strip_pl=False,
        retain_combo_listings=True,
    )
    work = work[work["SKU"].astype(str).str.len() > 0]
    if group_by_parent:
        work["SKU"] = work["SKU"].map(lambda s: get_parent_sku(s))

    work = work.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_sku"] = work["SKU"].astype(str)
    if platform_day_keys:
        if "_Source" in work.columns:
            work["_plat"] = work["_Source"].map(_sales_source_to_platform)
            keys = list(zip(work["_plat"], work["_sku"], work["_day"]))
            keep = np.array(
                [
                    (not p) or ((p, s, d) not in platform_day_keys)
                    for p, s, d in keys
                ],
                dtype=bool,
            )
            # Rows with unknown source: fall back to any-platform (SKU, day) claim
            # so unified rows without Source still avoid double-count.
            unknown = work["_plat"].eq("") | work["_plat"].isna()
            if unknown.any():
                sku_days_any = {(sku, day) for _p, sku, day in platform_day_keys}
                any_hit = np.array(
                    [
                        (s, d) in sku_days_any
                        for s, d in zip(work["_sku"], work["_day"])
                    ],
                    dtype=bool,
                )
                keep = np.where(unknown, ~any_hit, keep)
            work = work.loc[keep]
        else:
            sku_days_on_platform = {(sku, day) for _p, sku, day in platform_day_keys}
            keys = list(zip(work["_sku"], work["_day"]))
            keep = np.array([k not in sku_days_on_platform for k in keys], dtype=bool)
            work = work.loc[keep]
    work = work.drop(columns=["_day", "_sku", "_plat", "_Source"], errors="ignore")
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
            # Only count days with positive net contribution as "selling days".
            day_net = grp.groupby(grp["Date"].dt.normalize())["Qty"].sum()
            for d, net in day_net.items():
                if net > 0:
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
    """
    Aggregate quarterly pivot: Tier-1/2 bulk first, Tier-3 dailies gap-fill, then sales_df.
    """
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
    platform_day_keys: Set[PlatformDayKey] = set()

    _accumulate_tier1_platform_history(
        sku_mapping,
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
        platform_day_keys=platform_day_keys,
        progress_cb=progress_cb,
    )

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
        progress_cb(8, f"Gap-filling from {total} Tier-3 daily file(s)…")

    done = 0
    for plat, _sp, _ca in _PLATFORM_SPECS:
        rows = conn.execute(select_sql, (plat, s1, s0))
        for _fn, blob in rows:
            done += 1
            if progress_cb and total:
                pct = 20 + int(70 * done / max(total, 1))
                progress_cb(pct, f"Tier-3 gap-fill {plat} ({done}/{total})…")
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
                platform_day_keys=platform_day_keys,
                skip_days=platform_day_keys,
            )
            del d, blob
    conn.close()

    sales_df = _load_unified_sales_df()
    if not sales_df.empty:
        if progress_cb:
            progress_cb(94, "Merging unified sales history…")
        _accumulate_sales_df_shipments(
            sales_df,
            sku_mapping,
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
            platform_day_keys=platform_day_keys,
        )

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
    # Floor at 0: net-negative 90-day periods (more returns than sales) yield ADS=0,
    # not a negative rate — a negative ADS would give nonsensical Days_Left / PO Qty.
    pivot["ADS"] = (pivot["Units_90d"].clip(lower=0) / 90).round(3)
    pivot["Units_30d"] = pivot["OMS_SKU"].map(lambda s: units_30.get(s, 0)).astype(int)
    pivot["Freq_30d"] = pivot["OMS_SKU"].map(lambda s: len(days_30.get(s, set()))).astype(int)

    ads = pivot["ADS"]
    pivot["Status"] = np.select(
        [ads >= 1.0, ads >= 0.33, ads >= 0.10],
        ["Fast Moving", "Moderate", "Slow Selling"],
        default="Not Moving",
    )

    if not group_by_parent:
        pivot = apply_quarterly_bundled_fan_out(pivot, ordered_q_cols)

    return pivot

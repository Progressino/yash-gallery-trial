"""
Parse Daily Inventory History (Excel) for PO effective-days calculation.

The export is wide-format. Two sheets are recognised:

    * ``OMS`` (or ``OMS Inventory``) – warehouse on-hand snapshots
    * ``Amazon Inventory``           – FBA snapshots (PL-prefixed SKUs)

The first column stores the SKU code (header text varies, e.g. ``Item SkuCode``).
The second column is the parent style. Subsequent columns are daily snapshots —
the column **header** is the daily total inventory (an integer), and the
**first data row** of each column carries the actual date stamp (``YYYY-MM-DD``).

Output is a tall DataFrame with three columns: ``OMS_SKU`` / ``Date`` / ``Qty``.
PO engine uses it to count only the days a SKU actually had stock on hand;
days with zero (or missing) inventory are excluded from the ADS denominator.
"""
from __future__ import annotations

import io
from typing import BinaryIO, Optional

import numpy as np
import pandas as pd

from .helpers import (
    clean_sku,
    collapse_duplicate_trailing_size_suffix,
    normalize_id_token_for_mapping,
)
from .sku_status_lead import _strip_pl_sku


_TALL_COLS = ["OMS_SKU", "Date", "Qty"]


def _norm(s) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _looks_like_sku_header(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    n = _norm(value)
    return n in {
        "skucode",
        "itemskucode",
        "sku",
        "omssku",
        "item",
        "itemcode",
        "sellersku",
        "stylesku",
    }


def _is_date_value(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, (pd.Timestamp,)):
        return True
    # Wide inventory sheets use integer column headers for on-hand totals (e.g. 150).
    # Those must not be treated as Excel serial dates.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        fv = float(v)
        if fv < 40_000 or fv > 60_000:
            return False
        try:
            ts = pd.to_datetime(fv, unit="D", origin="1899-12-30", errors="coerce")
            if ts is None or pd.isna(ts):
                return False
            return 2015 <= int(ts.year) <= 2035
        except Exception:
            return False
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if ts is None or pd.isna(ts):
            return False
        return 2015 <= int(ts.year) <= 2035
    except Exception:
        return False


_VARIANT_LABELS = {"itemskucode", "skucode", "sellersku", "stylesku"}
_PARENT_LABELS = {"item", "itemcode", "parent", "parentsku", "parentstyle", "style", "sku", "omssku"}


def _detect_sku_column(df: pd.DataFrame) -> Optional[int]:
    """
    Return position of the SKU (variant) column.

    Real layout uses row 0 as a label row (``Item SkuCode``/``Item``); column
    headers are often noise (``Total Inv.``, ``SKU``). Treat row 0 as truth and
    bias toward variant-style markers; only fall back to column headers when
    row 0 has no recognisable label.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return None

    if df.shape[0] > 0:
        first = list(df.iloc[0].values)
        variant_first = next(
            (i for i, v in enumerate(first) if _norm(v) in _VARIANT_LABELS),
            None,
        )
        if variant_first is not None:
            return variant_first
        any_first = next((i for i, v in enumerate(first) if _looks_like_sku_header(v)), None)
        if any_first is not None:
            return any_first

    cols = list(df.columns)
    variant_col = next((i for i, c in enumerate(cols) if _norm(c) in _VARIANT_LABELS), None)
    if variant_col is not None:
        return variant_col
    any_col = next((i for i, c in enumerate(cols) if _looks_like_sku_header(c)), None)
    if any_col is not None:
        return any_col
    return 0  # fallback to leftmost column


def _detect_parent_column(df: pd.DataFrame, sku_idx: int) -> Optional[int]:
    """Best-effort: parent column sits next to SKU column with header 'Item' / 'SKUCode'."""
    candidates: list[int] = []
    if df.shape[0] > 0:
        first = df.iloc[0]
        for i, val in enumerate(first.values):
            if i == sku_idx:
                continue
            if _norm(val) in _PARENT_LABELS:
                candidates.append(i)
    if not candidates:
        for i, col in enumerate(df.columns):
            if i == sku_idx:
                continue
            if _norm(col) in _PARENT_LABELS:
                candidates.append(i)
    return candidates[0] if candidates else None


def _row_date_hints(df: pd.DataFrame, row_idx: int, skip_cols: set[int]) -> dict[int, pd.Timestamp]:
    out: dict[int, pd.Timestamp] = {}
    if row_idx < 0 or row_idx >= len(df):
        return out
    row = df.iloc[row_idx]
    for i, val in enumerate(row.values):
        if i in skip_cols:
            continue
        if _is_date_value(val):
            try:
                ts = pd.to_datetime(val, errors="coerce")
                if pd.notna(ts):
                    out[i] = pd.Timestamp(ts).normalize()
            except Exception:
                pass
    return out


def _build_column_date_map(
    df: pd.DataFrame,
    sku_idx: int,
    parent_idx: Optional[int],
) -> tuple[dict[int, pd.Timestamp], int]:
    """Map column index → snapshot date; return (date_map, first_data_row_index).

    Yash exports put dates in row 0 for most columns; a newly added "today"
  column may have its date only in row 1. Union hints from the first few rows
    and column headers instead of picking only row 0 OR only headers.
    """
    skip = {sku_idx}
    if parent_idx is not None:
        skip.add(parent_idx)

    header_dates: dict[int, pd.Timestamp] = {}
    for i, col in enumerate(df.columns):
        if i in skip:
            continue
        if isinstance(col, (pd.Timestamp,)) or _is_date_value(col):
            try:
                ts = pd.to_datetime(col, errors="coerce")
                if pd.notna(ts):
                    header_dates[i] = pd.Timestamp(ts).normalize()
            except Exception:
                pass

    row_maps: list[dict[int, pd.Timestamp]] = []
    for ridx in range(min(4, len(df))):
        rm = _row_date_hints(df, ridx, skip)
        if rm:
            row_maps.append(rm)

    date_map: dict[int, pd.Timestamp] = {}
    if row_maps:
        primary = max(row_maps, key=len)
        date_map.update(primary)
        for rm in row_maps:
            for i, d in rm.items():
                date_map.setdefault(i, d)
    for i, d in header_dates.items():
        date_map.setdefault(i, d)

    header_row_count = 0
    if date_map:
        positions = sorted(date_map.keys())
        for ridx in range(min(4, len(df))):
            row = df.iloc[ridx]
            hits = 0
            for i in positions:
                if i >= len(row):
                    continue
                if _is_date_value(row.iloc[i]):
                    hits += 1
            if hits >= max(2, len(positions) // 3):
                header_row_count = ridx + 1

    return date_map, header_row_count


def merge_inventory_history(
    existing: Optional[pd.DataFrame],
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """Union SKU-day rows; keep max qty when both frames have the same key."""
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame(columns=_TALL_COLS)
    if existing is None or existing.empty:
        return incoming[_TALL_COLS].copy()
    ex = existing[[c for c in _TALL_COLS if c in existing.columns]].copy()
    inc = incoming[_TALL_COLS].copy()
    combined = pd.concat([ex, inc], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.normalize()
    combined["Qty"] = pd.to_numeric(combined["Qty"], errors="coerce")
    combined = combined.dropna(subset=["Date", "OMS_SKU"])
    combined = combined[combined["OMS_SKU"].astype(str).str.len() > 0]
    combined = combined.dropna(subset=["Qty"])
    if combined.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    return (
        combined.groupby(["OMS_SKU", "Date"], as_index=False)["Qty"]
        .max()
        .sort_values(["OMS_SKU", "Date"])
        .reset_index(drop=True)
    )


def inventory_history_max_date(df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    mx = pd.to_datetime(df["Date"], errors="coerce").max()
    return pd.Timestamp(mx).normalize() if pd.notna(mx) else None


def _parse_one_sheet(df: pd.DataFrame, mapping: dict, sheet_name: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    df = df.copy()

    sku_idx = _detect_sku_column(df)
    if sku_idx is None:
        return pd.DataFrame(columns=_TALL_COLS)
    parent_idx = _detect_parent_column(df, sku_idx)

    date_map, header_row_count = _build_column_date_map(df, sku_idx, parent_idx)
    if not date_map:
        # Sheet shape unrecognised — fail silently for this sheet, parser keeps trying others.
        return pd.DataFrame(columns=_TALL_COLS)

    # Slice to data rows only (skip label/date header rows).
    body = df.iloc[header_row_count:] if header_row_count > 0 else df
    if body.empty:
        return pd.DataFrame(columns=_TALL_COLS)

    date_positions = sorted(i for i in date_map if i < body.shape[1])
    if not date_positions:
        return pd.DataFrame(columns=_TALL_COLS)

    block = body.iloc[:, [sku_idx] + date_positions].copy()
    block.columns = ["_raw_sku"] + [
        pd.Timestamp(date_map[i]).strftime("%Y-%m-%d") for i in date_positions
    ]
    tall = block.melt(id_vars="_raw_sku", var_name="Date", value_name="Qty")
    tall["Date"] = pd.to_datetime(tall["Date"], errors="coerce").dt.normalize()
    tall = tall.dropna(subset=["Date"])
    tall = tall.dropna(subset=["_raw_sku"])
    tall["_raw_sku"] = tall["_raw_sku"].astype(str).str.strip()
    tall = tall[tall["_raw_sku"].str.len() > 0]
    tall = tall[~tall["_raw_sku"].str.lower().isin({"nan", "none", "<na>", "nat"})]
    if tall.empty:
        return pd.DataFrame(columns=_TALL_COLS)

    def _canon(v: str) -> str:
        tok = normalize_id_token_for_mapping(v)
        clean = clean_sku(tok or v)
        if not clean:
            clean = str(v).strip().upper()
        return collapse_duplicate_trailing_size_suffix(_strip_pl_sku(clean, mapping))

    unique_raw = tall["_raw_sku"].unique()
    canon_map = {r: _canon(r) for r in unique_raw}
    tall["OMS_SKU"] = tall["_raw_sku"].map(canon_map)
    tall = tall[tall["OMS_SKU"].astype(str).str.len() > 0]
    tall["Qty"] = pd.to_numeric(tall["Qty"], errors="coerce")
    # Excel/pandas sometimes round-trips integer NA as ``iinfo(int64).min`` — those
    # show up as huge negatives. Coerce those back to NaN so they're treated as
    # "no snapshot for that day", not "zero stock".
    tall.loc[tall["Qty"] < -1.0, "Qty"] = pd.NA
    # Blank cells (= no snapshot was taken on that day, e.g. Sundays in many
    # warehouses) must NOT count as out-of-stock. Drop them so coverage_days
    # naturally reflects only the days actually sampled; the engine then
    # extrapolates Eff_Days from in-stock-rate × ADS_WINDOW.
    tall = tall.dropna(subset=["Qty"])
    if tall.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    tall["Qty"] = tall["Qty"].astype(float).clip(lower=0.0)
    return tall[_TALL_COLS].reset_index(drop=True)


def parse_daily_inventory_history_dataframes(
    sheet_dfs: dict[str, pd.DataFrame],
    sku_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    mapping = sku_mapping if sku_mapping is not None else {}
    parts: list[pd.DataFrame] = []
    for name, df in sheet_dfs.items():
        if df is None or df.empty:
            continue
        parts.append(_parse_one_sheet(df, mapping, sheet_name=name))
    if not parts:
        return pd.DataFrame(columns=_TALL_COLS)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return out
    # Multiple sheets can list the same SKU — keep the maximum stock seen across
    # sources for the same date (warehouse + FBA snapshots can both be > 0).
    out = (
        out.groupby(["OMS_SKU", "Date"], as_index=False)["Qty"].max()
        .sort_values(["OMS_SKU", "Date"])
        .reset_index(drop=True)
    )
    return out


def _sheet_looks_like_inventory(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    hints = ("oms", "amazon", "inventory", "fba", "warehouse", "stock", "on hand", "onhand")
    return any(h in n for h in hints)


def parse_daily_inventory_history_upload(
    file: BinaryIO,
    filename: str,
    sku_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    name = (filename or "").lower()
    raw = file.read() if hasattr(file, "read") else file
    if not raw:
        return pd.DataFrame(columns=_TALL_COLS)
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        return parse_daily_inventory_history_dataframes({"csv": df}, sku_mapping=sku_mapping)
    xl = pd.ExcelFile(io.BytesIO(raw))
    sheet_names = list(xl.sheet_names)
    inv_names = [sn for sn in sheet_names if _sheet_looks_like_inventory(sn)]
    to_read = inv_names if inv_names else sheet_names
    sheets: dict[str, pd.DataFrame] = {}
    for sn in to_read:
        try:
            sheets[sn] = xl.parse(sn)
        except Exception:
            continue
    return parse_daily_inventory_history_dataframes(sheets, sku_mapping=sku_mapping)


#: Operational threshold for "in stock" when counting Eff_Days. A day
#: counts toward Eff_Days when on-hand >= IN_STOCK_MIN_QTY (defaults to 1.0,
#: meaning any positive stock counts).
IN_STOCK_MIN_QTY: float = 1.0


def trim_inventory_history_for_po(
    inv_history: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    po_skus: Optional[set] = None,
) -> pd.DataFrame:
    """Keep only SKU-day rows needed for one PO run (avoids processing multi-year baselines)."""
    if inv_history is None or inv_history.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    ws = pd.Timestamp(window_start).normalize()
    we = pd.Timestamp(window_end).normalize()
    if ws > we:
        return pd.DataFrame(columns=_TALL_COLS)

    dates = pd.to_datetime(inv_history["Date"], errors="coerce").dt.normalize()
    mask = (dates >= ws) & (dates <= we)
    if po_skus:
        mask = mask & inv_history["OMS_SKU"].astype(str).isin(po_skus)
    if not bool(mask.any()):
        return pd.DataFrame(columns=_TALL_COLS)

    out = inv_history.loc[mask, ["OMS_SKU", "Date", "Qty"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["Qty"] = pd.to_numeric(out["Qty"], errors="coerce")
    out = out.dropna(subset=["Date", "Qty", "OMS_SKU"])
    out = out[out["OMS_SKU"].astype(str).str.len() > 0]
    if out.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    return out.reset_index(drop=True)


def effective_days_from_history(
    inv_history: pd.DataFrame,
    cutoff_start: pd.Timestamp,
    cutoff_end: pd.Timestamp,
    min_qty: float = IN_STOCK_MIN_QTY,
) -> pd.DataFrame:
    """
    Count days within ``[cutoff_start, cutoff_end]`` where each SKU had ``Qty >= min_qty``.

    By default ``min_qty == IN_STOCK_MIN_QTY`` (1.0) — any positive on-hand
    counts toward Eff_Days.

    Returns: ``OMS_SKU``, ``Eff_Days_Inventory`` (int).
    """
    if inv_history is None or inv_history.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Eff_Days_Inventory"])
    df = inv_history.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Eff_Days_Inventory"])
    cs = pd.Timestamp(cutoff_start).normalize()
    ce = pd.Timestamp(cutoff_end).normalize()
    if cs > ce:
        return pd.DataFrame(columns=["OMS_SKU", "Eff_Days_Inventory"])
    mask = (df["Date"] >= cs) & (df["Date"] <= ce)
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Eff_Days_Inventory"])
    sub["Qty"] = pd.to_numeric(sub["Qty"], errors="coerce")
    # Drop "no snapshot" rows so we never count blank cells as out-of-stock days.
    sub = sub.dropna(subset=["Qty"])
    if sub.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Eff_Days_Inventory"])
    sub["_has_stock"] = (sub["Qty"] >= float(min_qty)).astype(int)
    out = (
        sub.groupby("OMS_SKU", as_index=False)["_has_stock"]
        .sum()
        .rename(columns={"_has_stock": "Eff_Days_Inventory"})
    )
    out["Eff_Days_Inventory"] = out["Eff_Days_Inventory"].astype(int)
    return out


def extend_history_with_sales(
    inv_history: pd.DataFrame,
    sales_df: Optional[pd.DataFrame],
    cap_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Roll the uploaded baseline inventory forward using daily sales activity.

    User intent: upload the Daily Inventory History sheet *once*. From there,
    each day's sales upload should let the engine work out which SKUs were
    in stock without another inventory upload.

    For every SKU present in ``inv_history``, derive a synthetic snapshot
    for each day from ``(sheet_max_date + 1)`` up to ``cap_date`` (default:
    today, normalised). New on-hand is::

        new_qty = max(0, prev_qty - Σ Units_Effective for that day)

    Shipments are positive and pull stock down; refunds/cancellations are
    negative and push it back up. Stock is floored at 0 (we never report
    negative on-hand).

    SKUs without a baseline snapshot are skipped — we have no starting
    on-hand to roll forward.

    Returns the union of baseline + derived rows, with an extra ``Source``
    column (``"uploaded"`` vs ``"derived"``). Callers that only need the
    three core columns can drop ``Source`` after the merge.
    """
    if inv_history is None or inv_history.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Date", "Qty", "Source"])

    base = inv_history.copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce").dt.normalize()
    base["Qty"] = pd.to_numeric(base["Qty"], errors="coerce")
    base = base.dropna(subset=["Date", "Qty", "OMS_SKU"])
    base = base[base["OMS_SKU"].astype(str).str.len() > 0]
    if base.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Date", "Qty", "Source"])
    base["Qty"] = base["Qty"].astype(float).clip(lower=0.0)
    if "Source" not in base.columns:
        base["Source"] = "uploaded"

    sheet_max = base["Date"].max()
    if cap_date is None:
        cap_date = pd.Timestamp.now().normalize()
    cap_date = pd.Timestamp(cap_date).normalize()

    out_cols = ["OMS_SKU", "Date", "Qty", "Source"]
    if sheet_max >= cap_date:
        return base[out_cols].reset_index(drop=True)

    # Last seen snapshot per SKU = starting point for the roll-forward.
    last_snap = (
        base.sort_values(["OMS_SKU", "Date"])
        .groupby("OMS_SKU", as_index=False)
        .tail(1)
        .set_index("OMS_SKU")["Qty"]
    )
    sku_list = last_snap.index.to_numpy()

    # Aggregate net sales (signed Units_Effective) per (SKU, day) within window.
    sales_net = None
    if sales_df is not None and not sales_df.empty:
        s = sales_df.copy()
        sku_col = "Sku" if "Sku" in s.columns else "OMS_SKU"
        date_col = "TxnDate" if "TxnDate" in s.columns else "Date"
        eff_col = "Units_Effective" if "Units_Effective" in s.columns else "Quantity"
        if sku_col in s.columns and date_col in s.columns and eff_col in s.columns:
            s = s[[sku_col, date_col, eff_col]].copy()
            s.columns = ["OMS_SKU", "Date", "Net_Units"]
            s["Date"] = pd.to_datetime(s["Date"], errors="coerce").dt.normalize()
            s["Net_Units"] = pd.to_numeric(s["Net_Units"], errors="coerce").fillna(0.0)
            s = s.dropna(subset=["Date"])
            s = s[(s["Date"] > sheet_max) & (s["Date"] <= cap_date)]
            if not s.empty:
                sales_net = (
                    s.groupby(["OMS_SKU", "Date"], as_index=False)["Net_Units"].sum()
                )
                # Never roll inventory past the last day we actually have sales for.
                smax = pd.to_datetime(s["Date"], errors="coerce").max()
                if pd.notna(smax):
                    cap_date = min(cap_date, pd.Timestamp(smax).normalize())

    if sheet_max >= cap_date:
        return base[out_cols].reset_index(drop=True)

    # Only synthesize snapshots on days with sales activity — do not fill every
    # calendar day with flat on-hand (that inflated Eff_Days when today's sales
    # were not uploaded yet).
    if sales_net is not None and not sales_net.empty:
        active = sorted(
            pd.to_datetime(sales_net["Date"], errors="coerce")
            .dropna()
            .dt.normalize()
            .unique()
        )
        days = pd.DatetimeIndex(
            [d for d in active if sheet_max < pd.Timestamp(d) <= cap_date]
        )
    else:
        days = pd.DatetimeIndex([])

    if len(days) == 0:
        return base[out_cols].reset_index(drop=True)

    if sales_net is not None and not sales_net.empty:
        pivot = (
            sales_net.set_index(["OMS_SKU", "Date"])["Net_Units"]
            .unstack(fill_value=0.0)
            .reindex(index=sku_list, columns=days, fill_value=0.0)
        )
        net_matrix = pivot.to_numpy(dtype=float)  # shape: (n_sku, n_days)
    else:
        net_matrix = np.zeros((len(sku_list), len(days)), dtype=float)

    # Iterate days (small N — usually a handful) but vectorise across SKUs.
    prev_qty = last_snap.reindex(sku_list).fillna(0.0).to_numpy(dtype=float)
    derived_rows: list[pd.DataFrame] = []
    for di, d in enumerate(days):
        net_d = net_matrix[:, di]
        new_qty = np.maximum(0.0, prev_qty - net_d)
        derived_rows.append(
            pd.DataFrame(
                {
                    "OMS_SKU": sku_list,
                    "Date": pd.Timestamp(d),
                    "Qty": new_qty,
                    "Source": "derived",
                }
            )
        )
        prev_qty = new_qty

    derived = pd.concat(derived_rows, ignore_index=True)
    full = pd.concat([base[out_cols], derived[out_cols]], ignore_index=True)
    return full.reset_index(drop=True)


def coverage_days_within(
    inv_history: pd.DataFrame,
    cutoff_start: pd.Timestamp,
    cutoff_end: pd.Timestamp,
) -> int:
    """Number of distinct snapshot dates the history sheet provides within ``[cs, ce]``.

    Used by the PO engine to decide whether the sheet covers the full ADS window
    or only a sub-range — partial-coverage SKUs must be *extrapolated*, not
    capped at the sheet's day count (otherwise a sheet with 24 days collapses
    every SKU's Eff_Days to 24 even when the SKU was in-stock all 30 days).
    """
    if inv_history is None or inv_history.empty:
        return 0
    df = inv_history.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return 0
    cs = pd.Timestamp(cutoff_start).normalize()
    ce = pd.Timestamp(cutoff_end).normalize()
    if cs > ce:
        return 0
    mask = (df["Date"] >= cs) & (df["Date"] <= ce)
    sub = df.loc[mask]
    if sub.empty:
        return 0
    return int(sub["Date"].dt.normalize().nunique())


__all__ = [
    "parse_daily_inventory_history_dataframes",
    "parse_daily_inventory_history_upload",
    "effective_days_from_history",
    "coverage_days_within",
    "extend_history_with_sales",
]


# Suppress unused np import lint when the module shrinks during refactors.
_unused_np = np  # type: ignore[unused-ignore]

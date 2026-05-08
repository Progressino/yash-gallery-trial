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

from .helpers import clean_sku, normalize_id_token_for_mapping
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
    try:
        ts = pd.to_datetime(v, errors="coerce")
        return ts is not None and not pd.isna(ts)
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


def _parse_one_sheet(df: pd.DataFrame, mapping: dict, sheet_name: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_TALL_COLS)
    df = df.copy()

    sku_idx = _detect_sku_column(df)
    if sku_idx is None:
        return pd.DataFrame(columns=_TALL_COLS)
    parent_idx = _detect_parent_column(df, sku_idx)

    # Excel quirk: first data row carries the date in date columns. Build the
    # column→date map from row 0 (and the header itself when it's already a date).
    header_dates: dict[int, pd.Timestamp] = {}
    for i, col in enumerate(df.columns):
        if i in (sku_idx, parent_idx):
            continue
        if isinstance(col, (pd.Timestamp,)) or _is_date_value(col):
            try:
                ts = pd.to_datetime(col, errors="coerce")
                if pd.notna(ts):
                    header_dates[i] = pd.Timestamp(ts).normalize()
            except Exception:
                pass

    first_row_dates: dict[int, pd.Timestamp] = {}
    if df.shape[0] > 0:
        first = df.iloc[0]
        for i, val in enumerate(first.values):
            if i in (sku_idx, parent_idx):
                continue
            if _is_date_value(val):
                try:
                    ts = pd.to_datetime(val, errors="coerce")
                    if pd.notna(ts):
                        first_row_dates[i] = pd.Timestamp(ts).normalize()
                except Exception:
                    pass

    use_first_row_for_dates = bool(first_row_dates) and len(first_row_dates) >= len(header_dates)
    date_map = first_row_dates if use_first_row_for_dates else header_dates
    if not date_map:
        # Sheet shape unrecognised — fail silently for this sheet, parser keeps trying others.
        return pd.DataFrame(columns=_TALL_COLS)

    # Slice to data rows only.
    body = df.iloc[1:] if use_first_row_for_dates else df
    if body.empty:
        return pd.DataFrame(columns=_TALL_COLS)

    sku_series = body.iloc[:, sku_idx].astype(object)

    rows = []
    sku_index = sku_series.index
    for col_pos, dt in date_map.items():
        if col_pos >= body.shape[1]:
            continue
        col_vals = pd.to_numeric(body.iloc[:, col_pos], errors="coerce")
        if col_vals.isna().all():
            continue
        sub = pd.DataFrame(
            {
                "_raw_sku": sku_series.values,
                "Qty": col_vals.values,
                "Date": pd.Timestamp(dt),
            },
            index=sku_index,
        )
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=_TALL_COLS)
    tall = pd.concat(rows, ignore_index=True)
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
        return _strip_pl_sku(clean, mapping)

    tall["OMS_SKU"] = tall["_raw_sku"].map(_canon)
    tall = tall[tall["OMS_SKU"].astype(str).str.len() > 0]
    tall["Qty"] = pd.to_numeric(tall["Qty"], errors="coerce").fillna(0.0)
    # Excel/pandas sometimes round-trips integer NA as ``iinfo(int64).min`` — those
    # show up as huge negatives. Treat anything < 0 as "no stock for that day".
    tall["Qty"] = tall["Qty"].clip(lower=0.0)
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
    sheets: dict[str, pd.DataFrame] = {}
    for sn in xl.sheet_names:
        try:
            sheets[sn] = xl.parse(sn)
        except Exception:
            continue
    return parse_daily_inventory_history_dataframes(sheets, sku_mapping=sku_mapping)


def effective_days_from_history(
    inv_history: pd.DataFrame,
    cutoff_start: pd.Timestamp,
    cutoff_end: pd.Timestamp,
    min_qty: float = 1.0,
) -> pd.DataFrame:
    """
    Count days within ``[cutoff_start, cutoff_end]`` where each SKU had ``Qty >= min_qty``.

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
    sub["Qty"] = pd.to_numeric(sub["Qty"], errors="coerce").fillna(0.0)
    sub["_has_stock"] = (sub["Qty"] >= float(min_qty)).astype(int)
    out = (
        sub.groupby("OMS_SKU", as_index=False)["_has_stock"]
        .sum()
        .rename(columns={"_has_stock": "Eff_Days_Inventory"})
    )
    out["Eff_Days_Inventory"] = out["Eff_Days_Inventory"].astype(int)
    return out


__all__ = [
    "parse_daily_inventory_history_dataframes",
    "parse_daily_inventory_history_upload",
    "effective_days_from_history",
]


# Suppress unused np import lint when the module shrinks during refactors.
_unused_np = np  # type: ignore[unused-ignore]

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
import os
import re
from datetime import date, datetime
from typing import BinaryIO, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .helpers import (
    clean_sku,
    collapse_duplicate_trailing_size_suffix,
    normalize_id_token_for_mapping,
)
from .sku_status_lead import _strip_pl_sku

_IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_VIEW_DAYS = int(os.environ.get("DAILY_INV_VIEW_DAYS", "30"))


_TALL_COLS = ["OMS_SKU", "Date", "Qty"]

# Yash wide-matrix exports: ``28-5-26`` = 28 May 2026 (day-month-year).
_DMY_HEADER_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$")


def _safe_normalize(ts) -> pd.Timestamp | None:
    """Normalize a timestamp; return None for NaT / unparsable values."""
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return None
    if pd.isna(t):
        return None
    return t.normalize()


def _parse_inventory_snapshot_date(value) -> pd.Timestamp | None:
    """Parse snapshot dates from wide-matrix headers (D-M-YY) and other exports."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return _safe_normalize(value)
    if isinstance(value, (datetime, date)):
        return _safe_normalize(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fv = float(value)
        if 40_000 <= fv <= 60_000:
            try:
                ts = pd.to_datetime(fv, unit="D", origin="1899-12-30", errors="coerce")
                if ts is not None and pd.notna(ts) and 2015 <= int(ts.year) <= 2035:
                    return _safe_normalize(ts)
            except Exception:
                pass
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat", "days"}:
        return None
    m = _DMY_HEADER_RE.match(s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            ts = pd.Timestamp(year=year, month=month, day=day)
            if 2015 <= int(ts.year) <= 2035:
                return _safe_normalize(ts)
        except ValueError:
            return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        try:
            ts = pd.to_datetime(s[:10], errors="coerce")
            if ts is not None and pd.notna(ts) and 2015 <= int(ts.year) <= 2035:
                return _safe_normalize(ts)
        except Exception:
            return None
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if ts is not None and pd.notna(ts) and 2015 <= int(ts.year) <= 2035:
            return pd.Timestamp(ts).normalize()
    except Exception:
        return None
    return None


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
    return _parse_inventory_snapshot_date(v) is not None


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
            ts = _parse_inventory_snapshot_date(val)
            if ts is not None:
                out[i] = ts
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
            ts = _parse_inventory_snapshot_date(col)
            if ts is not None:
                header_dates[i] = ts

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


def _trim_date_map_to_window(
    date_map: dict[int, pd.Timestamp],
    max_days: int,
) -> dict[int, pd.Timestamp]:
    """Drop snapshot columns outside the last ``max_days`` calendar days (anchored on max date)."""
    if max_days <= 0 or not date_map:
        return date_map
    max_date = max(pd.Timestamp(d).normalize() for d in date_map.values())
    cutoff = max_date - pd.Timedelta(days=max_days - 1)
    return {
        i: d
        for i, d in date_map.items()
        if pd.Timestamp(d).normalize() >= cutoff
    }


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


def today_ist_timestamp() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_IST).normalize().tz_localize(None)


def filter_inventory_history_window(
    df: pd.DataFrame,
    *,
    days: int | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Keep SKU-day rows in the last ``days`` calendar days ending on ``end_date`` (default today IST)."""
    if df is None or df.empty or "Date" not in df.columns:
        return df if df is not None else pd.DataFrame(columns=_TALL_COLS)
    span = int(days if days is not None else _DEFAULT_VIEW_DAYS)
    if span <= 0:
        return df.copy()
    try:
        end = pd.Timestamp(str(end_date or "")[:10]).normalize() if end_date else today_ist_timestamp()
    except Exception:
        end = today_ist_timestamp()
    if pd.isna(end):
        end = today_ist_timestamp()
    start = end - pd.Timedelta(days=max(0, span - 1))
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["Date"])
    mask = (work["Date"] >= start) & (work["Date"] <= end)
    return work.loc[mask].reset_index(drop=True)


def append_snapshot_inventory_to_history(sess) -> dict:
    """
    After a daily snapshot inventory upload, append one SKU-day column to history.

    Uses ``Total_Inventory`` per variant SKU and the inferred snapshot date.
    """
    import pandas as pd

    variant = getattr(sess, "inventory_df_variant", None)
    if variant is None or getattr(variant, "empty", True):
        return {"appended": False, "reason": "empty_snapshot"}
    snap = str(getattr(sess, "inventory_snapshot_date", "") or "").strip()[:10]
    if len(snap) != 10:
        snap = str(today_ist_timestamp().date())

    work = variant.copy()
    if "OMS_SKU" not in work.columns:
        return {"appended": False, "reason": "no_sku_column"}
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    qty_col = "Total_Inventory" if "Total_Inventory" in work.columns else None
    if not qty_col:
        return {"appended": False, "reason": "no_total_inventory"}
    work["Qty"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0.0)
    work = work[work["OMS_SKU"].str.len() > 0]
    if work.empty:
        return {"appended": False, "reason": "no_rows"}

    incoming = pd.DataFrame(
        {
            "OMS_SKU": work["OMS_SKU"],
            "Date": pd.Timestamp(snap),
            "Qty": work["Qty"],
            "Source": "daily_snapshot",
        }
    )
    existing = getattr(sess, "daily_inventory_history_df", None)
    merged = merge_inventory_history(existing, incoming)
    merged = filter_inventory_history_window(merged, days=_DEFAULT_VIEW_DAYS)
    sess.daily_inventory_history_df = merged
    sess._quarterly_cache.clear()
    return {
        "appended": True,
        "snapshot_date": snap,
        "rows": int(len(merged)),
        "skus": int(merged["OMS_SKU"].nunique()) if not merged.empty else 0,
        "days": int(merged["Date"].nunique()) if not merged.empty else 0,
    }


def _column_usecols_for_inventory_sheet(
    raw: bytes,
    sheet_name: str,
    max_days: int | None,
) -> list[int] | None:
    """Peek sheet headers; return column indices to read (SKU + recent date columns only)."""
    try:
        import openpyxl
    except ImportError:
        return None
    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        try:
            if sheet_name not in wb.sheetnames:
                return None
            ws = wb[sheet_name]
            rows: list[list] = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= 5:
                    break
                rows.append(list(row) if row else [])
        finally:
            wb.close()
    except Exception:
        return None
    if not rows:
        return None
    width = max(len(r) for r in rows)
    header_df = pd.DataFrame([r + [None] * (width - len(r)) for r in rows])
    sku_idx = _detect_sku_column(header_df)
    if sku_idx is None:
        return None
    parent_idx = _detect_parent_column(header_df, sku_idx)
    date_map, _ = _build_column_date_map(header_df, sku_idx, parent_idx)
    if max_days is not None and max_days > 0:
        date_map = _trim_date_map_to_window(date_map, max_days)
    if not date_map:
        return None
    return [sku_idx] + sorted(i for i in date_map if i != sku_idx)


def _read_inventory_history_sheet(
    raw: bytes,
    sheet_name: str,
    *,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Read only SKU + recent date columns from a wide inventory sheet."""
    usecols = _column_usecols_for_inventory_sheet(raw, sheet_name, max_days)
    if usecols is None:
        return pd.read_excel(
            io.BytesIO(raw),
            sheet_name=sheet_name,
            header=None,
            dtype=str,
            engine="openpyxl",
        )
    return pd.read_excel(
        io.BytesIO(raw),
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        dtype=str,
        engine="openpyxl",
    )


def _parse_one_sheet(
    df: pd.DataFrame,
    mapping: dict,
    sheet_name: str = "",
    *,
    max_days: int | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_TALL_COLS)

    sku_idx = _detect_sku_column(df)
    if sku_idx is None:
        return pd.DataFrame(columns=_TALL_COLS)
    parent_idx = _detect_parent_column(df, sku_idx)

    date_map, header_row_count = _build_column_date_map(df, sku_idx, parent_idx)
    if max_days is not None and max_days > 0:
        date_map = _trim_date_map_to_window(date_map, max_days)
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
    *,
    max_days: int | None = None,
) -> pd.DataFrame:
    mapping = sku_mapping if sku_mapping is not None else {}
    parts: list[pd.DataFrame] = []
    for name, df in sheet_dfs.items():
        if df is None or df.empty:
            continue
        parts.append(_parse_one_sheet(df, mapping, sheet_name=name, max_days=max_days))
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
    *,
    max_days: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    def _progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    name = (filename or "").lower()
    raw = file.read() if hasattr(file, "read") else file
    if not raw:
        return pd.DataFrame(columns=_TALL_COLS)
    if name.endswith(".csv"):
        _progress("Parsing CSV…")
        df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
        return parse_daily_inventory_history_dataframes(
            {"csv": df}, sku_mapping=sku_mapping, max_days=max_days,
        )
    _progress("Opening workbook…")
    name_l = (filename or "").lower()
    is_xlsx = name_l.endswith(".xlsx") or (not name_l.endswith(".xls") and not name_l.endswith(".csv"))
    if is_xlsx:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet_names = list(xl.sheet_names)
        inv_names = [sn for sn in sheet_names if _sheet_looks_like_inventory(sn)]
        to_read = inv_names if inv_names else sheet_names
        sheets: dict[str, pd.DataFrame] = {}
        for i, sn in enumerate(to_read):
            _progress(f"Reading sheet {i + 1}/{len(to_read)}: {sn}…")
            try:
                sheets[sn] = _read_inventory_history_sheet(raw, sn, max_days=max_days)
            except Exception:
                try:
                    sheets[sn] = xl.parse(sn, dtype=str)
                except Exception:
                    continue
    else:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet_names = list(xl.sheet_names)
        inv_names = [sn for sn in sheet_names if _sheet_looks_like_inventory(sn)]
        to_read = inv_names if inv_names else sheet_names
        sheets = {}
        for i, sn in enumerate(to_read):
            _progress(f"Reading sheet {i + 1}/{len(to_read)}: {sn}…")
            try:
                sheets[sn] = xl.parse(sn, dtype=str)
            except Exception:
                continue
    _progress("Melting date columns…")
    return parse_daily_inventory_history_dataframes(
        sheets, sku_mapping=sku_mapping, max_days=max_days,
    )


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


def latest_inventory_qty_by_sku(
    history_df: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Latest on-hand qty per SKU from the wide inventory history matrix."""
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["OMS_SKU", "History_Qty"])
    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Date", "OMS_SKU", "Qty"])
    if df.empty:
        return pd.DataFrame(columns=["OMS_SKU", "History_Qty"])
    if as_of is not None:
        cap = pd.Timestamp(as_of).normalize()
        df = df[df["Date"] <= cap]
        if df.empty:
            return pd.DataFrame(columns=["OMS_SKU", "History_Qty"])
    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date]
    out = (
        latest.groupby("OMS_SKU", as_index=False)["Qty"]
        .max()
        .rename(columns={"Qty": "History_Qty"})
    )
    out["OMS_SKU"] = out["OMS_SKU"].astype(str).str.strip().str.upper()
    return out


def overlay_inventory_variant_from_history(
    inv_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    *,
    snapshot_date: str | None = None,
    reference_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    When the wide history matrix is fresher than the daily snapshot, use its
    latest column for PO on-hand (Total_Inventory / OMS_Inventory).
    """
    from .inventory_staleness import data_is_stale, today_ist_iso

    meta: dict = {"applied": False, "skus_updated": 0, "reason": ""}
    if inv_df is None or getattr(inv_df, "empty", True):
        meta["reason"] = "empty_inventory"
        return inv_df, meta
    if history_df is None or getattr(history_df, "empty", True):
        meta["reason"] = "empty_history"
        return inv_df, meta

    ref = (reference_date or today_ist_iso()).strip()[:10]
    hist_max = inventory_history_max_date(history_df)
    if hist_max is None:
        meta["reason"] = "no_history_dates"
        return inv_df, meta
    hist_max_s = str(hist_max.date())
    if data_is_stale(ref, hist_max_s, max_expected_lag_days=1):
        meta["reason"] = "history_stale"
        return inv_df, meta

    snap_s = str(snapshot_date or "").strip()[:10]
    snap_stale = not snap_s or data_is_stale(ref, snap_s, max_expected_lag_days=1)
    if not snap_stale and snap_s >= hist_max_s:
        meta["reason"] = "snapshot_fresh"
        return inv_df, meta

    latest = latest_inventory_qty_by_sku(history_df, as_of=hist_max)
    if latest.empty:
        meta["reason"] = "no_latest_rows"
        return inv_df, meta

    out = inv_df.copy()
    if "OMS_SKU" not in out.columns:
        meta["reason"] = "no_sku_column"
        return inv_df, meta
    out["OMS_SKU"] = out["OMS_SKU"].astype(str).str.strip().str.upper()
    merged = out.merge(latest, on="OMS_SKU", how="left")
    has_hist = merged["History_Qty"].notna()
    if not bool(has_hist.any()):
        meta["reason"] = "no_overlap"
        return inv_df, meta

    hist_qty = pd.to_numeric(merged["History_Qty"], errors="coerce").fillna(0.0)
    for col in ("Total_Inventory", "OMS_Inventory"):
        if col in merged.columns:
            cur = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
            merged[col] = np.where(has_hist, hist_qty, cur)
        elif col == "Total_Inventory":
            merged["Total_Inventory"] = np.where(has_hist, hist_qty, 0.0)
    if "Total_Inventory" not in merged.columns:
        merged["Total_Inventory"] = np.where(has_hist, hist_qty, 0.0)
    merged.drop(columns=["History_Qty"], inplace=True)

    meta["applied"] = True
    meta["skus_updated"] = int(has_hist.sum())
    meta["history_as_of"] = hist_max_s
    meta["reason"] = "history_newer_than_snapshot"
    return merged, meta


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


def inventory_history_summary(
    df: pd.DataFrame,
    *,
    days: int | None = None,
    end_date: str | None = None,
) -> dict:
    view = filter_inventory_history_window(df, days=days, end_date=end_date)
    if view is None or view.empty:
        return {
            "loaded": False,
            "rows": 0,
            "skus": 0,
            "days": 0,
            "min_date": "",
            "max_date": "",
            "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
            "window_end": str(end_date or today_ist_timestamp().date()),
        }
    dates = pd.to_datetime(view["Date"], errors="coerce").dt.normalize()
    min_d = dates.min()
    max_d = dates.max()
    return {
        "loaded": True,
        "rows": int(len(view)),
        "skus": int(view["OMS_SKU"].astype(str).nunique()),
        "days": int(dates.nunique()),
        "min_date": str(pd.Timestamp(min_d).date()) if pd.notna(min_d) else "",
        "max_date": str(pd.Timestamp(max_d).date()) if pd.notna(max_d) else "",
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(end_date or today_ist_timestamp().date()),
    }


def list_inventory_history_dates(df: pd.DataFrame, limit: int = 120) -> list[dict]:
    if df is None or df.empty:
        return []
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["Date"])
    if work.empty:
        return []
    grouped = (
        work.groupby("Date", as_index=False)
        .agg(rows=("OMS_SKU", "count"), skus=("OMS_SKU", "nunique"))
        .sort_values("Date", ascending=False)
    )
    if limit > 0:
        grouped = grouped.head(int(limit))
    return [
        {
            "date": str(pd.Timestamp(r["Date"]).date()),
            "rows": int(r["rows"]),
            "skus": int(r["skus"]),
        }
        for _, r in grouped.iterrows()
    ]


def inventory_rows_for_date(
    df: pd.DataFrame,
    date_iso: str,
    *,
    q: str = "",
    limit: int = 500,
    offset: int = 0,
) -> dict:
    if df is None or df.empty:
        return {
            "loaded": False,
            "date": date_iso,
            "rows": [],
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
        }
    try:
        target = pd.Timestamp(date_iso).normalize()
    except Exception:
        return {
            "loaded": True,
            "date": date_iso,
            "rows": [],
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
            "message": "Invalid date.",
        }

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    sub = work[work["Date"] == target].copy()
    if sub.empty:
        return {
            "loaded": True,
            "date": str(target.date()),
            "rows": [],
            "total": 0,
            "limit": int(limit),
            "offset": int(offset),
        }

    sub["OMS_SKU"] = sub["OMS_SKU"].astype(str).str.strip()
    sub["Qty"] = pd.to_numeric(sub["Qty"], errors="coerce").fillna(0.0)
    needle = (q or "").strip().upper()
    if needle:
        sub = sub[sub["OMS_SKU"].str.upper().str.contains(needle, na=False)]
    sub = sub.sort_values(["Qty", "OMS_SKU"], ascending=[False, True])
    total = int(len(sub))
    page = sub.iloc[max(0, int(offset)) : max(0, int(offset)) + max(1, int(limit))]
    rows = [
        {
            "sku": str(r["OMS_SKU"]),
            "qty": float(r["Qty"]),
            "in_stock": bool(float(r["Qty"]) >= IN_STOCK_MIN_QTY),
            "source": str(r.get("Source", "uploaded") or "uploaded"),
        }
        for _, r in page.iterrows()
    ]
    return {
        "loaded": True,
        "date": str(target.date()),
        "rows": rows,
        "total": total,
        "limit": int(limit),
        "offset": int(offset),
        "in_stock_min_qty": float(IN_STOCK_MIN_QTY),
    }


def inventory_history_wide_matrix(
    df: pd.DataFrame,
    *,
    q: str = "",
    limit: int = 150,
    offset: int = 0,
    days: int | None = None,
    end_date: str | None = None,
) -> dict:
    """Pivot tall history to Excel-style wide matrix: SKU rows × date columns."""
    empty = {
        "loaded": False,
        "dates": [],
        "rows": [],
        "total": 0,
        "limit": int(limit),
        "offset": int(offset),
        "in_stock_min_qty": float(IN_STOCK_MIN_QTY),
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(end_date or today_ist_timestamp().date()),
    }
    if df is None or df.empty:
        return empty

    work = filter_inventory_history_window(df, days=days, end_date=end_date)
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["Date", "OMS_SKU"])
    if work.empty:
        return {**empty, "loaded": True}

    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip()
    work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0.0)
    needle = (q or "").strip().upper()
    if needle:
        work = work[work["OMS_SKU"].str.upper().str.contains(needle, na=False)]
    if work.empty:
        return {**empty, "loaded": True}

    work = (
        work.sort_values("Qty", ascending=False)
        .drop_duplicates(subset=["OMS_SKU", "Date"], keep="first")
    )
    dates_sorted = sorted(work["Date"].unique())
    date_strs = [str(pd.Timestamp(d).date()) for d in dates_sorted]

    sku_list = sorted(work["OMS_SKU"].astype(str).unique())
    total = int(len(sku_list))
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    page_skus = sku_list[start:end]
    if not len(page_skus):
        return {**empty, "loaded": True, "dates": date_strs, "total": total}

    page_work = work[work["OMS_SKU"].isin(page_skus)]
    pivot = page_work.pivot(index="OMS_SKU", columns="Date", values="Qty")
    pivot = pivot.reindex(index=page_skus, columns=dates_sorted).fillna(0.0)

    rows = [
        {
            "sku": str(sku),
            "qtys": [float(row.get(d, 0.0) or 0.0) for d in dates_sorted],
        }
        for sku, row in pivot.iterrows()
    ]
    return {
        "loaded": True,
        "dates": date_strs,
        "rows": rows,
        "total": total,
        "limit": int(limit),
        "offset": start,
        "in_stock_min_qty": float(IN_STOCK_MIN_QTY),
        "window_days": int(days if days is not None else _DEFAULT_VIEW_DAYS),
        "window_end": str(end_date or today_ist_timestamp().date()),
    }


_DAILY_INV_META_FILENAME = "daily_inventory_history_meta.json"


def _warm_cache_dir() -> "Path":
    from pathlib import Path

    import os

    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def daily_inventory_history_meta_bundle(sess) -> dict:
    """Serializable metadata for warm-cache / disk restore (shared across users)."""
    df = getattr(sess, "daily_inventory_history_df", None)
    rows = int(len(df)) if df is not None and not getattr(df, "empty", True) else 0
    skus = int(df["OMS_SKU"].astype(str).nunique()) if rows and "OMS_SKU" in df.columns else 0
    mx = inventory_history_max_date(df)
    mn = None
    if df is not None and not getattr(df, "empty", True) and "Date" in df.columns:
        mn = pd.to_datetime(df["Date"], errors="coerce").min()
    return {
        "daily_inventory_history_uploaded_at": str(
            getattr(sess, "daily_inventory_history_uploaded_at", "") or ""
        ),
        "daily_inventory_history_filename": str(
            getattr(sess, "daily_inventory_history_filename", "") or ""
        ),
        "daily_inventory_history_rows": rows,
        "daily_inventory_history_skus": skus,
        "daily_inventory_history_max_date": str(pd.Timestamp(mx).date()) if mx is not None else "",
        "daily_inventory_history_min_date": str(pd.Timestamp(mn).date()) if pd.notna(mn) else "",
    }


def apply_daily_inventory_history_meta(sess, meta: dict) -> None:
    """Copy upload metadata onto a session (not the dataframe)."""
    if not isinstance(meta, dict):
        return
    for key in (
        "daily_inventory_history_uploaded_at",
        "daily_inventory_history_filename",
    ):
        if meta.get(key):
            setattr(sess, key, str(meta[key]))


def read_daily_inventory_history_disk_meta() -> dict | None:
    try:
        path = _warm_cache_dir() / _DAILY_INV_META_FILENAME
        if not path.is_file():
            return None
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def persist_daily_inventory_history_meta(sess) -> bool:
    meta = daily_inventory_history_meta_bundle(sess)
    if not meta.get("daily_inventory_history_uploaded_at") and not meta.get("daily_inventory_history_rows"):
        return False
    try:
        import json

        path = _warm_cache_dir()
        path.mkdir(parents=True, exist_ok=True)
        (path / _DAILY_INV_META_FILENAME).write_text(
            json.dumps(meta, default=str),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def upload_timestamp_epoch(value: str) -> float:
    """Parse uploaded_at strings (IST local or UTC ISO) for comparison."""
    s = str(value or "").strip()
    if not s:
        return 0.0
    try:
        from datetime import datetime

        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_IST).timestamp()
        return dt.timestamp()
    except Exception:
        return 0.0


def daily_inventory_meta_is_newer(meta: dict | None, sess) -> bool:
    """True when shared meta is strictly newer than what the session holds."""
    if not isinstance(meta, dict) or not meta:
        return False
    disk_at = upload_timestamp_epoch(str(meta.get("daily_inventory_history_uploaded_at") or ""))
    sess_at = upload_timestamp_epoch(str(getattr(sess, "daily_inventory_history_uploaded_at", "") or ""))
    if disk_at > sess_at + 0.5:
        return True
    disk_max = str(meta.get("daily_inventory_history_max_date") or "").strip()
    if not disk_max:
        return False
    df = getattr(sess, "daily_inventory_history_df", None)
    sess_max = inventory_history_max_date(df)
    if sess_max is None:
        return True
    try:
        return pd.Timestamp(disk_max).normalize() > pd.Timestamp(sess_max).normalize()
    except Exception:
        return False


__all__ = [
    "parse_daily_inventory_history_dataframes",
    "parse_daily_inventory_history_upload",
    "effective_days_from_history",
    "latest_inventory_qty_by_sku",
    "overlay_inventory_variant_from_history",
    "coverage_days_within",
    "extend_history_with_sales",
    "append_snapshot_inventory_to_history",
    "filter_inventory_history_window",
    "inventory_history_summary",
    "list_inventory_history_dates",
    "inventory_rows_for_date",
    "inventory_history_wide_matrix",
    "daily_inventory_history_meta_bundle",
    "apply_daily_inventory_history_meta",
    "read_daily_inventory_history_disk_meta",
    "persist_daily_inventory_history_meta",
    "upload_timestamp_epoch",
    "daily_inventory_meta_is_newer",
]


# Suppress unused np import lint when the module shrinks during refactors.
_unused_np = np  # type: ignore[unused-ignore]

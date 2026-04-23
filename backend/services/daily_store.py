"""
Persistent daily sales store — SQLite-backed.

Saves up to 30 days of daily order uploads per platform, keyed by
(platform, file_date, filename).  Survives server restarts.

DB path:  /data/daily_sales.db  (production VPS)
          ./daily_sales.db      (local dev fallback)
"""
import io
import os
import re
import sqlite3
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .helpers import (
    apply_dsr_segment_to_df_inplace,
    clean_line_id_series,
    infer_dsr_label_from_upload_filename,
)

# Filename tail for ``infer_dsr_label_from_upload_filename`` — applied on every ``save_daily_file``.
_DSR_TAIL_FOR_PLATFORM = {
    "amazon": "Amazon",
    "myntra": "Myntra",
    "meesho": "Meesho",
    "flipkart": "Flipkart",
    "snapdeal": "Snapdeal",
}

# Meesho LineKeys from TCS / ERP export / CSV fallbacks (not marketplace sub-order ids).
_MEE_SYN_LINEKEY = re.compile(r"^(MEETCS\||MEEEXP\||MEECSV\|)", re.I)
# Flipkart Order Export synthetic ids: product_Sku_YYYYMMDD (no pipes). earn_more uses FKEM|…
_FK_ORDER_EXPORT_LINEKEY = re.compile(r"^[^|]+_[^|]+_\d{8}$")


def _resolve_db_path() -> Path:
    env = (os.environ.get("DAILY_SALES_DB") or "").strip()
    if env:
        return Path(env)
    if Path("/data").exists():
        return Path("/data/daily_sales.db")
    return Path("daily_sales.db")


_DB_PATH = _resolve_db_path()
def _max_files_per_platform() -> int:
    raw = (os.environ.get("DAILY_UPLOADS_MAX_PER_PLATFORM") or "200").strip()
    try:
        n = int(raw)
        return max(30, min(n, 5000))
    except ValueError:
        return 200


# ── Schema ────────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_uploads (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            platform     TEXT    NOT NULL,
            file_date    DATE    NOT NULL,
            filename     TEXT    NOT NULL,
            uploaded_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            rows         INTEGER NOT NULL DEFAULT 0,
            data_parquet BLOB    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_platform_date "
        "ON daily_uploads (platform, file_date)"
    )
    # Migration: add date_from / date_to columns if missing
    try:
        conn.execute("ALTER TABLE daily_uploads ADD COLUMN date_from DATE")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE daily_uploads ADD COLUMN date_to DATE")
    except Exception:
        pass
    conn.commit()
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_file_date(filename: str, df: pd.DataFrame) -> str:
    """
    Derive the report date from filename patterns or from the earliest
    date value in the DataFrame.  Returns ISO date string (YYYY-MM-DD).
    """
    # Pattern: YYYY-MM-DD anywhere in the filename
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if m:
        try:
            d = datetime.date.fromisoformat(m.group(1))
            if 2020 <= d.year <= 2030:
                return str(d)
        except ValueError:
            pass

    # Pattern: DD-MM-YYYY or DD_MM_YYYY
    m = re.search(r"(\d{2})[_-](\d{2})[_-](\d{4})", filename)
    if m:
        try:
            d = datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if 2020 <= d.year <= 2030:
                return str(d)
        except ValueError:
            pass

    # Pattern: D-Mon-YYYY or D Mon YYYY (e.g. "6-Mar-2026", "9 Mar 2026")
    _MON = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    m = re.search(r"(\d{1,2})[-\s]([A-Za-z]{3})[-\s](\d{4})", filename)
    if m:
        try:
            mon = _MON.get(m.group(2).lower())
            if mon:
                d = datetime.date(int(m.group(3)), mon, int(m.group(1)))
                if 2020 <= d.year <= 2030:
                    return str(d)
        except ValueError:
            pass

    # "1 APR to 14" / "01-Apr to 14-Apr" (year missing in path) — use calendar from row dates.
    mrng = re.search(
        r"(?i)(\d{1,2})\D{0,3}(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\D+to\D+(\d{1,2})",
        filename.replace("_", " "),
    )
    if mrng:
        try:
            mon = _MON.get(mrng.group(2).lower())
            if mon and not df.empty:
                for col in ("Date", "TxnDate", "_Date"):
                    if col in df.columns:
                        dates = pd.to_datetime(df[col], errors="coerce").dropna()
                        if not dates.empty:
                            y = int(dates.min().year)
                            d0 = datetime.date(y, mon, int(mrng.group(1)))
                            if 2020 <= d0.year <= 2030:
                                return str(d0)
                        break
        except (ValueError, IndexError):
            pass

    # Fall back to earliest date in the DataFrame
    for col in ("Date", "TxnDate", "_Date", "Customer Shipment Date", "Order Date"):
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if not dates.empty:
                    earliest = dates.min().date()
                    if 2020 <= earliest.year <= 2030:
                        return str(earliest)
            except Exception:
                pass

    return str(datetime.date.today())


def _extract_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    """Extract actual min/max date from a DataFrame. Returns (date_from, date_to) ISO strings."""
    for col in ("Date", "TxnDate", "_Date"):
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if not dates.empty:
                    return str(dates.min().date()), str(dates.max().date())
            except Exception:
                pass
    today = str(datetime.date.today())
    return today, today


def _df_to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    # Strip tz info from any tz-aware datetime columns (parquet requires uniform tz)
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            try:
                df2[col] = df2[col].dt.tz_localize(None)
            except Exception:
                try:
                    df2[col] = df2[col].dt.tz_convert(None)
                except Exception:
                    pass
    df2.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    return buf.getvalue()


def _dedup_linekey_legacy_shadow(d: pd.DataFrame) -> pd.DataFrame:
    """
    After a warm-cache + Tier-3 merge, the same marketplace line can appear twice: once from an
    older snapshot with a weak/parent id (empty LineKey) and again with a stable LineKey.
    Drop the legacy shadow rows when a LineKey-backed twin exists for the same
    (day, SKU, txn, qty, raw status). Used for Myntra, Meesho, and Flipkart.
    """
    if d.empty or "LineKey" not in d.columns:
        return d
    if not all(c in d.columns for c in ("OMS_SKU", "Date", "TxnType", "Quantity")):
        return d
    work = d.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_q"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    if "RawStatus" in work.columns:
        work["_rs"] = work["RawStatus"].astype(str).str.strip()
    else:
        work["_rs"] = ""
    lk = work["LineKey"].fillna("").astype(str).str.strip()
    has_lk = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
    work["_prio"] = has_lk.astype(int)

    drop_idx: List[Any] = []
    for _, g in work.groupby(["_day", "OMS_SKU", "TxnType", "_q", "_rs"], sort=False):
        if len(g) <= 1:
            continue
        pr = g["_prio"]
        if (pr == 1).any() and (pr == 0).any():
            drop_idx.extend(g.index[pr == 0].tolist())
        elif (pr == 0).all() and len(g) > 1:
            drop_idx.extend(g.index[1:].tolist())

    if not drop_idx:
        return d
    return d.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


def _meesho_synthetic_line_mask(lk: pd.Series) -> pd.Series:
    s = lk.fillna("").astype(str).str.strip()
    return s.str.match(_MEE_SYN_LINEKEY, na=False)


def _dedup_meesho_cross_source_overlay(d: pd.DataFrame) -> pd.DataFrame:
    """
    The same Meesho shipment often appears twice: supplier TCS (MEETCS|…) vs ERP export
    (MEEEXP|…) vs daily CSV (MEECSV|…) with different LineKeys but the same day / SKU /
    txn / qty. Prefer real sub-order LineKeys; among synthetics prefer TCS > export > CSV.

    Fingerprint includes RawStatus + rounded invoice amount so unrelated orders sharing
    SKU/qty/day are not merged. When mixing real + synthetic, only drop a **single**
    synthetic twin (classic duplicate row); if multiple synthetics remain, do not drop —
    avoids under-counting distinct orders that shared a coarse bucket.
    """
    if d.empty or not all(
        c in d.columns for c in ("Date", "OMS_SKU", "TxnType", "Quantity", "LineKey")
    ):
        return d
    work = d.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_q"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    work["_skuN"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    if "RawStatus" in work.columns:
        work["_rs"] = work["RawStatus"].astype(str).str.strip()
    else:
        work["_rs"] = ""
    if "Invoice_Amount" in work.columns:
        work["_amt"] = (
            pd.to_numeric(work["Invoice_Amount"], errors="coerce").fillna(0).round().astype("int64")
        )
    else:
        work["_amt"] = 0
    lk = work["LineKey"].fillna("").astype(str).str.strip()
    syn_m = _meesho_synthetic_line_mask(lk)
    real_m = (~syn_m) & lk.ne("") & ~lk.str.lower().isin(["nan", "none"])

    drop_idx: List[Any] = []
    for _, g in work.groupby(["_day", "_skuN", "TxnType", "_q", "_rs", "_amt"], sort=False):
        idx = g.index.tolist()
        if len(idx) <= 1:
            continue
        sm = syn_m.loc[idx]
        rm = real_m.loc[idx]
        if rm.any() and sm.any():
            if int(sm.sum()) == 1:
                drop_idx.extend([i for i in idx if bool(sm.loc[i])])
            continue
        if sm.all():
            lu = lk.loc[idx].str.upper()
            pri_series = pd.Series(0, index=idx, dtype="int32")
            for i in idx:
                u = lu.loc[i]
                if u.startswith("MEETCS|"):
                    pri_series.loc[i] = 0
                elif u.startswith("MEEEXP|"):
                    pri_series.loc[i] = 1
                else:
                    pri_series.loc[i] = 2
            best = int(pri_series.min())
            if (pri_series == best).all():
                continue
            drop_idx.extend([i for i in idx if int(pri_series.loc[i]) > best])

    if not drop_idx:
        return d
    return d.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


def _dedup_meesho_suborder_cross_source(d: pd.DataFrame) -> pd.DataFrame:
    """
    Supplier TCS uses ``sub_order_num`` as the id; daily CSV should prefer Sub Order as
    ``LineKey`` (packet id is fallback). Same
    (day, SKU, txn, qty, sub-order) must collapse to one row: drop synthetic LineKey
    twins, then exact duplicates from re-uploads.
    """
    if d.empty or not all(
        c in d.columns for c in ("Date", "OMS_SKU", "TxnType", "Quantity", "LineKey", "OrderId")
    ):
        return d
    work = d.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_q"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    work["_skuN"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    if "MeeshoSubOrder" in work.columns:
        work["_sub"] = clean_line_id_series(work["MeeshoSubOrder"])
    else:
        work["_sub"] = clean_line_id_series(work["OrderId"])
    lk = work["LineKey"].fillna("").astype(str).str.strip()
    syn_m = lk.str.match(_MEE_SYN_LINEKEY, na=False)

    drop_idx: List[Any] = []
    sub_ok = work["_sub"].ne("") & ~work["_sub"].str.lower().isin(["nan", "none"])
    if sub_ok.any():
        for _, g in work.loc[sub_ok].groupby(["_day", "_skuN", "TxnType", "_q", "_sub"], sort=False):
            idx = g.index.tolist()
            if len(idx) <= 1:
                continue
            sm = syn_m.reindex(idx)
            if sm.any() and (~sm).any():
                drop_idx.extend([i for i in idx if bool(sm.loc[i])])

    out = d.drop(index=drop_idx, errors="ignore").reset_index(drop=True) if drop_idx else d
    if out.empty:
        return out

    w2 = out.copy()
    w2["_day"] = pd.to_datetime(w2["Date"], errors="coerce").dt.normalize()
    w2["_q"] = pd.to_numeric(w2["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    w2["_skuN"] = w2["OMS_SKU"].astype(str).str.strip().str.upper()
    if "MeeshoSubOrder" in w2.columns:
        w2["_sub"] = clean_line_id_series(w2["MeeshoSubOrder"])
    else:
        w2["_sub"] = clean_line_id_series(w2["OrderId"])
    ok = w2["_sub"].ne("") & ~w2["_sub"].str.lower().isin(["nan", "none"])
    if not ok.any():
        return out
    cols_tmp = ["_day", "_q", "_skuN", "_sub"]
    part = (
        w2.loc[ok]
        .drop_duplicates(subset=["_day", "_skuN", "TxnType", "_q", "_sub"], keep="last")
        .drop(columns=cols_tmp, errors="ignore")
    )
    rest = w2.loc[~ok].drop(columns=cols_tmp, errors="ignore")
    return pd.concat([rest, part], ignore_index=True)


def _flipkart_synthetic_line_mask(lk: pd.Series) -> pd.Series:
    s = lk.fillna("").astype(str).str.strip()
    m_exp = s.str.match(_FK_ORDER_EXPORT_LINEKEY, na=False)
    m_em = s.str.startswith("FKEM|")
    return m_exp | m_em


def _dedup_flipkart_cross_source_overlay(d: pd.DataFrame) -> pd.DataFrame:
    """
    Flipkart Sales Report rows carry real Order IDs; Order Export and earn_more_report use
    synthetic LineKeys. Same (day, SKU, txn, qty, status, amount) can appear in multiple
    files. When a real row and synthetics share a bucket, drop only a **single** synthetic
    twin (Meesho-style); dropping every synthetic under-counted vs overlapping exports.
    Among all-synthetic buckets, prefer earn_more over order-export LineKeys.
    """
    if d.empty or not all(
        c in d.columns for c in ("Date", "OMS_SKU", "TxnType", "Quantity", "LineKey")
    ):
        return d
    work = d.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_q"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    work["_skuN"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    if "RawStatus" in work.columns:
        work["_rs"] = work["RawStatus"].astype(str).str.strip()
    else:
        work["_rs"] = ""
    if "Invoice_Amount" in work.columns:
        work["_amt"] = (
            pd.to_numeric(work["Invoice_Amount"], errors="coerce").fillna(0).round().astype("int64")
        )
    else:
        work["_amt"] = 0
    lk = work["LineKey"].fillna("").astype(str).str.strip()
    syn_m = _flipkart_synthetic_line_mask(lk)
    real_m = (~syn_m) & lk.ne("") & ~lk.str.lower().isin(["nan", "none"])

    drop_idx: List[Any] = []
    for _, g in work.groupby(["_day", "_skuN", "TxnType", "_q", "_rs", "_amt"], sort=False):
        idx = g.index.tolist()
        if len(idx) <= 1:
            continue
        sm = syn_m.loc[idx]
        rm = real_m.loc[idx]
        if rm.any() and sm.any():
            # Same idea as Meesho overlay: drop only a single obvious synthetic twin.
            # Multiple synthetics in one coarse bucket may be distinct orders — dropping
            # all of them under-counted vs seller Excel when exports overlap.
            if int(sm.sum()) == 1:
                drop_idx.extend([i for i in idx if bool(sm.loc[i])])
            continue
        if sm.all():
            pri_series = pd.Series(2, index=idx, dtype="int32")
            for i in idx:
                u = lk.loc[i]
                if u.startswith("FKEM|"):
                    pri_series.loc[i] = 1
                elif bool(re.match(_FK_ORDER_EXPORT_LINEKEY, u)):
                    pri_series.loc[i] = 0
                else:
                    pri_series.loc[i] = 2
            best = int(pri_series.min())
            if (pri_series == best).all():
                continue
            drop_idx.extend([i for i in idx if int(pri_series.loc[i]) > best])

    if not drop_idx:
        return d
    return d.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


def _dedup_myntra_parent_order_shadow(d: pd.DataFrame) -> pd.DataFrame:
    """
    PPMP / seller CSVs sometimes repeat the same line once with ``order line id`` and
    again with only ``store order id`` as the LineKey. Same calendar fingerprint
    (day, SKU, txn, qty) — drop the parent-only row **only** when a line-level row in
    the same group shares that ``ParentOrderId`` (same marketplace order). Otherwise
    unrelated orders that happen to share SKU/qty/day would incorrectly lose rows.
    """
    if d.empty or "ParentOrderId" not in d.columns:
        return d
    if not all(c in d.columns for c in ("Date", "OMS_SKU", "TxnType", "Quantity", "LineKey")):
        return d
    work = d.copy()
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_q"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    work["_skuN"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    lk = clean_line_id_series(work["LineKey"])
    par = clean_line_id_series(work["ParentOrderId"])

    drop_idx: List[Any] = []
    for _, g in work.groupby(["_day", "_skuN", "TxnType", "_q"], sort=False):
        idx = g.index.tolist()
        if len(idx) <= 1:
            continue
        parent_rows = [
            i for i in idx
            if lk.loc[i] and par.loc[i] and lk.loc[i] == par.loc[i]
        ]
        if not parent_rows:
            continue
        line_rows = [
            i for i in idx
            if lk.loc[i] and par.loc[i] and lk.loc[i] != par.loc[i]
        ]
        if not line_rows:
            continue
        parents_with_line_child = {par.loc[j] for j in line_rows if par.loc[j]}
        for i in parent_rows:
            if lk.loc[i] in parents_with_line_child:
                drop_idx.append(i)

    if not drop_idx:
        return d
    return d.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


# Columns that differ when the *same* fulfilment line is re-exported under another id
# (packet id vs order line id, synthetic LineKey vs marketplace id, overlapping ZIPs).
_CROSS_EXPORT_ID_COLS = frozenset({
    "LineKey",
    "OrderId",
    "Month",
    "Month_Label",
})


def _dedup_cross_export_id_twins(d: pd.DataFrame) -> pd.DataFrame:
    """
    After LineKey / OrderId dedup, still remove rows whose **substantive** fields are
    identical — duplicate monthly + daily uploads, PPMP vs seller report, or two id
    columns populated differently for the same line (different LineKey only).

    Excludes identity-ish columns from the fingerprint so true twin rows collapse.
    """
    if d.empty or len(d) < 2:
        return d
    subset = [c for c in d.columns if c not in _CROSS_EXPORT_ID_COLS]
    if len(subset) < 4:
        return d
    return d.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)


def _dedup_shipment_superseded_by_same_day_refund(d: pd.DataFrame) -> pd.DataFrame:
    """
    Tier-3 merges (concat existing + new upload) keep Shipment and Refund as distinct keys,
    so a line that moved to **RTO / return** can leave a stale **Shipment** row from an older
    snapshot alongside the current **Refund** row — same marketplace line id, same SKU/qty,
    same reporting day. Drop the shadow Shipment so gross/return counts match seller exports.

    Used for **Myntra** and **Meesho**. Different calendar days (ship in January, return in
    February) are left intact.
    """
    if d.empty or "TxnType" not in d.columns:
        return d
    required = ("OrderId", "Date", "OMS_SKU", "Quantity", "TxnType")
    if not all(c in d.columns for c in required):
        return d
    work = d.copy()
    oid = work["OrderId"].astype(str).str.strip()
    if "LineKey" in work.columns:
        lk = work["LineKey"].fillna("").astype(str).str.strip()
        use_lk = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
        oid = oid.where(~use_lk, lk)
    work["_ded_id"] = oid
    work["_day"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["_sku"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    work["_qty"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0).round().astype("int64")
    work["_txn"] = work["TxnType"].astype(str).str.strip()

    drop_idx: List[Any] = []
    for key, g in work.groupby(["_ded_id", "_day", "_sku", "_qty"], sort=False):
        ded_id = key[0]
        if not ded_id or str(ded_id).lower() in ("nan", "none", ""):
            continue
        txns = set(g["_txn"])
        if "Shipment" in txns and "Refund" in txns:
            drop_idx.extend(g.index[g["_txn"] == "Shipment"].tolist())

    if not drop_idx:
        return d
    return d.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


def _dedup_platform_df(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Deduplicate a concatenated platform DataFrame to remove inflated rows
    caused by overlapping file uploads.
    - Amazon (mtr_df): MTR rows (Invoice_Number filled) take priority over
      FBA Shipment Report rows (no Invoice_Number) for the same Order_Id.
    - Other platforms: rows WITH a real OrderId are deduped by
      (OrderId, SKU, TxnType, Date) — composite key preserves multi-SKU orders
      (same store order id, different SKUs) while preventing re-upload duplicates.
      Rows WITHOUT an OrderId (aggregated/summary data) are kept as-is.
    - Myntra / Meesho / Flipkart: prefer ``LineKey`` when present; see
      ``_dedup_linekey_legacy_shadow``. Strong LineKeys dedupe without OMS_SKU for Myntra /
      Meesho (line-level ids). Flipkart includes ``OMS_SKU`` in the strong key because
      marketplace Order IDs are often shared across multiple line SKUs.
    - Final pass ``_dedup_cross_export_id_twins`` for those channels: rows whose substantive
      columns match but ``LineKey`` / ``OrderId`` differ (duplicate ZIPs, packet vs line id)
      collapse to one row — avoids ~2× counts when exports disagree on ids only.
    """
    if df.empty:
        return df
    try:
        if platform == "amazon" and "Invoice_Number" in df.columns and "Order_Id" in df.columns:
            from .mtr import dedup_amazon_mtr_dataframe
            return dedup_amazon_mtr_dataframe(df)
        elif "OrderId" in df.columns:
            d = df.copy()
            oid_raw = d["OrderId"].astype(str).str.strip()
            if "LineKey" in d.columns:
                lk = d["LineKey"].fillna("").astype(str).str.strip()
                use_lk = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
                d["_ded_id"] = oid_raw.where(~use_lk, lk)
            else:
                d["_ded_id"] = oid_raw
            has_id = d["_ded_id"].ne("") & ~d["_ded_id"].str.lower().isin(["nan", "none"])
            sku_col = "OMS_SKU" if "OMS_SKU" in d.columns else ("SKU" if "SKU" in d.columns else None)
            date_col = "Date" if "Date" in d.columns else None
            txn_col  = "TxnType" if "TxnType" in d.columns else None
            if date_col:
                d["_ded_date"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
            if "Quantity" in d.columns:
                d["_qtyk"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0).round().astype("int64")
            key = ["_ded_id"]
            if sku_col:
                key.append(sku_col)
            if txn_col:
                key.append(txn_col)
            if date_col:
                key.append("_ded_date")
            if "Quantity" in d.columns:
                key.append("_qtyk")

            with_id = d[has_id]
            parts: List[pd.DataFrame] = []
            if not with_id.empty:
                if "LineKey" in d.columns:
                    lk = with_id["LineKey"].fillna("").astype(str).str.strip()
                    sk_mask = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
                else:
                    sk_mask = pd.Series(False, index=with_id.index)
                ws = with_id.loc[sk_mask]
                ww = with_id.loc[~sk_mask]
                if not ws.empty:
                    key_s = ["_ded_id"]
                    if txn_col and txn_col in ws.columns:
                        key_s.append(txn_col)
                    # Flipkart LineKey is often marketplace Order ID shared by multiple line SKUs —
                    # include SKU so we do not collapse multi-item orders to one row.
                    if platform == "flipkart" and sku_col and sku_col in ws.columns:
                        key_s.append(sku_col)
                    parts.append(ws.drop_duplicates(subset=key_s, keep="last"))
                if not ww.empty:
                    key_w = [c for c in key if c in ww.columns]
                    parts.append(ww.drop_duplicates(subset=key_w, keep="last"))
            with_id_out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            no_id = d[~has_id]
            out = pd.concat([with_id_out, no_id], ignore_index=True)
            out = out.drop(columns=["_ded_date", "_qtyk", "_ded_id"], errors="ignore")
            if platform in ("myntra", "meesho", "flipkart"):
                out = _dedup_linekey_legacy_shadow(out)
            if platform == "myntra":
                out = _dedup_myntra_parent_order_shadow(out)
                out = _dedup_shipment_superseded_by_same_day_refund(out)
            if platform == "meesho":
                out = _dedup_meesho_cross_source_overlay(out)
                out = _dedup_meesho_suborder_cross_source(out)
                out = _dedup_shipment_superseded_by_same_day_refund(out)
            elif platform == "flipkart":
                out = _dedup_flipkart_cross_source_overlay(out)
            if platform in ("myntra", "meesho", "flipkart"):
                out = _dedup_cross_export_id_twins(out)
            return out
    except Exception:
        pass
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def save_daily_file(
    platform: str,
    filename: str,
    df: pd.DataFrame,
) -> Tuple[str, int]:
    """
    Persist a daily upload.
    - Stamps ``DSR_Segment`` from ``filename`` when it matches
      ``…_<Label> <Amazon|Myntra|…>…`` so SQLite + brand rollups never miss seller tags.
    - Replaces any existing entries for the same platform whose date range
      overlaps with the new file's actual data date range (prevents duplication
      when re-uploading or uploading wider date-range reports).
    - Auto-trims: only the latest ``DAILY_UPLOADS_MAX_PER_PLATFORM`` entries per platform.
    Returns (file_date, rows_saved).
    """
    if df.empty:
        return str(datetime.date.today()), 0

    tail = _DSR_TAIL_FOR_PLATFORM.get(platform)
    if tail:
        apply_dsr_segment_to_df_inplace(df, filename, tail)

    file_date = _extract_file_date(filename, df)
    date_from, date_to = _extract_date_range(df)
    parquet_bytes = _df_to_parquet(df)

    conn = _get_conn()

    # Delete only entries with the same filename (exact re-upload replacement).
    # Do NOT delete based on date range overlap — multiple seller accounts legitimately
    # upload different files covering the same date range. Deduplication at load time
    # (via _dedup_platform_df) prevents double-counting when the same orders appear
    # in multiple files.
    conn.execute(
        "DELETE FROM daily_uploads WHERE platform=? AND filename=?",
        (platform, filename),
    )

    conn.execute(
        "INSERT INTO daily_uploads (platform, file_date, filename, rows, data_parquet, date_from, date_to) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (platform, file_date, filename, len(df), parquet_bytes, date_from, date_to),
    )
    # Trim: keep only the latest N blobs per platform (see DAILY_UPLOADS_MAX_PER_PLATFORM).
    cap = _max_files_per_platform()
    conn.execute(
        """DELETE FROM daily_uploads
           WHERE platform=? AND id NOT IN (
               SELECT id FROM daily_uploads
               WHERE platform=?
               ORDER BY file_date DESC, id DESC
               LIMIT ?
           )""",
        (platform, platform, cap),
    )
    conn.commit()
    conn.close()
    return file_date, len(df)


def load_platform_data(platform: str, months: int | None = None) -> pd.DataFrame:
    """Load and concatenate stored daily data for one platform, with deduplication.

    ``months=None`` (default): load **all** blobs for the platform. Tier-1 bulk uploads
    often carry ``file_date`` from the earliest row in the archive (e.g. 2024); a rolling
    window would silently drop multi-year history and make yearly data look “not uploaded”.
    Pass a positive ``months`` only if you need an intentional memory-bound window.
    """
    conn = _get_conn()
    if months is None:
        rows = conn.execute(
            "SELECT filename, data_parquet FROM daily_uploads "
            "WHERE platform=? ORDER BY file_date ASC",
            (platform,),
        ).fetchall()
    else:
        cutoff = (datetime.date.today() - datetime.timedelta(days=months * 30)).isoformat()
        rows = conn.execute(
            "SELECT filename, data_parquet FROM daily_uploads "
            "WHERE platform=? AND file_date >= ? ORDER BY file_date ASC",
            (platform, cutoff),
        ).fetchall()
    conn.close()

    dfs = []
    tail = _DSR_TAIL_FOR_PLATFORM.get(platform)
    cutoff_ts = None
    if months is not None:
        try:
            cutoff_ts = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=months * 30))
        except Exception:
            cutoff_ts = None
    for (filename, blob) in rows:
        try:
            d = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
            # Legacy repair: old blobs (saved before DSR stamping) can still have blank
            # segments. Fill only missing values from the upload filename label.
            if tail and not d.empty:
                label = infer_dsr_label_from_upload_filename(filename, tail)
                if label:
                    if "DSR_Segment" not in d.columns:
                        d["DSR_Segment"] = label
                    else:
                        seg = d["DSR_Segment"].fillna("").astype(str).str.strip()
                        miss = seg.str.len().eq(0) | seg.str.casefold().isin({"all", "nan", "none"})
                        if miss.any():
                            d.loc[miss, "DSR_Segment"] = label
            # Row-level recency filter: file_date can be recent while rows inside are older.
            # Keep restore bounded by actual transaction dates when months is requested.
            if cutoff_ts is not None and not d.empty:
                dt_col = None
                for c in ("TxnDate", "Date", "Order Date", "order_date"):
                    if c in d.columns:
                        dt_col = c
                        break
                if dt_col is not None:
                    _dt = pd.to_datetime(d[dt_col], errors="coerce")
                    d = d[_dt >= cutoff_ts]
            if not d.empty:
                dfs.append(d)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return _dedup_platform_df(combined, platform)


def load_all_platforms() -> Dict[str, pd.DataFrame]:
    """Return {platform: df} for all platforms that have stored data."""
    result: Dict[str, pd.DataFrame] = {}
    for p in ("amazon", "myntra", "meesho", "flipkart", "snapdeal"):
        df = load_platform_data(p)
        if not df.empty:
            result[p] = df
    return result


def backfill_dsr_segments_in_store(*, dry_run: bool = True) -> dict:
    """
    One-time repair for legacy daily blobs saved before DSR filename stamping.
    Fills missing ``DSR_Segment`` values from the upload filename label per platform.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, platform, filename, data_parquet FROM daily_uploads ORDER BY id ASC"
    ).fetchall()

    scanned = len(rows)
    changed_files = 0
    changed_rows = 0
    skipped = 0

    for row_id, platform, filename, blob in rows:
        tail = _DSR_TAIL_FOR_PLATFORM.get(str(platform).strip().lower())
        if not tail:
            skipped += 1
            continue
        label = infer_dsr_label_from_upload_filename(filename, tail)
        if not label:
            skipped += 1
            continue
        try:
            d = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
        except Exception:
            skipped += 1
            continue
        if d.empty:
            skipped += 1
            continue

        changed_here = 0
        if "DSR_Segment" not in d.columns:
            changed_here = len(d)
            d["DSR_Segment"] = label
        else:
            seg = d["DSR_Segment"].fillna("").astype(str).str.strip()
            miss = seg.str.len().eq(0) | seg.str.casefold().isin({"all", "nan", "none"})
            changed_here = int(miss.sum())
            if changed_here > 0:
                d.loc[miss, "DSR_Segment"] = label

        if changed_here > 0:
            changed_files += 1
            changed_rows += changed_here
            if not dry_run:
                conn.execute(
                    "UPDATE daily_uploads SET data_parquet=?, rows=? WHERE id=?",
                    (_df_to_parquet(d), len(d), row_id),
                )

    if not dry_run:
        conn.commit()
    conn.close()
    return {
        "dry_run": dry_run,
        "scanned_files": scanned,
        "changed_files": changed_files,
        "changed_rows": changed_rows,
        "skipped_files": skipped,
    }


def merge_platform_data(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
    platform: str,
    *,
    source_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge two platform DataFrames with proper deduplication.
    Safe to call from any module. Uses _dedup_platform_df internally (including when
    ``existing`` is empty so the first upload still gets Meesho/Flipkart overlays).
    - Amazon: invoice/order/qty keys + PL SKU normalisation so overlapping Tier-1 ZIPs
      do not double-count the same shipment.
    - Other platforms: OrderId + SKU + txn + calendar day (and qty when present);
      newer upload wins (keep last after concat [existing, new]).
    - ``source_filename``: when set, stamps ``DSR_Segment`` from the filename (defense when
      merge runs without a prior ``save_daily_file`` on the same frame).
    """
    if not new_df.empty and source_filename:
        tail = _DSR_TAIL_FOR_PLATFORM.get(platform)
        if tail:
            apply_dsr_segment_to_df_inplace(new_df, source_filename, tail)

    if existing.empty:
        return _dedup_platform_df(new_df.copy() if not new_df.empty else new_df, platform)
    if new_df.empty:
        return existing
    combined = pd.concat([existing, new_df], ignore_index=True)
    return _dedup_platform_df(combined, platform)


def list_uploads() -> List[dict]:
    """Return metadata for all uploads (newest first), no blob."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, platform, file_date, filename, uploaded_at, rows, date_from, date_to
        FROM daily_uploads
        ORDER BY datetime(uploaded_at) DESC
        """
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0], "platform": r[1], "file_date": r[2],
            "filename": r[3], "uploaded_at": r[4], "rows": r[5],
            "date_from": r[6], "date_to": r[7],
        }
        for r in rows
    ]


def get_summary() -> dict:
    """Per-platform: min_date, max_date, total_rows, file_count.

    min/max prefer **actual** row date ranges (``date_from`` / ``date_to`` per upload)
    so Flipkart "1–14 Apr" files do not all look like a single ``file_date`` day.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT platform,
               MIN(COALESCE(NULLIF(TRIM(COALESCE(date_from, '')), ''), file_date)),
               MAX(COALESCE(NULLIF(TRIM(COALESCE(date_to, '')), ''), file_date)),
               SUM(rows), COUNT(*)
        FROM daily_uploads
        GROUP BY platform
        """
    ).fetchall()
    conn.close()
    return {
        r[0]: {
            "min_date":   r[1],
            "max_date":   r[2],
            "total_rows": r[3],
            "file_count": r[4],
        }
        for r in rows
    }


def delete_upload(upload_id: int) -> bool:
    """Delete one upload by id. Returns True if a row was deleted."""
    conn = _get_conn()
    conn.execute("DELETE FROM daily_uploads WHERE id=?", (upload_id,))
    conn.commit()
    changed = conn.total_changes > 0
    conn.close()
    return changed


def clear_all_daily_uploads() -> int:
    """Remove every Tier-3 daily snapshot from SQLite. Returns number of rows deleted."""
    conn = _get_conn()
    cur = conn.execute("SELECT COUNT(*) FROM daily_uploads")
    n = int(cur.fetchone()[0])
    conn.execute("DELETE FROM daily_uploads")
    conn.commit()
    conn.close()
    return n

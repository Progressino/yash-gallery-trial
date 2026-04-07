"""
Persistent daily sales store — SQLite-backed.

Saves up to 30 days of daily order uploads per platform, keyed by
(platform, file_date, filename).  Survives server restarts.

DB path:  /data/daily_sales.db  (production VPS)
          ./daily_sales.db      (local dev fallback)
"""
import io
import re
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_DB_PATH = Path("/data/daily_sales.db") if Path("/data").exists() else Path("daily_sales.db")
_MAX_FILES = 60   # keep at most 60 entries per platform


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
    """
    if df.empty:
        return df
    try:
        if platform == "amazon" and "Invoice_Number" in df.columns and "Order_Id" in df.columns:
            from .mtr import dedup_amazon_mtr_dataframe
            return dedup_amazon_mtr_dataframe(df)
        elif "OrderId" in df.columns:
            d = df.copy()
            has_id = d["OrderId"].astype(str).str.strip() != ""
            sku_col = "OMS_SKU" if "OMS_SKU" in d.columns else ("SKU" if "SKU" in d.columns else None)
            date_col = "Date" if "Date" in d.columns else None
            txn_col  = "TxnType" if "TxnType" in d.columns else None
            if date_col:
                d["_ded_date"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
            if "Quantity" in d.columns:
                d["_qtyk"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0).round().astype("int64")
            key = ["OrderId"]
            if sku_col:
                key.append(sku_col)
            if txn_col:
                key.append(txn_col)
            if date_col:
                key.append("_ded_date")
            if "Quantity" in d.columns:
                key.append("_qtyk")
            with_id = d[has_id].drop_duplicates(subset=key, keep="last")
            no_id = d[~has_id]
            out = pd.concat([with_id, no_id], ignore_index=True)
            out = out.drop(columns=["_ded_date", "_qtyk"], errors="ignore")
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
    - Replaces any existing entries for the same platform whose date range
      overlaps with the new file's actual data date range (prevents duplication
      when re-uploading or uploading wider date-range reports).
    - Auto-trims: only the latest _MAX_FILES entries per platform are kept.
    Returns (file_date, rows_saved).
    """
    if df.empty:
        return str(datetime.date.today()), 0

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
    # Trim: keep only the latest _MAX_FILES per platform
    conn.execute(
        """DELETE FROM daily_uploads
           WHERE platform=? AND id NOT IN (
               SELECT id FROM daily_uploads
               WHERE platform=?
               ORDER BY file_date DESC, id DESC
               LIMIT ?
           )""",
        (platform, platform, _MAX_FILES),
    )
    conn.commit()
    conn.close()
    return file_date, len(df)


def load_platform_data(platform: str, months: int = 24) -> pd.DataFrame:
    """Load and concatenate stored daily data for one platform, with deduplication.
    Only loads files whose file_date falls within the last `months` months (default 48)
    to cap memory usage on large datasets.
    """
    conn = _get_conn()
    cutoff = (datetime.date.today() - datetime.timedelta(days=months * 30)).isoformat()
    rows = conn.execute(
        "SELECT data_parquet FROM daily_uploads "
        "WHERE platform=? AND file_date >= ? ORDER BY file_date ASC",
        (platform, cutoff),
    ).fetchall()
    conn.close()

    dfs = []
    for (blob,) in rows:
        try:
            dfs.append(pd.read_parquet(io.BytesIO(blob), engine="pyarrow"))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return _dedup_platform_df(combined, platform)


def load_all_platforms() -> Dict[str, pd.DataFrame]:
    """Return {platform: df} for all platforms that have stored data."""
    result: Dict[str, pd.DataFrame] = {}
    for p in ("amazon", "myntra", "meesho", "flipkart"):
        df = load_platform_data(p)
        if not df.empty:
            result[p] = df
    return result


def merge_platform_data(existing: pd.DataFrame, new_df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Merge two platform DataFrames with proper deduplication.
    Safe to call from any module. Uses _dedup_platform_df internally.
    - Amazon: invoice/order/qty keys + PL SKU normalisation so overlapping Tier-1 ZIPs
      do not double-count the same shipment.
    - Other platforms: OrderId + SKU + txn + calendar day (and qty when present);
      newer upload wins (keep last after concat [existing, new]).
    """
    if existing.empty:
        return new_df.copy() if not new_df.empty else new_df
    if new_df.empty:
        return existing
    combined = pd.concat([existing, new_df], ignore_index=True)
    return _dedup_platform_df(combined, platform)


def list_uploads() -> List[dict]:
    """Return metadata for all uploads (newest first), no blob."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, platform, file_date, filename, uploaded_at, rows, date_from, date_to "
        "FROM daily_uploads "
        "ORDER BY file_date DESC, uploaded_at DESC"
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
    """Per-platform: min_date, max_date, total_rows, file_count."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT platform, MIN(file_date), MAX(file_date), SUM(rows), COUNT(*) "
        "FROM daily_uploads GROUP BY platform"
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

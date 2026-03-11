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
_MAX_FILES = 30   # keep at most 30 entries per platform


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


# ── Public API ────────────────────────────────────────────────────────────────

def save_daily_file(
    platform: str,
    filename: str,
    df: pd.DataFrame,
) -> Tuple[str, int]:
    """
    Persist a daily upload.
    - Deduplicates: existing row with same (platform, file_date, filename) is replaced.
    - Auto-trims: only the latest _MAX_FILES entries per platform are kept.
    Returns (file_date, rows_saved).
    """
    if df.empty:
        return str(datetime.date.today()), 0

    file_date = _extract_file_date(filename, df)
    parquet_bytes = _df_to_parquet(df)

    conn = _get_conn()
    # Replace if already exists
    conn.execute(
        "DELETE FROM daily_uploads WHERE platform=? AND file_date=? AND filename=?",
        (platform, file_date, filename),
    )
    conn.execute(
        "INSERT INTO daily_uploads (platform, file_date, filename, rows, data_parquet) "
        "VALUES (?, ?, ?, ?, ?)",
        (platform, file_date, filename, len(df), parquet_bytes),
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


def load_platform_data(platform: str) -> pd.DataFrame:
    """Load and concatenate all stored daily data for one platform."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT data_parquet FROM daily_uploads "
        "WHERE platform=? ORDER BY file_date ASC",
        (platform,),
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
    return pd.concat(dfs, ignore_index=True)


def load_all_platforms() -> Dict[str, pd.DataFrame]:
    """Return {platform: df} for all platforms that have stored data."""
    result: Dict[str, pd.DataFrame] = {}
    for p in ("amazon", "myntra", "meesho", "flipkart"):
        df = load_platform_data(p)
        if not df.empty:
            result[p] = df
    return result


def list_uploads() -> List[dict]:
    """Return metadata for all uploads (newest first), no blob."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, platform, file_date, filename, uploaded_at, rows "
        "FROM daily_uploads "
        "ORDER BY file_date DESC, uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0], "platform": r[1], "file_date": r[2],
            "filename": r[3], "uploaded_at": r[4], "rows": r[5],
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

#!/usr/bin/env python3
"""Repair Myntra Tier-3 parquets: single-day Seller Orders → report day, all rows Shipment."""
from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import clear_tier3_range_cache, save_daily_file  # noqa: E402
from backend.services.myntra import _myntra_seller_single_report_day  # noqa: E402


def main() -> int:
    db = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/daily_sales.db")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT filename, data_parquet FROM daily_uploads WHERE platform='myntra'"
    ).fetchall()
    conn.close()
    fixed = 0
    for filename, blob in rows:
        report_day = _myntra_seller_single_report_day(filename)
        if report_day is None:
            continue
        df = pd.read_parquet(io.BytesIO(blob))
        if df.empty or "Date" not in df.columns:
            continue
        before = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df = df.copy()
        df["Date"] = report_day
        if "TxnType" in df.columns and not filename.upper().startswith("RT "):
            df["TxnType"] = "Shipment"
        save_daily_file("myntra", filename, df)
        after_day = pd.Timestamp(report_day).normalize()
        moved = int((before != after_day).sum()) if before.notna().any() else len(df)
        print("fixed", filename.split("/")[-1][:70], "rows", len(df), "report_day", after_day.date(), "moved", moved)
        fixed += 1
    clear_tier3_range_cache()
    print("done", fixed, "uploads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

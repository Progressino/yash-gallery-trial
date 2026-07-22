#!/usr/bin/env python3
"""Re-save Myntra Tier-3 daily uploads from local CSV paths (prod repair after date-rule change)."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import clear_tier3_range_cache, save_daily_file  # noqa: E402
from backend.services.myntra import _parse_myntra_csv  # noqa: E402
from backend.services.sku_mapping import load_sku_mapping_from_disk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_files", nargs="+", help="Seller_Orders_Report CSV files")
    ap.add_argument("--db", default="/data/daily_sales.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    mapping = load_sku_mapping_from_disk() or {}
    conn = sqlite3.connect(args.db)
    known = {
        Path(row[0].replace("\\", "/").split("/")[-1]).name: row[0]
        for row in conn.execute(
            "SELECT filename FROM daily_uploads WHERE platform='myntra'"
        ).fetchall()
    }
    conn.close()
    updated = 0
    for path_str in args.csv_files:
        p = Path(path_str)
        if not p.is_file():
            print("MISSING", p)
            continue
        tier_name = known.get(p.name)
        if not tier_name:
            print("NO TIER3 MATCH", p.name)
            continue
        raw = p.read_bytes()
        df, msg = _parse_myntra_csv(raw, p.name, mapping)
        if df.empty:
            print("PARSE FAIL", p.name, msg)
            continue
        day = df["Date"].dt.normalize()
        import pandas as pd

        d0, d1 = day.min(), day.max()
        print(
            "OK",
            p.name,
            "rows",
            len(df),
            "range",
            str(pd.Timestamp(d0).date()),
            str(pd.Timestamp(d1).date()),
            "tier3",
            tier_name[:70],
        )
        if not args.dry_run:
            save_daily_file("myntra", tier_name, df)
            updated += 1
    if not args.dry_run and updated:
        clear_tier3_range_cache()
    print("updated", updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

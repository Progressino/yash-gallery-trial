import os
from pathlib import Path

import pandas as pd

import backend.main as m
from backend.services.daily_store import get_summary

CUT = pd.Timestamp("2025-01-01")
KEYS = ["mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"]


def line(k, df):
    if isinstance(df, pd.DataFrame) and not df.empty and "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        lo = str(d.min())[:10] if d.notna().any() else "NA"
        hi = str(d.max())[:10] if d.notna().any() else "NA"
        pre = int((d < CUT).sum())
        print(f"  {k}: {len(df):,} rows  {lo}->{hi}  pre2025={pre:,}")
    else:
        print(f"  {k}: empty/absent")


wc = m._warm_cache or {}
print("=== WARM CACHE frames ===")
for k in KEYS:
    line(k, wc.get(k))

print("=== DISK parquet ===")
cd = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
for k in KEYS:
    p = cd / f"{k}.parquet"
    if p.is_file():
        line(k + ".parquet", pd.read_parquet(p))
    else:
        print(f"  {k}.parquet: MISSING")

print("=== TIER-3 summary ===")
s = get_summary()
for k in ["amazon", "myntra", "meesho", "flipkart", "snapdeal"]:
    v = s.get(k, {})
    print(f"  {k}: rows={v.get('total_rows')} files={v.get('file_count')} {v.get('min_date')}->{v.get('max_date')}")

"""One-shot: patch warm sales_df.parquet from Tier-3 uploads (post-coalesce-fix)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/srv")

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from backend.services.daily_store import load_platform_data
from backend.services.sales import (
    build_sales_df,
    patch_sales_df_after_daily_upload,
    sales_date_window_from_platform_dfs,
    txn_reporting_naive_ist,
)
from backend.services.sku_mapping import load_sku_mapping_from_disk

WARM = Path("/data/warm_cache/sales_df.parquet")

sku_mapping = load_sku_mapping_from_disk() or {}
print("sku_mapping", len(sku_mapping))

plat_frames = {}
for p in ("amazon", "myntra", "meesho", "flipkart", "snapdeal"):
    df = load_platform_data(p, months=3, dedup=False, max_files=500)
    if df is not None and not df.empty:
        plat_frames[p] = df
        print(p, len(df))

d0, d1 = sales_date_window_from_platform_dfs(plat_frames)
print("window", d0, d1)

fresh = build_sales_df(
    mtr_df=plat_frames.get("amazon", pd.DataFrame()),
    myntra_df=plat_frames.get("myntra", pd.DataFrame()),
    meesho_df=plat_frames.get("meesho", pd.DataFrame()),
    flipkart_df=plat_frames.get("flipkart", pd.DataFrame()),
    snapdeal_df=plat_frames.get("snapdeal", pd.DataFrame()),
    sku_mapping=sku_mapping,
)
print("fresh rows", len(fresh))

existing = pd.read_parquet(WARM)
print("existing rows", len(existing))

new_sales = patch_sales_df_after_daily_upload(existing, fresh, set(plat_frames), d0, d1)
print("new rows", len(new_sales))

# Sanity: Amazon net shipments for key dates must match Tier-3 uploads.
t = txn_reporting_naive_ist(new_sales["TxnDate"])
src = new_sales["Source"].astype(str).str.strip()
q = pd.to_numeric(new_sales["Quantity"], errors="coerce").fillna(0)
txn = new_sales["Transaction Type"].astype(str).str.strip().str.lower()
for day in ("2026-06-20", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19"):
    m = (src == "Amazon") & (t == pd.Timestamp(day)) & (txn == "shipment")
    print(day, int(q[m].sum()))

bak = WARM.with_suffix(f".parquet.bak-tier3sync-{time.strftime('%Y%m%d%H%M')}")
WARM.rename(bak)
new_sales.to_parquet(WARM, index=False)
print("saved", WARM, "backup", bak)

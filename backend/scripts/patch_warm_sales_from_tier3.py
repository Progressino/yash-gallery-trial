#!/usr/bin/env python3
"""Patch warm sales_df from Tier-3; fix _Combo_Fan bool for parquet write."""
from __future__ import annotations

import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, "/srv")
warnings.filterwarnings("ignore")

import pandas as pd

from backend.services.combo_sku_map import combo_fan_mask
from backend.services.daily_store import load_platform_data
from backend.services.sales import (
    build_sales_df,
    patch_sales_df_after_daily_upload,
    sales_date_window_from_platform_dfs,
)
from backend.services.sku_mapping import load_sku_mapping_from_disk

WARM = Path("/data/warm_cache/sales_df.parquet")
sku_mapping = load_sku_mapping_from_disk() or {}
print("sku_mapping", len(sku_mapping), flush=True)

plat_frames = {}
for p in ("amazon", "myntra", "meesho", "flipkart", "snapdeal"):
    df = load_platform_data(p, months=3, dedup=False, max_files=500)
    if df is not None and not df.empty:
        plat_frames[p] = df
        print(p, len(df), flush=True)

d0, d1 = sales_date_window_from_platform_dfs(plat_frames)
print("window", d0, d1, flush=True)

fresh = build_sales_df(
    mtr_df=plat_frames.get("amazon", pd.DataFrame()),
    myntra_df=plat_frames.get("myntra", pd.DataFrame()),
    meesho_df=plat_frames.get("meesho", pd.DataFrame()),
    flipkart_df=plat_frames.get("flipkart", pd.DataFrame()),
    snapdeal_df=plat_frames.get("snapdeal", pd.DataFrame()),
    sku_mapping=sku_mapping,
)
print("fresh", len(fresh), "max", None if fresh.empty else fresh["TxnDate"].max(), flush=True)

existing = pd.read_parquet(WARM)
print("existing", len(existing), "max", existing["TxnDate"].max(), flush=True)

new_sales = patch_sales_df_after_daily_upload(existing, fresh, set(plat_frames), d0, d1)
print("patched", len(new_sales), "max", new_sales["TxnDate"].max(), flush=True)

if "_Combo_Fan" in new_sales.columns:
    new_sales["_Combo_Fan"] = combo_fan_mask(new_sales["_Combo_Fan"]).astype(bool)

stamp = time.strftime("%Y%m%d%H%M")
tmp = WARM.with_name(f"sales_df.parquet.tmp-patch-{stamp}")
bak = WARM.with_name(f"sales_df.parquet.bak-prepatch-{stamp}")
new_sales.to_parquet(tmp, index=False)
WARM.rename(bak)
tmp.rename(WARM)
print("saved", WARM, "backup", bak, flush=True)

sales = new_sales.copy()
sales["TxnDate"] = pd.to_datetime(sales["TxnDate"], errors="coerce")
sales["Sku"] = sales["Sku"].astype(str).str.strip().str.upper()
sales["TT"] = sales["Transaction Type"].astype(str).str.strip().str.lower()
mustard = sales[sales["Sku"].str.startswith("165YK251MUSTRAD")]
maxd = sales["TxnDate"].max().normalize()
win = mustard[
    (mustard["TxnDate"] >= maxd - timedelta(days=29))
    & (mustard["TxnDate"] <= maxd)
    & (mustard["TT"] == "shipment")
]
print(
    "30d end",
    maxd,
    "XL",
    int(win[win["Sku"] == "165YK251MUSTRAD-XL"]["Quantity"].sum()),
    "all",
    int(win["Quantity"].sum()),
    flush=True,
)
for d in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"):
    day = mustard[
        (mustard["TxnDate"].dt.normalize() == pd.Timestamp(d)) & (mustard["TT"] == "shipment")
    ]
    print(
        d,
        int(day["Quantity"].sum()),
        "XL",
        int(day[day["Sku"] == "165YK251MUSTRAD-XL"]["Quantity"].sum()),
        flush=True,
    )

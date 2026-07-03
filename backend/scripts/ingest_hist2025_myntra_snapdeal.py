#!/usr/bin/env python3
"""Ingest Myntra + Snapdeal 2025 historical RARs into Tier-1; keep 2026+ from current disk."""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from backend.services.daily_store import merge_platform_data
from backend.services.helpers import _coerce_df_for_parquet
from backend.services.myntra import load_myntra_from_zip
from backend.services.snapdeal import load_snapdeal_from_zip
from backend.services.sku_mapping import load_sku_mapping_from_disk

D = Path("/root/hist2025")
MIN_DATE = pd.Timestamp("2025-01-01")
CUT_2026 = pd.Timestamp("2026-01-01")
MANUAL = {
    "myntra": {"2025Q1": 15037},
    "snapdeal": {"2025Q1": 5980},
}


def quarterly_ship(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty:
        return {}
    d = pd.to_datetime(df["Date"], errors="coerce")
    s = df[df["TxnType"].astype(str).str.strip().eq("Shipment")].copy()
    ds = pd.to_datetime(s["Date"], errors="coerce")
    s = s[ds.notna()].copy()
    ds = ds[ds.notna()]
    s["_q"] = ds.dt.year.astype(str) + "Q" + ds.dt.quarter.astype(str)
    q = pd.to_numeric(s["Quantity"], errors="coerce").fillna(0)
    return s.assign(qq=q).groupby("_q")["qq"].sum().astype(int).to_dict()


def print_plat(name: str, attr: str, before: pd.DataFrame, after: pd.DataFrame):
    bq, aq = quarterly_ship(before), quarterly_ship(after)
    qs = sorted(set(bq) | set(aq))
    print(f"\n{'='*60}\n{name.upper()}  rows {len(before):,} -> {len(after):,}")
    print(f"{'quarter':8}{'before':>10}{'after':>10}{'delta':>10}")
    for q in qs:
        b, a = int(bq.get(q, 0)), int(aq.get(q, 0))
        print(f"{q:8}{b:>10}{a:>10}{a-b:>10}")
    m = MANUAL.get(name, {})
    if "2025Q1" in m:
        print(f"  manual Q1 target={m['2025Q1']:,}  after={int(aq.get('2025Q1',0)):,}")


def ingest_myntra(mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr = "myntra_df"
    before = pd.read_parquet(f"/data/warm_cache/{attr}.parquet")
    os.environ["MYNTRA_USE_DISPATCH_DATE"] = "1"
    merged = pd.DataFrame()
    for z in ["Myntra PPMP 2025.zip", "Myntra Sjit 2025.zip"]:
        p = D / z
        df, n, skipped = load_myntra_from_zip(p.read_bytes(), mapping)
        print(f"  parsed {z}: {len(df):,} rows skipped={skipped[:2]}")
        merged = merge_platform_data(merged, df, "myntra", source_filename=z)
    d = pd.to_datetime(merged["Date"], errors="coerce")
    new_2025 = merged[(d >= MIN_DATE) & (d < CUT_2026)].copy()
    ud = pd.to_datetime(before["Date"], errors="coerce")
    keep_2026 = before[ud >= CUT_2026].copy()
    after = pd.concat([new_2025, keep_2026], ignore_index=True)
    return before, after


def ingest_snapdeal(mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr = "snapdeal_df"
    before = pd.read_parquet(f"/data/warm_cache/{attr}.parquet")
    merged = pd.DataFrame()
    for z in ["Snapdeal AG 2025.zip", "Snapdeal PE 2025.zip", "Snapdeal YG 2025.zip"]:
        p = D / z
        df, n, skipped, info = load_snapdeal_from_zip(p.read_bytes(), mapping, filename=z)
        print(f"  parsed {z}: {len(df):,} rows files={n} skipped={len(skipped)}")
        if skipped:
            print(f"    skip sample: {skipped[:3]}")
        merged = merge_platform_data(merged, df, "snapdeal", source_filename=z)
    d = pd.to_datetime(merged["Date"], errors="coerce")
    new_2025 = merged[(d >= MIN_DATE) & (d < CUT_2026)].copy()
    ud = pd.to_datetime(before["Date"], errors="coerce")
    keep_2026 = before[ud >= CUT_2026].copy()
    after = pd.concat([new_2025, keep_2026], ignore_index=True)
    return before, after


def write_frame(attr: str, df: pd.DataFrame):
    path = Path(f"/data/warm_cache/{attr}.parquet")
    _coerce_df_for_parquet(df).to_parquet(path, index=False)
    print(f"  WROTE {path} ({len(df):,} rows)")


def main() -> int:
    dry = "--dry-run" in sys.argv
    t0 = time.time()
    mapping = load_sku_mapping_from_disk() or {}
    print(f"SKU map: {len(mapping):,} keys  dry_run={dry}")

    print("\n--- MYNTRA ---")
    my_b, my_a = ingest_myntra(mapping)
    print_plat("myntra", "myntra_df", my_b, my_a)

    print("\n--- SNAPDEAL ---")
    sd_b, sd_a = ingest_snapdeal(mapping)
    print_plat("snapdeal", "snapdeal_df", sd_b, sd_a)

    if not dry:
        write_frame("myntra_df", my_a)
        write_frame("snapdeal_df", sd_a)
    print(f"\nDone in {round(time.time()-t0,1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

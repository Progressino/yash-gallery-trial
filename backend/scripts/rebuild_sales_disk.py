#!/usr/bin/env python3
"""Rebuild sales_df.parquet from warm-cache platform parquets (standalone, OOM-safe)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

from backend.services.helpers import _coerce_df_for_parquet
from backend.services.sales import build_sales_df
from backend.services.sku_mapping import (
    load_bundled_sku_mapping,
    load_sku_mapping_from_disk,
    merge_sku_mapping_upload,
)


def main() -> int:
    cache = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    t0 = time.time()
    print("Loading platform parquets...", flush=True)

    def _load(name: str) -> pd.DataFrame:
        p = cache / f"{name}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    mtr = _load("mtr_df")
    myntra = _load("myntra_df")
    flipkart = _load("flipkart_df")
    meesho = _load("meesho_df")
    snapdeal = _load("snapdeal_df")
    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )
    print(f"mtr {len(mtr):,} rows", flush=True)

    sales = build_sales_df(
        mtr_df=mtr,
        myntra_df=myntra,
        meesho_df=meesho,
        flipkart_df=flipkart,
        snapdeal_df=snapdeal,
        sku_mapping=mapping,
    )
    if sales.empty:
        print("ERROR: sales_df build returned empty", flush=True)
        return 1

    print(f"sales_df {len(sales):,} rows in {time.time() - t0:.0f}s", flush=True)
    _coerce_df_for_parquet(sales).to_parquet(cache / "sales_df.parquet", index=False)

    gen_file = cache / "warm_cache_generation"
    gen = 5
    if gen_file.exists():
        try:
            gen = int(gen_file.read_text().strip()) + 1
        except Exception:
            pass
    gen_file.write_text(str(gen))
    print(f"warm_cache_generation={gen}", flush=True)

    if len(sys.argv) > 1:
        sku = sys.argv[1].upper()
        sales["TxnDate"] = pd.to_datetime(sales["TxnDate"])
        q1 = sales[
            (sales.Sku.astype(str).str.upper() == sku)
            & (sales.TxnDate >= "2025-01-01")
            & (sales.TxnDate < "2025-04-01")
        ]
        print(
            "Q1 verify:",
            q1.groupby(["Source", "Transaction Type"])["Quantity"].sum().to_dict(),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

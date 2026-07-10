#!/usr/bin/env python3
"""Refresh Meesho SKU/OMS via sub-order backfill + mapping; rebuild sales_df."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.services.helpers import _coerce_df_for_parquet
from backend.services.meesho import refresh_meesho_dataframe_oms_inplace
from backend.services.sales import build_sales_df
from backend.services.sku_mapping import (
    load_bundled_sku_mapping,
    load_sku_mapping_from_disk,
    merge_sku_mapping_upload,
)


def _blank_mask(df: pd.DataFrame) -> pd.Series:
    sku = df["SKU"].astype(str).str.strip().str.upper() if "SKU" in df.columns else pd.Series("", index=df.index)
    oms = (
        df["OMS_SKU"].astype(str).str.strip().str.upper()
        if "OMS_SKU" in df.columns
        else pd.Series("", index=df.index)
    )
    bad = {"", "NAN", "NONE", "NAT", "MEESHO_TOTAL"}
    return sku.isin(bad) & oms.isin(bad)


def main() -> int:
    cache = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    path = cache / "meesho_df.parquet"
    if not path.is_file():
        print("ERROR: missing", path, flush=True)
        return 1

    t0 = time.time()
    meesho = pd.read_parquet(path)
    blank0 = int(_blank_mask(meesho).sum())
    print(f"meesho rows={len(meesho):,} blank_sku={blank0:,}", flush=True)

    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )
    refresh_meesho_dataframe_oms_inplace(meesho, mapping)
    blank1 = int(_blank_mask(meesho).sum())
    print(
        f"after refresh blank_sku={blank1:,} recovered={blank0 - blank1:,}",
        flush=True,
    )

    _coerce_df_for_parquet(meesho).to_parquet(path, index=False)
    print(f"wrote {path}", flush=True)

    def _load(name: str) -> pd.DataFrame:
        p = cache / f"{name}.parquet"
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    print("Rebuilding sales_df…", flush=True)
    sales = build_sales_df(
        mtr_df=_load("mtr_df"),
        myntra_df=_load("myntra_df"),
        meesho_df=meesho,
        flipkart_df=_load("flipkart_df"),
        snapdeal_df=_load("snapdeal_df"),
        sku_mapping=mapping,
    )
    if sales.empty:
        print("ERROR: empty sales_df", flush=True)
        return 1
    _coerce_df_for_parquet(sales).to_parquet(cache / "sales_df.parquet", index=False)

    mee = sales[sales["Source"].astype(str) == "Meesho"]
    total_sku = mee["Sku"].astype(str).str.upper()
    mee_total = int(total_sku.eq("MEESHO_TOTAL").sum())
    print(
        f"sales_df={len(sales):,} meesho={len(mee):,} MEESHO_TOTAL={mee_total:,} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )

    # Remaining blank months (still no SKU after recovery)
    d = pd.to_datetime(meesho["Date"], errors="coerce")
    still = meesho.loc[_blank_mask(meesho)].copy()
    if not still.empty:
        still["_m"] = d.loc[still.index].dt.to_period("M").astype(str)
        print("Remaining blank Meesho months (re-upload Order Report CSV):", flush=True)
        print(still.groupby("_m").size().to_string(), flush=True)

    gen_file = cache / "warm_cache_generation"
    gen = 1
    if gen_file.exists():
        try:
            gen = int(gen_file.read_text().strip()) + 1
        except Exception:
            pass
    gen_file.write_text(str(gen))
    print(f"warm_cache_generation={gen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

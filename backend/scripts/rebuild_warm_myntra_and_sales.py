#!/usr/bin/env python3
"""Merge Tier-3 Myntra into warm myntra_df and rebuild sales_df.parquet on prod."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import (  # noqa: E402
    clear_tier3_range_cache,
    load_platform_data,
    merge_platform_data,
)
from backend.services.helpers import _coerce_df_for_parquet
from backend.services.sku_mapping import (
    load_bundled_sku_mapping,
    load_sku_mapping_from_disk,
    merge_sku_mapping_upload,
)
from backend.services.sales import build_sales_df


def _load_warm(name: str) -> pd.DataFrame:
    p = Path("/data/warm_cache") / f"{name}.parquet"
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def main() -> int:
    clear_tier3_range_cache()
    warm_myn = _load_warm("myntra_df")
    tier3_myn = load_platform_data("myntra", months=None, dedup=True)
    print("warm myntra rows", len(warm_myn), "tier3 myntra rows", len(tier3_myn))
    if tier3_myn.empty:
        print("ERROR: tier3 myntra empty")
        return 1
    merged = merge_platform_data(warm_myn, tier3_myn, "myntra")
    print("merged myntra rows", len(merged))
    out = Path("/data/warm_cache/myntra_df.parquet")
    _coerce_df_for_parquet(merged).to_parquet(out, index=False)

    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )

    def _load(name: str) -> pd.DataFrame:
        return _load_warm(name)

    sales = build_sales_df(
        mtr_df=_load("mtr_df"),
        myntra_df=merged,
        meesho_df=_load("meesho_df"),
        flipkart_df=_load("flipkart_df"),
        snapdeal_df=_load("snapdeal_df"),
        sku_mapping=mapping,
    )
    if sales.empty:
        print("ERROR: sales build empty")
        return 1
    _coerce_df_for_parquet(sales).to_parquet(
        Path("/data/warm_cache/sales_df.parquet"), index=False
    )
    gen_file = Path("/data/warm_cache/warm_cache_generation")
    gen = 1
    if gen_file.exists():
        try:
            gen = int(gen_file.read_text().strip()) + 1
        except Exception:
            pass
    gen_file.write_text(str(gen))
    print("sales_df rows", len(sales), "warm_cache_generation", gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

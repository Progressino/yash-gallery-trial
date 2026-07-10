#!/usr/bin/env python3
"""Apply OMS_SKU from Meesho_Blank_OMS_Report.xlsx onto warm-cache meesho_df; rebuild sales."""
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


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})


def main() -> int:
    report = Path(os.environ.get("MEESHO_OMS_REPORT", "/tmp/Meesho_Blank_OMS_Report.xlsx"))
    cache = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    meesho_path = cache / "meesho_df.parquet"
    if not report.is_file():
        print("ERROR: missing report", report, flush=True)
        return 1
    if not meesho_path.is_file():
        print("ERROR: missing", meesho_path, flush=True)
        return 1

    t0 = time.time()
    fill = pd.read_excel(report, sheet_name="Blank_Rows")
    fill["OrderId"] = _clean(fill["OrderId"])
    fill["OMS_SKU"] = _clean(fill["OMS_SKU"])
    fill = fill[fill["OrderId"].ne("") & fill["OMS_SKU"].ne("")].copy()
    # Prefer first OMS per OrderId when duplicates exist
    oid_map = (
        fill.drop_duplicates(subset=["OrderId"], keep="first")
        .set_index("OrderId")["OMS_SKU"]
        .to_dict()
    )
    print(f"report rows={len(fill):,} unique OrderId maps={len(oid_map):,}", flush=True)

    meesho = pd.read_parquet(meesho_path)
    blank0 = int(_blank_mask(meesho).sum())
    print(f"meesho rows={len(meesho):,} blank_before={blank0:,}", flush=True)

    if "OrderId" not in meesho.columns:
        print("ERROR: meesho_df missing OrderId", flush=True)
        return 1
    if "SKU" not in meesho.columns:
        meesho["SKU"] = ""
    if "OMS_SKU" not in meesho.columns:
        meesho["OMS_SKU"] = ""

    oid = _clean(meesho["OrderId"])
    sku = _clean(meesho["SKU"])
    oms = _clean(meesho["OMS_SKU"])
    need = _blank_mask(meesho)
    recovered = oid.map(oid_map).fillna("")
    hit = need & recovered.ne("")
    meesho.loc[hit, "SKU"] = recovered.loc[hit]
    meesho.loc[hit, "OMS_SKU"] = recovered.loc[hit]
    print(f"applied_from_report={int(hit.sum()):,}", flush=True)

    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )
    refresh_meesho_dataframe_oms_inplace(meesho, mapping)

    # User report is authoritative for previously-blank OrderIds — re-apply after mapping.
    oid = _clean(meesho["OrderId"])
    recovered2 = oid.map(oid_map).fillna("")
    force = recovered2.ne("")
    meesho.loc[force, "OMS_SKU"] = recovered2.loc[force]
    sku_now = _clean(meesho["SKU"])
    bad = {"", "NAN", "NONE", "NAT", "MEESHO_TOTAL"}
    fill_sku = force & sku_now.str.upper().isin(bad)
    meesho.loc[fill_sku, "SKU"] = recovered2.loc[fill_sku]
    print(f"reapplied_report_oms={int(force.sum()):,}", flush=True)

    blank1 = int(_blank_mask(meesho).sum())
    print(f"blank_after={blank1:,} recovered_total={blank0 - blank1:,}", flush=True)

    _coerce_df_for_parquet(meesho).to_parquet(meesho_path, index=False)
    print(f"wrote {meesho_path}", flush=True)

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
    sku_u = mee["Sku"].astype(str).str.strip().str.upper()
    totalish = int(sku_u.isin({"", "NAN", "NONE", "MEESHO_TOTAL"}).sum())
    print(
        f"sales_df={len(sales):,} meesho={len(mee):,} blank_or_total_sku={totalish:,} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )

    # Sample verification: top filled OMS should appear in Meesho sales
    samples = list(fill["OMS_SKU"].value_counts().head(5).index)
    for s in samples:
        n = int((sku_u == str(s).strip().upper()).sum())
        print(f"  sample {s}: meesho_sales_rows={n}", flush=True)

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

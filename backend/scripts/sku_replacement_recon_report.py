#!/usr/bin/env python3
"""Build a reconciliation report of Replace-SKU sheets vs an inventory matrix CSV.

Usage:
  python -m backend.scripts.sku_replacement_recon_report \\
      --matrix /path/inventory-matrix-oms.csv \\
      --replace '/path/Replace sku 1.xlsx' '/path/Replace sku 2.xlsx' \\
      --out /tmp/sku_recon.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--replace", nargs="+", required=True)
    p.add_argument("--out", default="sku_replacement_recon.csv")
    p.add_argument("--date-col", default="")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.services.inventory import _inventory_alias_oms_key
    from backend.services.sku_mapping import merge_sku_mapping_upload, parse_sku_mapping

    mapping: dict = {}
    for path in args.replace:
        raw = Path(path).read_bytes()
        mapping = merge_sku_mapping_upload(mapping, parse_sku_mapping(raw))

    m = pd.read_csv(args.matrix)
    date_cols = [c for c in m.columns if re.match(r"\d{1,2}-\d{1,2}-\d{2}$", str(c))]
    if not date_cols:
        print("No D-M-YY date columns found", file=sys.stderr)
        return 2
    dc = args.date_col or date_cols[-1]
    skus = m[~m["SKU"].astype(str).str.lower().str.contains("total")].copy()
    skus["Original_SKU"] = skus["SKU"].astype(str).str.strip().str.upper()
    skus["Original_Qty"] = pd.to_numeric(skus[dc], errors="coerce").fillna(0.0)
    skus["Replacement_SKU"] = skus["Original_SKU"].map(
        lambda s: _inventory_alias_oms_key(s, mapping)
    )
    skus["Mapped"] = skus["Original_SKU"] != skus["Replacement_SKU"]
    skus["Reason"] = skus.apply(
        lambda r: (
            "identity"
            if not r["Mapped"]
            else ("replace_map" if r["Original_SKU"] in mapping else "alias_spelling")
        ),
        axis=1,
    )
    consol = (
        skus.groupby("Replacement_SKU", as_index=False)["Original_Qty"]
        .sum()
        .rename(columns={"Original_Qty": "Final_Consolidated_Qty"})
    )
    out = skus.merge(consol, on="Replacement_SKU", how="left")
    unmapped = out[~out["Mapped"] & out["Original_SKU"].isin(mapping.keys())]
    ambig = []  # resolved map has no multi terminal by construction
    report_path = Path(args.out)
    out[
        [
            "Original_SKU",
            "Replacement_SKU",
            "Original_Qty",
            "Final_Consolidated_Qty",
            "Reason",
        ]
    ].to_csv(report_path, index=False)
    print(f"date_col={dc}")
    print(f"skus={len(out)} units_sum={out['Original_Qty'].sum():.0f}")
    print(f"mapped_rows={int(out['Mapped'].sum())}")
    print(f"consolidated_skus={out['Replacement_SKU'].nunique()}")
    print(f"map_keys={len(mapping)}")
    print(f"wrote {report_path}")
    if len(unmapped):
        print(f"WARN unexpected: {len(unmapped)} map-keys not remapped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

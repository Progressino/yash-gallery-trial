#!/usr/bin/env python3
"""Parse & check Meesho/Snapdeal archive ZIPs (local test, no upload)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.meesho import load_meesho_from_zip
from backend.services.snapdeal import load_snapdeal_from_zip
from backend.services.sku_mapping import load_bundled_sku_mapping

ARCHIVES = [
    ("meesho", Path("/Users/samraisinghani/Downloads/Meesho PE.zip")),
    ("meesho", Path("/Users/samraisinghani/Downloads/Meesho AG.zip")),
    ("snapdeal", Path("/Users/samraisinghani/Downloads/Snapdeal PE.zip")),
    ("snapdeal", Path("/Users/samraisinghani/Downloads/Snapdeal AG.zip")),
    ("snapdeal", Path("/Users/samraisinghani/Downloads/Snapdeal YG.zip")),
]

# Historical cutoff: PO uses last 8 quarters — drop 2024, keep 2025-01-01 onward.
CUTOFF = pd.Timestamp("2025-01-01")


def parse(platform: str, path: Path, mapping: dict):
    raw = path.read_bytes()
    if platform == "meesho":
        df, n, skipped = load_meesho_from_zip(raw, source_filename=path.name)
        return df, n, skipped
    df, n, skipped, _info = load_snapdeal_from_zip(raw, mapping, filename=path.name)
    return df, n, skipped


def monthly(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty:
        return {}
    d = pd.to_datetime(df["Date"], errors="coerce")
    w = df.loc[d.notna()].copy()
    if w.empty:
        return {}
    w["_m"] = d.loc[w.index].dt.to_period("M").astype(str)
    ship = w[w["TxnType"].astype(str).str.lower().isin(["shipment", "sale"])]
    return {k: int(v) for k, v in ship.groupby("_m")["Quantity"].sum().items()}


def main() -> int:
    mapping = load_bundled_sku_mapping()
    print(f"SKU mapping keys: {len(mapping):,}\n")

    for platform, path in ARCHIVES:
        if not path.is_file():
            print(f"MISSING: {path}")
            continue
        df, n, skipped = parse(platform, path, mapping)
        if df is None or df.empty:
            print(f"[{path.name}] NO DATA — {skipped[:4]}\n")
            continue
        d = pd.to_datetime(df["Date"], errors="coerce")
        kept = df.loc[d >= CUTOFF]
        dk = pd.to_datetime(kept["Date"], errors="coerce")
        dropped_2024 = int((d < CUTOFF).sum())
        print(f"[{path.name}] ({platform})")
        print(f"  parsed rows : {len(df):,}  files={n}")
        print(f"  full range  : {str(d.min())[:10]} → {str(d.max())[:10]}")
        print(f"  2024 rows   : {dropped_2024:,} (will be excluded)")
        print(f"  KEPT (2025+): {len(kept):,}  range {str(dk.min())[:10]} → {str(dk.max())[:10]}")
        if "OMS_SKU" in kept.columns:
            unmapped = kept["OMS_SKU"].astype(str).str.upper()
            n_unmapped = int((unmapped.eq("") | unmapped.isin(["NAN", "NONE", "UNKNOWN"])).sum())
            print(f"  SKUs        : {kept['OMS_SKU'].nunique():,} unique | unmapped rows: {n_unmapped:,}")
        if "TxnType" in kept.columns:
            print(f"  TxnType     : {kept['TxnType'].astype(str).value_counts().to_dict()}")
        ms = monthly(kept)
        print(f"  ship months : {len(ms)} | {list(ms.items())[:3]}…")
        if skipped:
            print(f"  skipped     : {len(skipped)}")
            for s in skipped[:4]:
                print(f"    - {s}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Import Tier-1 yearly bulk ZIPs into daily_sales.db (same SQLite as Tier-3).

Uses the same parsers as the Upload Data page (/upload/mtr, /myntra, …).
Requires ``--sku-mapping`` for Myntra and Flipkart (same rule as the UI).

Run from repo root after setting the DB path if needed:

  export DAILY_SALES_DB=/path/to/daily_sales.db
  PYTHONPATH=. python3 scripts/import_local_tier1_yearly_folder.py \\
    --sku-mapping "/path/to/SKU_Mapping.xlsx" \\
    "/path/to/Yearly"

Snapdeal / Amazon / Meesho ZIPs do not need the mapping on upload.

This complements ``import_local_tier3_folder.py`` — use the same ``DAILY_SALES_DB`` so
Tier-1 history and Tier-3 dailies live in one file before copying to the server.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _classify_zip(name: str) -> str:
    n = name.lower()
    if "snapdeal" in n:
        return "snapdeal"
    if "flipkart" in n:
        return "flipkart"
    if "meesho" in n:
        return "meesho"
    if "myntra" in n or "ppmp" in n or "sjit" in n:
        return "myntra"
    if "amzon" in n or "amazon" in n or "mtr" in n:
        return "amazon"
    return "unknown"


def _process_zip(path: Path, sku_mapping: dict) -> tuple[str, list[str], list[str]]:
    """Returns (kind, messages_ok, warnings)."""
    import pandas as pd

    from backend.routers.upload import _RAR_MAGIC, _extract_rar_files
    from backend.services.daily_store import save_daily_file
    from backend.services.flipkart import load_flipkart_from_zip
    from backend.services.meesho import load_meesho_from_zip
    from backend.services.mtr import load_mtr_from_extracted_files, load_mtr_from_zip
    from backend.services.myntra import load_myntra_from_zip
    from backend.services.snapdeal import load_snapdeal_from_zip

    name = path.name
    kind = _classify_zip(name)
    ok_msg: list[str] = []
    warn: list[str] = []

    if kind == "unknown":
        return kind, [], [f"{name}: cannot classify ZIP — rename or upload manually"]

    raw = path.read_bytes()
    try:
        if kind == "amazon":
            if raw[:6] == _RAR_MAGIC or name.lower().endswith(".rar"):
                inner = _extract_rar_files(raw)
                df, csv_count, skipped = load_mtr_from_extracted_files(inner)
                del inner
            else:
                df, csv_count, skipped = load_mtr_from_zip(raw)
            if df.empty:
                return kind, [], [f"{name}: no MTR data; {'; '.join(skipped[:3])}"]
            save_daily_file("amazon", name, df)
            ok_msg.append(f"Amazon MTR {name}: {len(df):,} rows ({csv_count} parts)")
            if skipped:
                warn.append(f"{name}: {'; '.join(skipped[:3])}")

        elif kind == "myntra":
            if not sku_mapping:
                return kind, [], [f"{name}: need --sku-mapping for Myntra PPMP ZIPs"]
            df, csv_count, skipped = load_myntra_from_zip(raw, sku_mapping)
            if df.empty:
                return kind, [], [f"{name}: no Myntra rows; {'; '.join(skipped[:3])}"]
            save_daily_file("myntra", name, df)
            ok_msg.append(f"Myntra {name}: {len(df):,} rows ({csv_count} CSVs)")
            if skipped:
                warn.append(f"{name}: {'; '.join(skipped[:3])}")

        elif kind == "meesho":
            df, zip_count, skipped = load_meesho_from_zip(raw)
            if df.empty:
                return kind, [], [f"{name}: no Meesho rows; {'; '.join(skipped[:3])}"]
            save_daily_file("meesho", name, df)
            ok_msg.append(f"Meesho {name}: {len(df):,} rows ({zip_count} ZIPs)")
            if skipped:
                warn.append(f"{name}: {'; '.join(skipped[:3])}")

        elif kind == "flipkart":
            if not sku_mapping:
                return kind, [], [f"{name}: need --sku-mapping for Flipkart master ZIPs"]
            df, xlsx_count, skipped = load_flipkart_from_zip(raw, sku_mapping)
            if df.empty:
                return kind, [], [f"{name}: no Flipkart rows; {'; '.join(skipped[:3])}"]
            save_daily_file("flipkart", name, df)
            ok_msg.append(f"Flipkart {name}: {len(df):,} rows ({xlsx_count} XLSX)")
            if skipped:
                warn.append(f"{name}: {'; '.join(skipped[:3])}")

        elif kind == "snapdeal":
            df, fc, skipped, _pi = load_snapdeal_from_zip(raw, sku_mapping, name)
            if df.empty:
                return kind, [], [f"{name}: no Snapdeal rows; {'; '.join(skipped[:3])}"]
            save_daily_file("snapdeal", name, df)
            ok_msg.append(f"Snapdeal {name}: {len(df):,} rows ({fc} files)")
            if skipped:
                warn.append(f"{name}: {'; '.join(skipped[:3])}")

    except Exception as e:
        return kind, [], [f"{name}: {e}"]
    finally:
        del raw
        gc.collect()

    return kind, ok_msg, warn


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Tier-1 yearly ZIP folder into daily_sales.db")
    parser.add_argument("yearly_dir", type=Path, help="Folder containing *.zip (and optional *.rar) bulk exports")
    parser.add_argument("--db", type=Path, help="SQLite path (sets DAILY_SALES_DB before imports)")
    parser.add_argument(
        "--sku-mapping",
        type=Path,
        help="SKU mapping Excel (required for Myntra/Flipkart; Amazon/Meesho/Snapdeal still run if omitted)",
    )
    args = parser.parse_args()

    yearly = args.yearly_dir.expanduser().resolve()
    if not yearly.is_dir():
        print(f"Not a directory: {yearly}", file=sys.stderr)
        return 2

    if args.sku_mapping and not args.sku_mapping.is_file():
        print(f"SKU mapping not found: {args.sku_mapping}", file=sys.stderr)
        return 2

    if args.db:
        os.environ["DAILY_SALES_DB"] = str(args.db.resolve())

    import importlib

    from backend.services.sku_mapping import parse_sku_mapping

    sku_mapping: dict = {}
    if args.sku_mapping:
        sku_raw = args.sku_mapping.read_bytes()
        sku_mapping = parse_sku_mapping(sku_raw, args.sku_mapping.name) or {}
        if not sku_mapping:
            print("SKU mapping parsed empty — check the file.", file=sys.stderr)
            return 2

    import backend.services.daily_store as ds

    importlib.reload(ds)

    from backend.services.daily_store import get_summary, load_all_platforms

    zips = sorted(yearly.glob("*.zip")) + sorted(yearly.glob("*.rar"))
    if not zips:
        print(f"No .zip/.rar files in {yearly}", file=sys.stderr)
        return 2

    print(f"Importing {len(zips)} archive(s) → {ds._DB_PATH}")
    any_ok = False
    for zp in zips:
        kind, ok_msg, warn = _process_zip(zp, sku_mapping)
        for m in ok_msg:
            print(f"  ✓ {m}")
            any_ok = True
        for m in warn:
            print(f"  ! {m}")

    print("\nTier store summary:")
    for plat, meta in sorted(get_summary().items()):
        print(f"  {plat}: files={meta['file_count']}, rows_in_db={meta['total_rows']}, "
              f"dates {meta['min_date']}..{meta['max_date']}")

    print("\nCombined rows (after dedup):")
    for plat, df in sorted(load_all_platforms().items()):
        print(f"  {plat}: {len(df):,}")

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

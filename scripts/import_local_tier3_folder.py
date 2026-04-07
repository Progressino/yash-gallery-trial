#!/usr/bin/env python3
"""
Import a folder of Tier-3 daily files into daily_sales.db (offline).

- Optional --fresh: clears all rows in the Tier-3 SQLite DB first.
- Dedupes by filename: if the same basename appears in subfolders, keeps the largest file.
- Uses the same detection + parsers as /api/upload/daily-auto; results are deduped on read via
  daily_store._dedup_platform_df.

Set DAILY_SALES_DB before running if the DB path is not the default:

  export DAILY_SALES_DB=/data/daily_sales.db   # typical path inside VPS bind mount

Run from repository root:

  PYTHONPATH=. python scripts/import_local_tier3_folder.py --fresh \\
    "/Users/you/Downloads/Sales 1-Mar-23"

Tier-1 (yearly zips) cannot be fully represented in this SQLite file alone — upload those via
the app “Upload Data” page after SKU mapping (see --print-tier1 with --yearly-dir).
"""
from __future__ import annotations

import argparse
import gc
import io
import os
import sys
from pathlib import Path

# Repo root: parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_deduped_files(folder: Path) -> list[Path]:
    allowed = {".csv", ".xlsx", ".xls", ".zip", ".rar"}
    by_name: dict[str, Path] = {}
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        name = p.name
        cur = by_name.get(name)
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if cur is None:
            by_name[name] = p
            continue
        try:
            if sz > cur.stat().st_size:
                by_name[name] = p
        except OSError:
            pass
    out = sorted(by_name.values(), key=lambda x: x.name.lower())
    return out


def _print_tier1_checklist(yearly_dir: Path) -> None:
    if not yearly_dir.is_dir():
        print(f"Yearly dir not found: {yearly_dir}", file=sys.stderr)
        return
    zips = sorted(yearly_dir.glob("*.zip"))
    print("\n── Tier 1 — upload these on “Upload Data” after SKU mapping (order flexible per platform card) ──")
    for z in zips:
        print(f"  • {z.name}")
    print("Then use “Load Cache” / work in app; Tier-3 DB you just imported layers on top at merge time.\n")


def _process_one(fname: str, raw: bytes, sku_mapping: dict) -> tuple[list[str], list[str]]:
    """Returns (detected_labels, warnings)."""
    import pandas as pd

    from backend.routers.upload import _RAR_MAGIC, _detect_platform, _extract_rar_files
    from backend.services.daily_store import load_platform_data as _load_platform
    from backend.services.daily_store import save_daily_file
    from backend.services.meesho import load_meesho_from_zip, parse_meesho_csv
    from backend.services.mtr import load_mtr_from_zip, parse_mtr_csv
    from backend.services.myntra import _parse_myntra_csv
    from backend.services.snapdeal import load_snapdeal_from_zip

    detected: list[str] = []
    warnings: list[str] = []

    def handle_one(name: str, data: bytes) -> None:
        platform = _detect_platform(name, data)
        try:
            if platform == "amazon_mtr_zip":
                df_mtr, _n, sk_mtr = load_mtr_from_zip(data)
                if not df_mtr.empty:
                    save_daily_file("amazon", name, df_mtr)
                    detected.append(f"Amazon MTR ({name})")
                    if sk_mtr:
                        warnings.append(f"{name}: {'; '.join(sk_mtr[:2])}")
                else:
                    warnings.append(f"{name}: No Amazon MTR CSVs")

            elif platform == "amazon_b2c":
                df, msg = parse_mtr_csv(data, name)
                if not df.empty:
                    save_daily_file("amazon", name, df)
                    detected.append(f"Amazon ({name})")
                    if msg != "OK":
                        warnings.append(f"{name}: {msg}")
                else:
                    warnings.append(f"{name}: {msg}")

            elif platform == "amazon_b2b":
                df, msg = parse_mtr_csv(data, name)
                if not df.empty:
                    save_daily_file("amazon", name, df)
                    detected.append(f"Amazon B2B ({name})")
                    if msg != "OK":
                        warnings.append(f"{name}: {msg}")
                else:
                    warnings.append(f"{name}: {msg}")

            elif platform == "myntra":
                df, msg = _parse_myntra_csv(data, name, sku_mapping)
                if not df.empty:
                    save_daily_file("myntra", name, df)
                    detected.append(f"Myntra ({name})")
                    if msg != "OK":
                        warnings.append(f"{name}: {msg}")
                else:
                    warnings.append(f"{name}: {msg}")

            elif platform == "meesho":
                df, _c, skip = load_meesho_from_zip(data)
                if not df.empty:
                    save_daily_file("meesho", name, df)
                    detected.append(f"Meesho zip ({name})")
                    if skip:
                        warnings.append(f"{name}: {'; '.join(skip[:2])}")
                else:
                    warnings.append(f"{name}: Meesho zip empty")

            elif platform == "meesho_csv":
                df, msg = parse_meesho_csv(data)
                if not df.empty:
                    save_daily_file("meesho", name, df)
                    detected.append(f"Meesho csv ({name})")
                    if msg != "OK":
                        warnings.append(f"{name}: {msg}")
                else:
                    warnings.append(f"{name}: {msg}")

            elif platform == "snapdeal":
                df_sd, _fc, skipped_sd, _info = load_snapdeal_from_zip(data, sku_mapping or {}, name)
                if df_sd.empty:
                    warnings.append(f"{name}: Snapdeal — {'; '.join(skipped_sd[:3])}")
                else:
                    warnings.append(
                        f"{name}: Snapdeal extracted {len(df_sd)} rows — Tier-3 SQLite only stores "
                        "amazon/myntra/meesho/flipkart; import Snapdeal via app session or extend daily_store."
                    )

            elif platform == "flipkart":
                from backend.services.flipkart import (
                    _parse_flipkart_earn_more,
                    _parse_flipkart_orders_sheet,
                    _parse_flipkart_xlsx,
                )

                try:
                    xl_sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
                except Exception:
                    xl_sheets = []
                if "Sales Report" in xl_sheets:
                    df = _parse_flipkart_xlsx(data, name, sku_mapping)
                elif "Orders" in xl_sheets:
                    df = _parse_flipkart_orders_sheet(data, name, sku_mapping)
                elif "earn_more_report" in xl_sheets:
                    df = _parse_flipkart_earn_more(data, name, sku_mapping)
                else:
                    df = _parse_flipkart_xlsx(data, name, sku_mapping)
                    if df.empty:
                        warnings.append(f"{name}: Flipkart — unknown sheet layout {xl_sheets[:5]}")
                        return
                if not df.empty:
                    save_daily_file("flipkart", name, df)
                    detected.append(f"Flipkart ({name})")
                else:
                    warnings.append(f"{name}: Flipkart empty")

            else:
                warnings.append(f"{name}: unknown format (detector={platform})")

        except Exception as e:
            warnings.append(f"{name}: {e}")

    if raw[:6] == _RAR_MAGIC or fname.lower().endswith(".rar"):
        try:
            for inner_name, inner_bytes in _extract_rar_files(raw):
                handle_one(inner_name, inner_bytes)
        except Exception as e:
            warnings.append(f"{fname} (RAR): {e}")
    else:
        handle_one(fname, raw)

    return detected, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Tier-3 folder into daily_sales.db")
    parser.add_argument("folder", type=Path, help="Folder with daily CSV/XLSX/ZIP exports")
    parser.add_argument("--fresh", action="store_true", help="DELETE all Tier-3 rows before import")
    parser.add_argument("--db", type=Path, help="SQLite path (sets DAILY_SALES_DB); default from env or backend rules")
    parser.add_argument("--sku-mapping", type=Path, help="Optional SKU mapping .xlsx for Myntra/Flipkart OMS columns")
    parser.add_argument("--yearly-dir", type=Path, help="If set with --print-tier1, list Tier-1 zips to upload in app")
    parser.add_argument("--print-tier1", action="store_true", help="Print Tier-1 upload checklist")
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 2

    if args.db:
        os.environ["DAILY_SALES_DB"] = str(args.db.resolve())

    # Import after env is set so daily_store picks up DAILY_SALES_DB
    import importlib

    import backend.services.daily_store as daily_store_mod

    importlib.reload(daily_store_mod)

    from backend.services.daily_store import clear_all_daily_uploads, get_summary, load_all_platforms

    sku_mapping: dict = {}
    if args.sku_mapping and args.sku_mapping.is_file():
        from backend.services.sku_mapping import parse_sku_mapping

        raw = args.sku_mapping.read_bytes()
        sku_mapping = parse_sku_mapping(raw, args.sku_mapping.name) or {}

    if args.fresh:
        n = clear_all_daily_uploads()
        print(f"Cleared Tier-3 DB ({n} previous row(s) in daily_uploads).")

    files = _collect_deduped_files(folder)
    print(f"Importing {len(files)} file(s) (basename-deduped) → {daily_store_mod._DB_PATH}")

    all_det: list[str] = []
    all_warn: list[str] = []
    for fp in files:
        raw = fp.read_bytes()
        try:
            det, warn = _process_one(fp.name, raw, sku_mapping)
            all_det.extend(det)
            all_warn.extend(warn)
        finally:
            del raw
            gc.collect()

    if all_det:
        print("Loaded:")
        for x in all_det:
            print(f"  ✓ {x}")
    if all_warn:
        print("Warnings:")
        for x in all_warn:
            print(f"  ! {x}")

    summary = get_summary()
    print("\nTier-3 summary (pre-merge; dedup happens when building platform DFs):")
    for plat, meta in sorted(summary.items()):
        print(f"  {plat}: {meta}")

    plat_dfs = load_all_platforms()
    print("\nCombined row counts after SQL dedup step:")
    for plat, df in sorted(plat_dfs.items()):
        print(f"  {plat}: {len(df):,} rows")

    if args.print_tier1 and args.yearly_dir:
        _print_tier1_checklist(args.yearly_dir.expanduser().resolve())

    return 0 if all_det else 1


if __name__ == "__main__":
    raise SystemExit(main())

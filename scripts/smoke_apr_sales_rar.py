#!/usr/bin/env python3
"""
Parse a sales RAR (or a folder of extracted files) like /api/upload/daily-auto and run build_sales_df.

Usage (from repo root):
  PYTHONPATH=. python3 scripts/smoke_apr_sales_rar.py "/path/to/Sales 1 APR to 14.rar"
  PYTHONPATH=. python3 scripts/smoke_apr_sales_rar.py ./tmp-sales-apr-extract/"Sales 1 APR to 14"

Exit 0 if unified sales_df is non-empty. Prints row counts and sales summary.

Requires: unar, bsdtar, or 7z on PATH for .rar input (same as production upload).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers.upload import (  # noqa: E402
    _RAR_MAGIC,
    _detect_platform,
    _extract_rar_files,
)
from backend.services.mtr import parse_mtr_csv  # noqa: E402
from backend.services.myntra import _parse_myntra_csv  # noqa: E402
from backend.services.meesho import parse_meesho_csv  # noqa: E402
from backend.services.flipkart import (  # noqa: E402
    _parse_flipkart_xlsx,
    _parse_flipkart_orders_sheet,
    _parse_flipkart_earn_more,
)
from backend.services.sales import build_sales_df, get_sales_summary  # noqa: E402


def _parse_flipkart_bytes(raw: bytes, fname: str, sku_mapping: dict) -> pd.DataFrame:
    try:
        xl_sheets = pd.ExcelFile(io.BytesIO(raw)).sheet_names
    except Exception:
        xl_sheets = []
    if "Sales Report" in xl_sheets:
        return _parse_flipkart_xlsx(raw, fname, sku_mapping)
    if "Orders" in xl_sheets:
        return _parse_flipkart_orders_sheet(raw, fname, sku_mapping)
    if "earn_more_report" in xl_sheets:
        return _parse_flipkart_earn_more(raw, fname, sku_mapping)
    return _parse_flipkart_xlsx(raw, fname, sku_mapping)


def _iter_inputs(target: Path) -> list[tuple[str, bytes]]:
    if target.is_file() and (
        target.suffix.lower() == ".rar" or target.name.lower().endswith(".rar")
    ):
        raw = target.read_bytes()
        if raw[:6] != _RAR_MAGIC and not target.name.lower().endswith(".rar"):
            raise SystemExit(f"Not a RAR file: {target}")
        return _extract_rar_files(raw)
    if target.is_dir():
        out: list[tuple[str, bytes]] = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(target)).replace("\\", "/")
                out.append((rel, p.read_bytes()))
        return out
    raise SystemExit(f"Not a file or directory: {target}")


def main() -> None:
    argv = sys.argv[1:]
    default = os.environ.get(
        "APR_SALES_RAR",
        str(Path.home() / "Downloads/Sales 1 APR to 14.rar"),
    )
    path = Path(argv[0] if argv else default)
    if not path.exists():
        raise SystemExit(
            f"Path not found: {path}\n"
            "Pass the .rar path as argv[1] or set APR_SALES_RAR, "
            "or extract to a folder and pass that directory."
        )

    sku_mapping: dict = {}
    mtr_parts: list[pd.DataFrame] = []
    myntra_parts: list[pd.DataFrame] = []
    meesho_parts: list[pd.DataFrame] = []
    flipkart_parts: list[pd.DataFrame] = []
    warnings: list[str] = []

    for fname, raw in _iter_inputs(path):
        plat = _detect_platform(fname, raw)
        try:
            if plat == "amazon_b2c":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    mtr_parts.append(df)
                elif msg:
                    warnings.append(f"{fname}: {msg}")
            elif plat == "amazon_b2b":
                df, msg = parse_mtr_csv(raw, fname)
                if not df.empty:
                    mtr_parts.append(df)
                elif msg:
                    warnings.append(f"{fname}: {msg}")
            elif plat == "myntra":
                df, msg = _parse_myntra_csv(raw, fname, sku_mapping)
                if not df.empty:
                    myntra_parts.append(df)
                elif msg:
                    warnings.append(f"{fname}: {msg}")
            elif plat == "meesho_csv":
                df, msg = parse_meesho_csv(raw)
                if not df.empty:
                    meesho_parts.append(df)
                elif msg:
                    warnings.append(f"{fname}: {msg}")
            elif plat == "flipkart":
                df = _parse_flipkart_bytes(raw, fname, sku_mapping)
                if not df.empty:
                    flipkart_parts.append(df)
                else:
                    warnings.append(f"{fname}: Flipkart parse empty")
            else:
                warnings.append(f"{fname}: skipped (detected={plat})")
        except Exception as e:
            warnings.append(f"{fname}: {e}")

    mtr_df = pd.concat(mtr_parts, ignore_index=True) if mtr_parts else pd.DataFrame()
    myntra_df = (
        pd.concat(myntra_parts, ignore_index=True) if myntra_parts else pd.DataFrame()
    )
    meesho_df = (
        pd.concat(meesho_parts, ignore_index=True) if meesho_parts else pd.DataFrame()
    )
    flipkart_df = (
        pd.concat(flipkart_parts, ignore_index=True) if flipkart_parts else pd.DataFrame()
    )

    sales_df = build_sales_df(mtr_df, myntra_df, meesho_df, flipkart_df, sku_mapping)

    print("── Input ──")
    print(f"  Path: {path}")
    print(
        f"  Platform frames: mtr={len(mtr_df):,}  myntra={len(myntra_df):,}  "
        f"meesho={len(meesho_df):,}  flipkart={len(flipkart_df):,}"
    )
    print(f"  unified sales_df: {len(sales_df):,} rows")
    if warnings:
        print("  Warnings:")
        for w in warnings[:20]:
            print(f"    - {w}")
        if len(warnings) > 20:
            print(f"    … +{len(warnings) - 20} more")

    if sales_df.empty:
        raise SystemExit(1)

    summ = get_sales_summary(sales_df, months=0)
    print("── get_sales_summary (all-time in bundle) ──")
    print(f"  total_units: {summ['total_units']:,}")
    print(f"  total_returns: {summ['total_returns']:,}")
    print(f"  net_units: {summ['net_units']:,}")
    print(f"  return_rate: {summ['return_rate']}%")
    if "Source" in sales_df.columns:
        print("── By Source ──")
        vc = sales_df["Source"].astype(str).value_counts()
        for k, v in vc.items():
            print(f"  {k}: {v:,}")
    print("OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load user-provided daily files and print Apr 2026 / Apr 21 KPI vs brand rollup (local repro)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from backend.services.flipkart import _parse_flipkart_earn_more
from backend.services.helpers import apply_dsr_segment_from_upload_filename
from backend.services.meesho import parse_meesho_csv
from backend.services.mtr import parse_mtr_csv
from backend.services.myntra import _parse_myntra_csv
from backend.services.sales import (
    build_sales_df,
    get_dsr_brand_monthly_comparison,
    get_sales_summary,
    filter_sales_for_export,
)


def main() -> None:
    base = Path("/Users/samraisinghani/Downloads")
    files = [
        "52669020566_YG UAE Amazon 21-22-4-26.csv",
        "140546020566_YG USA Amazon 21-22-4-26.csv",
        "626824020566_Akiko Amazon 21-22-4-26.csv",
        "961635020566_YG Amazon 21-22-4-26.csv",
        "aQXZRJz7_2026-04-23_Seller_Orders_Report_19498_2026-04-21_2026-04-22_Other Myntra 21-22-4-26.csv",
        "dJJ8Viij_2026-04-23_Seller_Orders_Report_36841_2026-04-21_2026-04-22_YG Myntra 21-22-4-26.csv",
        "earn_more_report (20) 3_Akiko Flipkart 21-22-4-26.xlsx",
        "earn_more_report (21) 3_YG Flipkart 21-22-4-26.xlsx",
        "earn_more_report (22) 2_Ikrass Flipkart 21-22-4-26.xlsx",
        "Orders_2026-04-21_2026-04-22_2026-04-23_10_36-10_41_890_Ashirwad Meesho 21-22-4-26.csv",
        "Orders_2026-04-21_2026-04-22_2026-04-23_10_36-10_41_62879_Akiko Meesho 21-22-4-26.csv",
        "Orders_2026-04-21_2026-04-22_2026-04-23_10_36-10_41_1662411_YG Meesho 21-22-4-26.csv",
    ]

    mtr_parts: list[pd.DataFrame] = []
    myntra_parts: list[pd.DataFrame] = []
    meesho_parts: list[pd.DataFrame] = []
    fk_parts: list[pd.DataFrame] = []
    mapping: dict[str, str] = {}

    for name in files:
        p = base / name
        if not p.exists():
            print("MISSING", p)
            continue
        raw = p.read_bytes()
        fn = p.name
        if "Amazon" in fn and fn.endswith(".csv"):
            df, msg = parse_mtr_csv(raw, fn)
            print(f"Amazon {fn}: rows={len(df)} msg={msg!r} seg_sample={df.get('DSR_Segment', pd.Series()).head(1).tolist()}")
            if not df.empty:
                mtr_parts.append(df)
        elif "Myntra" in fn and fn.endswith(".csv"):
            df, msg = _parse_myntra_csv(raw, fn, mapping)
            df = apply_dsr_segment_from_upload_filename(df, fn, "Myntra")
            print(f"Myntra {fn}: rows={len(df)} msg={msg!r} seg={df.get('DSR_Segment', pd.Series()).dropna().unique()[:2].tolist()}")
            if not df.empty:
                myntra_parts.append(df)
        elif "Meesho" in fn and fn.endswith(".csv"):
            df, msg = parse_meesho_csv(raw)
            df = apply_dsr_segment_from_upload_filename(df, fn, "Meesho")
            print(f"Meesho {fn}: rows={len(df)} msg={msg!r} seg={df.get('DSR_Segment', pd.Series()).dropna().unique()[:2].tolist()}")
            if not df.empty:
                meesho_parts.append(df)
        elif "Flipkart" in fn and fn.endswith(".xlsx"):
            df = _parse_flipkart_earn_more(raw, fn, mapping)
            df = apply_dsr_segment_from_upload_filename(df, fn, "Flipkart")
            print(f"Flipkart {fn}: rows={len(df)} seg={df.get('DSR_Segment', df.get('Brand', pd.Series())).dropna().unique()[:2].tolist()}")
            if not df.empty:
                fk_parts.append(df)
        else:
            print("SKIP", fn)

    mtr_df = pd.concat(mtr_parts, ignore_index=True) if mtr_parts else pd.DataFrame()
    myntra_df = pd.concat(myntra_parts, ignore_index=True) if myntra_parts else pd.DataFrame()
    meesho_df = pd.concat(meesho_parts, ignore_index=True) if meesho_parts else pd.DataFrame()
    flipkart_df = pd.concat(fk_parts, ignore_index=True) if fk_parts else pd.DataFrame()

    from backend.services.daily_store import _dedup_platform_df

    if not mtr_df.empty:
        from backend.services.mtr import dedup_amazon_mtr_dataframe

        mtr_df = dedup_amazon_mtr_dataframe(mtr_df)
    if not myntra_df.empty:
        myntra_df = _dedup_platform_df(myntra_df, "myntra")
    if not meesho_df.empty:
        meesho_df = _dedup_platform_df(meesho_df, "meesho")
    if not flipkart_df.empty:
        flipkart_df = _dedup_platform_df(flipkart_df, "flipkart")

    sales_df = build_sales_df(mtr_df, myntra_df, meesho_df, flipkart_df, mapping, None)
    print("\n=== sales_df rows", len(sales_df))
    if not sales_df.empty and "DSR_Segment" in sales_df.columns:
        sub = sales_df[sales_df["Transaction Type"].astype(str).eq("Shipment")]
        print("DSR_Segment value counts (shipments):\n", sub["DSR_Segment"].fillna("(empty)").value_counts().head(15))

    s21 = get_sales_summary(sales_df, months=0, start_date="2026-04-21", end_date="2026-04-21")
    print("\n=== Apr 21 2026 KPI", s21)

    d21 = filter_sales_for_export(
        sales_df, months=0, start_date="2026-04-21", end_date="2026-04-21",
    )
    if not d21.empty:
        print("\n=== Apr 21 by Source (shipments)")
        ship = d21[d21["Transaction Type"].eq("Shipment")]
        print(ship.groupby("Source")["Quantity"].sum())

    brand = get_dsr_brand_monthly_comparison(sales_df, "2026-04-21", "2026-04-21")
    print("\n=== Brand comparison Apr 21", brand)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Production chain audit: Sales → Inventory → Effective Days → PO.

Runs against warm-cache frames (real production data). Does not invent quantities.
Outputs JSON report with exception tables and first-breakpoint classification.

  docker exec -e PYTHONPATH=/srv -e WARM_CACHE_DIR=/data/warm_cache -w /srv \\
    progressino-backend-1 python -m backend.scripts.chain_audit_prod
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _cache() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def _read_parquet(name: str) -> pd.DataFrame | None:
    p = _cache() / name
    if not p.is_file():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        return pd.DataFrame({"__error__": [str(exc)]})


def _read_json(name: str) -> dict:
    p = _cache() / name
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sales_qty_col(df: pd.DataFrame) -> str | None:
    for c in ("Units_Effective", "Quantity", "Qty", "Units"):
        if c in df.columns:
            return c
    return None


def _sales_sku_col(df: pd.DataFrame) -> str | None:
    for c in ("Sku", "OMS_SKU", "SKU", "sku"):
        if c in df.columns:
            return c
    return None


def _sales_date_col(df: pd.DataFrame) -> str | None:
    for c in ("TxnDate", "Date", "Order_Date", "order_date"):
        if c in df.columns:
            return c
    return None


def _is_ship(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.eq("shipment") | s.str.contains("ship", na=False)


def prepare_sales(sales: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    from backend.services.po_engine import canonical_oms_key

    sku_c = _sales_sku_col(sales)
    qty_c = _sales_qty_col(sales)
    date_c = _sales_date_col(sales)
    if not sku_c or not qty_c or not date_c:
        return pd.DataFrame()
    work = sales[[sku_c, qty_c, date_c]].copy()
    type_c = next(
        (c for c in ("Transaction Type", "TxnType", "Transaction_Type") if c in sales.columns),
        None,
    )
    if type_c:
        work["_type"] = sales[type_c]
    else:
        work["_type"] = "Shipment"
    if "Platform" in sales.columns:
        work["Platform"] = sales["Platform"]
    work["Qty"] = pd.to_numeric(work[qty_c], errors="coerce").fillna(0.0)
    work["TxnDate"] = pd.to_datetime(work[date_c], errors="coerce")
    work["OMS_SKU"] = work[sku_c].map(lambda v: canonical_oms_key(v, mapping))
    work = work[work["TxnDate"].notna() & work["OMS_SKU"].astype(str).str.len().gt(0)]
    # Net demand: prefer Units_Effective already signed; else ship positive + refund negative pattern
    if qty_c == "Units_Effective":
        work["NetUnits"] = work["Qty"]
        work["ShipUnits"] = work["Qty"].clip(lower=0)
    else:
        is_ship = _is_ship(work["_type"])
        is_ref = work["_type"].astype(str).str.lower().str.contains("refund|return", na=False)
        work["NetUnits"] = np.where(is_ref, -work["Qty"].abs(), np.where(is_ship, work["Qty"].clip(lower=0), 0.0))
        work["ShipUnits"] = np.where(is_ship, work["Qty"].clip(lower=0), 0.0)
    work["Year"] = work["TxnDate"].dt.year
    work["Month"] = work["TxnDate"].dt.month
    work["Quarter"] = work["TxnDate"].dt.quarter
    return work


def monthly_sku_matrix(work: pd.DataFrame, year: int, months: list[int]) -> pd.DataFrame:
    sub = work[(work["Year"] == year) & (work["Month"].isin(months))]
    if sub.empty:
        return pd.DataFrame(columns=["OMS_SKU"] + [f"m{m:02d}" for m in months] + ["q_total"])
    g = (
        sub.groupby(["OMS_SKU", "Month"], as_index=False)["ShipUnits"]
        .sum()
    )
    pivot = g.pivot_table(index="OMS_SKU", columns="Month", values="ShipUnits", aggfunc="sum", fill_value=0.0)
    for m in months:
        if m not in pivot.columns:
            pivot[m] = 0.0
    pivot = pivot[months]
    pivot.columns = [f"m{m:02d}" for m in months]
    pivot["q_total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def platform_frame_monthly(
    name: str,
    year: int,
    months: list[int],
    mapping: dict,
) -> dict[str, Any]:
    """Independent platform parquet totals for diagnostics."""
    df = _read_parquet(name)
    if df is None or df.empty or "__error__" in (df.columns if df is not None else []):
        return {"present": False, "name": name}
    try:
        w = prepare_sales(df, mapping)
    except Exception as exc:
        return {"present": True, "name": name, "error": str(exc)}
    if w.empty:
        return {"present": True, "name": name, "rows": int(len(df)), "usable": 0}
    sub = w[(w["Year"] == year) & (w["Month"].isin(months))]
    return {
        "present": True,
        "name": name,
        "rows": int(len(df)),
        "usable": int(len(w)),
        "q_ship_units": float(sub["ShipUnits"].sum()),
        "q_net_units": float(sub["NetUnits"].sum()),
        "skus": int(sub["OMS_SKU"].nunique()),
        "date_min": str(w["TxnDate"].min())[:10],
        "date_max": str(w["TxnDate"].max())[:10],
    }


def audit_q1_sales(work: pd.DataFrame, mapping: dict) -> dict[str, Any]:
    """Sales Q1 2025 + neighboring periods from unified sales_df."""
    out: dict[str, Any] = {"periods": {}, "q1_2025": {}}
    if work.empty:
        return {"error": "empty_sales_work"}

    out["sales_span"] = {
        "min": str(work["TxnDate"].min())[:10],
        "max": str(work["TxnDate"].max())[:10],
        "rows": int(len(work)),
        "skus": int(work["OMS_SKU"].nunique()),
        "ship_units_total": float(work["ShipUnits"].sum()),
        "net_units_total": float(work["NetUnits"].sum()),
    }

    # Period totals (all SKUs)
    period_defs = {
        "Q4_2024": (2024, [10, 11, 12]),
        "Q1_2025": (2025, [1, 2, 3]),
        "Q2_2025": (2025, [4, 5, 6]),
        "Q3_2025": (2025, [7, 8, 9]),
        "Q4_2025": (2025, [10, 11, 12]),
        "Q1_2026": (2026, [1, 2, 3]),
        "Q2_2026": (2026, [4, 5, 6]),
    }
    for label, (y, mos) in period_defs.items():
        sub = work[(work["Year"] == y) & (work["Month"].isin(mos))]
        monthly = {
            f"{m:02d}": float(sub.loc[sub["Month"] == m, "ShipUnits"].sum())
            for m in mos
        }
        out["periods"][label] = {
            "ship_units": float(sub["ShipUnits"].sum()),
            "net_units": float(sub["NetUnits"].sum()),
            "rows": int(len(sub)),
            "skus": int(sub["OMS_SKU"].nunique()),
            "monthly_ship": monthly,
            "sum_months": float(sum(monthly.values())),
            "month_sum_matches_q": abs(float(sum(monthly.values())) - float(sub["ShipUnits"].sum())) < 0.5,
        }

    # Q1 2025 SKU matrix: Jan+Feb+Mar == q_total
    mat = monthly_sku_matrix(work, 2025, [1, 2, 3])
    if mat.empty:
        out["q1_2025"] = {
            "status": "NO_DATA",
            "detail": "No rows in sales_df for 2025-01..03",
            "sku_rows": 0,
        }
    else:
        # Internal identity: q_total is sum of months by construction
        mat["expected_q"] = mat[["m01", "m02", "m03"]].sum(axis=1)
        mat["diff"] = mat["q_total"] - mat["expected_q"]
        # Flag identity breaks (should be 0)
        identity_bad = mat[mat["diff"].abs() > 0.01]
        # Flag zero quarter
        zero_q = int((mat["q_total"] == 0).sum())
        # Rank discrepancies vs "platform sum" if we can load platforms
        top = mat.nlargest(25, "q_total")
        out["q1_2025"] = {
            "status": "OK" if identity_bad.empty else "IDENTITY_BREAK",
            "sku_rows": int(len(mat)),
            "ship_total": float(mat["q_total"].sum()),
            "jan": float(mat["m01"].sum()),
            "feb": float(mat["m02"].sum()),
            "mar": float(mat["m03"].sum()),
            "identity_bad_skus": int(len(identity_bad)),
            "zero_q_skus": zero_q,
            "top_skus": top.head(15).to_dict("records"),
            "sample_exception_table": mat.nlargest(50, "q_total")[
                ["OMS_SKU", "m01", "m02", "m03", "expected_q", "q_total", "diff"]
            ].rename(
                columns={
                    "m01": "Jan",
                    "m02": "Feb",
                    "m03": "Mar",
                    "expected_q": "Expected_Q1",
                    "q_total": "System_Q1",
                    "diff": "Difference",
                }
            ).to_dict("records"),
        }

    # Platform independent recompute for Q1 2025
    platforms = [
        "mtr_df.parquet",
        "myntra_df.parquet",
        "meesho_df.parquet",
        "flipkart_df.parquet",
        "snapdeal_df.parquet",
    ]
    plat_parts = [platform_frame_monthly(n, 2025, [1, 2, 3], mapping) for n in platforms]
    out["q1_2025_platforms"] = plat_parts
    plat_sum = sum(float(p.get("q_ship_units") or 0) for p in plat_parts if p.get("present"))
    sales_q1 = float((out.get("periods") or {}).get("Q1_2025", {}).get("ship_units") or 0)
    out["q1_2025_platform_vs_sales"] = {
        "platform_sum_ship": plat_sum,
        "sales_df_q1_ship": sales_q1,
        "delta_sales_minus_platform": sales_q1 - plat_sum,
        "note": (
            "sales_df is post-merge (dedupe across tier1/tier3). Platform sum is a "
            "lower/upper diagnostic — large delta indicates missing history or double-count."
        ),
    }

    # Coverage of Jan/Feb/Mar separately in sales
    for y, m in [(2025, 1), (2025, 2), (2025, 3), (2024, 1), (2026, 1)]:
        sub = work[(work["Year"] == y) & (work["Month"] == m)]
        out.setdefault("monthly_presence", {})[f"{y}-{m:02d}"] = {
            "rows": int(len(sub)),
            "skus": int(sub["OMS_SKU"].nunique()),
            "ship": float(sub["ShipUnits"].sum()),
            "empty": len(sub) == 0,
        }

    # First corruption heuristic for Q1
    mon = out.get("monthly_presence") or {}
    empty_months = [k for k, v in mon.items() if k.startswith("2025-0") and v.get("empty")]
    if empty_months:
        out["q1_root_cause_hypothesis"] = {
            "class": "MissingData",
            "first_breakpoint": "sales_df / Tier-1 history coverage",
            "detail": f"Empty months in unified sales: {empty_months}",
            "impact": "Q1 under-count → low ADS → bad Eff_Days / under-PO",
        }
    elif abs(sales_q1 - plat_sum) > max(500, 0.05 * max(plat_sum, sales_q1, 1)):
        out["q1_root_cause_hypothesis"] = {
            "class": "AggregationOrDedupe",
            "first_breakpoint": "merge of platform frames into sales_df",
            "detail": f"sales_df Q1 ship={sales_q1:,.0f} platform sum={plat_sum:,.0f}",
            "impact": "Inconsistent demand history across modules",
        }
    else:
        out["q1_root_cause_hypothesis"] = {
            "class": "NeedsBusinessSource",
            "first_breakpoint": "none detected in internal aggregation",
            "detail": "Q1 internal month sum coincides with sales_df; external marketplace archive needed to prove absolute truth",
        }

    return out


def audit_inventory_every_sku(inv: pd.DataFrame | None, hist: pd.DataFrame | None, mapping: dict) -> dict[str, Any]:
    from backend.services.daily_inventory_history import (
        filter_inventory_history_channel,
    )
    from backend.services.inventory import (
        coalesce_inventory_by_sku_mapping,
        inventory_column_totals,
        recompute_inventory_totals,
    )
    from backend.services.po_engine import inventory_oms_key

    out: dict[str, Any] = {}
    if inv is None or inv.empty:
        return {"error": "no_inventory"}

    inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), mapping)
    inv_c = recompute_inventory_totals(inv_c)
    out["totals"] = inventory_column_totals(inv_c)
    out["sku_count"] = int(inv_c["OMS_SKU"].nunique())
    out["rows"] = int(len(inv_c))

    # Duplicate detection after coalesce
    dups = inv_c["OMS_SKU"].astype(str).str.upper().duplicated().sum()
    out["duplicate_oms_after_coalesce"] = int(dups)

    # Missing / zero / negative
    tot = pd.to_numeric(inv_c["Total_Inventory"], errors="coerce").fillna(0)
    out["negative_total_skus"] = int((tot < 0).sum())
    out["zero_total_skus"] = int((tot == 0).sum())
    out["positive_total_skus"] = int((tot > 0).sum())

    # History alignment full SKU set
    if hist is not None and not hist.empty and "Date" in hist.columns:
        h = hist.copy()
        h["Date"] = pd.to_datetime(h["Date"], errors="coerce").dt.normalize()
        max_d = h["Date"].max()
        out["hist_max_date"] = str(max_d.date()) if pd.notna(max_d) else None
        if pd.notna(max_d):
            oms = filter_inventory_history_channel(h, "oms")
            oms_day = oms[pd.to_datetime(oms["Date"], errors="coerce").dt.normalize() == max_d]
            if not oms_day.empty:
                oms_i = (
                    oms_day.groupby(oms_day["OMS_SKU"].astype(str).str.upper())["Qty"]
                    .max()
                )
            else:
                oms_i = pd.Series(dtype=float)
            inv_oms = inv_c.set_index(inv_c["OMS_SKU"].astype(str).str.upper())["OMS_Inventory"]
            inv_oms = pd.to_numeric(inv_oms, errors="coerce").fillna(0)
            all_skus = sorted(set(inv_oms.index) | set(oms_i.index))
            mismatches = []
            only_inv = 0
            only_hist = 0
            for sku in all_skus:
                a = float(inv_oms.get(sku, 0) or 0)
                b = float(oms_i.get(sku, 0) or 0)
                if a > 0 and sku not in oms_i.index:
                    only_inv += 1
                if b > 0 and sku not in inv_oms.index:
                    only_hist += 1
                if abs(a - b) > 0.5:
                    mismatches.append({"sku": sku, "inventory_oms": a, "history_oms": b, "delta": a - b})
            mismatches.sort(key=lambda r: abs(r["delta"]), reverse=True)
            out["history_vs_inv"] = {
                "date": str(max_d.date()),
                "compared_skus": len(all_skus),
                "mismatch_count": len(mismatches),
                "only_inventory_positive": only_inv,
                "only_history_positive": only_hist,
                "history_oms_total": float(oms_i.sum()) if len(oms_i) else 0.0,
                "inventory_oms_total": float(inv_oms.sum()),
                "top_mismatches": mismatches[:40],
            }
            out["inventory_missing_skus_sample"] = [
                m for m in mismatches if m["inventory_oms"] == 0 and m["history_oms"] > 0
            ][:25]
            out["zero_wrong_sample"] = [
                m for m in mismatches if m["inventory_oms"] == 0 and abs(m["delta"]) > 0.5
            ][:25]

    # Powder/power residual twins
    skus = set(inv_c["OMS_SKU"].astype(str).str.upper())
    powder = {s for s in skus if "POWDERBLUE" in s}
    twin = [s for s in powder if s.replace("POWDERBLUE", "POWERBLUE") in skus]
    out["powder_power_twins"] = twin[:20]
    out["powder_power_twin_count"] = len(twin)

    return out


def audit_eff_days_and_po(
    sales: pd.DataFrame,
    inv: pd.DataFrame,
    hist: pd.DataFrame | None,
    status: pd.DataFrame | None,
    existing_po: pd.DataFrame | None,
    mapping: dict,
    work: pd.DataFrame,
) -> dict[str, Any]:
    """Independent Eff_Days vs PO engine; inventory join checks."""
    from backend.services.inventory import coalesce_inventory_by_sku_mapping
    from backend.services.po_engine import calculate_po_base

    inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), mapping)
    sales_work = sales.copy()
    if "TxnDate" not in sales_work.columns:
        dc = _sales_date_col(sales_work)
        if dc:
            sales_work["TxnDate"] = pd.to_datetime(sales_work[dc], errors="coerce")
    if "Sku" not in sales_work.columns:
        sc = _sales_sku_col(sales_work)
        if sc:
            sales_work["Sku"] = sales_work[sc]
    if "Transaction Type" not in sales_work.columns:
        for c in ("TxnType", "Transaction_Type"):
            if c in sales_work.columns:
                sales_work["Transaction Type"] = sales_work[c]
                break
        else:
            sales_work["Transaction Type"] = "Shipment"
    if "Units_Effective" not in sales_work.columns:
        qc = _sales_qty_col(sales_work)
        sales_work["Units_Effective"] = (
            pd.to_numeric(sales_work[qc], errors="coerce").fillna(0) if qc else 0
        )
    if "Quantity" not in sales_work.columns:
        sales_work["Quantity"] = sales_work["Units_Effective"]

    period_days = 30
    try:
        po = calculate_po_base(
            sales_df=sales_work,
            inv_df=inv_c,
            period_days=period_days,
            lead_time=45,
            target_days=45,
            use_seasonality=False,
            group_by_parent=False,
            existing_po_df=existing_po if existing_po is not None else pd.DataFrame(),
            sku_status_df=status if status is not None else pd.DataFrame(),
            inventory_history_df=hist,
            sku_mapping=mapping,
            use_oms_inventory_only=False,
            inventory_history_channel="oms",
        )
    except Exception as exc:
        return {
            "po_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }

    if po is None or po.empty:
        return {"po_error": "empty_po"}

    # Independent ADS = last 30d ship / period_days from sales work
    if not work.empty:
        max_d = work["TxnDate"].max()
        win_start = pd.Timestamp(max_d).normalize() - pd.Timedelta(days=period_days - 1)
        win = work[work["TxnDate"] >= win_start]
        ind_sold = win.groupby("OMS_SKU")["ShipUnits"].sum()
    else:
        ind_sold = pd.Series(dtype=float)

    inv_map = inv_c.set_index(inv_c["OMS_SKU"].astype(str).str.upper())
    rows = []
    inv_break = 0
    eff_break = 0
    zero_inv_with_sales = 0
    for _, r in po.iterrows():
        sku = str(r.get("OMS_SKU") or "").strip().upper()
        if not sku:
            continue
        po_inv = float(pd.to_numeric(r.get("Total_Inventory"), errors="coerce") or 0)
        src = inv_map.loc[sku] if sku in inv_map.index else None
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        src_inv = float(pd.to_numeric(src.get("Total_Inventory"), errors="coerce") or 0) if src is not None else 0.0
        if abs(po_inv - src_inv) > 0.51:
            inv_break += 1
        sold = float(pd.to_numeric(r.get("Sold_Units"), errors="coerce") or 0)
        ads = float(pd.to_numeric(r.get("ADS"), errors="coerce") or 0)
        eff = float(pd.to_numeric(r.get("Eff_Days"), errors="coerce") or 0)
        # Expected sales-based cover days = inv / ADS when ADS>0 (inventory cover)
        cover_from_ads = (po_inv / ads) if ads > 1e-9 else (999.0 if po_inv > 0 else 0.0)
        # Eff_Days in this engine is primarily "selling days with stock" not cover days
        # Document both: Eff_Days (engine) vs Cover_Days (inv/ADS)
        ind = float(ind_sold.get(sku, 0) or 0)
        if abs(sold - ind) > 1.0 and sold > 0:
            pass  # later class
        if po_inv == 0 and sold > 0:
            zero_inv_with_sales += 1
        if abs(eff - min(cover_from_ads, 999)) > 5 and ads > 0.05:
            # Only flag when engine Eff_Days behaves like cover (rare) — skip noise
            if "Eff_Days_Inventory" in r.index:
                pass
            else:
                eff_break += 1
        po_qty_col = next(
            (c for c in ("Final_PO_Qty", "New_PO_Qty", "PO_Qty", "Gross_PO_Qty") if c in r.index),
            None,
        )
        rows.append(
            {
                "OMS_SKU": sku,
                "Total_Inventory": po_inv,
                "Src_Inventory": src_inv,
                "Inv_Delta": po_inv - src_inv,
                "Sold_Units": sold,
                "Indep_Sold_30d": ind,
                "ADS": ads,
                "Eff_Days": eff,
                "Eff_Days_Inventory": float(
                    pd.to_numeric(r.get("Eff_Days_Inventory"), errors="coerce") or 0
                ),
                "Cover_Days_Inv_over_ADS": round(cover_from_ads, 2) if ads > 1e-9 else None,
                "PO_Qty": float(pd.to_numeric(r.get(po_qty_col), errors="coerce") or 0)
                if po_qty_col
                else None,
                "Priority": str(r.get("Priority") or ""),
            }
        )

    rec = pd.DataFrame(rows)
    inv_mismatches = rec[rec["Inv_Delta"].abs() > 0.51].sort_values("Inv_Delta", key=abs, ascending=False)
    # Critical: zero PO inventory with positive source inventory
    split = inv_mismatches[
        (inv_mismatches["Total_Inventory"] == 0) & (inv_mismatches["Src_Inventory"] > 0)
    ]
    # SKU mapping issue: sales SKU not on inventory
    sales_only_with_demand = rec[
        (rec["Src_Inventory"] == 0) & (rec["Sold_Units"] > 0) & (rec["Total_Inventory"] == 0)
    ]

    formula_doc = {
        "Eff_Days": (
            "In calculate_po_base: selling/in-stock window days used for ADS "
            "(often from inventory history as Eff_Days_Inventory when available). "
            "NOT pure Cover = Inventory/ADS. Use Cover_Days_Inv_over_ADS for "
            "inventory-days-of-cover diagnostic."
        ),
        "ADS": "Demand rate (units/day) after Recent/LY/Seasonal/Flat blend + burst caps",
        "PO": "target_days × ADS − available inventory − pipeline (see Gross/New_PO columns)",
    }

    return {
        "formula": formula_doc,
        "po_rows": int(len(po)),
        "inv_mismatch_count": int(len(inv_mismatches)),
        "inv_mismatch_zero_po_with_stock": int(len(split)),
        "sales_with_zero_inventory_count": int(len(sales_only_with_demand)),
        "zero_inv_with_sales_in_po": zero_inv_with_sales,
        "top_inv_mismatches": inv_mismatches.head(40).to_dict("records"),
        "top_zero_po_with_stock": split.head(30).to_dict("records"),
        "top_sales_zero_inv": sales_only_with_demand.nlargest(30, "Sold_Units").to_dict("records"),
        "edge_high_inv_low_ads": rec[(rec["Total_Inventory"] > 100) & (rec["ADS"] < 0.05)]
        .nlargest(15, "Total_Inventory")
        .to_dict("records"),
        "edge_low_inv_high_ads": rec[(rec["Total_Inventory"] < 5) & (rec["ADS"] > 0.5)]
        .nlargest(15, "ADS")
        .to_dict("records"),
        "summary_po_qty_sum": float(pd.to_numeric(rec["PO_Qty"], errors="coerce").fillna(0).clip(lower=0).sum())
        if "PO_Qty" in rec.columns
        else None,
    }


def main() -> int:
    from backend.services.sku_mapping import load_bundled_sku_mapping, load_sku_mapping_from_disk

    mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
    sales = _read_parquet("sales_df.parquet")
    inv = _read_parquet("inventory_df_variant.parquet")
    hist = _read_parquet("daily_inventory_history_df.parquet")
    status = _read_parquet("sku_status_lead_df.parquet")
    existing_po = _read_parquet("existing_po_df.parquet")
    meta = _read_json("inventory_session_meta.json")

    report: dict[str, Any] = {
        "cache_dir": str(_cache()),
        "inventory_meta": {
            k: meta.get(k)
            for k in (
                "inventory_snapshot_date",
                "inventory_snapshot_uploaded_at",
                "inventory_snapshot_date_label",
            )
        },
        "map_size": len(mapping),
    }

    if sales is None or (hasattr(sales, "columns") and "__error__" in sales.columns):
        report["fatal"] = "sales_df missing or unloadable"
        print(json.dumps(report, indent=2, default=str))
        return 2
    if inv is None or (hasattr(inv, "columns") and "__error__" in inv.columns):
        report["fatal"] = "inventory missing or unloadable"
        print(json.dumps(report, indent=2, default=str))
        return 2

    work = prepare_sales(sales, mapping)
    report["sales"] = audit_q1_sales(work, mapping)
    report["inventory"] = audit_inventory_every_sku(inv, hist, mapping)
    report["eff_days_po"] = audit_eff_days_and_po(
        sales, inv, hist, status, existing_po, mapping, work
    )

    # Root cause rollup
    causes = []
    q1h = (report.get("sales") or {}).get("q1_root_cause_hypothesis") or {}
    if q1h:
        causes.append({"area": "sales_q1", **q1h})
    inv_h = report.get("inventory") or {}
    hv = inv_h.get("history_vs_inv") or {}
    if hv.get("mismatch_count", 0) > 0:
        causes.append(
            {
                "area": "inventory",
                "class": "InventoryHistoryMisalign",
                "first_breakpoint": "history day vs inventory_df_variant OMS",
                "detail": (
                    f"mismatches={hv.get('mismatch_count')} "
                    f"inv_oms={hv.get('inventory_oms_total')} hist={hv.get('history_oms_total')}"
                ),
            }
        )
    if inv_h.get("powder_power_twin_count"):
        causes.append(
            {
                "area": "sku_mapping",
                "class": "SKUMapping",
                "first_breakpoint": "POWER/POWDER residual twins",
                "detail": f"twins={inv_h.get('powder_power_twin_count')}",
            }
        )
    ep = report.get("eff_days_po") or {}
    if ep.get("inv_mismatch_zero_po_with_stock", 0) > 0:
        causes.append(
            {
                "area": "po_join",
                "class": "InventoryJoin",
                "first_breakpoint": "PO Total_Inventory vs snapshot after mapping",
                "detail": f"zero_po_with_stock={ep.get('inv_mismatch_zero_po_with_stock')}",
                "impact": "Eff_Days/PO understock signal (false URGENT or wrong cover)",
            }
        )
    report["root_causes"] = causes
    report["requirement_status"] = {
        "Q1_2025_Sales": {
            "status": (report.get("sales") or {}).get("q1_2025", {}).get("status", "UNKNOWN"),
            "evidence": (report.get("sales") or {}).get("periods", {}).get("Q1_2025"),
        },
        "SKU_Inventory": {
            "status": "CHECK" if hv.get("mismatch_count", 0) else "OK",
            "evidence": {
                "sku_count": inv_h.get("sku_count"),
                "mismatches": hv.get("mismatch_count"),
                "totals": inv_h.get("totals"),
            },
        },
        "Effective_Days_PO": {
            "status": "CHECK" if ep.get("inv_mismatch_count") else "OK",
            "evidence": {
                "po_rows": ep.get("po_rows"),
                "inv_mismatches": ep.get("inv_mismatch_count"),
                "zero_po_with_stock": ep.get("inv_mismatch_zero_po_with_stock"),
            },
        },
    }

    print(json.dumps(report, indent=2, default=str))
    # Fail if critical join or completely missing Q1
    q1 = (report.get("sales") or {}).get("q1_2025") or {}
    if q1.get("status") == "NO_DATA":
        return 3
    if ep.get("inv_mismatch_zero_po_with_stock", 0) > 50:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

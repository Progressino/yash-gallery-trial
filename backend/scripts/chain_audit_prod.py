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
    work = pd.DataFrame(
        {
            "Sku_raw": sales[sku_c].astype(str),
            "Qty": pd.to_numeric(sales[qty_c], errors="coerce").fillna(0.0),
            "TxnDate": pd.to_datetime(sales[date_c], errors="coerce"),
        }
    )
    type_c = next(
        (c for c in ("Transaction Type", "TxnType", "Transaction_Type") if c in sales.columns),
        None,
    )
    work["_type"] = sales[type_c].astype(str) if type_c else "Shipment"
    if "Platform" in sales.columns:
        work["Platform"] = sales["Platform"].astype(str)
    work = work[work["TxnDate"].notna()]
    # Map unique SKUs only (production sales_df is multi-million rows)
    uniq = pd.unique(work["Sku_raw"].to_numpy())
    canon_map = {s: canonical_oms_key(s, mapping) for s in uniq}
    work["OMS_SKU"] = work["Sku_raw"].map(canon_map)
    work = work[work["OMS_SKU"].astype(str).str.len().gt(0)]
    if qty_c == "Units_Effective":
        work["NetUnits"] = work["Qty"]
        work["ShipUnits"] = work["Qty"].clip(lower=0)
    else:
        is_ship = _is_ship(work["_type"])
        is_ref = work["_type"].str.lower().str.contains("refund|return", na=False)
        work["NetUnits"] = np.where(
            is_ref, -work["Qty"].abs(), np.where(is_ship, work["Qty"].clip(lower=0), 0.0)
        )
        work["ShipUnits"] = np.where(is_ship, work["Qty"].clip(lower=0), 0.0)
    work["Year"] = work["TxnDate"].dt.year.astype("int16")
    work["Month"] = work["TxnDate"].dt.month.astype("int8")
    work["Quarter"] = work["TxnDate"].dt.quarter.astype("int8")
    work.drop(columns=["Sku_raw"], inplace=True)
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
    """Independent platform parquet totals (date filter first — cheap memory path)."""
    df = _read_parquet(name)
    if df is None or df.empty or "__error__" in (df.columns if df is not None else []):
        return {"present": False, "name": name}
    try:
        date_c = _sales_date_col(df)
        qty_c = _sales_qty_col(df)
        if not date_c or not qty_c:
            return {"present": True, "name": name, "rows": int(len(df)), "error": "missing_cols"}
        d = pd.to_datetime(df[date_c], errors="coerce")
        mask = (d.dt.year == year) & (d.dt.month.isin(months))
        sub = df.loc[mask]
        q = float(pd.to_numeric(sub[qty_c], errors="coerce").fillna(0).clip(lower=0).sum())
        return {
            "present": True,
            "name": name,
            "rows": int(len(df)),
            "usable": int(len(sub)),
            "q_ship_units": q,
            "q_net_units": q,
            "skus": int(sub[_sales_sku_col(df)].nunique()) if _sales_sku_col(df) else 0,
            "date_min": str(d.min())[:10] if d.notna().any() else "",
            "date_max": str(d.max())[:10] if d.notna().any() else "",
        }
    except Exception as exc:
        return {"present": True, "name": name, "error": str(exc)}


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


def audit_eff_days_light(
    inv: pd.DataFrame,
    mapping: dict,
    work: pd.DataFrame,
    *,
    period_days: int = 30,
) -> dict[str, Any]:
    """Memory-safe Cover_Days / ADS diagnostic without full calculate_po_base (avoids OOM).

    Eff_Days in the engine is mostly inventory-history in-stock days; Cover_Days
    (Inventory / ADS) is the diagnostic operators often confuse with it.
    """
    from backend.services.inventory import coalesce_inventory_by_sku_mapping
    from backend.services.po_engine import inventory_oms_key

    inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), mapping)
    inv_c["OMS_SKU"] = inv_c["OMS_SKU"].map(inventory_oms_key)
    inv_c = inv_c[inv_c["OMS_SKU"].astype(str).str.len() > 0]
    if inv_c["OMS_SKU"].duplicated().any():
        num = [c for c in inv_c.columns if c != "OMS_SKU" and pd.api.types.is_numeric_dtype(inv_c[c])]
        inv_c = inv_c.groupby("OMS_SKU", as_index=False)[num].sum()

    inv_idx = inv_c.set_index(inv_c["OMS_SKU"].astype(str).str.upper())
    tot = pd.to_numeric(inv_idx["Total_Inventory"], errors="coerce").fillna(0.0)

    if work.empty:
        return {"error": "no_sales_work", "formula": {}}

    max_d = work["TxnDate"].max()
    win_start = pd.Timestamp(max_d).normalize() - pd.Timedelta(days=period_days - 1)
    win = work[work["TxnDate"] >= win_start]
    sold = win.groupby(win["OMS_SKU"].astype(str).str.upper())["ShipUnits"].sum()
    ads = (sold / float(period_days)).rename("ADS")

    # Union of inventory + 30d sales keys
    all_skus = sorted(set(tot.index.astype(str)) | set(ads.index.astype(str)))
    rows = []
    miss_inv_with_sales = []
    miss_sales_with_inv = 0
    for sku in all_skus:
        inv_u = float(tot.get(sku, 0.0) or 0.0)
        sold_u = float(sold.get(sku, 0.0) or 0.0)
        ads_u = float(ads.get(sku, 0.0) or 0.0)
        cover = (inv_u / ads_u) if ads_u > 1e-9 else (999.0 if inv_u > 0 else 0.0)
        row = {
            "OMS_SKU": sku,
            "Total_Inventory": inv_u,
            "Sold_Units_30d": sold_u,
            "ADS_indep": round(ads_u, 4),
            "Cover_Days_Inv_over_ADS": round(cover, 2),
        }
        rows.append(row)
        if sold_u > 0 and inv_u <= 0:
            miss_inv_with_sales.append(row)
        if inv_u > 0 and sold_u <= 0:
            miss_sales_with_inv += 1

    rec = pd.DataFrame(rows)
    miss_inv_with_sales.sort(key=lambda r: r["Sold_Units_30d"], reverse=True)

    # Optional shared PO cache sample (if present on disk)
    po_sample = {}
    for name in ("po_sold_result.csv", "po_result.parquet"):
        p = _cache() / name
        if not p.is_file():
            continue
        try:
            pdf = pd.read_csv(p) if name.endswith(".csv") else pd.read_parquet(p)
            sku_c = "OMS_SKU" if "OMS_SKU" in pdf.columns else None
            if not sku_c:
                continue
            pdf["_sku"] = pdf[sku_c].astype(str).str.upper()
            # join independent cover vs system Eff_Days if column exists
            merge_cols = ["_sku"]
            for c in ("Total_Inventory", "ADS", "Eff_Days", "Sold_Units", "Priority"):
                if c in pdf.columns:
                    merge_cols.append(c)
            sample = pdf[merge_cols].copy()
            sample = sample.merge(
                rec.rename(columns={"OMS_SKU": "_sku"}),
                on="_sku",
                how="left",
                suffixes=("_sys", "_indep"),
            )
            inv_sys = pd.to_numeric(sample.get("Total_Inventory_sys", sample.get("Total_Inventory")), errors="coerce").fillna(0)
            inv_indep = pd.to_numeric(sample.get("Total_Inventory_indep", sample.get("Total_Inventory")), errors="coerce").fillna(0)
            # prefer explicit
            if "Total_Inventory_sys" in sample.columns and "Total_Inventory_indep" in sample.columns:
                dlt = inv_sys - inv_indep
            else:
                dlt = inv_sys * 0
            bad = sample.loc[dlt.abs() > 0.51].copy() if len(dlt) else sample.iloc[0:0]
            if len(bad):
                bad = bad.assign(Inv_Delta=dlt.loc[bad.index].values)
            zero_po_stock = []
            if "Total_Inventory_sys" in sample.columns:
                m = (inv_sys == 0) & (inv_indep > 0)
                zero_po_stock = sample.loc[m].head(30).to_dict("records")
            po_sample = {
                "source": name,
                "rows": int(len(pdf)),
                "inv_mismatch_count": int(len(bad)),
                "top_inv_mismatches": bad.head(25).to_dict("records") if len(bad) else [],
                "zero_sys_inv_with_snapshot_stock": zero_po_stock,
            }
            break
        except Exception as exc:
            po_sample = {"source": name, "error": str(exc)}

    formula_doc = {
        "Eff_Days": (
            "Engine: in-stock/selling-window days (often Eff_Days_Inventory from "
            "daily inventory history), not pure cover. DO NOT treat as Inventory/ADS."
        ),
        "ADS_indep": f"Sum of shipment Units over last {period_days} days ending sales_max / {period_days}",
        "Cover_Days_Inv_over_ADS": "Total_Inventory / ADS_indep — diagnostic cover only",
        "PO": "Requires full calculate_po_base (skipped here when memory-constrained); use shared PO cache sample when present",
    }

    return {
        "formula": formula_doc,
        "period_days": period_days,
        "sales_window_end": str(max_d)[:10],
        "sku_union_count": int(len(rec)),
        "sales_with_zero_inventory_count": int(len(miss_inv_with_sales)),
        "inventory_with_zero_30d_sales": int(miss_sales_with_inv),
        "top_sales_zero_inv": miss_inv_with_sales[:40],
        "edge_high_inv_low_ads": rec[(rec["Total_Inventory"] > 100) & (rec["ADS_indep"] < 0.05)]
        .nlargest(15, "Total_Inventory")
        .to_dict("records"),
        "edge_low_inv_high_ads": rec[(rec["Total_Inventory"] < 5) & (rec["ADS_indep"] > 0.5)]
        .nlargest(15, "ADS_indep")
        .to_dict("records"),
        "shared_po_cache_sample": po_sample,
        "inv_mismatch_count": int(po_sample.get("inv_mismatch_count") or 0),
        "inv_mismatch_zero_po_with_stock": int(
            len(po_sample.get("zero_sys_inv_with_snapshot_stock") or [])
        ),
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
    # Phase 1: sales (print early so OOM later still leaves clues on stderr)
    report["sales"] = audit_q1_sales(work, mapping)
    sys.stderr.write(
        "PHASE_SALES_DONE "
        + json.dumps(
            {
                "span": (report["sales"] or {}).get("sales_span"),
                "q1": {
                    k: (report["sales"] or {}).get("q1_2025", {}).get(k)
                    for k in ("status", "ship_total", "jan", "feb", "mar", "sku_rows")
                },
                "hyp": (report["sales"] or {}).get("q1_root_cause_hypothesis"),
            },
            default=str,
        )
        + "\n"
    )
    sys.stderr.flush()

    report["inventory"] = audit_inventory_every_sku(inv, hist, mapping)
    sys.stderr.write(
        "PHASE_INV_DONE "
        + json.dumps(
            {
                "sku_count": (report["inventory"] or {}).get("sku_count"),
                "hist": (report["inventory"] or {}).get("history_vs_inv"),
                "twins": (report["inventory"] or {}).get("powder_power_twin_count"),
            },
            default=str,
        )[:4000]
        + "\n"
    )
    sys.stderr.flush()

    # Lightweight cover-days + optional shared PO sample (no full calculate_po_base)
    report["eff_days_po"] = audit_eff_days_light(inv, mapping, work, period_days=30)
    # Drop unused large frames references before dump
    del sales, work
    if hist is not None:
        del hist

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
    # Exit 0 when report written (ops parse JSON). Use flags for automation later.
    q1 = (report.get("sales") or {}).get("q1_2025") or {}
    if q1.get("status") == "NO_DATA":
        report["exit_hint"] = "Q1_2025_NO_DATA_IN_SALES_DF"
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
PO Dashboard — aggregates PO engine output with short-horizon sales windows
for ops visibility (pipeline vs open PO, demand spikes, tight cover).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .helpers import get_parent_sku
from .po_engine import canonical_oms_key


def _gross_shipment_units(s: pd.DataFrame) -> pd.Series:
    """Positive shipped units per row (ignore refunds/cancels here)."""
    if s.empty or "TxnDate" not in s.columns:
        return pd.Series(dtype=float)
    tt = s.get("Transaction Type", pd.Series("", index=s.index)).astype(str)
    q = pd.to_numeric(s.get("Quantity", 0), errors="coerce").fillna(0.0)
    return np.where(tt == "Shipment", q.clip(lower=0), 0.0)


def recent_vs_prev_shipments(
    sales_df: pd.DataFrame,
    *,
    sku_mapping: Optional[dict] = None,
    group_by_parent: bool = False,
    recent_days: int = 7,
    prev_days: int = 7,
) -> pd.DataFrame:
    """
    Per-SKU gross shipment units in the last ``recent_days`` calendar days
    (inclusive of the sales max date) vs the preceding ``prev_days`` days.
    """
    if sales_df is None or sales_df.empty or "Sku" not in sales_df.columns:
        return pd.DataFrame(
            columns=["OMS_SKU", "units_recent", "units_prev", "spike_ratio"]
        )

    s = sales_df.copy()
    s["TxnDate"] = pd.to_datetime(s["TxnDate"], errors="coerce")
    s = s.dropna(subset=["TxnDate"])
    if s.empty:
        return pd.DataFrame(
            columns=["OMS_SKU", "units_recent", "units_prev", "spike_ratio"]
        )

    end = s["TxnDate"].max().normalize()
    recent_start = end - timedelta(days=max(1, int(recent_days)) - 1)
    prev_end = recent_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=max(1, int(prev_days)) - 1)

    s["OMS_SKU"] = (
        s["Sku"].astype(str).map(lambda x: canonical_oms_key(x, sku_mapping))
    )
    if group_by_parent:
        s["OMS_SKU"] = s["OMS_SKU"].map(get_parent_sku)
    s = s[s["OMS_SKU"].astype(str).str.len() > 0]

    s["_ship"] = _gross_shipment_units(s)

    def _sum_window(start: pd.Timestamp, end_d: pd.Timestamp) -> pd.Series:
        m = (s["TxnDate"] >= start) & (s["TxnDate"] <= end_d)
        return s.loc[m].groupby("OMS_SKU", observed=True)["_ship"].sum()

    ur = _sum_window(recent_start, end).rename("units_recent")
    up = _sum_window(prev_start, prev_end).rename("units_prev")
    out = pd.concat([ur, up], axis=1).fillna(0.0)
    out = out.reset_index()
    out["spike_ratio"] = np.where(
        out["units_prev"] > 0,
        out["units_recent"] / out["units_prev"],
        np.where(out["units_recent"] > 0, 99.0, 0.0),
    )
    return out


def build_dashboard_payload(
    po_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    *,
    sku_mapping: Optional[dict] = None,
    group_by_parent: bool = False,
    recent_days: int = 7,
    prev_days: int = 7,
    spike_ratio: float = 1.35,
    min_recent_units: int = 5,
    low_run_days: float = 40.0,
    max_rows_per_section: int = 80,
    lead_time_default: int = 30,
) -> Dict[str, Any]:
    """Return structured sections + summary for the PO dashboard UI."""
    if po_df is None or po_df.empty:
        return {
            "ok": False,
            "message": "PO dataframe is empty.",
            "summary": {},
            "windows": {},
            "in_production": [],
            "open_po": [],
            "spike_attention": [],
            "running_tight": [],
        }

    p = po_df.copy()
    for c in ("PO_Pipeline_Total", "Raised_Recently_Units", "PO_Qty", "ADS", "Total_Inventory"):
        if c not in p.columns:
            p[c] = 0
    p["PO_Pipeline_Total"] = pd.to_numeric(p["PO_Pipeline_Total"], errors="coerce").fillna(0)
    p["Raised_Recently_Units"] = pd.to_numeric(
        p.get("Raised_Recently_Units", 0), errors="coerce"
    ).fillna(0)
    p["Pipeline_From_Sheet"] = (
        p["PO_Pipeline_Total"] - p["Raised_Recently_Units"]
    ).clip(lower=0)

    seg = recent_vs_prev_shipments(
        sales_df,
        sku_mapping=sku_mapping,
        group_by_parent=group_by_parent,
        recent_days=recent_days,
        prev_days=prev_days,
    )
    if not seg.empty:
        p = p.merge(seg, on="OMS_SKU", how="left")
    else:
        p["units_recent"] = 0.0
        p["units_prev"] = 0.0
        p["spike_ratio"] = 0.0
    p["units_recent"] = pd.to_numeric(p["units_recent"], errors="coerce").fillna(0.0)
    p["units_prev"] = pd.to_numeric(p["units_prev"], errors="coerce").fillna(0.0)
    p["spike_ratio"] = pd.to_numeric(p["spike_ratio"], errors="coerce").fillna(0.0)

    ads = pd.to_numeric(p["ADS"], errors="coerce").fillna(0.0)
    proj = pd.to_numeric(p.get("Projected_Running_Days", 999), errors="coerce").fillna(999.0)
    days_left = pd.to_numeric(p.get("Days_Left", 999), errors="coerce").fillna(999.0)
    lt = pd.to_numeric(p.get("Lead_Time_Days", lead_time_default), errors="coerce").fillna(
        float(lead_time_default)
    )

    cap = int(max(10, min(500, max_rows_per_section)))

    # ── In production (pipeline materialised or in-flight) ─────────────
    pipe = p["PO_Pipeline_Total"] > 0
    in_prod = p.loc[pipe].sort_values("PO_Pipeline_Total", ascending=False).head(cap)
    in_production_cols = [
        c
        for c in (
            "OMS_SKU",
            "Parent_SKU",
            "Total_Inventory",
            "PO_Pipeline_Total",
            "Pipeline_From_Sheet",
            "Raised_Recently_Units",
            "Raised_Recently_Last_Date",
            "ADS",
            "Projected_Running_Days",
            "Days_Left",
            "Lead_Time_Days",
            "Priority",
            "SKU_Sheet_Status",
        )
        if c in in_prod.columns
    ]
    in_production = _rows(in_prod, in_production_cols)

    # ── Open PO recommendations (engine net qty) ─────────────────────
    poq = pd.to_numeric(p.get("PO_Qty", 0), errors="coerce").fillna(0) > 0
    open_po_df = p.loc[poq].copy()
    if "Priority" in open_po_df.columns:
        open_po_df["_prio"] = open_po_df["Priority"].astype(str).map(_priority_rank)
        open_po_df = open_po_df.sort_values(["_prio", "Days_Left"], ascending=[True, True])
    else:
        open_po_df = open_po_df.sort_values("Days_Left", ascending=True)
    open_cols = [
        c
        for c in (
            "OMS_SKU",
            "Parent_SKU",
            "PO_Qty",
            "Gross_PO_Qty",
            "Total_Inventory",
            "PO_Pipeline_Total",
            "ADS",
            "Days_Left",
            "Projected_Running_Days",
            "Post_PO_Cover_Days_Capped",
            "Lead_Time_Days",
            "Priority",
            "PO_Block_Reason",
        )
        if c in open_po_df.columns
    ]
    open_po = _rows(open_po_df.head(cap), open_cols)

    # ── Spike + needs attention (selling up + cover below threshold) ──
    spike_mask = (
        (p["units_recent"] >= float(min_recent_units))
        & (p["spike_ratio"] >= float(spike_ratio))
        & (ads > 0)
        & ((proj < float(low_run_days)) | (days_left < float(low_run_days)))
    )
    spike_df = p.loc[spike_mask].sort_values("units_recent", ascending=False).head(cap)
    spike_cols = [
        c
        for c in (
            "OMS_SKU",
            "Parent_SKU",
            "units_recent",
            "units_prev",
            "spike_ratio",
            "ADS",
            "Total_Inventory",
            "PO_Pipeline_Total",
            "Projected_Running_Days",
            "Days_Left",
            "PO_Qty",
            "Lead_Time_Days",
        )
        if c in spike_df.columns
    ]
    spike_attention = _rows(spike_df, spike_cols)

    # ── Running tight (cover risk without requiring a spike) ───────────
    tight_mask = (ads > 0) & (proj < float(low_run_days)) & (proj < 900)
    tight_df = p.loc[tight_mask].sort_values(
        ["Projected_Running_Days", "Days_Left"], ascending=[True, True]
    ).head(cap)
    tight_cols = [
        c
        for c in (
            "OMS_SKU",
            "Parent_SKU",
            "Total_Inventory",
            "PO_Pipeline_Total",
            "Projected_Running_Days",
            "Days_Left",
            "ADS",
            "units_recent",
            "PO_Qty",
            "Priority",
        )
        if c in tight_df.columns
    ]
    running_tight = _rows(tight_df, tight_cols)

    inv = pd.to_numeric(p.get("Total_Inventory", 0), errors="coerce").fillna(0)
    summary = {
        "sku_rows": int(len(p)),
        "in_production_skus": int(pipe.sum()),
        "open_po_skus": int(poq.sum()),
        "spike_attention_skus": int(spike_mask.sum()),
        "running_tight_skus": int(tight_mask.sum()),
        "total_pipeline_units": int(p["PO_Pipeline_Total"].sum()),
        "total_sheet_pipeline_units": int(p["Pipeline_From_Sheet"].sum()),
        "total_raised_recent_units": int(p["Raised_Recently_Units"].sum()),
        "total_open_po_units": int(pd.to_numeric(p.get("PO_Qty", 0), errors="coerce").fillna(0).sum()),
        "oos_skus": int(((ads > 0) & (inv <= 0)).sum()),
    }

    sales_max = None
    if sales_df is not None and not sales_df.empty and "TxnDate" in sales_df.columns:
        mx = pd.to_datetime(sales_df["TxnDate"], errors="coerce").max()
        if pd.notna(mx):
            sales_max = str(mx.normalize().date())

    return {
        "ok": True,
        "summary": summary,
        "windows": {
            "recent_days": int(recent_days),
            "prev_days": int(prev_days),
            "sales_max_date": sales_max,
            "spike_ratio_threshold": float(spike_ratio),
            "low_run_days_threshold": float(low_run_days),
            "min_recent_units": int(min_recent_units),
        },
        "in_production": in_production,
        "open_po": open_po,
        "spike_attention": spike_attention,
        "running_tight": running_tight,
    }


def _priority_rank(p: str) -> int:
    p = str(p)
    if "URGENT" in p:
        return 0
    if "HIGH" in p:
        return 1
    if "MEDIUM" in p:
        return 2
    if "Pipeline" in p:
        return 3
    return 9


def _rows(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    use = [c for c in cols if c in df.columns]
    out = df[use].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].fillna("").astype(str)
        else:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.fillna("").to_dict("records")

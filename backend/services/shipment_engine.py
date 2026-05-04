"""
Shipment engine.
Calculates marketplace-wise shipment recommendation from OMS to channel inventory.
"""
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd


_MKT_CONFIG: Dict[str, Dict[str, str]] = {
    "amazon": {
        "source": "Amazon",
        "inventory_col": "Amazon_Inventory",
        "inventory_fallback_cols": [],
        "in_transit_col": "FBA_InTransit",
    },
    "flipkart": {
        "source": "Flipkart",
        "inventory_col": "Flipkart_Inventory",
        "inventory_fallback_cols": [],
        "in_transit_col": "Flipkart_InTransit",
    },
    "myntra": {
        "source": "Myntra",
        "inventory_col": "Myntra_Other_Inventory",
        "inventory_fallback_cols": ["Myntra_Inventory"],
        "in_transit_col": "Myntra_InTransit",
    },
    "meesho": {
        "source": "Meesho",
        "inventory_col": "Meesho_Inventory",
        "inventory_fallback_cols": [],
        "in_transit_col": "Meesho_InTransit",
    },
}


def calculate_shipment_plan(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    marketplace: str,
    period_days: int = 30,
    lead_time: int = 7,
    target_days: int = 14,
    min_denominator: int = 7,
    safety_pct: float = 10.0,
    round_to: int = 5,
    demand_basis: str = "Sold",
    cap_to_oms_inventory: bool = True,
) -> pd.DataFrame:
    def _num_series(col_name: str) -> pd.Series:
        if col_name in out.columns:
            return pd.to_numeric(out[col_name], errors="coerce").fillna(0)
        return pd.Series(0, index=out.index, dtype=float)

    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    key = str(marketplace).strip().lower()
    cfg = _MKT_CONFIG.get(key)
    if not cfg:
        raise ValueError(f"Unsupported marketplace: {marketplace}")

    src = cfg["source"]
    inv_col = cfg["inventory_col"]
    inv_fallback_cols = cfg.get("inventory_fallback_cols", [])
    in_transit_col = cfg["in_transit_col"]

    if "TxnDate" not in sales_df.columns or "Sku" not in sales_df.columns:
        return pd.DataFrame()

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])
    if "Source" in df.columns:
        df = df[df["Source"].astype(str).str.strip().str.lower() == src.lower()]
    if df.empty:
        return pd.DataFrame()

    max_date = df["TxnDate"].max()
    cutoff = max_date - timedelta(days=period_days)
    recent = df[df["TxnDate"] >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame()

    sold = recent[recent["Transaction Type"] == "Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]
    summary = sold.merge(net, on="OMS_SKU", how="outer").fillna(0)

    if demand_basis == "Net":
        act = recent
    else:
        act = recent[recent["Transaction Type"].astype(str).str.strip().str.lower() == "shipment"].copy()
    if act.empty:
        sku_span = pd.DataFrame(columns=["OMS_SKU", "_eff_days_active"])
    else:
        sku_span = (
            act.groupby("Sku", as_index=False)
            .agg(ts_min=("TxnDate", "min"), ts_max=("TxnDate", "max"))
            .rename(columns={"Sku": "OMS_SKU"})
        )
        sku_span["_eff_days_active"] = (sku_span["ts_max"] - sku_span["ts_min"]).dt.days + 1

    out = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna({"Sold_Units": 0, "Net_Units": 0})
    out = out.merge(sku_span[["OMS_SKU", "_eff_days_active"]], on="OMS_SKU", how="left")
    out["Eff_Days"] = (
        pd.to_numeric(out["_eff_days_active"], errors="coerce")
        .fillna(float(period_days))
        .clip(lower=float(min_denominator), upper=float(period_days))
    )
    out.drop(columns=["_eff_days_active"], inplace=True, errors="ignore")

    demand_units = out["Net_Units"].clip(lower=0) if demand_basis == "Net" else out["Sold_Units"]
    out["ADS"] = (demand_units / out["Eff_Days"]).fillna(0).round(3)

    market_inv = _num_series(inv_col)
    for fb_col in inv_fallback_cols:
        fb_series = _num_series(fb_col)
        market_inv = np.where(market_inv > 0, market_inv, fb_series)
    out["Marketplace_Inventory"] = pd.Series(market_inv, index=out.index).fillna(0).astype(float)
    out["In_Transit"] = _num_series(in_transit_col)
    out["OMS_Inventory"] = _num_series("OMS_Inventory")

    coverage_days = max(int(lead_time) + int(target_days), 1)
    out["Required_Units"] = (out["ADS"] * coverage_days).round(2)
    out["Safety_Units"] = (out["Required_Units"] * (float(safety_pct) / 100.0)).round(2)
    out["Target_Stock"] = (out["Required_Units"] + out["Safety_Units"]).round(2)

    gross_need = (out["Target_Stock"] - out["Marketplace_Inventory"] - out["In_Transit"]).clip(lower=0)
    if cap_to_oms_inventory:
        gross_need = np.minimum(gross_need, out["OMS_Inventory"].clip(lower=0))

    step = max(int(round_to or 1), 1)
    out["Suggested_Shipment_Qty"] = (np.ceil(gross_need / step) * step).astype(int)
    out["Days_Cover_After_Shipment"] = np.where(
        out["ADS"] > 0,
        ((out["Marketplace_Inventory"] + out["In_Transit"] + out["Suggested_Shipment_Qty"]) / out["ADS"]).round(1),
        999.0,
    )

    out["Priority"] = np.select(
        [
            (out["ADS"] > 0) & (out["Marketplace_Inventory"] <= 0) & (out["Suggested_Shipment_Qty"] > 0),
            (out["Suggested_Shipment_Qty"] > 0),
        ],
        ["🔴 URGENT", "🟡 SHIP"],
        default="⚪ OK",
    )

    return out[
        [
            "Priority",
            "OMS_SKU",
            "Sold_Units",
            "Net_Units",
            "ADS",
            "OMS_Inventory",
            "Marketplace_Inventory",
            "In_Transit",
            "Required_Units",
            "Safety_Units",
            "Target_Stock",
            "Suggested_Shipment_Qty",
            "Days_Cover_After_Shipment",
        ]
    ]

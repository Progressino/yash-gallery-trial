"""
PO Engine — extracted 1-for-1 from app.py.
calculate_quarterly_history + calculate_po_base.
"""
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .helpers import map_to_oms_sku, get_parent_sku
from .myntra import myntra_to_sales_rows


def get_indian_fy_quarter(date: pd.Timestamp) -> tuple:
    m = date.month
    y = date.year
    if m >= 4:
        fy = y + 1
        q  = 1 if m <= 6 else 2 if m <= 9 else 3
    else:
        fy = y
        q  = 4
    return fy, q


_Q_LABELS = {1: "Apr–Jun", 2: "Jul–Sep", 3: "Oct–Dec", 4: "Jan–Mar"}


def quarter_col_name(fy: int, q: int) -> str:
    cal_year = fy - 1 if q in (1, 2, 3) else fy
    return f"{_Q_LABELS[q]} {cal_year}"


def _mtr_to_sales_df_local(mtr_df, sku_mapping, group_by_parent=False):
    if mtr_df.empty:
        return pd.DataFrame()
    m = mtr_df[["Date", "SKU", "Transaction_Type", "Quantity"]].copy()
    m = m.rename(columns={"Date": "TxnDate", "SKU": "Sku", "Transaction_Type": "Transaction Type"})
    m["TxnDate"]  = pd.to_datetime(m["TxnDate"], errors="coerce")
    m["Quantity"] = pd.to_numeric(m["Quantity"], errors="coerce").fillna(0)
    m = m.dropna(subset=["TxnDate"])
    m["Sku"] = m["Sku"].apply(lambda x: map_to_oms_sku(x, sku_mapping))
    if group_by_parent:
        m["Sku"] = m["Sku"].apply(get_parent_sku)
    m["Units_Effective"] = np.where(
        m["Transaction Type"] == "Refund", -m["Quantity"],
        np.where(m["Transaction Type"] == "Cancel", 0, m["Quantity"])
    )
    return m[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective"]]


def calculate_quarterly_history(
    sales_df: pd.DataFrame,
    mtr_df: Optional[pd.DataFrame] = None,
    myntra_df: Optional[pd.DataFrame] = None,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> pd.DataFrame:
    parts = []

    if not sales_df.empty and "Sku" in sales_df.columns:
        tmp = sales_df[["Sku", "TxnDate", "Quantity", "Transaction Type"]].copy()
        tmp.columns = ["SKU", "Date", "Qty", "TxnType"]
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["Qty"]  = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
        parts.append(tmp.dropna(subset=["Date"]))

    if mtr_df is not None and not mtr_df.empty:
        mtr_sku_col  = next((c for c in mtr_df.columns if c in ["SKU", "Sku", "OMS_SKU"]),  None)
        mtr_date_col = next((c for c in mtr_df.columns if c in ["Date", "TxnDate"]),         None)
        mtr_qty_col  = next((c for c in mtr_df.columns if c in ["Quantity", "Qty"]),          None)
        mtr_txn_col  = next((c for c in mtr_df.columns if c in ["Transaction_Type", "Transaction Type", "TxnType"]), None)
        if mtr_sku_col and mtr_date_col and mtr_qty_col:
            tmp = mtr_df[[mtr_sku_col, mtr_date_col, mtr_qty_col]].copy()
            tmp.columns = ["SKU", "Date", "Qty"]
            tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
            tmp["TxnType"] = mtr_df[mtr_txn_col].values if mtr_txn_col else "Shipment"
            if sku_mapping:
                tmp["SKU"] = tmp["SKU"].apply(lambda x: map_to_oms_sku(x, sku_mapping))
            parts.append(tmp.dropna(subset=["Date"]))

    if myntra_df is not None and not myntra_df.empty:
        myn_sku_col  = next((c for c in myntra_df.columns if c in ["OMS_SKU", "Sku", "SKU"]), None)
        myn_date_col = next((c for c in myntra_df.columns if c in ["Date", "TxnDate"]),        None)
        myn_qty_col  = next((c for c in myntra_df.columns if c in ["Quantity", "Qty"]),        None)
        myn_txn_col  = next((c for c in myntra_df.columns if c in ["TxnType", "Transaction Type"]), None)
        if myn_sku_col and myn_date_col and myn_qty_col:
            tmp = myntra_df[[myn_sku_col, myn_date_col, myn_qty_col]].copy()
            tmp.columns = ["SKU", "Date", "Qty"]
            tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
            tmp["TxnType"] = myntra_df[myn_txn_col].values if myn_txn_col else "Shipment"
            parts.append(tmp.dropna(subset=["Date"]))

    if not parts:
        return pd.DataFrame()

    hist = pd.concat(parts, ignore_index=True)
    hist = hist[hist["TxnType"].astype(str).str.strip() == "Shipment"]
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"])
    hist["Qty"] = pd.to_numeric(hist["Qty"], errors="coerce").fillna(0)
    hist = hist[hist["Qty"] > 0]
    if hist.empty:
        return pd.DataFrame()

    if group_by_parent:
        hist["SKU"] = hist["SKU"].apply(get_parent_sku)

    fy_q = hist["Date"].apply(get_indian_fy_quarter)
    hist["FY"] = fy_q.apply(lambda x: x[0])
    hist["QN"] = fy_q.apply(lambda x: x[1])

    today         = pd.Timestamp.today()
    cur_fy, cur_q = get_indian_fy_quarter(today)
    quarter_seq   = []
    fy_i, q_i     = cur_fy, cur_q
    for _ in range(n_quarters):
        quarter_seq.append((fy_i, q_i))
        q_i -= 1
        if q_i == 0:
            q_i = 4
            fy_i -= 1
    quarter_seq = list(reversed(quarter_seq))

    hist["col"] = hist.apply(lambda r: quarter_col_name(int(r["FY"]), int(r["QN"])), axis=1)
    grp   = hist.groupby(["SKU", "col"])["Qty"].sum().reset_index()
    pivot = grp.pivot_table(index="SKU", columns="col", values="Qty",
                            aggfunc="sum", fill_value=0).reset_index()
    pivot = pivot.rename(columns={"SKU": "OMS_SKU"})
    pivot.columns.name = None

    ordered_q_cols = []
    for fy_j, q_j in quarter_seq:
        col = quarter_col_name(fy_j, q_j)
        ordered_q_cols.append(col)
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["OMS_SKU"] + ordered_q_cols]

    last4 = ordered_q_cols[-4:]
    pivot["Avg_Monthly"] = (pivot[last4].mean(axis=1) / 3).round(1)

    cutoff_90 = today - timedelta(days=90)
    r90 = hist[hist["Date"] >= cutoff_90].groupby("SKU")["Qty"].sum().reset_index()
    r90.columns = ["OMS_SKU", "Units_90d"]
    pivot = pivot.merge(r90, on="OMS_SKU", how="left").fillna({"Units_90d": 0})
    pivot["ADS"] = (pivot["Units_90d"] / 90).round(3)

    cutoff_30 = today - timedelta(days=30)
    r30 = hist[hist["Date"] >= cutoff_30].groupby("SKU")["Qty"].sum().reset_index()
    r30.columns = ["OMS_SKU", "Units_30d"]
    pivot = pivot.merge(r30, on="OMS_SKU", how="left").fillna({"Units_30d": 0})

    f30 = (
        hist[hist["Date"] >= cutoff_30]
        .assign(_day=lambda d: d["Date"].dt.normalize())
        .groupby("SKU")["_day"].nunique()
        .reset_index()
    )
    f30.columns = ["OMS_SKU", "Freq_30d"]
    pivot = pivot.merge(f30, on="OMS_SKU", how="left").fillna({"Freq_30d": 0})
    pivot["Freq_30d"] = pivot["Freq_30d"].astype(int)

    def _status(ads):
        if ads >= 1.0:  return "Fast Moving"
        if ads >= 0.33: return "Moderate"
        if ads >= 0.10: return "Slow Selling"
        return "Not Moving"

    pivot["Status"]    = pivot["ADS"].apply(_status)
    pivot["Units_90d"] = pivot["Units_90d"].astype(int)
    pivot["Units_30d"] = pivot["Units_30d"].astype(int)
    return pivot


def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    lead_time: int,
    target_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7,
    grace_days: int = 7,
    safety_pct: float = 20.0,
    use_seasonality: bool = False,
    seasonal_weight: float = 0.5,
    mtr_df: Optional[pd.DataFrame] = None,
    myntra_df: Optional[pd.DataFrame] = None,
    sku_mapping: Optional[Dict[str, str]] = None,
    group_by_parent: bool = False,
    existing_po_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])

    max_date = df["TxnDate"].max()
    cutoff   = max_date - timedelta(days=period_days)
    recent   = df[df["TxnDate"] >= cutoff].copy()

    sold = recent[recent["Transaction Type"] == "Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]
    returns = recent[recent["Transaction Type"] == "Refund"].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU", "Return_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)
    po_df   = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna(
        {"Sold_Units": 0, "Return_Units": 0, "Net_Units": 0}
    )

    denom        = max(period_days, min_denominator)
    demand_units = po_df["Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["Sold_Units"]
    po_df["Recent_ADS"] = (demand_units / denom).fillna(0)

    if use_seasonality and sku_mapping:
        hist_parts = [df]

        if mtr_df is not None and not mtr_df.empty:
            mtr_sales = _mtr_to_sales_df_local(mtr_df, sku_mapping, group_by_parent)
            if not mtr_sales.empty:
                hist_parts.append(mtr_sales)

        if myntra_df is not None and not myntra_df.empty:
            myn_sales = myntra_to_sales_rows(myntra_df)
            if not myn_sales.empty:
                myn_sales["TxnDate"] = pd.to_datetime(myn_sales["TxnDate"], errors="coerce")
                if group_by_parent:
                    myn_sales["Sku"] = myn_sales["Sku"].apply(get_parent_sku)
                hist_parts.append(myn_sales)

        if len(hist_parts) > 1:
            hist_df = pd.concat(hist_parts, ignore_index=True)
            hist_df = hist_df.drop_duplicates(
                subset=["Sku", "TxnDate", "Transaction Type"], keep="last"
            )
        else:
            hist_df = df

        ly_trailing_end   = max_date - timedelta(days=365)
        ly_trailing_start = ly_trailing_end - timedelta(days=period_days)
        ly_fwd_start      = (max_date + timedelta(days=lead_time)) - timedelta(days=365)
        ly_fwd_end        = (max_date + timedelta(days=lead_time + max(target_days, period_days))) - timedelta(days=365)

        ly_sales_trailing = hist_df[
            (hist_df["TxnDate"] >= ly_trailing_start) & (hist_df["TxnDate"] < ly_trailing_end)
        ].copy()
        ly_sales_fwd = hist_df[
            (hist_df["TxnDate"] >= ly_fwd_start) & (hist_df["TxnDate"] < ly_fwd_end)
        ].copy()

        if not ly_sales_trailing.empty:
            ly_sales      = ly_sales_trailing
            ly_days_count = max((ly_trailing_end - ly_trailing_start).days, min_denominator)
        elif not ly_sales_fwd.empty:
            ly_sales      = ly_sales_fwd
            ly_days_count = max((ly_fwd_end - ly_fwd_start).days, min_denominator)
        else:
            ly_broad_start = max_date - timedelta(days=730)
            ly_broad_end   = max_date - timedelta(days=365)
            ly_sales       = hist_df[
                (hist_df["TxnDate"] >= ly_broad_start) & (hist_df["TxnDate"] < ly_broad_end)
            ].copy()
            ly_days_count = max((ly_broad_end - ly_broad_start).days, min_denominator)

        if not ly_sales.empty:
            ly_sold = (ly_sales[ly_sales["Transaction Type"] == "Shipment"]
                       .groupby("Sku")["Quantity"].sum().reset_index())
            ly_sold.columns = ["OMS_SKU", "LY_Sold_Units"]
            ly_net = ly_sales.groupby("Sku")["Units_Effective"].sum().reset_index()
            ly_net.columns = ["OMS_SKU", "LY_Net_Units"]
            ly_summary = ly_sold.merge(ly_net, on="OMS_SKU", how="outer").fillna(0)
            po_df = pd.merge(po_df, ly_summary, on="OMS_SKU", how="left").fillna(0)
            ly_demand     = po_df["LY_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["LY_Sold_Units"]
            po_df["LY_ADS"] = (ly_demand / ly_days_count).round(3)
            po_df["ADS"] = np.where(
                po_df["LY_ADS"] > 0,
                (po_df["Recent_ADS"] * (1 - seasonal_weight)) + (po_df["LY_ADS"] * seasonal_weight),
                po_df["Recent_ADS"],
            )
        else:
            po_df["ADS"]    = po_df["Recent_ADS"]
            po_df["LY_ADS"] = 0.0
    else:
        po_df["ADS"]    = po_df["Recent_ADS"]
        po_df["LY_ADS"] = 0

    inv_col = "Total_Inventory" if "Total_Inventory" in po_df.columns else po_df.columns[1]

    # Days of stock remaining
    po_df["Days_Left"] = np.where(
        po_df["ADS"] > 0,
        (po_df[inv_col] / po_df["ADS"]).round(1),
        999.0,
    )

    # Safety-stock-aware PO calculation, rounded up to nearest 5
    lead_demand  = po_df["ADS"] * lead_time
    target_stock = po_df["ADS"] * target_days
    base_req     = lead_demand + target_stock
    safety       = base_req * (safety_pct / 100.0)
    total_req    = base_req + safety
    gross_po     = (total_req - po_df[inv_col]).clip(lower=0)
    po_df["Gross_PO_Qty"] = (np.ceil(gross_po / 5) * 5).astype(int)

    # Pipeline deduction from existing PO sheet
    if existing_po_df is not None and not existing_po_df.empty and "PO_Pipeline_Total" in existing_po_df.columns:
        po_df = pd.merge(
            po_df,
            existing_po_df[["OMS_SKU", "PO_Pipeline_Total"]],
            on="OMS_SKU", how="left",
        )
        po_df["PO_Pipeline_Total"] = pd.to_numeric(
            po_df["PO_Pipeline_Total"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        po_df["PO_Pipeline_Total"] = 0

    net_po = (po_df["Gross_PO_Qty"] - po_df["PO_Pipeline_Total"]).clip(lower=0)
    po_df["PO_Qty"] = (np.ceil(net_po / 5) * 5).astype(int)

    # Stockout flag
    po_df["Stockout_Flag"] = np.where(
        (po_df["ADS"] > 0) & (po_df[inv_col] <= 0), "OOS", ""
    )

    # Priority classification (vectorised)
    conditions = [
        (po_df["Days_Left"] < lead_time) & (po_df["PO_Qty"] > 0),
        (po_df["Days_Left"] < (lead_time + grace_days)) & (po_df["PO_Qty"] > 0),
        po_df["PO_Qty"] > 0,
        po_df["PO_Pipeline_Total"] > 0,
    ]
    choices = ["🔴 URGENT", "🟡 HIGH", "🟢 MEDIUM", "🔄 In Pipeline"]
    po_df["Priority"] = np.select(conditions, choices, default="⚪ OK")

    # Parent SKU — strip marketplace suffix + size/colour suffix
    po_df["Parent_SKU"] = po_df["OMS_SKU"].apply(get_parent_sku)

    return po_df

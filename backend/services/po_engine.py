"""
PO Engine — extracted 1-for-1 from app.py.
calculate_quarterly_history + calculate_po_base.
"""
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

import re

from .helpers import map_to_oms_sku, get_parent_sku
from .myntra import myntra_to_sales_rows

# Strip "PL" infix in Amazon seller SKUs (e.g. 1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL)
# Must match the same pattern used in inventory.py _resolve_amz_sku.
_PL_RE = re.compile(r'^(\d+)PL(YK)', re.I)


def _strip_pl(sku: str, mapping: Dict[str, str]) -> str:
    """Map an Amazon seller SKU to OMS SKU, stripping PL infix if needed."""
    raw = str(sku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)
    return mapping.get(stripped, mapping.get(raw, stripped))


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
    m["Sku"] = m["Sku"].apply(lambda x: _strip_pl(x, sku_mapping))
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
                # Use _strip_pl so "1001PLYKBEIGE-5XL" resolves to "1001YKBEIGE-5XL"
                # matching the same stripping done in inventory._resolve_amz_sku.
                tmp["SKU"] = tmp["SKU"].apply(lambda x: _strip_pl(x, sku_mapping))
            else:
                tmp["SKU"] = tmp["SKU"].apply(lambda x: _PL_RE.sub(r"\1\2", str(x).strip().upper()))
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

    # Normalize SKUs: strip PL infix from all sources (fixes stale sales_df with old PL SKUs)
    hist["SKU"] = hist["SKU"].apply(
        lambda x: _PL_RE.sub(r"\1\2", str(x).strip().upper()) if isinstance(x, str) else str(x)
    )

    if group_by_parent:
        hist["SKU"] = hist["SKU"].apply(get_parent_sku)

    # Vectorized FY/Quarter computation (avoids slow row-by-row .apply)
    _month = hist["Date"].dt.month
    _year  = hist["Date"].dt.year
    hist["FY"] = np.where(_month >= 4, _year + 1, _year)
    hist["QN"] = np.select(
        [(_month >= 4) & (_month <= 6),
         (_month >= 7) & (_month <= 9),
         _month >= 10],
        [1, 2, 3],
        default=4,
    )

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

    # Build quarter label via a lookup map (avoids row-by-row apply)
    _unique_fq = hist[["FY", "QN"]].drop_duplicates()
    _q_label_map = {
        (int(r.FY), int(r.QN)): quarter_col_name(int(r.FY), int(r.QN))
        for r in _unique_fq.itertuples(index=False)
    }
    hist["col"] = [_q_label_map[(int(fy), int(qn))]
                   for fy, qn in zip(hist["FY"], hist["QN"])]
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

    # ADS uses a fixed 30-day window regardless of period_days.
    # This reflects current sales velocity more accurately than a longer window.
    ADS_WINDOW = 30
    ads_cutoff = max_date - timedelta(days=ADS_WINDOW)
    ads_recent = df[df["TxnDate"] >= ads_cutoff].copy()

    ads_sold = (
        ads_recent[ads_recent["Transaction Type"] == "Shipment"]
        .groupby("Sku")["Quantity"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Quantity": "ADS_Sold_Units"})
    )
    ads_net = (
        ads_recent.groupby("Sku")["Units_Effective"].sum().reset_index()
        .rename(columns={"Sku": "OMS_SKU", "Units_Effective": "ADS_Net_Units"})
    )
    # Per-SKU effective days: from first sale within 30-day window to max_date.
    # Prevents diluting ADS for recently-launched SKUs.
    ads_first = (
        ads_recent.groupby("Sku")["TxnDate"].min()
        .reset_index()
        .rename(columns={"Sku": "OMS_SKU", "TxnDate": "ADS_First_Sale_Date"})
    )

    po_df = po_df.merge(ads_sold, on="OMS_SKU", how="left")
    po_df = po_df.merge(ads_net,  on="OMS_SKU", how="left")
    po_df = po_df.merge(ads_first, on="OMS_SKU", how="left")
    po_df[["ADS_Sold_Units", "ADS_Net_Units"]] = po_df[["ADS_Sold_Units", "ADS_Net_Units"]].fillna(0)

    po_df["Eff_Days"] = (
        (max_date - po_df["ADS_First_Sale_Date"]).dt.days
        .fillna(ADS_WINDOW)                           # no sales in 30d → ADS stays 0
        .clip(lower=min_denominator, upper=ADS_WINDOW)
        .astype(float)
    )
    ads_demand = po_df["ADS_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["ADS_Sold_Units"]
    po_df["Recent_ADS"] = (ads_demand / po_df["Eff_Days"]).fillna(0)

    if use_seasonality and sku_mapping:
        # sales_df is already the unified dataset across platforms.
        # Re-appending mtr/myntra and de-duplicating by (Sku, TxnDate, Transaction Type)
        # can collapse legitimate same-day multi-order sales and undercount YoY demand.
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

    # PO calculation — total coverage = lead_time + target_days
    # Rounded up to nearest 10 to match team's manual formula
    lead_demand  = po_df["ADS"] * lead_time
    target_stock = po_df["ADS"] * (target_days + grace_days)
    base_req     = lead_demand + target_stock
    safety       = base_req * (safety_pct / 100.0)
    total_req    = base_req + safety
    gross_po     = (total_req - po_df[inv_col]).clip(lower=0)
    po_df["Gross_PO_Qty"] = (np.ceil(gross_po / 10) * 10).astype(int)

    # Pipeline deduction from existing PO sheet
    if existing_po_df is not None and not existing_po_df.empty and "PO_Pipeline_Total" in existing_po_df.columns:
        # Pull PO_Pipeline_Total + any breakdown columns present in the uploaded sheet
        _breakdown_cols = [c for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch"]
                           if c in existing_po_df.columns]
        _merge_cols = ["OMS_SKU", "PO_Pipeline_Total"] + _breakdown_cols
        po_df = pd.merge(
            po_df,
            existing_po_df[_merge_cols],
            on="OMS_SKU", how="left",
        )
        po_df["PO_Pipeline_Total"] = pd.to_numeric(
            po_df["PO_Pipeline_Total"], errors="coerce"
        ).fillna(0).astype(int)
        for _bc in _breakdown_cols:
            po_df[_bc] = pd.to_numeric(po_df[_bc], errors="coerce").fillna(0).astype(int)
    else:
        po_df["PO_Pipeline_Total"] = 0

    # Days of stock remaining — includes pipeline so "Running Days" matches
    # the team's manual formula: (inventory + pipeline) / ADS
    po_df["Days_Left"] = np.where(
        po_df["ADS"] > 0,
        ((po_df[inv_col] + po_df["PO_Pipeline_Total"]) / po_df["ADS"]).round(1),
        999.0,
    )

    # ── Inject PO-sheet SKUs missing from inventory ──────────────
    # If a SKU has an active pipeline order but isn't in the inventory file
    # (e.g. out of stock, removed from listing), add it as a ghost row so it
    # still shows up as "🔄 In Pipeline" and isn't invisible to the user.
    if existing_po_df is not None and not existing_po_df.empty and "PO_Pipeline_Total" in existing_po_df.columns:
        missing_mask = (
            ~existing_po_df["OMS_SKU"].isin(po_df["OMS_SKU"])
            & (existing_po_df["PO_Pipeline_Total"] > 0)
        )
        missing_po = existing_po_df[missing_mask].copy()
        if not missing_po.empty:
            ghost = pd.DataFrame({"OMS_SKU": missing_po["OMS_SKU"].values})
            ghost["Total_Inventory"] = 0
            ghost["Sold_Units"]      = 0
            ghost["Return_Units"]    = 0
            ghost["Net_Units"]       = 0
            ghost["Recent_ADS"]      = 0.0
            ghost["ADS"]             = 0.0
            ghost["LY_ADS"]          = 0.0
            ghost["Days_Left"]       = 999.0
            ghost["Gross_PO_Qty"]    = 0
            for c in ["PO_Pipeline_Total"] + _breakdown_cols:
                ghost[c] = missing_po[c].values if c in missing_po.columns else 0
            # Fill any other columns po_df already has
            for c in po_df.columns:
                if c not in ghost.columns:
                    ghost[c] = 0
            po_df = pd.concat([po_df, ghost[po_df.columns]], ignore_index=True)

    net_po = (po_df["Gross_PO_Qty"] - po_df["PO_Pipeline_Total"]).clip(lower=0)
    po_df["PO_Qty"] = (np.ceil(net_po / 10) * 10).astype(int)

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

    # Cutting ratio for the Cutting Planner:
    # Prefer each size's share of net PO qty (actionable demand),
    # and fall back to ADS share when the parent has no PO demand.
    parent_po_sum = po_df.groupby("Parent_SKU")["PO_Qty"].transform("sum")
    parent_ads_sum = po_df.groupby("Parent_SKU")["ADS"].transform("sum")
    po_df["Cutting_Ratio"] = np.where(
        parent_po_sum > 0,
        (po_df["PO_Qty"] / parent_po_sum),
        np.where(
            parent_ads_sum > 0,
            (po_df["ADS"] / parent_ads_sum),
            0.0,
        ),
    ).round(4)

    # Projected Running Days = (current stock + full pipeline) / ADS
    # Shows how many days stock will last once all pipeline orders arrive
    total_supply = po_df[inv_col] + po_df["PO_Pipeline_Total"] + po_df["Gross_PO_Qty"]
    po_df["Projected_Running_Days"] = np.where(
        po_df["ADS"] > 0,
        (total_supply / po_df["ADS"]).round(1),
        999.0,
    )

    # Drop intermediate calc columns (datetime/float cols that break router serialisation)
    po_df.drop(
        columns=["ADS_First_Sale_Date", "ADS_Sold_Units", "ADS_Net_Units"],
        errors="ignore",
        inplace=True,
    )

    return po_df

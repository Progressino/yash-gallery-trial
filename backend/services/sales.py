"""
Sales aggregation — build + dedup logic extracted from app.py.
"""
import gc
from typing import Dict, List

import numpy as np
import pandas as pd

from .helpers import _downcast_sales, map_to_oms_sku
from .myntra import myntra_to_sales_rows
from .meesho import meesho_to_sales_rows
from .flipkart import flipkart_to_sales_rows


def _mtr_to_sales_df(
    mtr_df: pd.DataFrame,
    sku_mapping: Dict[str, str],
    group_by_parent: bool = False,
) -> pd.DataFrame:
    """Convert MTR DataFrame to sales rows format."""
    from .helpers import get_parent_sku

    if mtr_df.empty:
        return pd.DataFrame()

    m = mtr_df[["Date", "SKU", "Transaction_Type", "Quantity"]].copy()
    m = m.rename(columns={
        "Date":             "TxnDate",
        "SKU":              "Sku",
        "Transaction_Type": "Transaction Type",
    })
    m["TxnDate"]  = pd.to_datetime(m["TxnDate"], errors="coerce")
    m["Quantity"] = pd.to_numeric(m["Quantity"], errors="coerce").fillna(0)
    m = m.dropna(subset=["TxnDate"])
    m["Sku"] = m["Sku"].apply(lambda x: map_to_oms_sku(x, sku_mapping))

    if group_by_parent:
        m["Sku"] = m["Sku"].apply(get_parent_sku)

    m["Units_Effective"] = np.where(
        m["Transaction Type"] == "Refund",  -m["Quantity"],
        np.where(m["Transaction Type"] == "Cancel", 0, m["Quantity"])
    )
    return m[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective"]]


def build_sales_df(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    sku_mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    Concatenate all platform DataFrames into a unified sales_df and deduplicate.
    Mirrors the 'Load All Data' block in app.py.
    """
    sales_parts: List[pd.DataFrame] = []

    if not meesho_df.empty:
        sales_parts.append(_downcast_sales(meesho_to_sales_rows(meesho_df)))
    if not myntra_df.empty:
        sales_parts.append(_downcast_sales(myntra_to_sales_rows(myntra_df)))
    if not flipkart_df.empty:
        sales_parts.append(_downcast_sales(flipkart_to_sales_rows(flipkart_df)))
    if not mtr_df.empty and sku_mapping:
        _mtr_sales = _mtr_to_sales_df(mtr_df, sku_mapping)
        if not _mtr_sales.empty:
            _mtr_sales["Source"]  = "Amazon"
            _mtr_sales["OrderId"] = np.nan
            sales_parts.append(_downcast_sales(_mtr_sales))
        del _mtr_sales
        gc.collect()

    if not sales_parts:
        return pd.DataFrame()

    combined_sales = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)
    del sales_parts
    gc.collect()

    # Deduplicate: rows with valid OrderId by (OrderId, Source, Transaction Type)
    _oid_str   = combined_sales["OrderId"].astype(str).str.strip()
    _oid_valid = combined_sales["OrderId"].notna() & ~_oid_str.str.lower().isin(["", "nan", "none"])
    del _oid_str

    _with_oid    = combined_sales[_oid_valid].drop_duplicates(
        subset=["OrderId", "Source", "Transaction Type"], keep="last"
    )
    _without_oid = combined_sales[~_oid_valid].drop_duplicates(
        subset=["Sku", "TxnDate", "Source", "Transaction Type"], keep="last"
    )
    del combined_sales, _oid_valid
    gc.collect()

    result = pd.concat([_with_oid, _without_oid], ignore_index=True)
    del _with_oid, _without_oid
    gc.collect()

    return _downcast_sales(result)


def get_sales_summary(sales_df: pd.DataFrame, months: int = 3) -> dict:
    """Return KPI summary for the dashboard."""
    if sales_df.empty:
        return {"total_units": 0, "total_returns": 0, "net_units": 0, "return_rate": 0.0}

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])

    if months > 0:
        cutoff = df["TxnDate"].max() - pd.DateOffset(months=months)
        df = df[df["TxnDate"] >= cutoff]

    shipped  = df[df["Transaction Type"] == "Shipment"]["Quantity"].sum()
    returned = df[df["Transaction Type"] == "Refund"]["Quantity"].sum()
    net      = df["Units_Effective"].sum()
    rate     = float(returned / shipped * 100) if shipped > 0 else 0.0

    return {
        "total_units":  int(shipped),
        "total_returns": int(returned),
        "net_units":    int(net),
        "return_rate":  round(rate, 1),
    }


def get_sales_by_source(sales_df: pd.DataFrame) -> List[dict]:
    """Returns pie chart data: [{source, units}]."""
    if sales_df.empty:
        return []
    df = sales_df[sales_df["Transaction Type"] == "Shipment"].copy()
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str)
    grp = df.groupby("Source")["Quantity"].sum().reset_index()
    grp.columns = ["source", "units"]
    return grp.sort_values("units", ascending=False).to_dict("records")


def get_top_skus(sales_df: pd.DataFrame, limit: int = 20) -> List[dict]:
    """Returns top SKUs by net units sold."""
    if sales_df.empty or "Sku" not in sales_df.columns:
        return []
    df = sales_df[sales_df["Transaction Type"] == "Shipment"].copy()
    # Filter out aggregate/total rows (e.g. MEESHO_TOTAL, TOTAL, etc.)
    _sku_lower = df["Sku"].astype(str).str.lower()
    df = df[~(_sku_lower.str.contains("_total") | _sku_lower.str.endswith("total") | _sku_lower.str.startswith("total"))]
    grp = df.groupby("Sku")["Quantity"].sum().reset_index()
    grp.columns = ["sku", "units"]
    return grp.sort_values("units", ascending=False).head(limit).to_dict("records")


# ── AI Dashboard helpers ───────────────────────────────────────────────────────

def _compute_platform_metrics(
    df: pd.DataFrame,
    platform_name: str,
    sku_col: str,
    txn_col: str,
    ship_val: str = "Shipment",
    refund_val: str = "Refund",
) -> dict:
    """Shared computation for a single platform DataFrame."""
    stub = {
        "platform": platform_name, "loaded": False,
        "total_units": 0, "total_returns": 0, "return_rate": 0.0,
        "top_sku": "", "trend_direction": "flat",
        "monthly": [], "by_state": [],
    }
    if df.empty:
        return stub

    try:
        d = df.copy()
        d["_Date"] = pd.to_datetime(d.get("Date", d.get("_Date")), errors="coerce")
        d = d.dropna(subset=["_Date"])
        if d.empty:
            return stub

        shipped_mask  = d[txn_col].astype(str).str.strip() == ship_val
        refund_mask   = d[txn_col].astype(str).str.strip() == refund_val
        qty           = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)

        total_units   = int(qty[shipped_mask].sum())
        total_returns = int(qty[refund_mask].sum())
        return_rate   = round(total_returns / total_units * 100, 1) if total_units > 0 else 0.0

        # Top SKU
        top_grp = d[shipped_mask].copy()
        top_grp["_qty"] = qty[shipped_mask]
        if sku_col in top_grp.columns and not top_grp.empty:
            top_sku = top_grp.groupby(sku_col)["_qty"].sum().idxmax()
        else:
            top_sku = ""

        # Monthly (last 6 months)
        d["_Month"] = d["_Date"].dt.to_period("M").astype(str)
        monthly_grp = (
            d.groupby(["_Month", txn_col])["Quantity"]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum())
            .reset_index()
            .pivot_table(index="_Month", columns=txn_col, values="Quantity", fill_value=0)
            .reset_index()
        )
        monthly_grp.columns.name = None
        monthly_grp = monthly_grp.rename(columns={
            ship_val:  "shipments",
            refund_val: "refunds",
        })
        if "shipments" not in monthly_grp.columns:
            monthly_grp["shipments"] = 0
        if "refunds" not in monthly_grp.columns:
            monthly_grp["refunds"] = 0
        monthly_grp = monthly_grp.sort_values("_Month").tail(6)
        monthly_grp = monthly_grp.rename(columns={"_Month": "month"})
        keep_cols = [c for c in ["month", "shipments", "refunds"] if c in monthly_grp.columns]
        monthly = monthly_grp[keep_cols].to_dict("records")

        # Trend direction (last month vs 3 months ago)
        trend_direction = "flat"
        ships = monthly_grp["shipments"].tolist() if "shipments" in monthly_grp.columns else []
        if len(ships) >= 3:
            last, three_ago = ships[-1], ships[-3]
            if three_ago > 0:
                change = (last - three_ago) / three_ago
                if change > 0.10:
                    trend_direction = "up"
                elif change < -0.10:
                    trend_direction = "down"

        # By state
        by_state = []
        if "State" in d.columns:
            st = (
                d[shipped_mask].copy()
                .assign(_qty=qty[shipped_mask].values)
                .groupby("State")["_qty"].sum()
                .reset_index()
                .rename(columns={"State": "state", "_qty": "units"})
                .sort_values("units", ascending=False)
            )
            st["units"] = st["units"].astype(int)
            by_state = st.to_dict("records")

        return {
            "platform": platform_name, "loaded": True,
            "total_units": total_units, "total_returns": total_returns,
            "return_rate": return_rate, "top_sku": str(top_sku),
            "trend_direction": trend_direction,
            "monthly": monthly, "by_state": by_state,
        }
    except Exception:
        return stub


def get_platform_summary(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
) -> List[dict]:
    """Returns 4 platform summary dicts (always 4, even for unloaded platforms)."""
    # MTR has Date col already; add it as _Date alias
    mtr = mtr_df.copy() if not mtr_df.empty else mtr_df
    if not mtr.empty and "Date" in mtr.columns:
        mtr["_Date"] = mtr["Date"]

    results = [
        _compute_platform_metrics(mtr,        "Amazon",   "SKU",     "Transaction_Type"),
        _compute_platform_metrics(myntra_df,   "Myntra",   "OMS_SKU", "TxnType"),
        _compute_platform_metrics(meesho_df,   "Meesho",   "OMS_SKU", "TxnType"),
        _compute_platform_metrics(flipkart_df, "Flipkart", "OMS_SKU", "TxnType"),
    ]
    return results


def get_anomalies(
    mtr_df: pd.DataFrame,
    myntra_df: pd.DataFrame,
    meesho_df: pd.DataFrame,
    flipkart_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> List[dict]:
    """Runs 5 anomaly rules. Returns list sorted critical → warning → info."""
    SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
    alerts: List[dict] = []

    platform_dfs = [
        ("Amazon",   mtr_df,        "Transaction_Type", "Shipment", "Refund", "Date"),
        ("Myntra",   myntra_df,      "TxnType",          "Shipment", "Refund", "Date"),
        ("Meesho",   meesho_df,      "TxnType",          "Shipment", "Refund", "Date"),
        ("Flipkart", flipkart_df,    "TxnType",          "Shipment", "Refund", "Date"),
    ]

    for name, df, txn_col, ship_val, refund_val, date_col in platform_dfs:
        if df.empty:
            continue

        try:
            d = df.copy()
            qty = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
            shipped  = qty[d[txn_col].astype(str).str.strip() == ship_val].sum()
            returned = qty[d[txn_col].astype(str).str.strip() == refund_val].sum()

            # Rule 1: Return spike
            rate = returned / shipped * 100 if shipped > 0 else 0
            if rate > 30:
                alerts.append({
                    "type": "return_spike", "severity": "warning",
                    "platform": name,
                    "message": f"{name} return rate is {rate:.1f}% (threshold: 30%) — investigate product quality or sizing",
                    "sku": None,
                })
        except Exception:
            pass

        try:
            # Rule 2: Zero sales in last 30 days
            d2 = df.copy()
            d2["_date"] = pd.to_datetime(d2[date_col], errors="coerce")
            d2 = d2.dropna(subset=["_date"])
            if not d2.empty:
                max_date = d2["_date"].max()
                cutoff   = max_date - pd.Timedelta(days=30)
                recent   = d2[d2["_date"] >= cutoff]
                recent_shipped = recent[recent[txn_col].astype(str).str.strip() == ship_val]
                if len(recent_shipped) == 0:
                    alerts.append({
                        "type": "zero_sales", "severity": "warning",
                        "platform": name,
                        "message": f"{name} has no shipments in the last 30 days — check listing status",
                        "sku": None,
                    })
        except Exception:
            pass

        try:
            # Rule 4: Sales drop >50% month-over-month
            d3 = df.copy()
            d3["_date"] = pd.to_datetime(d3[date_col], errors="coerce")
            d3 = d3.dropna(subset=["_date"])
            d3["_month"] = d3["_date"].dt.to_period("M")
            qty3 = pd.to_numeric(d3["Quantity"], errors="coerce").fillna(0)
            monthly_shipped = (
                d3[d3[txn_col].astype(str).str.strip() == ship_val]
                .assign(_qty=qty3[d3[txn_col].astype(str).str.strip() == ship_val].values)
                .groupby("_month")["_qty"].sum()
                .sort_index()
            )
            if len(monthly_shipped) >= 2:
                last_m  = monthly_shipped.iloc[-1]
                prev_m  = monthly_shipped.iloc[-2]
                if prev_m > 0 and last_m < prev_m * 0.5:
                    pct = int((1 - last_m / prev_m) * 100)
                    alerts.append({
                        "type": "sales_drop", "severity": "warning",
                        "platform": name,
                        "message": (
                            f"{name} sales dropped {pct}% month-over-month "
                            f"({monthly_shipped.index[-2]}: {int(prev_m):,} → "
                            f"{monthly_shipped.index[-1]}: {int(last_m):,} units)"
                        ),
                        "sku": None,
                    })
        except Exception:
            pass

    # Rule 3: Stockout (cap at 5)
    try:
        if not inventory_df.empty and not sales_df.empty:
            inv = inventory_df.copy()
            inv_col = "Total_Inventory" if "Total_Inventory" in inv.columns else inv.columns[1]
            inv["_inv"] = pd.to_numeric(inv[inv_col], errors="coerce").fillna(0)
            zero_inv = inv[inv["_inv"] <= 0]["OMS_SKU"].tolist()

            if zero_inv and "Sku" in sales_df.columns:
                s = sales_df.copy()
                s["TxnDate"] = pd.to_datetime(s["TxnDate"], errors="coerce")
                max_d    = s["TxnDate"].max()
                cutoff90 = max_d - pd.Timedelta(days=90)
                recent_s = s[(s["TxnDate"] >= cutoff90) & (s["Transaction Type"] == "Shipment")]
                ads_map  = (
                    recent_s.groupby("Sku")["Quantity"]
                    .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum() / 90)
                    .to_dict()
                )
                stockouts = [
                    sku for sku in zero_inv
                    if ads_map.get(sku, 0) > 0
                ][:5]
                for sku in stockouts:
                    ads = ads_map[sku]
                    alerts.append({
                        "type": "stockout", "severity": "critical",
                        "platform": "All",
                        "message": (
                            f"SKU {sku} has 0 inventory but avg daily sales of "
                            f"{ads:.1f} units — restock urgently"
                        ),
                        "sku": sku,
                    })
    except Exception:
        pass

    # Rule 5: Single platform loaded
    try:
        loaded_count = sum(1 for _, df, *_ in platform_dfs if not df.empty)
        if loaded_count == 1:
            loaded_name = next(name for name, df, *_ in platform_dfs if not df.empty)
            others = [n for n, df, *_ in platform_dfs if df.empty]
            alerts.append({
                "type": "single_platform", "severity": "info",
                "platform": "All",
                "message": (
                    f"Only {loaded_name} is loaded. Upload "
                    f"{', '.join(others)} data for cross-platform insights"
                ),
                "sku": None,
            })
    except Exception:
        pass

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts

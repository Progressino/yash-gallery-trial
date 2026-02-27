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
    grp = df.groupby("Sku")["Quantity"].sum().reset_index()
    grp.columns = ["sku", "units"]
    return grp.sort_values("units", ascending=False).head(limit).to_dict("records")

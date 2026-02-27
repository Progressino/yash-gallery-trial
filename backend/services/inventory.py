"""
Inventory loader — extracted 1-for-1 from app.py.
"""
from typing import Dict, Optional

import pandas as pd

from .helpers import map_to_oms_sku, get_parent_sku, read_csv_safe


def load_inventory_consolidated(
    oms_bytes: Optional[bytes],
    fk_bytes: Optional[bytes],
    myntra_bytes: Optional[bytes],
    amz_bytes: Optional[bytes],
    mapping: Dict[str, str],
    group_by_parent: bool = False,
) -> pd.DataFrame:
    """
    Merge inventory from OMS, Flipkart, Myntra and Amazon CSVs.
    Mirrors load_inventory_consolidated() in app.py.
    """
    inv_dfs = []

    if oms_bytes:
        df = read_csv_safe(oms_bytes)
        if not df.empty and {"Item SkuCode", "Inventory"}.issubset(df.columns):
            df = df.rename(columns={"Item SkuCode": "OMS_SKU", "Inventory": "OMS_Inventory"})
            df["OMS_SKU"]       = df["OMS_SKU"].astype(str)
            df["OMS_Inventory"] = pd.to_numeric(df["OMS_Inventory"], errors="coerce").fillna(0)
            inv_dfs.append(df[["OMS_SKU", "OMS_Inventory"]].groupby("OMS_SKU").sum().reset_index())

    if fk_bytes:
        df = read_csv_safe(fk_bytes)
        if not df.empty and {"SKU", "Live on Website"}.issubset(df.columns):
            df["OMS_SKU"]       = df["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Flipkart_Live"] = pd.to_numeric(df["Live on Website"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Flipkart_Live"].sum().reset_index())

    if myntra_bytes:
        df = read_csv_safe(myntra_bytes)
        if not df.empty:
            sku_col = next((c for c in df.columns if "seller sku code" in c.lower() or "sku code" in c.lower()), None)
            inv_col = next((c for c in df.columns if "sellable inventory count" in c.lower()), None)
            if sku_col and inv_col:
                df["OMS_SKU"]          = df[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                df["Myntra_Inventory"] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)
                inv_dfs.append(df.groupby("OMS_SKU")["Myntra_Inventory"].sum().reset_index())

    if amz_bytes:
        df = read_csv_safe(amz_bytes)
        if not df.empty and {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
            if "Location" in df.columns:
                df = df[df["Location"] != "ZNNE"]
            df["OMS_SKU"]          = df["MSKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Amazon_Inventory"] = pd.to_numeric(df["Ending Warehouse Balance"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Amazon_Inventory"].sum().reset_index())

    if not inv_dfs:
        return pd.DataFrame()

    consolidated = inv_dfs[0]
    for d in inv_dfs[1:]:
        consolidated = pd.merge(consolidated, d, on="OMS_SKU", how="outer")

    inv_cols = [c for c in consolidated.columns if c.endswith("_Inventory") or c.endswith("_Live")]
    consolidated[inv_cols] = consolidated[inv_cols].fillna(0)

    mkt_cols = [c for c in inv_cols if "OMS" not in c]
    consolidated["Marketplace_Total"] = consolidated[mkt_cols].sum(axis=1) if mkt_cols else 0
    consolidated["Total_Inventory"]   = consolidated.get("OMS_Inventory", 0) + consolidated["Marketplace_Total"]

    if group_by_parent:
        consolidated["Parent_SKU"] = consolidated["OMS_SKU"].apply(get_parent_sku)
        consolidated = (
            consolidated.groupby("Parent_SKU")[inv_cols + ["Marketplace_Total", "Total_Inventory"]]
            .sum().reset_index()
            .rename(columns={"Parent_SKU": "OMS_SKU"})
        )

    return consolidated[consolidated["Total_Inventory"] > 0].reset_index(drop=True)

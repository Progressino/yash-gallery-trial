#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System - app.py (UPDATED FULL VERSION)

✅ Key fixes for WRONG PO LIST:
1) INVENTORY was ALWAYS grouped to Parent SKU inside load_inventory_consolidated()
   → This breaks Variant-mode PO (size/color SKUs) because sales are variant-level.
   ✅ Fix: inventory loader now supports group_by_parent = False/True.
   - Variant mode uses variant-level inventory
   - Parent mode aggregates to parent

2) ADS calculation was based on "Days_With_Sales" (selling days),
   which can inflate/deflate demand and produce wrong PO.
   ✅ Fix: ADS now uses a stable denominator (period_days) with optional minimum floor.

3) Demand basis added:
   - "Sold" (Shipments only)  ✅ Recommended for PO
   - "Net" (Shipments - Refunds)

4) Proper SKU key alignment:
   - Variant mode uses OMS_SKU as-is
   - Parent mode uses Parent_SKU for BOTH sales and inventory before PO

Everything else from your original app remains (tabs, theme, forecast, deep dive, downloads).
"""

import io
import zipfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

warnings.filterwarnings("ignore")

# ==========================================================
# 1) PAGE CONFIG & THEME
# ==========================================================
st.set_page_config(
    page_title="Yash Gallery ERP",
    page_icon="🚀",
    layout="wide"
)

# Professional Navy & White Theme
st.markdown(
    """
<style>
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #002B5B;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: #6B7280;
        font-size: 0.9rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #002B5B 0%, #1e40af 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        color: #4B5563;
        font-weight: 600;
        border: 1px solid #E5E7EB;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #002B5B !important;
        color: white !important;
        border: 1px solid #002B5B;
    }
    h1, h2, h3 { color: #002B5B !important; }
</style>
""",
    unsafe_allow_html=True
)

st.title("🚀 Yash Gallery Command Center")
st.caption("Complete ERP: Sales Analytics • Inventory Management • PO Engine • AI Forecasting")

# ==========================================================
# 2) SESSION STATE INITIALIZATION
# ==========================================================
def init_session_state():
    defaults = {
        "sku_mapping": {},
        "sales_df": pd.DataFrame(),
        "inventory_df_variant": pd.DataFrame(),  # ✅ NEW: variant-level inventory
        "inventory_df_parent": pd.DataFrame(),   # ✅ NEW: parent-level inventory
        "transfer_df": pd.DataFrame(),
        "amazon_date_basis": "Shipment Date",
        "include_replacements": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ==========================================================
# 3) CONFIGURATION DATACLASS
# ==========================================================
@dataclass(frozen=True)
class SalesConfig:
    date_basis: str = "Shipment Date"
    include_replacements: bool = False

# ==========================================================
# 4) UTILITY FUNCTIONS
# ==========================================================
def clean_sku(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip()

def get_parent_sku(oms_sku) -> str:
    if pd.isna(oms_sku):
        return oms_sku

    s = str(oms_sku).strip()

    # remove marketplace suffixes
    marketplace_suffixes = ["_Myntra", "_Flipkart", "_Amazon", "_Meesho",
                            "_MYNTRA", "_FLIPKART", "_AMAZON", "_MEESHO"]
    for suf in marketplace_suffixes:
        if s.endswith(suf):
            s = s.replace(suf, "")
            break

    # remove last hyphen-part if looks like size/color
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2:
            last = parts[-1].upper()

            size_patterns = {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL"}
            common_colors = {
                "RED","BLUE","GREEN","BLACK","WHITE","YELLOW","PINK","PURPLE","ORANGE","BROWN",
                "GREY","GRAY","NAVY","MAROON","BEIGE","CREAM","GOLD","SILVER","TAN","KHAKI",
                "OLIVE","TEAL","CORAL","PEACH"
            }

            is_size = (
                last in size_patterns
                or last.endswith("XL")
                or last.isdigit()
                or (len(last) <= 4 and any(c in last for c in ["S","M","L","X"]))
            )

            is_color = (last in common_colors) or any(c in last for c in common_colors)

            if is_size or is_color:
                s = "-".join(parts[:-1])

    return s

def map_to_oms_sku(seller_sku, mapping: Dict[str, str]) -> str:
    if pd.isna(seller_sku):
        return seller_sku
    c = clean_sku(seller_sku)
    return mapping.get(c, c)

def read_zip_csv(zip_file) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                return pd.DataFrame()
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading ZIP: {e}")
        return pd.DataFrame()

def read_csv_safe(file_obj) -> pd.DataFrame:
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

# ==========================================================
# 5) SKU MAPPING LOADER
# ==========================================================
@st.cache_data(show_spinner=False)
def load_sku_mapping(mapping_file) -> Dict[str, str]:
    mapping_dict = {}
    try:
        xls = pd.ExcelFile(mapping_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(mapping_file, sheet_name=sheet_name)

            if df.empty or len(df.columns) < 2:
                continue

            seller_col = None
            oms_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["seller", "myntra", "meesho", "snapdeal", "sku id"]) and "sku" in col_lower:
                    seller_col = col
                if "oms" in col_lower and "sku" in col_lower:
                    oms_col = col

            if seller_col is None and len(df.columns) > 1:
                seller_col = df.columns[1]
            if oms_col is None:
                oms_col = df.columns[-1]

            if seller_col and oms_col:
                for _, row in df.iterrows():
                    seller_sku = clean_sku(row.get(seller_col, ""))
                    oms_sku = clean_sku(row.get(oms_col, ""))
                    if seller_sku and oms_sku and seller_sku != "nan" and oms_sku != "nan":
                        mapping_dict[seller_sku] = oms_sku

        return mapping_dict
    except Exception as e:
        st.error(f"Error loading SKU mapping: {e}")
        return {}

# ==========================================================
# 6) SALES DATA LOADERS
# ==========================================================
def load_amazon_sales(zip_file, mapping: Dict[str, str], source: str, config: SalesConfig) -> pd.DataFrame:
    df = read_zip_csv(zip_file)
    if df.empty or "Sku" not in df.columns:
        return pd.DataFrame()

    df["OMS_SKU"] = df["Sku"].apply(lambda x: map_to_oms_sku(x, mapping))

    date_col = config.date_basis
    if date_col not in df.columns:
        date_col = "Shipment Date" if "Shipment Date" in df.columns else \
                   "Invoice Date" if "Invoice Date" in df.columns else \
                   "Order Date" if "Order Date" in df.columns else df.columns[0]

    df["TxnDate"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)

    def classify_txn(txn_type):
        s = str(txn_type).lower()
        if "refund" in s or "return" in s:
            return "Refund"
        if "cancel" in s:
            return "Cancel"
        if "freereplacement" in s or "replacement" in s:
            return "FreeReplacement"
        return "Shipment"

    df["TxnType"] = df.get("Transaction Type", "").apply(classify_txn)

    if not config.include_replacements:
        df.loc[df["TxnType"] == "FreeReplacement", "Quantity"] = 0

    df["Units_Effective"] = np.where(
        df["TxnType"] == "Refund", -df["Quantity"],
        np.where(df["TxnType"] == "Cancel", 0, df["Quantity"])
    )

    df["Source"] = source

    order_col = None
    for col in df.columns:
        if "order" in str(col).lower() and "id" in str(col).lower():
            order_col = col
            break
    df["OrderId"] = df[order_col] if order_col else np.nan

    result = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
    result.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
    return result.dropna(subset=["TxnDate"])

def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_file, sheet_name="Sales Report")
        if df.empty:
            return pd.DataFrame()

        df["SKU_Clean"] = df["SKU"].apply(clean_sku)
        df["OMS_SKU"] = df["SKU_Clean"].apply(lambda x: map_to_oms_sku(x, mapping))

        df["TxnDate"] = pd.to_datetime(df.get("Order Date"), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("Item Quantity", 0), errors="coerce").fillna(0)
        df["Source"] = "Flipkart"

        df["TxnType"] = df.get("Event Sub Type", "").apply(lambda x: "Refund" if "return" in str(x).lower() else "Shipment")
        df["Units_Effective"] = np.where(df["TxnType"] == "Refund", -df["Quantity"], df["Quantity"])
        df["OrderId"] = df.get("Order ID", df.get("Order Id", np.nan))

        result = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
        result.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}")
        return pd.DataFrame()

def load_meesho_sales(zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            excel_files = [
                f for f in z.namelist()
                if "tcs_sales" in f.lower() and f.lower().endswith(".xlsx") and "return" not in f.lower()
            ]
            if not excel_files:
                return pd.DataFrame()
            with z.open(excel_files[0]) as f:
                df = pd.read_excel(f)

        if df.empty:
            return pd.DataFrame()

        df["OMS_SKU"] = df.get("identifier").apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"] = pd.to_datetime(df.get("order_date"), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
        df["Source"] = "Meesho"
        df["TxnType"] = "Shipment"
        df["Units_Effective"] = df["Quantity"]
        df["OrderId"] = df.get("sub_order_num", np.nan)

        result = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
        result.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Meesho: {e}")
        return pd.DataFrame()

# ==========================================================
# 7) INVENTORY LOADERS (✅ FIXED)
# ==========================================================
def load_inventory_consolidated(
    oms_file, fk_file, myntra_file, amz_file,
    mapping: Dict[str, str],
    group_by_parent: bool = False
) -> pd.DataFrame:
    """
    Load and consolidate inventory from all sources.

    ✅ FIX:
    - If group_by_parent=False → keep variant-level OMS_SKU.
    - If group_by_parent=True  → aggregate to Parent SKU.

    Amazon: still excludes ZNNE.
    """
    inv_dfs = []

    # OMS Inventory
    if oms_file:
        df = read_csv_safe(oms_file)
        if not df.empty and {"Item SkuCode", "Inventory"}.issubset(df.columns):
            df = df.rename(columns={"Item SkuCode": "OMS_SKU", "Inventory": "OMS_Inventory"})
            df["OMS_SKU"] = df["OMS_SKU"].astype(str)
            df["OMS_Inventory"] = pd.to_numeric(df["OMS_Inventory"], errors="coerce").fillna(0)
            inv_dfs.append(df[["OMS_SKU", "OMS_Inventory"]].groupby("OMS_SKU").sum().reset_index())

    # Flipkart Inventory
    if fk_file:
        df = read_csv_safe(fk_file)
        if not df.empty and {"SKU", "Live on Website"}.issubset(df.columns):
            df["OMS_SKU"] = df["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Flipkart_Live"] = pd.to_numeric(df["Live on Website"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Flipkart_Live"].sum().reset_index())

    # Myntra Inventory
    if myntra_file:
        df = read_csv_safe(myntra_file)
        if not df.empty:
            sku_col = None
            inv_col = None

            for col in df.columns:
                if "seller sku code" in str(col).lower() or "sku code" in str(col).lower():
                    sku_col = col
                    break

            for col in df.columns:
                if "sellable inventory count" in str(col).lower():
                    inv_col = col
                    break

            if sku_col and inv_col:
                df["OMS_SKU"] = df[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                df["Myntra_Inventory"] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)
                inv_dfs.append(df.groupby("OMS_SKU")["Myntra_Inventory"].sum().reset_index())

    # Amazon Inventory (excluding ZNNE)
    if amz_file:
        df = read_csv_safe(amz_file)
        if not df.empty and {"MSKU", "Ending Warehouse Balance"}.issubset(df.columns):
            if "Location" in df.columns:
                original = len(df)
                df = df[df["Location"] != "ZNNE"]
                excluded = original - len(df)
                if excluded > 0:
                    st.sidebar.info(f"ℹ️ Excluded {excluded:,} ZNNE records (OMS duplicate)")

            df["OMS_SKU"] = df["MSKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Amazon_Inventory"] = pd.to_numeric(df["Ending Warehouse Balance"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Amazon_Inventory"].sum().reset_index())

    if not inv_dfs:
        return pd.DataFrame()

    consolidated = inv_dfs[0]
    for d in inv_dfs[1:]:
        consolidated = pd.merge(consolidated, d, on="OMS_SKU", how="outer")

    inv_cols = [c for c in consolidated.columns if c.endswith("_Inventory") or c.endswith("_Live")]
    consolidated[inv_cols] = consolidated[inv_cols].fillna(0)

    # totals
    marketplace_cols = [c for c in inv_cols if "OMS" not in c]
    consolidated["Marketplace_Total"] = consolidated[marketplace_cols].sum(axis=1) if marketplace_cols else 0
    consolidated["Total_Inventory"] = consolidated.get("OMS_Inventory", 0) + consolidated["Marketplace_Total"]

    # ✅ OPTIONAL parent grouping ONLY when needed
    if group_by_parent:
        consolidated["Parent_SKU"] = consolidated["OMS_SKU"].apply(get_parent_sku)
        consolidated = consolidated.groupby("Parent_SKU")[inv_cols + ["Marketplace_Total", "Total_Inventory"]].sum().reset_index()
        consolidated = consolidated.rename(columns={"Parent_SKU": "OMS_SKU"})

    # keep non-zero
    consolidated = consolidated[consolidated["Total_Inventory"] > 0]
    return consolidated

# ==========================================================
# 8) STOCK TRANSFER LOADER
# ==========================================================
def load_stock_transfer(zip_file) -> pd.DataFrame:
    df = read_zip_csv(zip_file)
    if df.empty:
        return pd.DataFrame()

    required = ["Invoice Date", "Ship From Fc", "Ship To Fc", "Quantity", "Transaction Type"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    result = df[required].copy()
    result["Invoice Date"] = pd.to_datetime(result["Invoice Date"], errors="coerce")
    result["Quantity"] = pd.to_numeric(result["Quantity"], errors="coerce").fillna(0)
    return result

# ==========================================================
# 9) PANEL-WISE PIVOT BUILDER
# ==========================================================
def build_panel_pivots(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w = df.copy()
    w["Panel"] = w["Source"].astype(str)

    sold = (
        w[w["Transaction Type"] == "Shipment"]
        .groupby(["Sku", "Panel"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    sold.columns = [f"{c} | Sold" for c in sold.columns]

    ret = (
        w[w["Transaction Type"] == "Refund"]
        .groupby(["Sku", "Panel"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    ret.columns = [f"{c} | Return" for c in ret.columns]

    net = (
        w.groupby(["Sku", "Panel"])["Units_Effective"]
        .sum()
        .unstack(fill_value=0)
    )
    net.columns = [f"{c} | Net" for c in net.columns]

    return sold, ret, net

# ==========================================================
# 10) SMART ADS CALCULATION (✅ FIXED)
# ==========================================================
def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    demand_basis: str = "Sold",     # "Sold" or "Net"
    min_denominator: int = 7
) -> pd.DataFrame:
    """
    ✅ FIX: ADS uses stable period denominator, NOT "days with sales".

    demand_basis:
      - "Sold" => sum Shipment Quantity
      - "Net"  => sum Units_Effective (Ship - Refund)

    Returns merged DF with:
      OMS_SKU, Total_Inventory, Total_Units_Sold, Returns, Net_Units, ADS
    """
    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])

    max_date = df["TxnDate"].max()
    cutoff = max_date - timedelta(days=period_days)

    recent = df[df["TxnDate"] >= cutoff].copy()

    # Sold and returns
    sold = recent[recent["Transaction Type"] == "Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]

    returns = recent[recent["Transaction Type"] == "Refund"].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU", "Return_Units"]

    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)

    # Merge with inventory
    po_df = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna({"Sold_Units": 0, "Return_Units": 0, "Net_Units": 0})

    denom = max(period_days, min_denominator)

    if demand_basis == "Net":
        demand_units = po_df["Net_Units"].clip(lower=0)  # avoid negative ADS
    else:
        demand_units = po_df["Sold_Units"]

    po_df["ADS"] = (demand_units / denom).fillna(0)

    # stockout flag (simple)
    po_df["Stockout_Flag"] = ""
    po_df.loc[(po_df["ADS"] > 0) & (po_df["Total_Inventory"] <= 0), "Stockout_Flag"] = "⚠️ OOS"

    return po_df

# ==========================================================
# 11) SIDEBAR - FILE UPLOADS
# ==========================================================
st.sidebar.markdown("## 📂 Data Upload")

try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.markdown("### Yash Gallery")

st.sidebar.divider()

map_file = st.sidebar.file_uploader(
    "1️⃣ SKU Mapping (Required)",
    type=["xlsx"],
    help="Copy_of_All_penal_replace_sku.xlsx"
)

st.sidebar.markdown("### ⚙️ Amazon Settings")
st.session_state.amazon_date_basis = st.sidebar.selectbox(
    "Date Basis", ["Shipment Date", "Invoice Date", "Order Date"], index=0
)
st.session_state.include_replacements = st.sidebar.checkbox("Include FreeReplacement", value=False)

st.sidebar.divider()

st.sidebar.markdown("### 2️⃣ Sales Data")
f_b2c = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=["zip"], key="b2c")
f_b2b = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=["zip"], key="b2b")
f_transfer = st.sidebar.file_uploader("Stock Transfer (ZIP)", type=["zip"], key="transfer")
f_fk = st.sidebar.file_uploader("Flipkart (Excel)", type=["xlsx"], key="fk")
f_meesho = st.sidebar.file_uploader("Meesho (ZIP)", type=["zip"], key="meesho")

st.sidebar.divider()

st.sidebar.markdown("### 3️⃣ Inventory Data")
i_oms = st.sidebar.file_uploader("OMS (CSV)", type=["csv"], key="oms")
i_fk = st.sidebar.file_uploader("Flipkart (CSV)", type=["csv"], key="fk_inv")
i_myntra = st.sidebar.file_uploader("Myntra (CSV)", type=["csv"], key="myntra")
i_amz = st.sidebar.file_uploader("Amazon (CSV)", type=["csv"], key="amz")

st.sidebar.divider()

if st.sidebar.button("🚀 Load All Data", use_container_width=True):
    if not map_file:
        st.sidebar.error("SKU Mapping required!")
    else:
        with st.spinner("Loading data..."):
            st.session_state.sku_mapping = load_sku_mapping(map_file)
            if st.session_state.sku_mapping:
                st.sidebar.success(f"✅ Mapping: {len(st.session_state.sku_mapping):,} SKUs")

            config = SalesConfig(
                date_basis=st.session_state.amazon_date_basis,
                include_replacements=st.session_state.include_replacements,
            )

            # SALES
            sales_parts = []
            if f_b2c:
                sales_parts.append(load_amazon_sales(f_b2c, st.session_state.sku_mapping, "Amazon B2C", config))
            if f_b2b:
                sales_parts.append(load_amazon_sales(f_b2b, st.session_state.sku_mapping, "Amazon B2B", config))
            if f_fk:
                sales_parts.append(load_flipkart_sales(f_fk, st.session_state.sku_mapping))
            if f_meesho:
                sales_parts.append(load_meesho_sales(f_meesho, st.session_state.sku_mapping))

            if sales_parts:
                st.session_state.sales_df = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)
                st.sidebar.success(f"✅ Sales: {len(st.session_state.sales_df):,} records")
            else:
                st.session_state.sales_df = pd.DataFrame()

            # INVENTORY (✅ load BOTH)
            st.session_state.inventory_df_variant = load_inventory_consolidated(
                i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=False
            )
            st.session_state.inventory_df_parent = load_inventory_consolidated(
                i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=True
            )

            if not st.session_state.inventory_df_variant.empty:
                st.sidebar.success(f"✅ Inventory (Variant): {len(st.session_state.inventory_df_variant):,} SKUs")
            if not st.session_state.inventory_df_parent.empty:
                st.sidebar.success(f"✅ Inventory (Parent): {len(st.session_state.inventory_df_parent):,} Styles")

            # TRANSFERS
            if f_transfer:
                st.session_state.transfer_df = load_stock_transfer(f_transfer)
                if not st.session_state.transfer_df.empty:
                    st.sidebar.success(f"✅ Transfers: {len(st.session_state.transfer_df):,} records")

        st.rerun()

# ==========================================================
# 12) GUARD
# ==========================================================
if not st.session_state.sku_mapping:
    st.info("👋 **Welcome!** Upload SKU Mapping file and click **Load All Data** to begin.")
    st.stop()

# ==========================================================
# 13) MAIN TABS
# ==========================================================
tab_dash, tab_inv, tab_po, tab_logistics, tab_forecast, tab_drill = st.tabs(
    ["📊 Dashboard", "📦 Inventory", "🎯 PO Engine", "🚚 Logistics", "📈 AI Forecast", "🔍 Deep Dive"]
)

# ----------------------------------------------------------
# TAB 1: DASHBOARD
# ----------------------------------------------------------
with tab_dash:
    st.subheader("📊 Sales Analytics Dashboard")
    df = st.session_state.sales_df

    if df.empty:
        st.warning("⚠️ No sales data loaded. Upload sales files and click Load Data.")
    else:
        col_period, col_grace = st.columns([3, 1])
        with col_period:
            period_option = st.selectbox(
                "Analysis Period",
                ["Last 7 Days", "Last 30 Days", "Last 60 Days", "Last 90 Days", "All Time"],
                index=1,
                key="dash_period",
            )
        with col_grace:
            grace_days = st.number_input(
                "Grace Period (Days)", min_value=0, max_value=14, value=7,
                help="Extra days to capture late transactions (recommended: 7 days)"
            )

        df = df.copy()
        df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
        max_date = df["TxnDate"].max()

        if period_option == "All Time":
            filtered_df = df
            min_date = df["TxnDate"].min()
            date_range_text = f"All Time: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            base_days = 7 if "7" in period_option else 30 if "30" in period_option else 60 if "60" in period_option else 90
            total_days = base_days + grace_days
            cutoff_date = max_date - timedelta(days=total_days)
            filtered_df = df[df["TxnDate"] >= cutoff_date]
            actual_min = filtered_df["TxnDate"].min()
            date_range_text = f"Period: {actual_min.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({base_days}+{grace_days})"

        st.info(f"📅 **{date_range_text}** | Transactions: {len(filtered_df):,}")

        filtered_df["Quantity"] = pd.to_numeric(filtered_df["Quantity"], errors="coerce").fillna(0)
        filtered_df["Units_Effective"] = pd.to_numeric(filtered_df["Units_Effective"], errors="coerce").fillna(0)

        sold_pcs = filtered_df[filtered_df["Transaction Type"] == "Shipment"]["Quantity"].sum()
        ret_pcs = filtered_df[filtered_df["Transaction Type"] == "Refund"]["Quantity"].sum()
        net_units = filtered_df["Units_Effective"].sum()

        orders = filtered_df[filtered_df["Transaction Type"] == "Shipment"]["OrderId"].nunique() if "OrderId" in filtered_df.columns else len(filtered_df[filtered_df["Transaction Type"] == "Shipment"])
        return_rate = (ret_pcs / sold_pcs * 100) if sold_pcs > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛒 Orders", f"{orders:,}")
        c2.metric("✅ Sold Pieces", f"{int(sold_pcs):,}")
        c3.metric("↩️ Returns", f"{int(ret_pcs):,}")
        c4.metric("📊 Return Rate", f"{return_rate:.1f}%")
        c5.metric("📦 Net Units", f"{int(net_units):,}")

        st.info(f"**Settings:** Amazon Date: {st.session_state.amazon_date_basis} | Include Replacements: {st.session_state.include_replacements}")

        st.divider()

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### 🏆 Top 20 Selling SKUs")
            top = (
                filtered_df[filtered_df["Transaction Type"] == "Shipment"]
                .groupby("Sku")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )
            fig = px.bar(top, x="Sku", y="Quantity", title="Top Sellers (Pieces)")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📊 Marketplace Split")
            source_summary = filtered_df.groupby("Source")["Quantity"].sum().reset_index()
            fig = px.pie(source_summary, values="Quantity", names="Source", hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 2: INVENTORY
# ----------------------------------------------------------
with tab_inv:
    st.subheader("📦 Consolidated Inventory")

    mode = st.radio("Inventory View", ["Variant (Size/Color)", "Parent (Style Only)"], horizontal=True)
    inv = st.session_state.inventory_df_variant if "Variant" in mode else st.session_state.inventory_df_parent

    if inv.empty:
        st.warning("⚠️ No inventory data loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(inv):,}")
        c2.metric("Total Units", f"{inv['Total_Inventory'].sum():,.0f}")
        if "OMS_Inventory" in inv.columns:
            c3.metric("OMS Warehouse", f"{inv['OMS_Inventory'].sum():,.0f}")
        c4.metric("Marketplaces", f"{inv['Marketplace_Total'].sum():,.0f}" if "Marketplace_Total" in inv.columns else "0")

        st.divider()
        search = st.text_input("🔍 Search SKU", placeholder="Type to filter...")
        display = inv.copy()
        if search:
            display = display[display["OMS_SKU"].str.contains(search, case=False, na=False)]
        st.dataframe(display, use_container_width=True, height=500)

# ----------------------------------------------------------
# TAB 3: PO ENGINE (✅ FIXED)
# ----------------------------------------------------------
with tab_po:
    st.subheader("🎯 Purchase Order Recommendations")

    if st.session_state.sales_df.empty:
        st.error("⚠️ Sales data required for PO calculations.")
        st.stop()

    if st.session_state.inventory_df_variant.empty and st.session_state.inventory_df_parent.empty:
        st.error("⚠️ Inventory data required for PO calculations.")
        st.stop()

    st.markdown("### 👁️ View Mode")
    col_view, col_info = st.columns([1, 3])
    with col_view:
        view_mode = st.radio(
            "Group By",
            ["By Variant (Size/Color)", "By Parent SKU (Style Only)"],
            key="po_view_mode",
        )

    with col_info:
        st.info(
            "✅ Variant mode = size/color level PO\n\n✅ Parent mode = style level PO (all sizes combined)\n\n"
            "This build now aligns sales + inventory correctly in both modes."
        )

    st.divider()

    st.markdown("### ⚙️ Configure PO Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)

    velocity = c1.selectbox("Velocity Period", ["Last 7 Days", "Last 30 Days", "Last 60 Days", "Last 90 Days"], key="po_velocity")
    base_days = 7 if "7" in velocity else 30 if "30" in velocity else 60 if "60" in velocity else 90

    grace_days = c2.number_input("Grace Days", 0, 14, 7)
    lead_time = c3.number_input("Lead Time (Days)", 1, 180, 15)
    target_days = c4.number_input("Target Stock (Days)", 0, 180, 60)
    safety_pct = c5.slider("Safety Stock %", 0, 100, 20)

    st.divider()

    demand_basis = st.selectbox(
        "Demand Basis for ADS (Important)",
        ["Sold", "Net"],
        index=0,
        help="Sold = shipments only (recommended for PO). Net = shipments - returns."
    )

    min_den = st.number_input(
        "Minimum ADS Denominator Days",
        min_value=1,
        max_value=60,
        value=7,
        help="If your period is small or sparse, this prevents ADS from becoming too large."
    )

    total_period = int(base_days + grace_days)

    # ✅ choose correct inventory by mode
    if "Parent" in view_mode:
        inv_for_po = st.session_state.inventory_df_parent.copy()

        # ✅ also convert sales to parent key
        sales_for_po = st.session_state.sales_df.copy()
        sales_for_po["Sku"] = sales_for_po["Sku"].apply(get_parent_sku)

    else:
        inv_for_po = st.session_state.inventory_df_variant.copy()
        sales_for_po = st.session_state.sales_df.copy()

    # build base
    po_df = calculate_po_base(
        sales_df=sales_for_po,
        inv_df=inv_for_po,
        period_days=total_period,
        demand_basis=demand_basis,
        min_denominator=int(min_den),
    )

    if po_df.empty:
        st.warning("No PO calculations available.")
        st.stop()

    # days left
    po_df["Days_Left"] = np.where(po_df["ADS"] > 0, po_df["Total_Inventory"] / po_df["ADS"], 999)

    # requirements
    po_df["Lead_Time_Demand"] = po_df["ADS"] * lead_time
    po_df["Target_Stock"] = po_df["ADS"] * target_days
    po_df["Base_Requirement"] = po_df["Lead_Time_Demand"] + po_df["Target_Stock"]
    po_df["Safety_Stock"] = po_df["Base_Requirement"] * (safety_pct / 100)
    po_df["Total_Required"] = po_df["Base_Requirement"] + po_df["Safety_Stock"]

    po_df["PO_Recommended"] = (po_df["Total_Required"] - po_df["Total_Inventory"]).clip(lower=0)
    po_df["PO_Recommended"] = (np.ceil(po_df["PO_Recommended"] / 5) * 5).astype(int)

    def get_priority(row):
        if row["Days_Left"] < lead_time and row["PO_Recommended"] > 0:
            return "🔴 URGENT"
        elif row["Days_Left"] < lead_time + 7 and row["PO_Recommended"] > 0:
            return "🟡 HIGH"
        elif row["PO_Recommended"] > 0:
            return "🟢 MEDIUM"
        return "⚪ OK"

    po_df["Priority"] = po_df.apply(get_priority, axis=1)

    po_needed = po_df[po_df["PO_Recommended"] > 0].sort_values(["Priority", "Days_Left"])

    urgent = len(po_needed[po_needed["Priority"] == "🔴 URGENT"])
    high = len(po_needed[po_needed["Priority"] == "🟡 HIGH"])
    medium = len(po_needed[po_needed["Priority"] == "🟢 MEDIUM"])
    total_units = po_needed["PO_Recommended"].sum()

    st.markdown("### 📊 PO Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔴 Urgent", urgent)
    m2.metric("🟡 High", high)
    m3.metric("🟢 Medium", medium)
    m4.metric("📦 Total Units", f"{total_units:,}")

    st.divider()

    search = st.text_input("🔍 Search SKU", placeholder="Filter by SKU...")
    if search:
        po_needed = po_needed[po_needed["OMS_SKU"].astype(str).str.contains(search, case=False, na=False)]
        st.caption(f"Showing {len(po_needed):,} matches")

    st.markdown("### 📋 Purchase Order List")

    display_cols = [
        "Priority",
        "OMS_SKU",
        "Total_Inventory",
        "Sold_Units",
        "Return_Units",
        "Net_Units",
        "ADS",
        "Days_Left",
        "Lead_Time_Demand",
        "Target_Stock",
        "Safety_Stock",
        "Total_Required",
        "PO_Recommended",
        "Stockout_Flag",
    ]
    display_cols = [c for c in display_cols if c in po_needed.columns]

    def highlight_priority(row):
        colors = []
        for col in row.index:
            if col == "Priority":
                if "🔴" in str(row[col]):
                    colors.append("background-color: #fee2e2; font-weight: bold")
                elif "🟡" in str(row[col]):
                    colors.append("background-color: #fef3c7")
                else:
                    colors.append("background-color: #d1fae5")
            elif col == "PO_Recommended":
                colors.append("background-color: #dbeafe; font-weight: bold; font-size: 1.1em")
            elif col == "Days_Left" and float(row[col]) < float(lead_time):
                colors.append("background-color: #fee2e2; font-weight: bold")
            else:
                colors.append("")
        return colors

    format_dict = {
        "ADS": "{:.3f}",
        "Days_Left": "{:.1f}",
        "Lead_Time_Demand": "{:.0f}",
        "Target_Stock": "{:.0f}",
        "Safety_Stock": "{:.0f}",
        "Total_Required": "{:.0f}",
        "PO_Recommended": "{:.0f}",
        "Total_Inventory": "{:.0f}",
        "Sold_Units": "{:.0f}",
        "Return_Units": "{:.0f}",
        "Net_Units": "{:.0f}",
    }

    st.dataframe(
        po_needed[display_cols].head(200).style.apply(highlight_priority, axis=1).format(format_dict),
        use_container_width=True,
        height=520,
    )
    st.caption(f"Showing top 200 of {len(po_needed):,} SKUs needing orders")

    st.divider()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = po_needed[display_cols].to_csv(index=False).encode("utf-8")
        suffix = "parent" if "Parent" in view_mode else "variant"
        st.download_button(
            "📥 Download PO (CSV)",
            csv,
            f"po_recommendations_{suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True,
        )

    with col_dl2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            po_needed[display_cols].to_excel(writer, sheet_name="PO_Recommendations", index=False)
        st.download_button(
            "📥 Download PO (Excel)",
            excel_buffer.getvalue(),
            f"po_recommendations_{suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ----------------------------------------------------------
# TAB 4: LOGISTICS
# ----------------------------------------------------------
with tab_logistics:
    st.subheader("🚚 Stock Transfers & FC Movements")

    transfer_df = st.session_state.transfer_df
    if transfer_df.empty:
        st.info("📦 Upload Amazon Stock Transfer file to view logistics data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Movements", f"{len(transfer_df):,}")
        c2.metric("Units Transferred", f"{transfer_df['Quantity'].sum():,.0f}")
        c3.metric("FC Transfers", f"{len(transfer_df[transfer_df['Transaction Type'] == 'FC_TRANSFER']):,}")
        c4.metric("Unique Routes", f"{len(transfer_df.groupby(['Ship From Fc','Ship To Fc'])):,}")

        st.divider()

        st.markdown("### 🔝 Top Transfer Routes")
        routes = transfer_df.groupby(["Ship From Fc", "Ship To Fc"]).agg({"Quantity": ["sum", "count"]}).reset_index()
        routes.columns = ["From FC", "To FC", "Units", "Transfers"]
        routes = routes.sort_values("Units", ascending=False).head(20)
        st.dataframe(routes, use_container_width=True)

# ----------------------------------------------------------
# TAB 5: AI FORECAST
# ----------------------------------------------------------
with tab_forecast:
    st.subheader("📈 AI Demand Forecasting")

    sales = st.session_state.sales_df
    if sales.empty:
        st.warning("⚠️ Upload sales data for forecasting.")
    else:
        sku = st.selectbox("Select SKU", [""] + sorted(sales["Sku"].dropna().unique().tolist()))
        days = st.slider("Forecast Days", 7, 90, 30)

        if sku:
            subset = sales[sales["Sku"] == sku].copy()
            subset["ds"] = pd.to_datetime(subset["TxnDate"]).dt.date
            daily = subset.groupby("ds")["Units_Effective"].sum().reset_index()
            daily.columns = ["ds", "y"]
            daily["ds"] = pd.to_datetime(daily["ds"])

            if len(daily) < 14:
                st.warning("Need at least 14 days of data.")
            else:
                try:
                    with st.spinner("Forecasting..."):
                        m = Prophet(daily_seasonality=False, weekly_seasonality=True)
                        m.fit(daily)
                        future = m.make_future_dataframe(periods=days)
                        forecast = m.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual"))
                    future_only = forecast[forecast["ds"] > daily["ds"].max()]
                    fig.add_trace(go.Scatter(x=future_only["ds"], y=future_only["yhat"], name="Forecast", line=dict(dash="dash")))
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"🤖 Predicted demand (next {days} days): **{int(future_only['yhat'].sum())} units**")
                except Exception as e:
                    st.error(f"Forecast error: {e}")

# ----------------------------------------------------------
# TAB 6: DEEP DIVE
# ----------------------------------------------------------
with tab_drill:
    st.subheader("🔍 Deep Dive & Panel Analysis")
    df = st.session_state.sales_df

    if df.empty:
        st.warning("⚠️ Upload sales data for deep dive.")
    else:
        colA, colB, colC = st.columns([3, 2, 1])

        with colA:
            search = st.text_input("Enter SKU", placeholder="e.g., 1065YK", key="drill_search")
        with colB:
            period = st.selectbox("Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"], index=1, key="drill_period")
        with colC:
            grace = st.number_input("Grace Days", 0, 14, 7, key="drill_grace")

        fdf = df.copy()
        fdf["TxnDate"] = pd.to_datetime(fdf["TxnDate"], errors="coerce")
        max_d = fdf["TxnDate"].max()

        if period != "All Time" and not pd.isna(max_d):
            base = 7 if "7" in period else 30 if "30" in period else 90
            cutoff = max_d - timedelta(days=base + grace)
            fdf = fdf[fdf["TxnDate"] >= cutoff]
            actual_min = fdf["TxnDate"].min()
            date_range_text = f"{actual_min.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')} ({base}+{grace})"
        else:
            date_range_text = f"All Time: {fdf['TxnDate'].min().strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"

        st.info(f"📅 **Period:** {date_range_text} | **Transactions:** {len(fdf):,}")

        if search:
            matches = fdf[fdf["Sku"].astype(str).str.contains(search, case=False, na=False)].copy()
            if matches.empty:
                st.warning("No matching SKUs found.")
            else:
                unique_skus = matches["Sku"].nunique()
                st.success(f"✅ Found **{unique_skus}** SKU variant(s) matching '{search}'")

                st.divider()

                sold = matches[matches["Transaction Type"] == "Shipment"]["Quantity"].sum()
                ret = matches[matches["Transaction Type"] == "Refund"]["Quantity"].sum()
                net = matches["Units_Effective"].sum()
                rate = (ret / sold * 100) if sold > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sold Pieces", f"{int(sold):,}")
                m2.metric("Returns", f"{int(ret):,}")
                m3.metric("Net Units", f"{int(net):,}")
                m4.metric("Return %", f"{rate:.1f}%")

                st.divider()

                st.markdown("### 🏪 Marketplace Breakdown (Panel-wise)")
                sold_p, ret_p, net_p = build_panel_pivots(matches)
                combined = pd.concat([sold_p, ret_p, net_p], axis=1).fillna(0).reset_index()
                if not combined.empty:
                    st.dataframe(combined, use_container_width=True, height=400)

                st.markdown("---")
                st.markdown("### 📜 Recent Transactions (Latest 100)")
                cols = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Source"]
                if "OrderId" in matches.columns:
                    cols.append("OrderId")
                if "Units_Effective" in matches.columns:
                    cols.append("Units_Effective")

                display_txns = matches.sort_values("TxnDate", ascending=False).head(100)[cols].copy()
                display_txns["TxnDate"] = pd.to_datetime(display_txns["TxnDate"]).dt.strftime("%Y-%m-%d")
                st.dataframe(display_txns, use_container_width=True, height=400)

                st.markdown("---")
                csv = matches[cols].to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Full Transaction History", csv, f"sku_drilldown_{search}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.info("💡 **Tip:** Type a SKU (or partial SKU) in the search box above to start analysis.")
            st.markdown("### 📦 Sample SKUs Available")
            sample_skus = fdf["Sku"].value_counts().head(20).reset_index()
            sample_skus.columns = ["SKU", "Transaction Count"]
            st.dataframe(sample_skus, use_container_width=True, height=300)

st.divider()
st.caption("💡 Yash Gallery ERP | UPDATED app.py | PO engine fixed (inventory alignment + ADS logic)")

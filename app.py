#!/usr/bin/env python3
"""
Progressino / Yash Gallery Complete ERP System (Refactored)
Merged + Refactored from:
- Yash Gallery ERP v3.0 (Fixed metrics + panel-wise drilldown)
- Progressino ERP UI/Theme + PO Engine corrections

Key Fixes Preserved:
1) Amazon Date Basis selection (Shipment/Invoice/Order)
2) Sold/Return metrics based on SUM(Quantity) (pieces), not row count
3) FreeReplacement handling include/exclude
4) Panel-wise (Marketplace-wise) SKU drilldown (Sold/Return/Net per panel)
5) PO engine ranges updated (Lead time up to 180, Target up to 180, Safety up to 100, Velocity modes)
"""

from __future__ import annotations

import io
import zipfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

warnings.filterwarnings("ignore")


# ==========================================================
# 1) APP CONFIG + THEME
# ==========================================================
APP_TITLE = "Progressino ERP"
APP_ICON = "🧠"
DEFAULT_LAYOUT = "wide"

st.set_page_config(page_title=APP_TITLE, layout=DEFAULT_LAYOUT, page_icon=APP_ICON)

# --- PROFESSIONAL NAVY & WHITE CSS ---
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
    .ai-box {
        background-color: #FFFFFF;
        border-left: 4px solid #6366F1;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .success-box { background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 12px; border-radius: 6px; color: #065F46; }
    .warning-box { background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; border-radius: 6px; color: #92400E; }
    h1, h2, h3 { color: #002B5B !important; letter-spacing: -0.5px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚀 Yash Gallery Command Center")
st.caption("Inventory + Sales + PO Recommendations with OMS SKU Mapping | Refactored Build")


# ==========================================================
# 2) SESSION STATE
# ==========================================================
def _init_state():
    if "sku_mapping" not in st.session_state:
        st.session_state.sku_mapping = {}
    if "sales_df" not in st.session_state:
        st.session_state.sales_df = pd.DataFrame()
    if "inventory_df" not in st.session_state:
        st.session_state.inventory_df = pd.DataFrame()
    if "transfer_df" not in st.session_state:
        st.session_state.transfer_df = pd.DataFrame()
    if "amazon_date_basis" not in st.session_state:
        st.session_state.amazon_date_basis = "Shipment Date"
    if "include_replacements" not in st.session_state:
        st.session_state.include_replacements = False


_init_state()


# ==========================================================
# 3) UTILITIES / HELPERS
# ==========================================================
@dataclass(frozen=True)
class SalesLoadConfig:
    date_basis: str = "Shipment Date"          # Shipment Date / Invoice Date / Order Date
    include_replacements: bool = False         # include FreeReplacement in sold qty


def _clean_sku(x) -> str:
    return str(x).strip().replace('"""', "").replace("SKU:", "").strip()


@st.cache_data(show_spinner=False)
def load_sku_mapping(mapping_file) -> Dict[str, str]:
    """
    Load SKU mapping from Excel with multiple sheets.
    Tries to auto-detect seller sku column and oms sku column.
    """
    mapping_dict: Dict[str, str] = {}
    try:
        xls = pd.ExcelFile(mapping_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(mapping_file, sheet_name=sheet_name)
            if df.empty or len(df.columns) < 2:
                continue

            # Detect columns
            seller_col = None
            oms_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if ("sku" in col_lower) and any(k in col_lower for k in ["seller", "myntra", "meesho", "snapdeal", "sku id"]):
                    seller_col = col
                if ("oms" in col_lower) and ("sku" in col_lower):
                    oms_col = col

            if seller_col is None:
                seller_col = df.columns[1] if len(df.columns) > 1 else None
            if oms_col is None:
                oms_col = df.columns[-1]

            if not seller_col or not oms_col:
                continue

            for _, row in df.iterrows():
                k = _clean_sku(row.get(seller_col, ""))
                v = _clean_sku(row.get(oms_col, ""))
                if k and v and k.lower() != "nan" and v.lower() != "nan":
                    mapping_dict[k] = v

        return mapping_dict
    except Exception:
        return {}


def map_to_oms_sku(seller_sku, mapping_dict: Dict[str, str]) -> Optional[str]:
    if pd.isna(seller_sku):
        return None
    clean = _clean_sku(seller_sku)
    return mapping_dict.get(clean, clean)


def get_parent_sku(oms_sku) -> str:
    if pd.isna(oms_sku):
        return oms_sku
    s = str(oms_sku)
    for suffix in ["_Myntra", "_Flipkart", "_Amazon", "_Meesho", "_MYNTRA", "_FLIPKART", "_AMAZON", "_MEESHO"]:
        if s.endswith(suffix):
            s = s.replace(suffix, "")
            break
    return s


def read_zip_first_csv(file_obj) -> pd.DataFrame:
    """Reads first CSV found inside ZIP."""
    try:
        with zipfile.ZipFile(file_obj, "r") as z:
            csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csvs:
                return pd.DataFrame()
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()


def read_csv_file(file_obj) -> pd.DataFrame:
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception:
        return pd.DataFrame()


def detect_order_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in ["order id", "order-id", "orderid"]:
            return c
    return None


def build_panel_pivots(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 3 pivots:
      - Sold pivot: SKU x Panel -> Shipment Qty (pieces)
      - Return pivot: SKU x Panel -> Return Qty (pieces)
      - Net pivot: SKU x Panel -> Net Units (Units_Effective)
    """
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
    ret = (
        w[w["Transaction Type"] == "Refund"]
        .groupby(["Sku", "Panel"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    net = (
        w.groupby(["Sku", "Panel"])["Units_Effective"]
        .sum()
        .unstack(fill_value=0)
    )

    sold.columns = [f"{c} | Sold" for c in sold.columns]
    ret.columns = [f"{c} | Return" for c in ret.columns]
    net.columns = [f"{c} | Net" for c in net.columns]
    return sold, ret, net


# ==========================================================
# 4) DATA LOADERS
# ==========================================================
def load_amazon_sales(zip_file, mapping: Dict[str, str], source: str, cfg: SalesLoadConfig) -> pd.DataFrame:
    """
    Amazon sales loader:
    - ZIP CSV
    - date basis selection
    - replacement include/exclude
    - Units_Effective: Refund negative, Cancel 0, Shipment positive
    """
    df = read_zip_first_csv(zip_file)
    if df.empty or "Sku" not in df.columns:
        return pd.DataFrame()

    df["Sku"] = df["Sku"].astype(str)
    df["OMS_SKU"] = df["Sku"].apply(lambda x: map_to_oms_sku(x, mapping))

    # date basis safe fallback
    preferred = cfg.date_basis
    if preferred not in df.columns:
        if "Shipment Date" in df.columns:
            preferred = "Shipment Date"
        elif "Invoice Date" in df.columns:
            preferred = "Invoice Date"
        elif "Order Date" in df.columns:
            preferred = "Order Date"
        else:
            preferred = df.columns[0]

    df["TxnDate"] = pd.to_datetime(df[preferred], errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)

    def classify_txn(x) -> str:
        s = str(x).lower()
        if "refund" in s or "return" in s:
            return "Refund"
        if "cancel" in s:
            return "Cancel"
        if "freereplacement" in s or "replacement" in s:
            return "FreeReplacement"
        return "Shipment"

    df["TxnType"] = df.get("Transaction Type", "").apply(classify_txn)

    # exclude replacements from sold qty if unchecked
    if not cfg.include_replacements:
        df.loc[df["TxnType"] == "FreeReplacement", "Quantity"] = 0

    df["Units_Effective"] = np.where(
        df["TxnType"] == "Refund",
        -df["Quantity"],
        np.where(df["TxnType"] == "Cancel", 0, df["Quantity"]),
    )

    df["Source"] = source

    # OrderId best effort
    oid_col = detect_order_id_col(df)
    df["OrderId"] = df[oid_col] if oid_col else np.nan

    out = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
    out.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
    out = out.dropna(subset=["TxnDate"])
    return out


def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_file, sheet_name="Sales Report")
        if df.empty:
            return pd.DataFrame()

        df["SKU_Clean"] = df["SKU"].astype(str).apply(_clean_sku)
        df["OMS_SKU"] = df["SKU_Clean"].apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"] = pd.to_datetime(df.get("Order Date"), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("Item Quantity", 0), errors="coerce").fillna(0)
        df["Source"] = "Flipkart"

        df["TxnType"] = df.get("Event Sub Type", "").apply(
            lambda x: "Refund" if "return" in str(x).lower() else "Shipment"
        )
        df["Units_Effective"] = np.where(df["TxnType"] == "Refund", -df["Quantity"], df["Quantity"])

        # OrderId
        if "Order ID" in df.columns:
            df["OrderId"] = df["Order ID"]
        elif "Order Id" in df.columns:
            df["OrderId"] = df["Order Id"]
        else:
            df["OrderId"] = np.nan

        out = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
        out.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
        return out.dropna(subset=["TxnDate"])
    except Exception:
        return pd.DataFrame()


def load_meesho_sales(zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """Meesho sales from ZIP (tcs_sales.xlsx)."""
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            excel_files = [f for f in z.namelist() if "tcs_sales" in f.lower() and f.lower().endswith(".xlsx") and "return" not in f.lower()]
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
        df["OrderId"] = df.get("order_id", np.nan)

        out = df[["OMS_SKU", "TxnDate", "TxnType", "Quantity", "Units_Effective", "Source", "OrderId"]].copy()
        out.columns = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]
        return out.dropna(subset=["TxnDate"])
    except Exception:
        return pd.DataFrame()


def load_stock_transfer(zip_file) -> pd.DataFrame:
    """Amazon stock transfer data (FC movements)."""
    df = read_zip_first_csv(zip_file)
    if df.empty:
        return pd.DataFrame()

    needed = ["Invoice Date", "Ship From Fc", "Ship To Fc", "Quantity", "Transaction Type"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    t = df[needed].copy()
    t["Invoice Date"] = pd.to_datetime(t["Invoice Date"], errors="coerce")
    t["Quantity"] = pd.to_numeric(t["Quantity"], errors="coerce").fillna(0)
    return t


def load_inventory_files(
    oms_csv,
    fk_csv,
    myntra_csv,
    amz_csv,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    Consolidated inventory:
    - OMS (Item SkuCode, Inventory)
    - Flipkart (SKU, Live on Website)
    - Myntra (seller sku code / sku code, sellable inventory count)
    - Amazon (MSKU, Ending Warehouse Balance) excluding ZNNE if present
    Grouped by Parent SKU.
    """
    dfs: List[pd.DataFrame] = []

    if oms_csv:
        d = read_csv_file(oms_csv)
        if not d.empty and {"Item SkuCode", "Inventory"}.issubset(d.columns):
            d = d.rename(columns={"Item SkuCode": "OMS_SKU", "Inventory": "OMS_Stock"})
            d["OMS_SKU"] = d["OMS_SKU"].astype(str)
            dfs.append(d[["OMS_SKU", "OMS_Stock"]].groupby("OMS_SKU", as_index=False).sum())

    if fk_csv:
        d = read_csv_file(fk_csv)
        if not d.empty and {"SKU", "Live on Website"}.issubset(d.columns):
            d["OMS_SKU"] = d["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            fk = d.groupby("OMS_SKU", as_index=False)["Live on Website"].sum().rename(columns={"Live on Website": "FK_Stock"})
            dfs.append(fk)

    if myntra_csv:
        d = read_csv_file(myntra_csv)
        if not d.empty:
            sku_col = None
            for c in d.columns:
                cl = str(c).lower()
                if cl in ["seller sku code", "sku code"] or ("sku" in cl and "code" in cl):
                    sku_col = c
                    break
            inv_col = None
            for c in d.columns:
                if str(c).lower().strip() == "sellable inventory count":
                    inv_col = c
                    break

            if sku_col and inv_col:
                d["OMS_SKU"] = d[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                myn = d.groupby("OMS_SKU", as_index=False)[inv_col].sum().rename(columns={inv_col: "MYN_Stock"})
                dfs.append(myn)

    if amz_csv:
        d = read_csv_file(amz_csv)
        if not d.empty and {"MSKU", "Ending Warehouse Balance"}.issubset(d.columns):
            if "Location" in d.columns:
                d = d[d["Location"] != "ZNNE"]
            d["OMS_SKU"] = d["MSKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            amz = d.groupby("OMS_SKU", as_index=False)["Ending Warehouse Balance"].sum().rename(columns={"Ending Warehouse Balance": "AMZ_Stock"})
            dfs.append(amz)

    if not dfs:
        return pd.DataFrame()

    final = dfs[0]
    for d in dfs[1:]:
        final = pd.merge(final, d, on="OMS_SKU", how="outer")

    stock_cols = [c for c in final.columns if c.endswith("_Stock")]
    final[stock_cols] = final[stock_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Parent grouping
    final["Parent"] = final["OMS_SKU"].apply(get_parent_sku)
    final = final.groupby("Parent", as_index=False)[stock_cols].sum().rename(columns={"Parent": "OMS_SKU"})
    final["Total_Inventory"] = final[stock_cols].sum(axis=1)

    return final[final["Total_Inventory"] > 0].copy()


# ==========================================================
# 5) SIDEBAR (UPLOADS + LOAD BUTTON)
# ==========================================================
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.markdown("## Yash ERP")

st.sidebar.divider()

map_file = st.sidebar.file_uploader("1) SKU Mapping (Required)", type=["xlsx"], help="e.g., Copy_of_All_penal_replace_sku.xlsx")

st.sidebar.markdown("### ⚙️ Amazon Sales Settings")
st.session_state.amazon_date_basis = st.sidebar.selectbox(
    "Amazon Date Basis",
    ["Shipment Date", "Invoice Date", "Order Date"],
    index=["Shipment Date", "Invoice Date", "Order Date"].index(st.session_state.amazon_date_basis)
    if st.session_state.amazon_date_basis in ["Shipment Date", "Invoice Date", "Order Date"]
    else 0,
)
st.session_state.include_replacements = st.sidebar.checkbox(
    "Include FreeReplacement in Sold Qty",
    value=bool(st.session_state.include_replacements),
)

st.sidebar.markdown("### 2) Sales Data")
f_b2c = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=["zip"])
f_b2b = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=["zip"])
f_transfer = st.sidebar.file_uploader("Amazon Stock Transfer (ZIP)", type=["zip"])
f_fk = st.sidebar.file_uploader("Flipkart Sales (Excel)", type=["xlsx"])
f_meesho = st.sidebar.file_uploader("Meesho Sales (ZIP)", type=["zip"])

st.sidebar.markdown("### 3) Inventory Data")
i_oms = st.sidebar.file_uploader("OMS Inventory (CSV)", type=["csv"])
i_fk = st.sidebar.file_uploader("Flipkart Inventory (CSV)", type=["csv"])
i_myntra = st.sidebar.file_uploader("Myntra Inventory (CSV)", type=["csv"])
i_amz = st.sidebar.file_uploader("Amazon Inventory (CSV)", type=["csv"])

st.sidebar.divider()


def run_load_pipeline():
    if not map_file:
        st.sidebar.error("Please upload SKU Mapping first.")
        return

    st.session_state.sku_mapping = load_sku_mapping(map_file)
    if not st.session_state.sku_mapping:
        st.sidebar.warning("Mapping loaded but looks empty. Check mapping file format.")
    else:
        st.sidebar.success(f"✅ Mapping loaded: {len(st.session_state.sku_mapping):,} rows")

    cfg = SalesLoadConfig(
        date_basis=st.session_state.amazon_date_basis,
        include_replacements=st.session_state.include_replacements,
    )

    sales_parts = []
    if f_b2c:
        sales_parts.append(load_amazon_sales(f_b2c, st.session_state.sku_mapping, "Amazon B2C", cfg))
    if f_b2b:
        sales_parts.append(load_amazon_sales(f_b2b, st.session_state.sku_mapping, "Amazon B2B", cfg))
    if f_fk:
        sales_parts.append(load_flipkart_sales(f_fk, st.session_state.sku_mapping))
    if f_meesho:
        sales_parts.append(load_meesho_sales(f_meesho, st.session_state.sku_mapping))

    if sales_parts:
        st.session_state.sales_df = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)
    else:
        st.session_state.sales_df = pd.DataFrame()

    st.session_state.inventory_df = load_inventory_files(i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping)

    st.session_state.transfer_df = load_stock_transfer(f_transfer) if f_transfer else pd.DataFrame()


if st.sidebar.button("🚀 Load Data"):
    run_load_pipeline()
    st.rerun()


# ==========================================================
# 6) GUARD (REQUIRE MAPPING)
# ==========================================================
if not st.session_state.sku_mapping:
    st.info("👋 Upload **SKU Mapping** and click **Load Data** to initialize the system.")
    st.stop()


# ==========================================================
# 7) TABS
# ==========================================================
tab_dashboard, tab_inventory, tab_po, tab_production, tab_forecast, tab_drill = st.tabs(
    ["📊 Dashboard", "📦 Inventory", "🎯 PO Engine", "🏭 Production", "📈 AI Forecast", "🔍 Deep Drilldown"]
)

# ----------------------------------------------------------
# TAB 1: DASHBOARD (FIXED METRICS: SUM QUANTITY, RETURN%)
# ----------------------------------------------------------
with tab_dashboard:
    df = st.session_state.sales_df

    if df.empty:
        st.warning("⚠️ No sales data loaded. Upload sales files and click Load Data.")
    else:
        df = df.copy()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        df["Units_Effective"] = pd.to_numeric(df["Units_Effective"], errors="coerce").fillna(0)

        sold_pcs = df[df["Transaction Type"] == "Shipment"]["Quantity"].sum()
        ret_pcs = df[df["Transaction Type"] == "Refund"]["Quantity"].sum()
        net_units = df["Units_Effective"].sum()

        # Orders: prefer OrderId unique if present
        if "OrderId" in df.columns:
            ship_orders = df[df["Transaction Type"] == "Shipment"]["OrderId"].nunique()
            orders = int(ship_orders) if ship_orders and not pd.isna(ship_orders) else int(len(df[df["Transaction Type"] == "Shipment"]))
        else:
            orders = int(len(df[df["Transaction Type"] == "Shipment"]))

        rate = (ret_pcs / sold_pcs * 100) if sold_pcs > 0 else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🛒 Orders", f"{orders:,}")
        k2.metric("✅ Sold Pieces", f"{int(sold_pcs):,}")
        k3.metric("↩️ Return Pieces", f"{int(ret_pcs):,}")
        k4.metric("📊 Return %", f"{rate:.1f}%")
        k5.metric("📦 Net Units", f"{int(net_units):,}")

        st.info(
            f"Amazon Date Basis: **{st.session_state.amazon_date_basis}** | "
            f"Include FreeReplacement: **{st.session_state.include_replacements}**"
        )

        st.divider()

        # Smart Narrative
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### 🧠 Smart Narrative")
            max_d = pd.to_datetime(df["TxnDate"]).max()
            if pd.isna(max_d):
                max_d = datetime.now()

            last7 = df[df["TxnDate"] > (max_d - timedelta(7))]["Units_Effective"].sum()
            prev7 = df[(df["TxnDate"] <= (max_d - timedelta(7))) & (df["TxnDate"] > (max_d - timedelta(14)))]["Units_Effective"].sum()

            if prev7 != 0:
                g = ((last7 - prev7) / abs(prev7)) * 100
                st.markdown(
                    f'<div class="ai-box">🚀 <b>Velocity:</b> Net units are <b>{g:.1f}%</b> '
                    f'{"up" if g > 0 else "down"} vs previous 7 days.</div>',
                    unsafe_allow_html=True,
                )

            # top SKU by net units
            top_sku = df.groupby("Sku")["Units_Effective"].sum().sort_values(ascending=False)
            if len(top_sku) > 0:
                st.markdown(
                    f'<div class="ai-box">🏆 <b>Top Seller:</b> SKU <b>{top_sku.index[0]}</b> is leading net units.</div>',
                    unsafe_allow_html=True,
                )

        with c2:
            src = df.groupby("Source")["Quantity"].sum().reset_index()
            if not src.empty:
                fig = px.pie(src, values="Quantity", names="Source", hole=0.4)
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🏆 Top 20 Selling SKUs (Pieces)")
        top_skus = (
            df[df["Transaction Type"] == "Shipment"]
            .groupby("Sku")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
            .rename(columns={"Quantity": "Sold_Pcs"})
        )
        fig = px.bar(top_skus, x="Sku", y="Sold_Pcs", title="Top 20 Sellers (Pieces)")
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------
# TAB 2: INVENTORY
# ----------------------------------------------------------
with tab_inventory:
    inv = st.session_state.inventory_df

    if inv.empty:
        st.warning("No inventory data loaded.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total SKUs", f"{len(inv):,}")
        c2.metric("Total Stock", f"{inv['Total_Inventory'].sum():,.0f}")

        search = st.text_input("🔍 Search Inventory SKU", placeholder="Type to filter...")
        show = inv.copy()
        if search:
            show = show[show["OMS_SKU"].astype(str).str.contains(search, case=False, na=False)]

        st.dataframe(show, use_container_width=True, height=520)


# ----------------------------------------------------------
# TAB 3: PO ENGINE (CORRECTIONS MERGED)
# - Lead time up to 180 (default 90)
# - Target buffer up to 180 (default 0)
# - Safety up to 100 (default 0)
# - Velocity modes: 7 / 30 / Full
# - Search filter added
# - Uses Net sold (Units_Effective) and current Total_Inventory
# ----------------------------------------------------------
with tab_po:
    st.subheader("🎯 Purchase Order Engine")

    if st.session_state.sales_df.empty:
        st.error("Sales data required for POs.")
    elif st.session_state.inventory_df.empty:
        st.error("Inventory required for POs.")
    else:
        sales = st.session_state.sales_df.copy()
        inv = st.session_state.inventory_df.copy()

        c1, c2, c3, c4 = st.columns(4)
        v_mode = c1.selectbox("Velocity Mode", ["Last 7 Days", "Last 30 Days", "Full History"])
        lead_time = c2.number_input("Lead Time / Production Days", 1, 180, 90)
        tgt = c3.number_input("Target Buffer (Days)", 0, 180, 0)
        safe = c4.slider("Safety Stock %", 0, 100, 0)

        st.divider()
        po_search = st.text_input("🔍 Filter PO by SKU", placeholder="Type SKU to find in list...")

        max_d = pd.to_datetime(sales["TxnDate"]).max()
        min_d = pd.to_datetime(sales["TxnDate"]).min()
        if pd.isna(max_d) or pd.isna(min_d):
            st.warning("Sales dates look invalid. Please check TxnDate parsing.")
            st.stop()

        if "7" in v_mode:
            days = 7
        elif "30" in v_mode:
            days = 30
        else:
            days = max((max_d - min_d).days, 1)

        start = max_d - timedelta(days=days)
        recent = sales[sales["TxnDate"] >= start].copy()

        stats = recent.groupby("Sku", as_index=False)["Units_Effective"].sum().rename(columns={"Sku": "OMS_SKU", "Units_Effective": "Net_Sold"})
        po = pd.merge(inv, stats, on="OMS_SKU", how="left").fillna(0)

        po["ADS"] = po["Net_Sold"] / float(days)
        po["Days_Cover"] = np.where(po["ADS"] > 0, po["Total_Inventory"] / po["ADS"], 999)

        po["Required"] = po["ADS"] * (tgt + lead_time) * (1 + safe / 100.0)
        po["Order_Qty"] = (po["Required"] - po["Total_Inventory"]).clip(lower=0).round(0).astype(int)

        def get_status(row):
            if row["Order_Qty"] > 0:
                return "🔴 Critical" if row["Days_Cover"] < lead_time else "🟡 Reorder"
            return "🟢 OK"

        po["Status"] = po.apply(get_status, axis=1)

        if po_search:
            po = po[po["OMS_SKU"].astype(str).str.contains(po_search, case=False, na=False)]

        final = po[po["Order_Qty"] > 0].sort_values("Order_Qty", ascending=False)

        st.markdown(f"**{len(final)} SKUs need attention**")

        def color_rows(row):
            if "Critical" in row["Status"]:
                return ["background-color: #fee2e2"] * len(row)
            return ["background-color: #fef3c7"] * len(row)

        cols = ["Status", "OMS_SKU", "Total_Inventory", "ADS", "Days_Cover", "Order_Qty"]
        st.dataframe(
            final[cols]
            .style.apply(color_rows, axis=1)
            .format("{:.2f}", subset=["ADS", "Days_Cover"]),
            use_container_width=True,
            height=600,
        )

        csv = final[cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download PO CSV", csv, f"po_recommendations_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# ----------------------------------------------------------
# TAB 4: PRODUCTION (PLACEHOLDER TOOL)
# ----------------------------------------------------------
with tab_production:
    st.subheader("🏭 Fabric Planning")
    st.info("Define BOM in 'Item Master' first (Future Module). For now, use the calculator below.")

    with st.form("fabric_calc"):
        qty = st.number_input("PCS to Make", min_value=0, value=500)
        cons = st.number_input("Consumption (Mtr)", min_value=0.0, value=2.5, step=0.1)
        st.markdown(f"**Total Fabric Needed:** {qty * cons:,.2f} Mtr")
        st.form_submit_button("Calculate")


# ----------------------------------------------------------
# TAB 5: AI FORECAST (Prophet)
# ----------------------------------------------------------
with tab_forecast:
    st.subheader("📈 AI Prediction Engine")

    sales = st.session_state.sales_df
    if sales.empty:
        st.warning("Upload sales data to generate forecasts.")
    else:
        sku = st.selectbox("Select SKU", [""] + sorted(sales["Sku"].dropna().astype(str).unique().tolist()))
        fc_days = st.slider("Forecast Period (Days)", 7, 90, 30)

        if sku:
            subset = sales[sales["Sku"].astype(str) == str(sku)].copy()
            subset["ds"] = pd.to_datetime(subset["TxnDate"]).dt.date
            daily = subset.groupby("ds", as_index=False)["Units_Effective"].sum()
            daily.columns = ["ds", "y"]
            daily["ds"] = pd.to_datetime(daily["ds"])

            if len(daily) < 14:
                st.warning("Need at least 14 days of data for forecasting.")
            else:
                try:
                    with st.spinner("Forecasting..."):
                        m = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=False,
                            changepoint_prior_scale=0.05,
                        )
                        m.fit(daily)
                        future = m.make_future_dataframe(periods=fc_days)
                        fcst = m.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual", mode="lines+markers"))
                    future_only = fcst[fcst["ds"] > daily["ds"].max()]
                    fig.add_trace(go.Scatter(x=future_only["ds"], y=future_only["yhat"], name="Forecast", mode="lines", line=dict(dash="dash")))
                    fig.add_trace(
                        go.Scatter(
                            x=list(future_only["ds"]) + list(future_only["ds"][::-1]),
                            y=list(future_only["yhat_upper"]) + list(future_only["yhat_lower"][::-1]),
                            fill="toself",
                            name="Confidence Interval",
                        )
                    )
                    fig.update_layout(title=f"Demand Forecast: {sku}", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                    total_pred = float(future_only["yhat"].sum())
                    st.success(f"🤖 Predicted Demand (Next {fc_days} Days): **{int(round(total_pred))} units**")

                except Exception as e:
                    st.error(f"Forecast error: {e}")


# ----------------------------------------------------------
# TAB 6: DEEP DRILLDOWN (PANEL-WISE)
# - search SKU (contains)
# - show sold/returns/net + panel pivots + recent txns
# ----------------------------------------------------------
with tab_drill:
    st.subheader("🔍 Deep Dive & Panel Analysis")

    df = st.session_state.sales_df
    if df.empty:
        st.warning("Upload sales data for drilldown.")
    else:
        colA, colB = st.columns([3, 1])
        with colA:
            drill_sku = st.text_input("Enter SKU for Drilldown", placeholder="e.g. 1065YK")
        with colB:
            period = st.selectbox("Time Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"], index=1)

        fdf = df.copy()
        max_d = pd.to_datetime(fdf["TxnDate"]).max()
        if period != "All Time" and not pd.isna(max_d):
            days = 7 if "7" in period else (30 if "30" in period else 90)
            fdf = fdf[fdf["TxnDate"] >= (max_d - timedelta(days=days))]

        if drill_sku:
            matches = fdf[fdf["Sku"].astype(str).str.contains(drill_sku, case=False, na=False)].copy()

            if matches.empty:
                st.warning("No matching SKU found.")
            else:
                sold = matches[matches["Transaction Type"] == "Shipment"]["Quantity"].sum()
                ret = matches[matches["Transaction Type"] == "Refund"]["Quantity"].sum()
                net = matches["Units_Effective"].sum()
                rr = (ret / sold * 100) if sold > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sold Pieces", f"{int(sold):,}")
                m2.metric("Return Pieces", f"{int(ret):,}")
                m3.metric("Net Units", f"{int(net):,}")
                m4.metric("Return %", f"{rr:.1f}%")

                st.divider()

                st.markdown("### 🏪 Marketplace Breakdown (Panel-wise)")
                ship_p, ret_p, net_p = build_panel_pivots(matches)

                combined = pd.concat([ship_p, ret_p, net_p], axis=1).fillna(0)
                combined = combined.reset_index().rename(columns={"Sku": "SKU"})
                st.dataframe(combined, use_container_width=True, height=420)

                st.markdown("### 📈 Panel Chart (Sold Pieces)")
                if not ship_p.empty:
                    chart_df = ship_p.sum(axis=0).reset_index()
                    chart_df.columns = ["Panel", "Sold_Pcs"]
                    chart_df["Panel"] = chart_df["Panel"].str.replace(" | Sold", "", regex=False)
                    fig = px.bar(chart_df.sort_values("Sold_Pcs", ascending=False), x="Panel", y="Sold_Pcs", title="Pieces Sold per Panel")
                    fig.update_xaxes(tickangle=-25)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📜 Recent Transactions")
                cols = ["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source"]
                if "OrderId" in matches.columns:
                    cols.append("OrderId")
                st.dataframe(matches.sort_values("TxnDate", ascending=False).head(50)[cols], use_container_width=True, height=420)

        else:
            st.info("Type SKU in the box above to start drilldown.")


st.divider()
st.caption("💡 All data displayed in OMS SKU format | Refactored ERP (Fixed metrics + Panel-wise drilldown + PO Engine updates)")

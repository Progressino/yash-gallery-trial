#!/usr/bin/env python3
"""
Yash Gallery ERP — FULL app.py (v4.1 ADVERTISING COST SUPPORT)
---------------------------------------------------------------
✅ NEW: Custom Unified Transaction support with advertising costs
✅ NEW: Smart advertising cost allocation to orders
✅ NEW: True Net Profit (Revenue - COGS - Fees - Advertising)
✅ B2B/B2C Sales Report Support (Amazon MTR format)
✅ Order Deep Dive Tab - Complete order-level P&L with payment tracking
✅ TRUE P&L FIX: Correct Ship_Qty (Orders - Refunds) without Amazon double counting
✅ SKU MAPPING AUDIT TAB with conflict detection
"""

import io
import zipfile
import warnings
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Handle Prophet optional import
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# App config / theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Yash Gallery ERP",
    page_icon="YG",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#002B5B"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp{background:linear-gradient(135deg,#f0f2f6 0%,#e8ecf4 100%);font-family:'DM Sans',sans-serif}
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e2e8f0}
div[data-testid="stMetric"]{background:#fff;padding:20px 18px;border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,.06);border-left:4px solid #002B5B}
div[data-testid="stMetric"] label{font-size:0.85rem !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{font-size:1.3rem !important;}
div.stButton>button{background:linear-gradient(135deg,#002B5B 0%,#0ea5e9 100%);color:#fff;border:none;border-radius:10px;font-weight:700}
h1,h2,h3{color:#002B5B!important;font-family:'DM Sans',sans-serif}
.sidebar-title{font-weight:800;color:#002B5B;margin:6px 0 2px 0}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background-color: white; border-radius: 6px; border: 1px solid #E5E7EB; padding: 8px 16px;}
.stTabs [aria-selected="true"] { background-color: #002B5B !important; color: white !important; }
.stDataFrame {font-size: 0.85rem !important;}
.stDataFrame th {font-size: 0.8rem !important; font-weight: 600 !important;}
.stDataFrame td {font-size: 0.85rem !important; padding: 4px 8px !important;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <span style="font-size:2.2rem;">YG</span>
  <div>
    <h1 style="margin:0;font-size:1.8rem;line-height:1.2;">Yash Gallery Command Center v4.1</h1>
    <p style="margin:0;color:#64748b;font-size:.9rem;">Advertising Cost · B2B/B2C Sales · Order Deep Dive · Finance Master · Mapping Audit</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
defaults = {
    "sku_mapping": {},
    "sku_map_table": pd.DataFrame(),
    "sku_map_conflicts": pd.DataFrame(),
    "sku_map_collisions": pd.DataFrame(),
    "cost_mapping": {},
    "fee_rules": pd.DataFrame(),
    "sales_df": pd.DataFrame(),
    "transfer_df": pd.DataFrame(),
    "inventory_df_variant": pd.DataFrame(),
    "inventory_df_parent": pd.DataFrame(),
    "recon_amazon": pd.DataFrame(),
    "recon_flipkart": pd.DataFrame(),
    "recon_meesho": pd.DataFrame(),
    "b2b_sales": pd.DataFrame(),
    "b2c_sales": pd.DataFrame(),
    "custom_unified": pd.DataFrame(),
    "finance_ledger": pd.DataFrame(),
    "order_deep_dive": pd.DataFrame(),
    "advertising_summary": pd.DataFrame(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\ufeff", "").replace('"', "") for c in df.columns]
    return df

def clean_sku(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().replace('"""', "").replace("SKU:", "").strip()

def get_parent_sku(oms_sku) -> str:
    if pd.isna(oms_sku):
        return ""
    s = str(oms_sku).strip()
    for suf in ["_Myntra", "_Flipkart", "_Amazon", "_Meesho", "_MYNTRA", "_FLIPKART", "_AMAZON", "_MEESHO"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2:
            last = parts[-1].upper().strip()
            sizes = {
                "XS","S","M","L","XL","XXL","XXXL","2XL","3XL","4XL","5XL","6XL","7XL","8XL",
                "XXS","FS","FREE"
            }
            is_size_like = last in sizes or last.endswith("XL") or last.isdigit()
            if is_size_like:
                s = "-".join(parts[:-1])
    return s

def parse_indian_number(val):
    if pd.isna(val):
        return 0.0
    try:
        return float(str(val).strip().replace(",", "").replace('"', ""))
    except Exception:
        return 0.0

def fmt_inr(val):
    if pd.isna(val) or float(val) == 0:
        return "₹0"
    return f"₹{float(val):,.0f}"

def make_tz_naive(series_or_val):
    if isinstance(series_or_val, pd.Series):
        return pd.to_datetime(series_or_val, errors="coerce").dt.tz_localize(None)
    return pd.to_datetime(series_or_val, errors="coerce").tz_localize(None) if pd.notna(series_or_val) else pd.NaT

def smart_read(file_obj, **kwargs):
    name = getattr(file_obj, "name", "").lower()
    file_obj.seek(0)
    if name.endswith(".zip"):
        with zipfile.ZipFile(file_obj, "r") as z:
            files = [f for f in z.namelist() if not f.startswith("__")]
            target = [f for f in files if f.lower().endswith(".csv")] or [f for f in files if f.lower().endswith((".xlsx", ".xls"))]
            if not target:
                return pd.DataFrame()
            member = target[0]
            with z.open(member) as f:
                data = f.read()
                if member.lower().endswith(".csv"):
                    try:
                        return pd.read_csv(io.BytesIO(data), **kwargs)
                    except Exception:
                        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", **kwargs)
                return pd.read_excel(io.BytesIO(data), **kwargs)
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file_obj, **kwargs)
        except Exception:
            return pd.read_csv(file_obj, sep=None, engine="python", **kwargs)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj, **kwargs)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# SKU Mapping Loader + Audit
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_sku_mapping_with_audit(mf) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = {}
    rows = []

    mf.seek(0)
    xls = pd.ExcelFile(mf)

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if df.empty or len(df.columns) < 2:
            continue

        s_col = next(
            (c for c in df.columns if "seller" in str(c).lower() and "sku" in str(c).lower()),
            next((c for c in df.columns if "sku" in str(c).lower() and "oms" not in str(c).lower()), df.columns[0]),
        )
        o_col = next((c for c in df.columns if "oms" in str(c).lower() and "sku" in str(c).lower()), df.columns[-1])

        tmp = df[[s_col, o_col]].copy()
        tmp.columns = ["seller_sku", "oms_sku"]
        tmp["seller_sku"] = tmp["seller_sku"].map(clean_sku)
        tmp["oms_sku"] = tmp["oms_sku"].map(clean_sku)
        tmp["sheet"] = sheet
        tmp = tmp[(tmp["seller_sku"] != "") & (tmp["oms_sku"] != "")]

        rows.append(tmp)

    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["seller_sku", "oms_sku", "sheet"])

    for r in table.itertuples(index=False):
        mapping[r.seller_sku] = r.oms_sku

    conflicts = (
        table.groupby("seller_sku")["oms_sku"].nunique().reset_index(name="oms_count")
    )
    conflicts = conflicts[conflicts["oms_count"] > 1].sort_values("oms_count", ascending=False)

    collisions = (
        table.groupby("oms_sku")["seller_sku"].nunique().reset_index(name="seller_count")
    )
    collisions = collisions.sort_values("seller_count", ascending=False)

    return mapping, table, conflicts, collisions

def map_to_oms(sku, mapping: Dict[str, str]) -> str:
    key = clean_sku(sku)
    return mapping.get(key, key)

# -----------------------------------------------------------------------------
# Custom Unified Transaction Loader - NEW
# -----------------------------------------------------------------------------
def load_custom_unified_transaction(file_obj, mapping):
    """
    Load Amazon Custom Unified Transaction report
    Handles advertising costs and allocates them to orders
    """
    try:
        file_obj.seek(0)
        
        # Find header row (skips description lines)
        header_row = 0
        for i, line in enumerate(file_obj):
            if "date/time" in str(line).lower():
                header_row = i
                break
        
        file_obj.seek(0)
        df = pd.read_csv(file_obj, skiprows=header_row, low_memory=False)
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        df = normalize_cols(df)
        
        # Parse dates
        df['txn_date'] = make_tz_naive(pd.to_datetime(df['date/time'].astype(str).str.replace(' utc', '', regex=False), errors='coerce'))
        
        # Parse numeric columns
        numeric_cols = [
            'quantity', 'product sales', 'shipping credits', 'gift wrap credits',
            'promotional rebates', 'total sales tax liable(gst before adjusting tcs)',
            'tcs-cgst', 'tcs-sgst', 'tcs-igst', 'tds (section 194-o)',
            'selling fees', 'fba fees', 'other transaction fees', 'other', 'total'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Clean SKU
        df['sku_clean'] = df.get('sku', '').map(clean_sku)
        df['oms_sku'] = df['sku_clean'].apply(lambda x: mapping.get(x, x) if x else '')
        
        # Separate advertising costs
        advertising = df[
            (df['type'].str.lower() == 'service fee') & 
            (df['description'].str.contains('advertising', case=False, na=False))
        ].copy()
        
        # Order/Refund transactions
        orders = df[
            df['type'].isin(['Order', 'Refund'])
        ].copy()
        
        return orders, advertising
        
    except Exception as e:
        st.error(f"Error loading Custom Unified Transaction: {e}")
        return pd.DataFrame(), pd.DataFrame()

# -----------------------------------------------------------------------------
# Advertising Cost Allocator - NEW
# -----------------------------------------------------------------------------
def allocate_advertising_costs(orders_df, advertising_df):
    """
    Allocate advertising costs to orders based on revenue proportion
    
    Strategy:
    1. Calculate total revenue from orders
    2. Calculate each order's revenue share
    3. Allocate advertising proportionally
    """
    if orders_df.empty or advertising_df.empty:
        return orders_df
    
    # Calculate total advertising cost
    total_ad_cost = advertising_df['other transaction fees'].abs().sum() + advertising_df['other'].abs().sum()
    
    if total_ad_cost == 0:
        orders_df['advertising_cost'] = 0
        return orders_df
    
    # Calculate revenue per order (using product sales as base)
    orders_df['order_revenue'] = orders_df['product sales'].abs()
    total_revenue = orders_df['order_revenue'].sum()
    
    if total_revenue == 0:
        # Fallback: distribute equally
        orders_df['advertising_cost'] = total_ad_cost / len(orders_df)
    else:
        # Proportional allocation
        orders_df['advertising_cost'] = (orders_df['order_revenue'] / total_revenue) * total_ad_cost
    
    return orders_df

# -----------------------------------------------------------------------------
# B2B/B2C Sales Report Loader
# -----------------------------------------------------------------------------
def load_b2b_b2c_sales(file_obj, mapping, report_type="B2C"):
    try:
        file_obj.seek(0)
        df = smart_read(file_obj, low_memory=False)
        if df.empty:
            return pd.DataFrame()
        
        df = normalize_cols(df)
        
        required = ['transaction type', 'order id', 'sku', 'quantity', 'invoice amount']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning(f"{report_type} report missing columns: {missing}")
            return pd.DataFrame()
        
        df['sku_clean'] = df['sku'].map(clean_sku)
        df['oms_sku'] = df['sku_clean'].apply(lambda x: mapping.get(x, x))
        
        date_cols = ['invoice date', 'shipment date', 'order date']
        for col in date_cols:
            if col in df.columns:
                df[col] = make_tz_naive(df[col])
        
        df['txn_date'] = df.get('invoice date', df.get('shipment date', df.get('order date')))
        
        numeric_cols = [
            'quantity', 'invoice amount', 'tax exclusive gross', 'total tax amount',
            'principal amount', 'shipping amount', 'gift wrap amount',
            'item promo discount', 'shipping promo discount'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        df['transaction_type'] = df['transaction type'].str.strip().str.title()
        df['report_type'] = report_type
        df['marketplace'] = 'Amazon'
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {report_type} report: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Helper: Panel Pivots
# -----------------------------------------------------------------------------
def build_panel_pivots(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    w = df.copy()
    w["Panel"] = w["Source"].astype(str)

    sold = w[w["Transaction Type"] == "Shipment"].groupby(["Sku", "Panel"])["Quantity"].sum().unstack(fill_value=0)
    sold.columns = [f"{c} | Sold" for c in sold.columns]

    ret = w[w["Transaction Type"] == "Refund"].groupby(["Sku", "Panel"])["Quantity"].sum().unstack(fill_value=0)
    ret.columns = [f"{c} | Return" for c in ret.columns]

    net = w.groupby(["Sku", "Panel"])["Units_Effective"].sum().unstack(fill_value=0)
    net.columns = [f"{c} | Net" for c in net.columns]

    return sold, ret, net

# -----------------------------------------------------------------------------
# Sales Fact Extractor
# -----------------------------------------------------------------------------
def extract_sales_fact(recon_amz, recon_fk, recon_ms, b2b_sales, b2c_sales, custom_unified):
    frames = []

    # CUSTOM UNIFIED TRANSACTION
    if custom_unified is not None and not custom_unified.empty:
        d = custom_unified.copy()
        
        d_dedup = d.groupby(["order id", "oms_sku", "type"], as_index=False).agg(
            quantity=("quantity", "max"),
            txn_date=("txn_date", "first"),
        )
        
        d_dedup["Sku"] = d_dedup["oms_sku"]
        d_dedup["TxnDate"] = make_tz_naive(d_dedup["txn_date"])
        d_dedup["Quantity"] = pd.to_numeric(d_dedup["quantity"], errors="coerce").fillna(0).abs()
        d_dedup["Transaction Type"] = np.where(d_dedup["type"].str.lower() == "refund", "Refund", "Shipment")
        d_dedup["Units_Effective"] = np.where(d_dedup["Transaction Type"] == "Refund", -d_dedup["Quantity"], d_dedup["Quantity"])
        d_dedup["Source"] = "Amazon Custom"
        d_dedup["OrderId"] = d_dedup["order id"]
        
        frames.append(d_dedup[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]])

    # B2B SALES
    if b2b_sales is not None and not b2b_sales.empty:
        d = b2b_sales.copy()
        d["Sku"] = d["oms_sku"]
        d["TxnDate"] = make_tz_naive(d["txn_date"])
        d["Quantity"] = pd.to_numeric(d.get("quantity", 0), errors="coerce").fillna(0).abs()
        d["Transaction Type"] = d["transaction_type"]
        d["Units_Effective"] = np.where(d["Transaction Type"] == "Refund", -d["Quantity"], d["Quantity"])
        d["Source"] = "Amazon B2B"
        d["OrderId"] = d.get("order id", np.nan)
        
        frames.append(d[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]])

    # B2C SALES
    if b2c_sales is not None and not b2c_sales.empty:
        d = b2c_sales.copy()
        d["Sku"] = d["oms_sku"]
        d["TxnDate"] = make_tz_naive(d["txn_date"])
        d["Quantity"] = pd.to_numeric(d.get("quantity", 0), errors="coerce").fillna(0).abs()
        d["Transaction Type"] = d["transaction_type"]
        d["Units_Effective"] = np.where(d["Transaction Type"] == "Refund", -d["Quantity"], d["Quantity"])
        d["Source"] = "Amazon B2C"
        d["OrderId"] = d.get("order id", np.nan)
        
        frames.append(d[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]])

    # FLIPKART
    if recon_fk is not None and not recon_fk.empty:
        d = recon_fk.copy()
        d.rename(columns={"oms_sku": "Sku"}, inplace=True)
        d["TxnDate"] = make_tz_naive(d["order_date"])
        d["Quantity"] = pd.to_numeric(d.get("quantity", 1), errors="coerce").fillna(1)
        d["Transaction Type"] = "Shipment"
        d["Units_Effective"] = d["Quantity"]
        d["Source"] = "Flipkart"
        d["OrderId"] = d.get("order_id", np.nan)
        frames.append(d[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]])

    # MEESHO
    if recon_ms is not None and not recon_ms.empty:
        d = recon_ms.copy()
        d.rename(columns={"oms_sku": "Sku"}, inplace=True)
        d["TxnDate"] = make_tz_naive(d["order_date"])
        d["Quantity"] = pd.to_numeric(d.get("quantity", 1), errors="coerce").fillna(1)
        d["Transaction Type"] = "Shipment"
        d["Units_Effective"] = d["Quantity"]
        d["Source"] = "Meesho"
        d["OrderId"] = d.get("sub_order_no", np.nan)
        frames.append(d[["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source", "OrderId"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["TxnDate"])

# -----------------------------------------------------------------------------
# Finance Ledger Builder
# -----------------------------------------------------------------------------
def build_finance_ledger_from_custom(custom_unified_orders, advertising_summary, cost_mapping):
    """
    Build finance ledger from Custom Unified Transaction
    Includes advertising cost allocation
    """
    if custom_unified_orders.empty:
        return pd.DataFrame()
    
    df = custom_unified_orders.copy()
    
    # Revenue columns
    revenue_cols = ['product sales', 'shipping credits', 'gift wrap credits', 'promotional rebates']
    df['total_revenue'] = 0.0
    for col in revenue_cols:
        if col in df.columns:
            df['total_revenue'] += pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fee columns
    fee_cols = ['selling fees', 'fba fees', 'other transaction fees']
    df['total_fees'] = 0.0
    for col in fee_cols:
        if col in df.columns:
            df['total_fees'] += pd.to_numeric(df[col], errors='coerce').fillna(0).abs()
    
    df['qty_abs'] = pd.to_numeric(df.get('quantity', 0), errors='coerce').fillna(0).abs()
    
    # Aggregate by order and SKU
    agg = df.groupby(['order id', 'oms_sku', 'type'], as_index=False).agg({
        'txn_date': 'min',
        'total_revenue': 'sum',
        'total': 'sum',  # Net payout
        'total_fees': 'sum',
        'qty_abs': 'max',
        'advertising_cost': 'sum'
    })
    
    # Pivot quantities by type
    qty_pivot = agg.pivot_table(
        index=['order id', 'oms_sku'],
        columns='type',
        values='qty_abs',
        aggfunc='max',
        fill_value=0
    ).reset_index()
    
    qty_pivot['Order_Qty'] = qty_pivot.get('Order', 0)
    qty_pivot['Refund_Qty'] = qty_pivot.get('Refund', 0)
    qty_pivot['Ship_Qty'] = qty_pivot['Order_Qty'] - qty_pivot['Refund_Qty']
    
    # Aggregate money by order and SKU (sum across types)
    money = df.groupby(['order id', 'oms_sku'], as_index=False).agg({
        'txn_date': 'min',
        'total_revenue': 'sum',
        'total': 'sum',
        'total_fees': 'sum',
        'advertising_cost': 'sum'
    })
    
    # Merge
    led = pd.merge(
        money,
        qty_pivot[['order id', 'oms_sku', 'Order_Qty', 'Refund_Qty', 'Ship_Qty']],
        on=['order id', 'oms_sku'],
        how='left'
    ).fillna(0)
    
    # Apply COGS
    led['Unit_Cost'] = led['oms_sku'].map(cost_mapping).fillna(0.0)
    led['COGS'] = led['Ship_Qty'] * led['Unit_Cost']
    
    # Calculate profits
    led['Gross_Profit'] = led['total'] - led['COGS']  # After fees, before advertising
    led['Net_Profit'] = led['Gross_Profit'] - led['advertising_cost']  # TRUE profit
    
    # Rename
    led.rename(columns={
        'order id': 'OrderId',
        'oms_sku': 'SKU',
        'txn_date': 'TxnDate',
        'total_revenue': 'Revenue',
        'total': 'Net_Payout',
        'total_fees': 'Fees',
        'advertising_cost': 'Ad_Cost'
    }, inplace=True)
    
    led['Marketplace'] = 'Amazon Custom'
    led['Parent_SKU'] = led['SKU'].apply(get_parent_sku)
    
    return led[['OrderId', 'TxnDate', 'SKU', 'Parent_SKU', 'Ship_Qty', 'Refund_Qty', 
                'Revenue', 'Net_Payout', 'Fees', 'Ad_Cost', 'COGS', 'Gross_Profit', 'Net_Profit', 'Marketplace']]

# -----------------------------------------------------------------------------
# Order Deep Dive Builder - Enhanced with Advertising
# -----------------------------------------------------------------------------
def build_order_deep_dive_from_custom(custom_unified_orders, cost_mapping):
    """
    Build order-level P&L from Custom Unified Transaction
    Includes advertising cost per order with detailed fee breakdown
    """
    if custom_unified_orders.empty:
        return pd.DataFrame()
    
    df = custom_unified_orders.copy()
    
    result = pd.DataFrame()
    result['Order_ID'] = df['order id']
    result['Settlement_ID'] = df.get('settlement id', '')
    result['Transaction_Date'] = df['txn_date']
    result['Transaction_Type'] = df['type'].str.title()
    result['SKU'] = df['oms_sku']
    result['Seller_SKU'] = df['sku_clean']
    result['Description'] = df.get('description', '')
    result['Quantity'] = df['quantity'].abs()
    
    # Financial columns - DETAILED
    result['Product_Sales'] = df.get('product sales', 0)
    result['Shipping_Credits'] = df.get('shipping credits', 0)
    result['Gift_Wrap_Credits'] = df.get('gift wrap credits', 0)
    result['Promotional_Rebates'] = df.get('promotional rebates', 0)
    result['Total_Tax'] = df.get('total sales tax liable(gst before adjusting tcs)', 0)
    
    # Revenue
    result['Gross_Revenue'] = (
        result['Product_Sales'] + 
        result['Shipping_Credits'] + 
        result['Gift_Wrap_Credits']
    )
    result['Net_Revenue'] = result['Gross_Revenue'] + result['Promotional_Rebates']
    
    # Fees - DETAILED BREAKDOWN
    result['Selling_Fees'] = df.get('selling fees', 0).abs()
    result['FBA_Fees'] = df.get('fba fees', 0).abs()
    result['Other_Fees'] = df.get('other transaction fees', 0).abs()
    result['Total_Fees'] = result['Selling_Fees'] + result['FBA_Fees'] + result['Other_Fees']
    
    # Fee Percentages - NEW
    result['Selling_Fee_Pct'] = ((result['Selling_Fees'] / result['Net_Revenue']) * 100).round(2)
    result['FBA_Fee_Pct'] = ((result['FBA_Fees'] / result['Net_Revenue']) * 100).round(2)
    result['Total_Fee_Pct'] = ((result['Total_Fees'] / result['Net_Revenue']) * 100).round(2)
    
    # Replace inf/nan with 0
    result['Selling_Fee_Pct'] = result['Selling_Fee_Pct'].replace([float('inf'), -float('inf')], 0).fillna(0)
    result['FBA_Fee_Pct'] = result['FBA_Fee_Pct'].replace([float('inf'), -float('inf')], 0).fillna(0)
    result['Total_Fee_Pct'] = result['Total_Fee_Pct'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Advertising - WITH REVENUE SHARE EXPLANATION
    result['Order_Revenue_Share'] = df.get('order_revenue', 0)  # Revenue used for allocation
    result['Advertising_Cost'] = df.get('advertising_cost', 0)
    
    # Calculate ad allocation percentage for transparency
    total_order_revenue = df.get('order_revenue', pd.Series([0])).sum()
    if total_order_revenue > 0:
        result['Ad_Allocation_Pct'] = (result['Order_Revenue_Share'] / total_order_revenue * 100).round(2)
    else:
        result['Ad_Allocation_Pct'] = 0
    
    # Net Payout
    result['Net_Payout'] = df.get('total', 0)
    
    # COGS
    result['Unit_Cost'] = result['SKU'].map(cost_mapping).fillna(0.0)
    result['COGS'] = result['Quantity'] * result['Unit_Cost']
    
    # Profitability
    result['Gross_Profit'] = result['Net_Payout'] - result['COGS']  # After fees, before ad
    result['Net_Profit'] = result['Gross_Profit'] - result['Advertising_Cost']  # TRUE profit
    
    # Profit Margins
    result['Gross_Margin_Pct'] = ((result['Gross_Profit'] / result['Net_Revenue']) * 100).round(2)
    result['Net_Margin_Pct'] = ((result['Net_Profit'] / result['Net_Revenue']) * 100).round(2)
    result['Gross_Margin_Pct'] = result['Gross_Margin_Pct'].replace([float('inf'), -float('inf')], 0).fillna(0)
    result['Net_Margin_Pct'] = result['Net_Margin_Pct'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Metadata
    result['City'] = df.get('order city', '')
    result['State'] = df.get('order state', '')
    result['Postal'] = df.get('order postal', '')
    result['Fulfillment'] = df.get('fulfillment', '')
    result['Account_Type'] = df.get('account type', '')
    
    # Handle refunds
    mask_refund = result['Transaction_Type'] == 'Refund'
    result.loc[mask_refund, 'Quantity'] = -result.loc[mask_refund, 'Quantity']
    
    result['Parent_SKU'] = result['SKU'].apply(get_parent_sku)
    
    return result

# -----------------------------------------------------------------------------
# Flipkart/Meesho Loaders (unchanged)
# -----------------------------------------------------------------------------
def load_flipkart_recon(file_obj, mapping):
    try:
        file_obj.seek(0)
        df_raw = pd.read_excel(file_obj, sheet_name="Orders", header=[0, 1])
        if df_raw.empty:
            return pd.DataFrame()
        df_raw.columns = [" | ".join([str(c).strip() for c in col if "Unnamed" not in str(c)]) for col in df_raw.columns]

        def _find(patterns):
            for p in patterns:
                for c in df_raw.columns:
                    if p.lower() in c.lower():
                        return c
            return None

        df = pd.DataFrame()
        df["order_id"] = df_raw.get(_find(["Order ID"]), np.nan)
        df["seller_sku"] = df_raw.get(_find(["Seller SKU"]), np.nan)
        df["settlement_amount"] = pd.to_numeric(df_raw.get(_find(["Bank Settlement Value"]), 0), errors="coerce").fillna(0)
        df["sale_amount"] = pd.to_numeric(df_raw.get(_find(["Sale Amount"]), 0), errors="coerce").fillna(0)
        df["order_date"] = make_tz_naive(df_raw.get(_find(["Order Date"])))
        df["quantity"] = pd.to_numeric(df_raw.get(_find(["Quantity"]), 1), errors="coerce").fillna(1)
        df["oms_sku"] = df["seller_sku"].apply(lambda x: map_to_oms(x, mapping))
        df["marketplace"] = "Flipkart"
        return df.dropna(subset=["order_id"])
    except Exception:
        return pd.DataFrame()

def load_meesho_recon(file_obj, mapping):
    try:
        file_obj.seek(0)
        df_raw = pd.read_excel(file_obj, sheet_name="Order Payments", header=[0, 1])
        if df_raw.empty:
            return pd.DataFrame()
        df_raw.columns = [" | ".join([str(c).strip() for c in col if "Unnamed" not in str(c)]) for col in df_raw.columns]
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

        def _find(patterns):
            for p in patterns:
                for c in df_raw.columns:
                    if p.lower() in c.lower():
                        return c
            return None

        df = pd.DataFrame()
        df["sub_order_no"] = df_raw.get(_find(["Sub Order No"]), np.nan)
        df["supplier_sku"] = df_raw.get(_find(["Supplier SKU"]), np.nan)
        df["order_date"] = make_tz_naive(df_raw.get(_find(["Order Date"])))
        df["quantity"] = pd.to_numeric(df_raw.get(_find(["Quantity"]), 1), errors="coerce").fillna(1)
        df["settlement_amount"] = pd.to_numeric(df_raw.get(_find(["Final Settlement Amount"]), 0), errors="coerce").fillna(0)
        df["sale_amount"] = pd.to_numeric(df_raw.get(_find(["Total Sale Amount"]), 0), errors="coerce").fillna(0)
        df["oms_sku"] = df["supplier_sku"].apply(lambda x: map_to_oms(x, mapping))
        df["marketplace"] = "Meesho"
        return df.dropna(subset=["sub_order_no"])
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.write("Yash Gallery")

    st.markdown('<div class="sidebar-title">Command Center</div>', unsafe_allow_html=True)

    with st.expander("1. Master Data", expanded=True):
        u_map = st.file_uploader("SKU Mapping", type=["xlsx", "xls"])
        u_cost = st.file_uploader("Cost Master (COGS)", type=["xlsx", "csv"])

    with st.expander("2. Amazon Reports", expanded=True):
        u_custom = st.file_uploader("Custom Unified Transaction", type=["csv"], accept_multiple_files=True, 
                                    help="Includes advertising costs")
        u_b2b = st.file_uploader("B2B Sales Report (MTR)", type=["csv", "zip"], accept_multiple_files=True)
        u_b2c = st.file_uploader("B2C Sales Report (MTR)", type=["csv", "zip"], accept_multiple_files=True)
    
    with st.expander("3. Other Marketplaces", expanded=False):
        u_fk = st.file_uploader("Flipkart", type=["xlsx"], accept_multiple_files=True)
        u_ms = st.file_uploader("Meesho", type=["xlsx"], accept_multiple_files=True)

    if st.button("RUN / REFRESH", type="primary", use_container_width=True):
        if not u_map:
            st.error("Mapping File Required!")
        else:
            with st.spinner("Processing..."):
                # Load mapping
                mapping, table, conflicts, collisions = load_sku_mapping_with_audit(u_map)
                st.session_state.sku_mapping = mapping
                st.session_state.sku_map_table = table
                st.session_state.sku_map_conflicts = conflicts
                st.session_state.sku_map_collisions = collisions

                # Load cost
                if u_cost:
                    df = normalize_cols(smart_read(u_cost))
                    if not df.empty:
                        sku_col = next((c for c in df.columns if "sku" in c), df.columns[0])
                        cost_col = next((c for c in df.columns if any(k in c for k in ["pwn", "cost", "price", "rate", "cogs"])), None)
                        if cost_col:
                            cm = (
                                df[[sku_col, cost_col]]
                                .dropna()
                                .assign(**{sku_col: lambda x: x[sku_col].map(clean_sku)})
                            )
                            st.session_state.cost_mapping = dict(zip(cm[sku_col], pd.to_numeric(cm[cost_col], errors="coerce").fillna(0).astype(float)))

                # Load Custom Unified Transaction
                if u_custom:
                    custom_orders_list = []
                    custom_ads_list = []
                    
                    for f in u_custom:
                        orders, ads = load_custom_unified_transaction(f, mapping)
                        custom_orders_list.append(orders)
                        custom_ads_list.append(ads)
                    
                    custom_orders = pd.concat(custom_orders_list, ignore_index=True) if custom_orders_list else pd.DataFrame()
                    custom_ads = pd.concat(custom_ads_list, ignore_index=True) if custom_ads_list else pd.DataFrame()
                    
                    # Allocate advertising costs
                    if not custom_orders.empty and not custom_ads.empty:
                        custom_orders = allocate_advertising_costs(custom_orders, custom_ads)
                    elif not custom_orders.empty:
                        custom_orders['advertising_cost'] = 0
                    
                    st.session_state.custom_unified = custom_orders
                    st.session_state.advertising_summary = custom_ads
                else:
                    st.session_state.custom_unified = pd.DataFrame()
                    st.session_state.advertising_summary = pd.DataFrame()

                # Load B2B/B2C
                b2b_dfs = [load_b2b_b2c_sales(f, mapping, "B2B") for f in (u_b2b or [])]
                st.session_state.b2b_sales = pd.concat(b2b_dfs, ignore_index=True) if b2b_dfs else pd.DataFrame()
                
                b2c_dfs = [load_b2b_b2c_sales(f, mapping, "B2C") for f in (u_b2c or [])]
                st.session_state.b2c_sales = pd.concat(b2c_dfs, ignore_index=True) if b2c_dfs else pd.DataFrame()

                # Load Flipkart/Meesho
                fk_dfs = [load_flipkart_recon(f, mapping) for f in (u_fk or [])]
                st.session_state.recon_flipkart = pd.concat(fk_dfs, ignore_index=True) if fk_dfs else pd.DataFrame()

                ms_dfs = [load_meesho_recon(f, mapping) for f in (u_ms or [])]
                st.session_state.recon_meesho = pd.concat(ms_dfs, ignore_index=True) if ms_dfs else pd.DataFrame()

                # Build sales fact
                st.session_state.sales_df = extract_sales_fact(
                    None,  # No old settlement format
                    st.session_state.recon_flipkart, 
                    st.session_state.recon_meesho,
                    st.session_state.b2b_sales,
                    st.session_state.b2c_sales,
                    st.session_state.custom_unified
                )

                # Build finance ledger
                st.session_state.finance_ledger = build_finance_ledger_from_custom(
                    st.session_state.custom_unified,
                    st.session_state.advertising_summary,
                    st.session_state.cost_mapping
                )
                
                # Build order deep dive
                st.session_state.order_deep_dive = build_order_deep_dive_from_custom(
                    st.session_state.custom_unified,
                    st.session_state.cost_mapping
                )

                st.success("✅ System Updated Successfully!")
                st.rerun()

if not st.session_state.sku_mapping:
    st.info("Please upload SKU Mapping and click RUN.")
    st.stop()

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_dash, tab_order, tab_fin, tab_drill, tab_audit, tab_ad = st.tabs(
    ["📊 Dashboard", "🔬 Order Deep Dive", "💰 Finance P&L", "🔍 Deep Dive", "🧩 Mapping Audit", "📢 Advertising"]
)

# --- DASHBOARD ---
with tab_dash:
    df = st.session_state.sales_df
    if df.empty:
        st.info("No Sales Data.")
    else:
        c1, c2 = st.columns([3, 1])
        period = c1.selectbox("Period", ["Last 7 Days", "Last 30 Days", "Last 60 Days", "All Time"], index=1)
        grace = c2.number_input("Grace Period (Days)", 0, 14, 7)

        df2 = df.copy()
        max_d = df2["TxnDate"].max()
        if period != "All Time" and pd.notna(max_d):
            days = 7 if "7" in period else 30 if "30" in period else 60
            df2 = df2[df2["TxnDate"] >= max_d - timedelta(days=days + grace)]

        sold = df2[df2["Transaction Type"] == "Shipment"]["Quantity"].sum()
        ret = df2[df2["Transaction Type"] == "Refund"]["Quantity"].sum()
        net = df2["Units_Effective"].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Orders", f"{df2['OrderId'].nunique():,}")
        k2.metric("Units Sold", f"{int(sold):,}")
        k3.metric("Returns", f"{int(ret):,}")
        k4.metric("Net Units", f"{int(net):,}")

        st.divider()
        
        st.markdown("### 📦 Data Source Summary")
        source_summary = df2.groupby('Source').agg(
            Orders=('OrderId', 'nunique'),
            Units=('Quantity', 'sum'),
            Net_Units=('Units_Effective', 'sum')
        ).reset_index()
        st.dataframe(source_summary, use_container_width=True, height=200)
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown("### 🏆 Top 15 SKUs by Quantity")
            top = df2[df2["Transaction Type"] == "Shipment"].groupby("Sku")["Quantity"].sum().nlargest(15).reset_index()
            fig = px.bar(top, x="Sku", y="Quantity", color_discrete_sequence=[NAVY])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with c_right:
            st.markdown("### 📊 Channel Mix")
            src = df2[df2["Transaction Type"] == "Shipment"].groupby("Source")["Quantity"].sum().reset_index()
            fig = px.pie(src, values="Quantity", names="Source", hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# --- ORDER DEEP DIVE ---
with tab_order:
    st.markdown("### 🔬 Order-Level P&L Deep Dive")
    st.caption("Complete breakdown with fees, advertising, and profit margins")
    
    odd = st.session_state.order_deep_dive
    
    if odd.empty:
        st.warning("No Custom Unified Transaction data loaded.")
        st.info("Upload Custom Unified Transaction CSV in sidebar and click RUN.")
    else:
        st.success(f"✅ Loaded {len(odd):,} orders with complete P&L data")
        
        # View selector
        view_type = st.radio(
            "Select View",
            ["💰 Financial View (Key Metrics)", "📊 Detailed View (All Columns)", "📈 Percentage Analysis"],
            horizontal=True
        )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            txn_filter = st.multiselect("Transaction Type", ["Order", "Refund"], default=["Order"])
        with col2:
            search_order = st.text_input("🔍 Order ID")
        with col3:
            search_sku = st.text_input("🔍 SKU")
        
        # Apply filters
        filtered = odd.copy()
        if txn_filter:
            filtered = filtered[filtered['Transaction_Type'].isin(txn_filter)]
        if search_order:
            filtered = filtered[filtered['Order_ID'].astype(str).str.contains(search_order, case=False, na=False)]
        if search_sku:
            filtered = filtered[filtered['SKU'].astype(str).str.contains(search_sku, case=False, na=False)]
        
        # Summary metrics
        st.markdown("#### 📊 Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Orders", f"{filtered['Order_ID'].nunique():,}")
        m2.metric("Total Revenue", f"₹{filtered['Net_Revenue'].sum():,.0f}")
        m3.metric("Total Fees", f"₹{filtered['Total_Fees'].sum():,.0f}")
        m4.metric("Total Ad Cost", f"₹{filtered['Advertising_Cost'].sum():,.0f}")
        m5.metric("Net Profit", f"₹{filtered['Net_Profit'].sum():,.0f}")
        
        st.divider()
        
        # Pagination for large datasets
        show_all = st.checkbox("Show all rows (may be slow for large datasets)", value=False)
        if not show_all and len(filtered) > 1000:
            st.info(f"📊 Dataset has {len(filtered):,} rows. Showing first 1,000 rows. Check 'Show all rows' to see everything.")
            display_data = filtered.head(1000)
        else:
            display_data = filtered
        
        # Different views
        if view_type == "💰 Financial View (Key Metrics)":
            # Clean financial view
            display_cols = {
                'Order_ID': 'Order ID',
                'Transaction_Date': 'Date',
                'SKU': 'SKU',
                'Quantity': 'Qty',
                'Net_Revenue': 'Revenue',
                'Total_Fees': 'Fees',
                'Advertising_Cost': 'Ad Cost',
                'COGS': 'COGS',
                'Gross_Profit': 'Gross Profit',
                'Net_Profit': 'Net Profit',
                'State': 'State'
            }
            
            display_df = display_data[[col for col in display_cols.keys() if col in display_data.columns]].copy()
            display_df = display_df.rename(columns=display_cols)
            
            # Format currency
            for col in ['Revenue', 'Fees', 'Ad Cost', 'COGS', 'Gross Profit', 'Net Profit']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)
            
        elif view_type == "📈 Percentage Analysis":
            # Percentage-focused view
            display_cols = {
                'Order_ID': 'Order ID',
                'Transaction_Date': 'Date',
                'SKU': 'SKU',
                'Net_Revenue': 'Revenue',
                'Selling_Fee_Pct': 'Commission %',
                'FBA_Fee_Pct': 'FBA %',
                'Total_Fee_Pct': 'Total Fee %',
                'Gross_Margin_Pct': 'Gross Margin %',
                'Net_Margin_Pct': 'Net Margin %',
                'Selling_Fees': 'Commission ₹',
                'FBA_Fees': 'FBA ₹',
                'Gross_Profit': 'Gross Profit ₹',
                'Net_Profit': 'Net Profit ₹'
            }
            
            display_df = display_data[[col for col in display_cols.keys() if col in display_data.columns]].copy()
            display_df = display_df.rename(columns=display_cols)
            
            # Format percentages
            for col in ['Commission %', 'FBA %', 'Total Fee %', 'Gross Margin %', 'Net Margin %']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%")
            
            # Format currency
            for col in ['Revenue', 'Commission ₹', 'FBA ₹', 'Gross Profit ₹', 'Net Profit ₹']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
            
            # Only apply styling if dataset is small enough (< 5000 rows)
            total_cells = len(display_df) * len(display_df.columns)
            if total_cells < 100000:  # Safe limit
                # Color code margins
                def color_margin(val):
                    if isinstance(val, str) and '%' in val:
                        try:
                            num = float(val.replace('%', ''))
                            if num < 0:
                                return 'background-color: #ffcccc'  # Red
                            elif num < 10:
                                return 'background-color: #fff3cd'  # Yellow
                            elif num >= 20:
                                return 'background-color: #d4edda'  # Green
                        except:
                            pass
                    return ''
                
                styled_df = display_df.style.applymap(color_margin, subset=['Gross Margin %', 'Net Margin %'] if 'Gross Margin %' in display_df.columns else [])
                st.dataframe(styled_df, use_container_width=True, height=600, hide_index=True)
                st.caption("🟢 Green: >20% margin | 🟡 Yellow: <10% margin | 🔴 Red: Negative margin")
            else:
                # Too many rows - show without styling but with info
                st.info(f"📊 Showing {len(display_df):,} rows. Color coding disabled for performance. Use filters to reduce rows for color highlighting.")
                st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)
                st.caption("💡 **Tip**: Filter by SKU or date range to enable color-coded margins")
            
        else:  # Detailed View
            # Show all columns
            st.info(f"📊 Showing {len(display_data):,} rows with all available columns. Scroll right to see everything.")
            st.dataframe(display_data, use_container_width=True, height=600, hide_index=True)
        
        # Export
        st.divider()
        st.markdown("#### 📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download Filtered Data (CSV)",
                csv_data,
                f"orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("📊 Quick Stats", use_container_width=True):
                st.write("**Average Metrics:**")
                st.write(f"- Avg Revenue: ₹{filtered['Net_Revenue'].mean():,.0f}")
                st.write(f"- Avg Commission %: {filtered['Selling_Fee_Pct'].mean():.1f}%")
                st.write(f"- Avg FBA %: {filtered['FBA_Fee_Pct'].mean():.1f}%")
                st.write(f"- Avg Gross Margin: {filtered['Gross_Margin_Pct'].mean():.1f}%")
                st.write(f"- Avg Net Margin: {filtered['Net_Margin_Pct'].mean():.1f}%")

# --- FINANCE & P/L ---
with tab_fin:
    # Create sub-tabs under Finance
    fin_order, fin_sku, fin_summary = st.tabs(["📋 Order-Level P&L", "📊 SKU P&L", "💼 Summary P&L"])
    
    # SUB-TAB 1: Order-Level P&L (was Order Deep Dive)
    with fin_order:
        st.markdown("###🔬 Order-Level P&L Deep Dive with Advertising Costs")
        st.caption("Complete view with COGS, fees, advertising costs, and TRUE net profit")
        
        odd = st.session_state.order_deep_dive
        
        if odd.empty:
            st.warning("No Custom Unified Transaction data loaded. Please upload and click RUN.")
            st.info("**Tip**: Upload the Custom Unified Transaction CSV file in sidebar section 2 (Amazon Reports)")
        else:
            st.info("✅ Order Deep Dive functionality is available! Data loaded successfully. Full functionality will be restored in next update.")
    
    # SUB-TAB 2: SKU P&L
    with fin_sku:
        led = st.session_state.finance_ledger
        if led.empty:
            st.warning("Finance Ledger is empty.")
        else:
            st.markdown("### 💰 SKU-Level P&L with Advertising Costs")
            
            c1, c2, c3 = st.columns(3)
            grp = c1.radio("P&L Level", ["Variant SKU", "Parent SKU"])
            q_sku = c2.text_input("Filter SKU")
            
            df_f = led.copy()
            if q_sku:
                df_f = df_f[df_f["SKU"].astype(str).str.contains(q_sku, case=False, na=False)]

            g_col = "Parent_SKU" if grp == "Parent SKU" else "SKU"

            agg = df_f.groupby(g_col, as_index=False).agg(
                Revenue=("Revenue", "sum"),
                Net_Payout=("Net_Payout", "sum"),
                Fees=("Fees", "sum"),
                Ad_Cost=("Ad_Cost", "sum"),
                COGS=("COGS", "sum"),
                Gross_Profit=("Gross_Profit", "sum"),
                Net_Profit=("Net_Profit", "sum"),
                Ship_Qty=("Ship_Qty", "sum"),
            )

            tot_rev = agg["Revenue"].sum()
            tot_gross = agg["Gross_Profit"].sum()
            tot_net = agg["Net_Profit"].sum()
            tot_ad = agg["Ad_Cost"].sum()
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Revenue", fmt_inr(tot_rev))
            k2.metric("Fees", fmt_inr(agg["Fees"].sum()))
            k3.metric("Ad Cost", fmt_inr(tot_ad))
            k4.metric("Gross Profit", fmt_inr(tot_gross))
            k5.metric("Net Profit", fmt_inr(tot_net), 
                     delta=f"{(tot_net / tot_rev * 100) if tot_rev else 0:.1f}%")

            st.dataframe(agg.sort_values("Net_Profit", ascending=False), use_container_width=True, height=520)
    
    # SUB-TAB 3: Summary P&L
    with fin_summary:
        st.markdown("### 💼 Financial Summary")
        st.info("Aggregate P&L summary coming soon - use SKU P&L tab for now")

# --- DEEP DIVE ---
with tab_drill:
    st.subheader("🔍 Deep Dive")
    df = st.session_state.sales_df
    if df.empty:
        st.warning("No sales data available.")
    else:
        q = st.text_input("Enter SKU to Analyze")
        if q:
            matches = df[df["Sku"].astype(str).str.contains(q, case=False, na=False)].copy()
            if matches.empty:
                st.warning("No matching SKUs found.")
            else:
                sold = matches[matches["Transaction Type"] == "Shipment"]["Quantity"].sum()
                ret = matches[matches["Transaction Type"] == "Refund"]["Quantity"].sum()
                net = matches["Units_Effective"].sum()

                m1, m2, m3 = st.columns(3)
                m1.metric("Sold", int(sold))
                m2.metric("Returns", int(ret))
                m3.metric("Net", int(net))

                st.markdown("### Panel Pivot")
                p_sold, p_ret, p_net = build_panel_pivots(matches)
                st.dataframe(pd.concat([p_sold, p_ret, p_net], axis=1).fillna(0), use_container_width=True)

                st.markdown("### Transactions")
                st.dataframe(matches.sort_values("TxnDate", ascending=False).head(200), use_container_width=True)

# --- MAPPING AUDIT ---
with tab_audit:
    st.subheader("🧩 SKU Mapping Audit")
    table = st.session_state.sku_map_table
    conflicts = st.session_state.sku_map_conflicts
    collisions = st.session_state.sku_map_collisions

    if table.empty:
        st.info("No mapping loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total mapping rows", f"{len(table):,}")
        c2.metric("Seller→OMS conflicts", f"{len(conflicts):,}")
        c3.metric("Top OMS collision", f"{int(collisions['seller_count'].max()) if not collisions.empty else 0}")

        st.markdown("### A) Conflicts (1 seller → many OMS)")
        if conflicts.empty:
            st.success("✅ No conflicts")
        else:
            st.warning("⚠️ Conflicts found")
            st.dataframe(conflicts.head(50), use_container_width=True)

        st.markdown("### B) Collisions (many sellers → 1 OMS)")
        thresh = st.slider("Threshold", 2, 50, 10)
        heavy = collisions[collisions["seller_count"] >= thresh].head(200)
        if heavy.empty:
            st.success("✅ No heavy collisions")
        else:
            st.error("⚠️ Heavy collisions detected")
            st.dataframe(heavy, use_container_width=True)

# --- ADVERTISING ---
with tab_ad:
    st.markdown("### 📢 Advertising Cost Analysis")
    
    ad_summary = st.session_state.advertising_summary
    
    if ad_summary.empty:
        st.info("No advertising cost data. Upload Custom Unified Transaction file.")
    else:
        # Total ad spend
        total_ad = ad_summary['other transaction fees'].abs().sum() + ad_summary['other'].abs().sum()
        
        st.metric("Total Advertising Spend", fmt_inr(total_ad))
        
        # Show advertising entries
        st.markdown("### 📋 Advertising Transactions")
        ad_display = ad_summary[['txn_date', 'settlement id', 'description', 'other transaction fees', 'other', 'total']].copy()
        ad_display.columns = ['Date', 'Settlement ID', 'Description', 'Ad Fee', 'Other', 'Total']
        
        for col in ['Ad Fee', 'Other', 'Total']:
            ad_display[col] = ad_display[col].apply(lambda x: fmt_inr(abs(x)))
        
        st.dataframe(ad_display, use_container_width=True)
        
        # Ad cost allocation summary
        if not st.session_state.order_deep_dive.empty:
            st.markdown("### 📊 Advertising Allocation")
            
            odd = st.session_state.order_deep_dive
            
            st.info(f"""
            **Allocation Method**: Proportional to revenue
            
            - Total orders: {len(odd):,}
            - Total advertising: {fmt_inr(total_ad)}
            - Avg per order: {fmt_inr(total_ad / len(odd)) if len(odd) > 0 else '₹0'}
            """)
            
            # Top SKUs by ad cost
            sku_ad = odd.groupby('SKU')['Advertising_Cost'].sum().nlargest(20).reset_index()
            sku_ad.columns = ['SKU', 'Total_Ad_Cost']
            
            fig = px.bar(sku_ad, x='SKU', y='Total_Ad_Cost', 
                        title='Top 20 SKUs by Advertising Cost')
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Yash Gallery ERP v4.1 | Now with TRUE Net Profit (after advertising costs)")

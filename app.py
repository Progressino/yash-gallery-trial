#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System - Final Refactored Version
Combines features from both implementations:
- Smart ADS calculation with stockout detection
- Panel-wise (marketplace) drilldown
- Amazon date basis selection (Shipment/Invoice/Order)
- FreeReplacement handling
- Logistics & Transfers
- Professional UI theme
- ZNNE exclusion
- Parent SKU grouping
"""

import io
import zipfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List

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
st.markdown("""
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
""", unsafe_allow_html=True)

st.title("🚀 Yash Gallery Command Center")
st.caption("Complete ERP: Sales Analytics • Inventory Management • PO Engine • AI Forecasting")

# ==========================================================
# 2) SESSION STATE INITIALIZATION
# ==========================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'sku_mapping': {},
        'sales_df': pd.DataFrame(),
        'inventory_df': pd.DataFrame(),
        'transfer_df': pd.DataFrame(),
        'amazon_date_basis': 'Shipment Date',
        'include_replacements': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==========================================================
# 3) CONFIGURATION DATACLASS
# ==========================================================
@dataclass(frozen=True)
class SalesConfig:
    """Configuration for sales data loading"""
    date_basis: str = "Shipment Date"  # Shipment Date / Invoice Date / Order Date
    include_replacements: bool = False  # Include FreeReplacement in sold qty

# ==========================================================
# 4) UTILITY FUNCTIONS
# ==========================================================
def clean_sku(sku) -> str:
    """Clean SKU string by removing quotes and prefixes"""
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip()

def get_parent_sku(oms_sku) -> str:
    """
    Remove marketplace suffixes AND size/color variants to get parent SKU.
    
    Examples:
    - 6017SKDRED-XXL → 6017SKDRED
    - 1065YKBLUE-L → 1065YKBLUE
    - 1309YKRED-3XL_Myntra → 1309YKRED
    """
    if pd.isna(oms_sku):
        return oms_sku
    
    s = str(oms_sku).strip()
    
    # First, remove marketplace suffixes
    marketplace_suffixes = ["_Myntra", "_Flipkart", "_Amazon", "_Meesho", 
                           "_MYNTRA", "_FLIPKART", "_AMAZON", "_MEESHO"]
    for suffix in marketplace_suffixes:
        if s.endswith(suffix):
            s = s.replace(suffix, "")
            break
    
    # Then, remove size/color variants (anything after last hyphen)
    # Common patterns: -XXL, -3XL, -L, -M, -S, -XS, -RED, -BLUE, etc.
    if '-' in s:
        # Split by hyphen and check if last part is a size/color
        parts = s.split('-')
        if len(parts) >= 2:
            last_part = parts[-1].upper()
            
            # Check if last part looks like a size or color variant
            # Sizes: S, M, L, XL, XXL, XXXL, 2XL, 3XL, 4XL, 5XL, 6XL, 28, 30, 32, etc.
            # Colors: RED, BLUE, GREEN, BLACK, WHITE, etc.
            size_patterns = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', '2XL', '3XL', '4XL', '5XL', '6XL']
            
            # Check if it's a size
            is_size = (
                last_part in size_patterns or  # Exact match
                last_part.endswith('XL') or    # XXL, XXXL, etc.
                last_part.isdigit() or         # 28, 30, 32 (numeric sizes)
                (len(last_part) <= 4 and any(c in last_part for c in ['S', 'M', 'L', 'X']))  # Contains size letters
            )
            
            # Check if it's likely a color (common color names)
            common_colors = ['RED', 'BLUE', 'GREEN', 'BLACK', 'WHITE', 'YELLOW', 'PINK', 'PURPLE', 
                           'ORANGE', 'BROWN', 'GREY', 'GRAY', 'NAVY', 'MAROON', 'BEIGE', 'CREAM',
                           'GOLD', 'SILVER', 'TAN', 'KHAKI', 'OLIVE', 'TEAL', 'CORAL', 'PEACH']
            is_color = last_part in common_colors or any(color in last_part for color in common_colors)
            
            # If last part is size or color, remove it
            if is_size or is_color:
                s = '-'.join(parts[:-1])
    
    return s

def map_to_oms_sku(seller_sku, mapping: Dict[str, str]) -> str:
    """Map seller SKU to OMS SKU using mapping dictionary"""
    if pd.isna(seller_sku):
        return seller_sku
    clean = clean_sku(seller_sku)
    return mapping.get(clean, clean)

def read_zip_csv(zip_file) -> pd.DataFrame:
    """Read first CSV from ZIP file"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                return pd.DataFrame()
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading ZIP: {e}")
        return pd.DataFrame()

def read_csv_safe(file_obj) -> pd.DataFrame:
    """Safely read CSV file"""
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
    """
    Load SKU mapping from Excel file with multiple sheets.
    Auto-detects seller SKU and OMS SKU columns.
    """
    mapping_dict = {}
    
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
                # Seller SKU column
                if any(k in col_lower for k in ['seller', 'myntra', 'meesho', 'snapdeal', 'sku id']):
                    if 'sku' in col_lower:
                        seller_col = col
                # OMS SKU column
                if 'oms' in col_lower and 'sku' in col_lower:
                    oms_col = col
            
            # Fallback to position
            if seller_col is None and len(df.columns) > 1:
                seller_col = df.columns[1]
            if oms_col is None:
                oms_col = df.columns[-1]
            
            if seller_col and oms_col:
                for _, row in df.iterrows():
                    seller_sku = clean_sku(row.get(seller_col, ""))
                    oms_sku = clean_sku(row.get(oms_col, ""))
                    
                    if seller_sku and oms_sku and seller_sku != 'nan' and oms_sku != 'nan':
                        mapping_dict[seller_sku] = oms_sku
        
        return mapping_dict
        
    except Exception as e:
        st.error(f"Error loading SKU mapping: {e}")
        return {}

# ==========================================================
# 6) SALES DATA LOADERS
# ==========================================================
def load_amazon_sales(zip_file, mapping: Dict[str, str], source: str, config: SalesConfig) -> pd.DataFrame:
    """
    Load Amazon sales from ZIP file.
    Handles date basis selection and FreeReplacement logic.
    """
    df = read_zip_csv(zip_file)
    
    if df.empty or 'Sku' not in df.columns:
        return pd.DataFrame()
    
    # Map to OMS SKU
    df['OMS_SKU'] = df['Sku'].apply(lambda x: map_to_oms_sku(x, mapping))
    
    # Date basis selection with fallback
    date_col = config.date_basis
    if date_col not in df.columns:
        date_col = 'Shipment Date' if 'Shipment Date' in df.columns else \
                   'Invoice Date' if 'Invoice Date' in df.columns else \
                   'Order Date' if 'Order Date' in df.columns else df.columns[0]
    
    df['TxnDate'] = pd.to_datetime(df[date_col], errors='coerce')
    df['Quantity'] = pd.to_numeric(df.get('Quantity', 0), errors='coerce').fillna(0)
    
    # Classify transaction type
    def classify_txn(txn_type):
        s = str(txn_type).lower()
        if 'refund' in s or 'return' in s:
            return 'Refund'
        if 'cancel' in s:
            return 'Cancel'
        if 'freereplacement' in s or 'replacement' in s:
            return 'FreeReplacement'
        return 'Shipment'
    
    df['TxnType'] = df.get('Transaction Type', '').apply(classify_txn)
    
    # Handle FreeReplacement
    if not config.include_replacements:
        df.loc[df['TxnType'] == 'FreeReplacement', 'Quantity'] = 0
    
    # Calculate effective units
    df['Units_Effective'] = np.where(
        df['TxnType'] == 'Refund', -df['Quantity'],
        np.where(df['TxnType'] == 'Cancel', 0, df['Quantity'])
    )
    
    df['Source'] = source
    
    # Order ID detection
    order_col = None
    for col in df.columns:
        if 'order' in str(col).lower() and 'id' in str(col).lower():
            order_col = col
            break
    df['OrderId'] = df[order_col] if order_col else np.nan
    
    result = df[['OMS_SKU', 'TxnDate', 'TxnType', 'Quantity', 'Units_Effective', 'Source', 'OrderId']].copy()
    result.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source', 'OrderId']
    
    return result.dropna(subset=['TxnDate'])

def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """Load Flipkart sales from Excel file"""
    try:
        df = pd.read_excel(xlsx_file, sheet_name='Sales Report')
        
        if df.empty:
            return pd.DataFrame()
        
        # Clean and map SKU
        df['SKU_Clean'] = df['SKU'].apply(clean_sku)
        df['OMS_SKU'] = df['SKU_Clean'].apply(lambda x: map_to_oms_sku(x, mapping))
        
        df['TxnDate'] = pd.to_datetime(df.get('Order Date'), errors='coerce')
        df['Quantity'] = pd.to_numeric(df.get('Item Quantity', 0), errors='coerce').fillna(0)
        df['Source'] = 'Flipkart'
        
        # Transaction type
        df['TxnType'] = df.get('Event Sub Type', '').apply(
            lambda x: 'Refund' if 'return' in str(x).lower() else 'Shipment'
        )
        
        df['Units_Effective'] = np.where(df['TxnType'] == 'Refund', -df['Quantity'], df['Quantity'])
        
        # Order ID
        df['OrderId'] = df.get('Order ID', df.get('Order Id', np.nan))
        
        result = df[['OMS_SKU', 'TxnDate', 'TxnType', 'Quantity', 'Units_Effective', 'Source', 'OrderId']].copy()
        result.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source', 'OrderId']
        
        return result.dropna(subset=['TxnDate'])
        
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}")
        return pd.DataFrame()

def load_meesho_sales(zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """Load Meesho sales from ZIP file"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            excel_files = [f for f in z.namelist() 
                          if 'tcs_sales' in f.lower() 
                          and f.lower().endswith('.xlsx') 
                          and 'return' not in f.lower()]
            
            if not excel_files:
                return pd.DataFrame()
            
            with z.open(excel_files[0]) as f:
                df = pd.read_excel(f)
        
        if df.empty:
            return pd.DataFrame()
        
        df['OMS_SKU'] = df.get('identifier').apply(lambda x: map_to_oms_sku(x, mapping))
        df['TxnDate'] = pd.to_datetime(df.get('order_date'), errors='coerce')
        df['Quantity'] = pd.to_numeric(df.get('quantity', 0), errors='coerce').fillna(0)
        df['Source'] = 'Meesho'
        df['TxnType'] = 'Shipment'
        df['Units_Effective'] = df['Quantity']
        df['OrderId'] = df.get('sub_order_num', np.nan)
        
        result = df[['OMS_SKU', 'TxnDate', 'TxnType', 'Quantity', 'Units_Effective', 'Source', 'OrderId']].copy()
        result.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source', 'OrderId']
        
        return result.dropna(subset=['TxnDate'])
        
    except Exception as e:
        st.error(f"Error loading Meesho: {e}")
        return pd.DataFrame()

# ==========================================================
# 7) INVENTORY LOADERS
# ==========================================================
def load_inventory_consolidated(oms_file, fk_file, myntra_file, amz_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Load and consolidate inventory from all sources.
    Excludes ZNNE from Amazon, groups by parent SKU.
    """
    inv_dfs = []
    
    # OMS Inventory
    if oms_file:
        df = read_csv_safe(oms_file)
        if not df.empty and {'Item SkuCode', 'Inventory'}.issubset(df.columns):
            df = df.rename(columns={'Item SkuCode': 'OMS_SKU', 'Inventory': 'OMS_Inventory'})
            df['OMS_SKU'] = df['OMS_SKU'].astype(str)
            df['OMS_Inventory'] = pd.to_numeric(df['OMS_Inventory'], errors='coerce').fillna(0)
            inv_dfs.append(df[['OMS_SKU', 'OMS_Inventory']].groupby('OMS_SKU').sum().reset_index())
    
    # Flipkart Inventory
    if fk_file:
        df = read_csv_safe(fk_file)
        if not df.empty and {'SKU', 'Live on Website'}.issubset(df.columns):
            df['OMS_SKU'] = df['SKU'].apply(lambda x: map_to_oms_sku(x, mapping))
            df['Flipkart_Live'] = pd.to_numeric(df['Live on Website'], errors='coerce').fillna(0)
            inv_dfs.append(df.groupby('OMS_SKU')['Flipkart_Live'].sum().reset_index())
    
    # Myntra Inventory
    if myntra_file:
        df = read_csv_safe(myntra_file)
        if not df.empty:
            # Detect SKU column
            sku_col = None
            for col in df.columns:
                if 'seller sku code' in str(col).lower() or 'sku code' in str(col).lower():
                    sku_col = col
                    break
            
            # Detect inventory column
            inv_col = None
            for col in df.columns:
                if 'sellable inventory count' in str(col).lower():
                    inv_col = col
                    break
            
            if sku_col and inv_col:
                df['OMS_SKU'] = df[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                df['Myntra_Inventory'] = pd.to_numeric(df[inv_col], errors='coerce').fillna(0)
                inv_dfs.append(df.groupby('OMS_SKU')['Myntra_Inventory'].sum().reset_index())
    
    # Amazon Inventory (excluding ZNNE)
    if amz_file:
        df = read_csv_safe(amz_file)
        if not df.empty and {'MSKU', 'Ending Warehouse Balance'}.issubset(df.columns):
            # CRITICAL: Exclude ZNNE location
            if 'Location' in df.columns:
                original_count = len(df)
                df = df[df['Location'] != 'ZNNE']
                excluded = original_count - len(df)
                if excluded > 0:
                    st.sidebar.info(f"ℹ️ Excluded {excluded:,} ZNNE records (OMS duplicate)")
            
            df['OMS_SKU'] = df['MSKU'].apply(lambda x: map_to_oms_sku(x, mapping))
            df['Amazon_Inventory'] = pd.to_numeric(df['Ending Warehouse Balance'], errors='coerce').fillna(0)
            inv_dfs.append(df.groupby('OMS_SKU')['Amazon_Inventory'].sum().reset_index())
    
    if not inv_dfs:
        return pd.DataFrame()
    
    # Merge all inventory sources
    consolidated = inv_dfs[0]
    for df in inv_dfs[1:]:
        consolidated = pd.merge(consolidated, df, on='OMS_SKU', how='outer')
    
    # Fill NaN with 0
    inv_cols = [c for c in consolidated.columns if c.endswith('_Inventory') or c.endswith('_Live')]
    consolidated[inv_cols] = consolidated[inv_cols].fillna(0)
    
    # Parent SKU grouping
    consolidated['Parent_SKU'] = consolidated['OMS_SKU'].apply(get_parent_sku)
    consolidated = consolidated.groupby('Parent_SKU')[inv_cols].sum().reset_index()
    consolidated = consolidated.rename(columns={'Parent_SKU': 'OMS_SKU'})
    
    # Calculate totals
    marketplace_cols = [c for c in inv_cols if 'OMS' not in c]
    consolidated['Marketplace_Total'] = consolidated[marketplace_cols].sum(axis=1) if marketplace_cols else 0
    
    if 'OMS_Inventory' in consolidated.columns:
        consolidated['Total_Inventory'] = consolidated['OMS_Inventory'] + consolidated['Marketplace_Total']
    else:
        consolidated['Total_Inventory'] = consolidated['Marketplace_Total']
    
    # Remove zero inventory rows
    consolidated = consolidated[consolidated['Total_Inventory'] > 0]
    
    return consolidated

# ==========================================================
# 8) STOCK TRANSFER LOADER
# ==========================================================
def load_stock_transfer(zip_file) -> pd.DataFrame:
    """Load Amazon stock transfer data"""
    df = read_zip_csv(zip_file)
    
    if df.empty:
        return pd.DataFrame()
    
    required = ['Invoice Date', 'Ship From Fc', 'Ship To Fc', 'Quantity', 'Transaction Type']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    
    result = df[required].copy()
    result['Invoice Date'] = pd.to_datetime(result['Invoice Date'], errors='coerce')
    result['Quantity'] = pd.to_numeric(result['Quantity'], errors='coerce').fillna(0)
    
    return result

# ==========================================================
# 9) PANEL-WISE PIVOT BUILDER
# ==========================================================
def build_panel_pivots(df: pd.DataFrame):
    """
    Build marketplace-wise pivots:
    - Sold: Shipment quantities by panel
    - Return: Refund quantities by panel
    - Net: Net units by panel
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    w = df.copy()
    w['Panel'] = w['Source'].astype(str)
    
    # Sold pivot
    sold = (w[w['Transaction Type'] == 'Shipment']
            .groupby(['Sku', 'Panel'])['Quantity']
            .sum()
            .unstack(fill_value=0))
    sold.columns = [f"{c} | Sold" for c in sold.columns]
    
    # Return pivot
    ret = (w[w['Transaction Type'] == 'Refund']
           .groupby(['Sku', 'Panel'])['Quantity']
           .sum()
           .unstack(fill_value=0))
    ret.columns = [f"{c} | Return" for c in ret.columns]
    
    # Net pivot
    net = (w.groupby(['Sku', 'Panel'])['Units_Effective']
           .sum()
           .unstack(fill_value=0))
    net.columns = [f"{c} | Net" for c in net.columns]
    
    return sold, ret, net

# ==========================================================
# 10) SMART ADS CALCULATION
# ==========================================================
def calculate_smart_ads(sales_df: pd.DataFrame, inv_df: pd.DataFrame, days_back: int):
    """
    Calculate Average Daily Sales with stockout detection.
    Only counts days when product was available for sale.
    """
    if sales_df.empty:
        return pd.DataFrame()
    
    max_date = sales_df['TxnDate'].max()
    cutoff_date = max_date - timedelta(days=days_back)
    recent = sales_df[sales_df['TxnDate'] >= cutoff_date].copy()
    
    # Get sales summary with unique selling days
    recent['Date'] = recent['TxnDate'].dt.date
    
    sales_summary = recent.groupby('Sku').agg({
        'Units_Effective': 'sum',
        'Quantity': 'count',
        'Date': 'nunique'
    }).reset_index()
    
    sales_summary.columns = ['OMS_SKU', 'Total_Units_Sold', 'Transactions', 'Days_With_Sales']
    
    # Merge with inventory
    po_df = pd.merge(inv_df, sales_summary, on='OMS_SKU', how='left')
    po_df[['Total_Units_Sold', 'Transactions', 'Days_With_Sales']] = \
        po_df[['Total_Units_Sold', 'Transactions', 'Days_With_Sales']].fillna(0)
    
    # Calculate active days (minimum 7 to avoid inflated ADS)
    po_df['Active_Days'] = po_df['Days_With_Sales'].clip(lower=7)
    
    # Smart ADS calculation
    po_df['ADS'] = np.where(
        po_df['Active_Days'] > 0,
        po_df['Total_Units_Sold'] / po_df['Active_Days'],
        0
    )
    
    # Stockout detection (simplified)
    po_df['Stockout_Flag'] = ''
    po_df.loc[(po_df['Active_Days'] < days_back) & (po_df['Total_Units_Sold'] > 0), 
              'Stockout_Flag'] = '⚠️ Low Avail'
    
    return po_df

# ==========================================================
# 11) SIDEBAR - FILE UPLOADS
# ==========================================================
st.sidebar.markdown("## 📂 Data Upload")

# Logo
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.markdown("### Yash Gallery")

st.sidebar.divider()

# SKU Mapping
map_file = st.sidebar.file_uploader(
    "1️⃣ SKU Mapping (Required)",
    type=['xlsx'],
    help="Copy_of_All_penal_replace_sku.xlsx"
)

# Amazon Settings
st.sidebar.markdown("### ⚙️ Amazon Settings")
st.session_state.amazon_date_basis = st.sidebar.selectbox(
    "Date Basis",
    ["Shipment Date", "Invoice Date", "Order Date"],
    index=0
)
st.session_state.include_replacements = st.sidebar.checkbox(
    "Include FreeReplacement",
    value=False
)

st.sidebar.divider()

# Sales Files
st.sidebar.markdown("### 2️⃣ Sales Data")
f_b2c = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=['zip'], key='b2c')
f_b2b = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=['zip'], key='b2b')
f_transfer = st.sidebar.file_uploader("Stock Transfer (ZIP)", type=['zip'], key='transfer')
f_fk = st.sidebar.file_uploader("Flipkart (Excel)", type=['xlsx'], key='fk')
f_meesho = st.sidebar.file_uploader("Meesho (ZIP)", type=['zip'], key='meesho')

st.sidebar.divider()

# Inventory Files
st.sidebar.markdown("### 3️⃣ Inventory Data")
i_oms = st.sidebar.file_uploader("OMS (CSV)", type=['csv'], key='oms')
i_fk = st.sidebar.file_uploader("Flipkart (CSV)", type=['csv'], key='fk_inv')
i_myntra = st.sidebar.file_uploader("Myntra (CSV)", type=['csv'], key='myntra')
i_amz = st.sidebar.file_uploader("Amazon (CSV)", type=['csv'], key='amz')

st.sidebar.divider()

# Load Button
if st.sidebar.button("🚀 Load All Data", use_container_width=True):
    if not map_file:
        st.sidebar.error("SKU Mapping required!")
    else:
        with st.spinner("Loading data..."):
            # Load SKU mapping
            st.session_state.sku_mapping = load_sku_mapping(map_file)
            
            if st.session_state.sku_mapping:
                st.sidebar.success(f"✅ Mapping: {len(st.session_state.sku_mapping):,} SKUs")
            
            # Create config
            config = SalesConfig(
                date_basis=st.session_state.amazon_date_basis,
                include_replacements=st.session_state.include_replacements
            )
            
            # Load sales
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
            
            # Load inventory
            st.session_state.inventory_df = load_inventory_consolidated(
                i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping
            )
            
            if not st.session_state.inventory_df.empty:
                st.sidebar.success(f"✅ Inventory: {len(st.session_state.inventory_df):,} SKUs")
            
            # Load transfers
            if f_transfer:
                st.session_state.transfer_df = load_stock_transfer(f_transfer)
                if not st.session_state.transfer_df.empty:
                    st.sidebar.success(f"✅ Transfers: {len(st.session_state.transfer_df):,} records")
        
        st.rerun()

# ==========================================================
# 12) GUARD - REQUIRE MAPPING
# ==========================================================
if not st.session_state.sku_mapping:
    st.info("👋 **Welcome!** Upload SKU Mapping file and click **Load All Data** to begin.")
    st.stop()

# ==========================================================
# 13) MAIN TABS
# ==========================================================
tab_dash, tab_inv, tab_po, tab_logistics, tab_forecast, tab_drill = st.tabs([
    "📊 Dashboard",
    "📦 Inventory",
    "🎯 PO Engine",
    "🚚 Logistics",
    "📈 AI Forecast",
    "🔍 Deep Dive"
])

# ----------------------------------------------------------
# TAB 1: DASHBOARD
# ----------------------------------------------------------
with tab_dash:
    st.subheader("📊 Sales Analytics Dashboard")
    
    df = st.session_state.sales_df
    
    if df.empty:
        st.warning("⚠️ No sales data loaded. Upload sales files and click Load Data.")
    else:
        # Add period selector with grace period
        col_period, col_grace = st.columns([3, 1])
        
        with col_period:
            period_option = st.selectbox(
                "Analysis Period",
                ["Last 7 Days", "Last 30 Days", "Last 60 Days", "Last 90 Days", "All Time"],
                index=1,  # Default to Last 30 Days
                key="dash_period"
            )
        
        with col_grace:
            grace_days = st.number_input(
                "Grace Period (Days)",
                min_value=0,
                max_value=14,
                value=7,
                help="Extra days to capture late transactions (recommended: 7 days)"
            )
        
        # Calculate date range with grace period
        df = df.copy()
        df['TxnDate'] = pd.to_datetime(df['TxnDate'], errors='coerce')
        max_date = df['TxnDate'].max()
        
        if period_option == "All Time":
            filtered_df = df
            min_date = df['TxnDate'].min()
            date_range_text = f"All Time: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            # Extract days from option
            base_days = 7 if "7" in period_option else \
                       30 if "30" in period_option else \
                       60 if "60" in period_option else 90
            
            # Add grace period buffer
            total_days = base_days + grace_days
            cutoff_date = max_date - timedelta(days=total_days)
            
            filtered_df = df[df['TxnDate'] >= cutoff_date]
            
            # Show actual date range
            actual_min = filtered_df['TxnDate'].min()
            date_range_text = f"Period: {actual_min.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} " \
                            f"({base_days} days + {grace_days} day grace period)"
        
        # Display date range prominently
        st.info(f"📅 **{date_range_text}** | Transactions: {len(filtered_df):,}")
        
        # Calculate metrics
        filtered_df['Quantity'] = pd.to_numeric(filtered_df['Quantity'], errors='coerce').fillna(0)
        filtered_df['Units_Effective'] = pd.to_numeric(filtered_df['Units_Effective'], errors='coerce').fillna(0)
        
        sold_pcs = filtered_df[filtered_df['Transaction Type'] == 'Shipment']['Quantity'].sum()
        ret_pcs = filtered_df[filtered_df['Transaction Type'] == 'Refund']['Quantity'].sum()
        net_units = filtered_df['Units_Effective'].sum()
        
        # Orders count
        if 'OrderId' in filtered_df.columns:
            orders = filtered_df[filtered_df['Transaction Type'] == 'Shipment']['OrderId'].nunique()
        else:
            orders = len(filtered_df[filtered_df['Transaction Type'] == 'Shipment'])
        
        return_rate = (ret_pcs / sold_pcs * 100) if sold_pcs > 0 else 0
        
        # Display metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛒 Orders", f"{orders:,}")
        c2.metric("✅ Sold Pieces", f"{int(sold_pcs):,}")
        c3.metric("↩️ Returns", f"{int(ret_pcs):,}")
        c4.metric("📊 Return Rate", f"{return_rate:.1f}%")
        c5.metric("📦 Net Units", f"{int(net_units):,}")
        
        # Settings info
        st.info(
            f"**Settings:** Amazon Date: {st.session_state.amazon_date_basis} | "
            f"Include Replacements: {st.session_state.include_replacements}"
        )
        
        st.divider()
        
        # Charts (use filtered_df instead of df)
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 🏆 Top 20 Selling SKUs")
            top = filtered_df[filtered_df['Transaction Type'] == 'Shipment'].groupby('Sku')['Quantity'].sum() \
                    .sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(top, x='Sku', y='Quantity', title='Top Sellers (Pieces)')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 📊 Marketplace Split")
            source_summary = filtered_df.groupby('Source')['Quantity'].sum().reset_index()
            fig = px.pie(source_summary, values='Quantity', names='Source', hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 2: INVENTORY
# ----------------------------------------------------------
with tab_inv:
    st.subheader("📦 Consolidated Inventory")
    
    inv = st.session_state.inventory_df
    
    if inv.empty:
        st.warning("⚠️ No inventory data loaded.")
    else:
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total SKUs", f"{len(inv):,}")
        c2.metric("Total Units", f"{inv['Total_Inventory'].sum():,.0f}")
        
        if 'OMS_Inventory' in inv.columns:
            c3.metric("OMS Warehouse", f"{inv['OMS_Inventory'].sum():,.0f}")
        if 'Marketplace_Total' in inv.columns:
            c4.metric("Marketplaces", f"{inv['Marketplace_Total'].sum():,.0f}")
        
        st.divider()
        
        # Search
        search = st.text_input("🔍 Search SKU", placeholder="Type to filter...")
        
        display = inv.copy()
        if search:
            display = display[display['OMS_SKU'].str.contains(search, case=False, na=False)]
        
        st.dataframe(display, use_container_width=True, height=500)

# ----------------------------------------------------------
# TAB 3: PO ENGINE
# ----------------------------------------------------------
with tab_po:
    st.subheader("🎯 Purchase Order Recommendations")
    
    if st.session_state.sales_df.empty or st.session_state.inventory_df.empty:
        st.error("⚠️ Both sales and inventory data required for PO calculations.")
    else:
        # View mode selector at the top
        st.markdown("### 👁️ View Mode")
        
        col_view, col_info = st.columns([1, 3])
        
        with col_view:
            view_mode = st.radio(
                "Group By",
                ["By Variant (Size/Color)", "By Parent SKU (Style Only)"],
                key="po_view_mode",
                help="Variant = individual sizes/colors | Parent = style without variants"
            )
        
        with col_info:
            if "Parent" in view_mode:
                st.info("""
                **📦 Parent SKU Mode:**
                - Groups all sizes/colors under parent style (e.g., 6017SKDRED-XXL → 6017SKDRED)
                - Shows combined PO for entire style
                - Useful for production planning and bulk ordering
                - Example: Order 500 units of 6017SKDRED (will cut into sizes later)
                """)
            else:
                st.info("""
                **🎯 Variant Mode:**
                - Shows individual size/color combinations
                - Precise PO for each variant
                - Useful for pre-cut inventory and specific size planning
                - Example: Order 50 units of 6017SKDRED-XXL specifically
                """)
        
        st.divider()
        
        st.markdown("""
        ### 📚 Understanding PO Parameters
        
        **Velocity Period**: Historical window to calculate Average Daily Sales (ADS)
        - *Example*: Last 30 Days means ADS = Total Sales / 30 days
        
        **Grace Days**: Buffer to capture late-recorded transactions
        - *Recommended*: 7 days to ensure complete data
        
        **Lead Time**: Days from placing order to receiving stock at your warehouse
        - *Example*: 15 days = time for production + shipping + quality check
        - *Impact*: Higher lead time = need more stock to cover the waiting period
        
        **Target Stock**: Extra inventory buffer beyond lead time demand
        - *Example*: 60 days = maintain 2 months of stock for safety
        - *Purpose*: Handle demand spikes, prevent stockouts, reduce order frequency
        
        **Safety Stock %**: Extra buffer to handle demand variability
        - *Example*: 20% means order 20% more than calculated need
        - *Purpose*: Protect against unexpected demand increases or supply delays
        
        ---
        
        **📊 PO Formula:**
        ```
        ADS = Average Daily Sales (from velocity period)
        
        Lead Time Stock = ADS × Lead Time Days
        Target Buffer = ADS × Target Stock Days
        Safety Buffer = (Lead Time Stock + Target Buffer) × Safety %
        
        Total Required = Lead Time Stock + Target Buffer + Safety Buffer
        PO Recommended = Total Required - Current Inventory
        ```
        
        **Example Calculation:**
        - ADS = 2 units/day
        - Lead Time = 15 days → Need 30 units for lead time period
        - Target = 60 days → Need 120 units buffer
        - Safety = 20% of (30 + 120) = 30 units
        - Total Required = 30 + 120 + 30 = 180 units
        - Current Stock = 50 units
        - **PO Recommended = 180 - 50 = 130 units**
        
        ✅ If lead time increases, PO increases (need more cover)  
        ✅ If target decreases, PO decreases (less buffer needed)  
        ✅ If safety % increases, PO increases (more protection)
        """)
        
        st.divider()
        
        # Configuration with detailed help text
        st.markdown("### ⚙️ Configure PO Parameters")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        velocity = c1.selectbox(
            "Velocity Period",
            ["Last 7 Days", "Last 30 Days", "Last 60 Days", "Last 90 Days"],
            key="po_velocity",
            help="Historical period to calculate average daily sales"
        )
        days = 7 if "7" in velocity else 30 if "30" in velocity else 60 if "60" in velocity else 90
        
        grace_days = c2.number_input(
            "Grace Days",
            0, 14, 7,
            help="Extra days to capture late transactions. Recommended: 7 days"
        )
        
        lead_time = c3.number_input(
            "Lead Time (Days)",
            1, 180, 15,
            help="⏱️ Days from order to delivery. Higher = need more stock during waiting period"
        )
        
        target_days = c4.number_input(
            "Target Stock (Days)",
            0, 180, 60,
            help="📦 Extra buffer beyond lead time. Higher = more safety stock, fewer orders"
        )
        
        safety_pct = c5.slider(
            "Safety Stock %",
            0, 100, 20,
            help="🛡️ Extra % to handle demand spikes. Higher = more protection but higher inventory cost"
        )
        
        st.divider()
        
        # Show current configuration summary
        st.info(f"""
        **Current Configuration:**
        - 📊 Velocity: {velocity} + {grace_days} day grace = **{days + grace_days} days** of sales data
        - ⏱️ Lead Time: **{lead_time} days** to receive stock after ordering
        - 📦 Target Buffer: **{target_days} days** of extra inventory
        - 🛡️ Safety Buffer: **{safety_pct}%** above calculated need
        - 📈 Total Coverage: **{lead_time + target_days} days** of inventory target
        """)
        
        # Apply grace period to velocity calculation
        total_period = days + grace_days
        
        # Calculate PO with grace period
        po_df = calculate_smart_ads(st.session_state.sales_df, st.session_state.inventory_df, total_period)
        
        if po_df.empty:
            st.warning("No PO calculations available.")
        else:
            # Add parent SKU column
            po_df['Parent_SKU'] = po_df['OMS_SKU'].apply(get_parent_sku)
            
            # FIXED: Include lead time in the calculation
            # Days left calculation
            po_df['Days_Left'] = np.where(
                po_df['ADS'] > 0,
                po_df['Total_Inventory'] / po_df['ADS'],
                999
            )
            
            # CORRECTED FORMULA:
            # 1. Lead Time Demand: Stock needed during lead time
            po_df['Lead_Time_Demand'] = po_df['ADS'] * lead_time
            
            # 2. Target Stock: Additional buffer stock
            po_df['Target_Stock'] = po_df['ADS'] * target_days
            
            # 3. Base Requirement: Lead time stock + Target buffer
            po_df['Base_Requirement'] = po_df['Lead_Time_Demand'] + po_df['Target_Stock']
            
            # 4. Safety Stock: % of base requirement
            po_df['Safety_Stock'] = po_df['Base_Requirement'] * (safety_pct / 100)
            
            # 5. Total Required: Base + Safety
            po_df['Total_Required'] = po_df['Base_Requirement'] + po_df['Safety_Stock']
            
            # 6. PO Needed: Total Required - Current Inventory
            po_df['PO_Recommended'] = (po_df['Total_Required'] - po_df['Total_Inventory']).clip(lower=0)
            
            # Round to nearest 5
            po_df['PO_Recommended'] = (np.ceil(po_df['PO_Recommended'] / 5) * 5).astype(int)
            
            # Priority based on days left vs lead time
            def get_priority(row):
                if row['Days_Left'] < lead_time and row['PO_Recommended'] > 0:
                    return '🔴 URGENT'
                elif row['Days_Left'] < lead_time + 7 and row['PO_Recommended'] > 0:
                    return '🟡 HIGH'
                elif row['PO_Recommended'] > 0:
                    return '🟢 MEDIUM'
                return '⚪ OK'
            
            po_df['Priority'] = po_df.apply(get_priority, axis=1)
            
            # GROUP BY PARENT SKU IF SELECTED
            if "Parent" in view_mode:
                # Show before grouping count
                variants_before = len(po_df)
                
                # Aggregate by parent SKU
                agg_cols = {
                    'Total_Inventory': 'sum',
                    'Total_Units_Sold': 'sum',
                    'ADS': 'sum',  # Sum ADS across variants
                    'Active_Days': 'mean',  # Average active days
                    'Lead_Time_Demand': 'sum',
                    'Target_Stock': 'sum',
                    'Base_Requirement': 'sum',
                    'Safety_Stock': 'sum',
                    'Total_Required': 'sum',
                    'PO_Recommended': 'sum'
                }
                
                # Group by parent
                po_grouped = po_df.groupby('Parent_SKU').agg(agg_cols).reset_index()
                po_grouped = po_grouped.rename(columns={'Parent_SKU': 'OMS_SKU'})
                
                # Recalculate days left and priority for grouped data
                po_grouped['Days_Left'] = np.where(
                    po_grouped['ADS'] > 0,
                    po_grouped['Total_Inventory'] / po_grouped['ADS'],
                    999
                )
                
                po_grouped['Priority'] = po_grouped.apply(get_priority, axis=1)
                
                # Count variants per parent
                variant_count = po_df.groupby('Parent_SKU').size().reset_index(name='Variant_Count')
                po_grouped = pd.merge(po_grouped, variant_count, left_on='OMS_SKU', right_on='Parent_SKU', how='left')
                po_grouped = po_grouped.drop('Parent_SKU', axis=1)
                
                # Stockout flag (if any variant has stockout)
                stockout_flag = po_df.groupby('Parent_SKU')['Stockout_Flag'].apply(
                    lambda x: '⚠️ Some Variants' if any('⚠️' in str(s) for s in x) else ''
                ).reset_index()
                po_grouped = pd.merge(po_grouped, stockout_flag, left_on='OMS_SKU', right_on='Parent_SKU', how='left')
                po_grouped = po_grouped.drop('Parent_SKU', axis=1)
                
                po_needed = po_grouped[po_grouped['PO_Recommended'] > 0].sort_values(['Priority', 'Days_Left'])
                
                # Show grouping success message with sample
                st.success(f"""
                ✅ **Parent SKU Mode Active:** 
                - Grouped {variants_before:,} variant SKUs into {len(po_grouped):,} parent styles
                - Example: `6017SKDRED-XXL`, `6017SKDRED-L`, `6017SKDRED-M` → `6017SKDRED`
                """)
                
                # Show a sample of grouping
                if len(po_df) > 0:
                    sample_parent = po_df['Parent_SKU'].value_counts().head(1)
                    if len(sample_parent) > 0:
                        sample_sku = sample_parent.index[0]
                        sample_variants = po_df[po_df['Parent_SKU'] == sample_sku]['OMS_SKU'].tolist()
                        if len(sample_variants) > 1:
                            st.info(f"**Sample Grouping:** {', '.join(sample_variants[:5])} → `{sample_sku}`")
                
            else:
                # Variant mode - use original data
                po_needed = po_df[po_df['PO_Recommended'] > 0].sort_values(['Priority', 'Days_Left'])
            
            # Summary
            urgent = len(po_needed[po_needed['Priority'] == '🔴 URGENT'])
            high = len(po_needed[po_needed['Priority'] == '🟡 HIGH'])
            medium = len(po_needed[po_needed['Priority'] == '🟢 MEDIUM'])
            total_units = po_needed['PO_Recommended'].sum()
            
            st.markdown("### 📊 PO Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔴 Urgent Orders", urgent, help="Stock will run out before lead time")
            m2.metric("🟡 High Priority", high, help="Stock will run out within lead time + 7 days")
            m3.metric("🟢 Medium Priority", medium, help="Needs reorder but not urgent")
            m4.metric("📦 Total Units to Order", f"{total_units:,}")
            
            st.divider()
            
            # Search
            search = st.text_input("🔍 Search SKU", placeholder="Filter by SKU code, color, or size...")
            if search:
                po_needed = po_needed[po_needed['OMS_SKU'].str.contains(search, case=False, na=False)]
                st.caption(f"Showing {len(po_needed):,} SKUs matching '{search}'")
            
            # Display
            st.markdown("### 📋 Purchase Order List")
            
            # Different columns based on view mode
            if "Parent" in view_mode:
                display_cols = ['Priority', 'OMS_SKU', 'Variant_Count', 'Total_Inventory', 'ADS', 
                               'Stockout_Flag', 'Days_Left', 'Lead_Time_Demand', 'Target_Stock', 
                               'Safety_Stock', 'Total_Required', 'PO_Recommended']
            else:
                display_cols = ['Priority', 'OMS_SKU', 'Total_Inventory', 'ADS', 'Active_Days', 
                               'Stockout_Flag', 'Days_Left', 'Lead_Time_Demand', 'Target_Stock', 
                               'Safety_Stock', 'Total_Required', 'PO_Recommended']
            
            display_cols = [c for c in display_cols if c in po_needed.columns]
            
            # Color coding
            def highlight_priority(row):
                colors = []
                for col in row.index:
                    if col == 'Priority':
                        if '🔴' in str(row[col]):
                            colors.append('background-color: #fee2e2; font-weight: bold')
                        elif '🟡' in str(row[col]):
                            colors.append('background-color: #fef3c7')
                        else:
                            colors.append('background-color: #d1fae5')
                    elif col == 'PO_Recommended':
                        colors.append('background-color: #dbeafe; font-weight: bold; font-size: 1.1em')
                    elif col == 'Stockout_Flag' and '⚠️' in str(row[col]):
                        colors.append('background-color: #fee2e2; color: #dc2626')
                    elif col == 'Days_Left' and row[col] < lead_time:
                        colors.append('background-color: #fee2e2; font-weight: bold')
                    elif col == 'Variant_Count':
                        colors.append('background-color: #f3f4f6; font-style: italic')
                    else:
                        colors.append('')
                return colors
            
            # Format dictionary
            format_dict = {
                'ADS': '{:.2f}',
                'Days_Left': '{:.1f}',
                'Lead_Time_Demand': '{:.0f}',
                'Target_Stock': '{:.0f}',
                'Safety_Stock': '{:.0f}',
                'Total_Required': '{:.0f}',
                'PO_Recommended': '{:.0f}',
                'Total_Inventory': '{:.0f}',
                'Active_Days': '{:.0f}',
                'Variant_Count': '{:.0f}'
            }
            
            st.dataframe(
                po_needed[display_cols].head(100).style.apply(highlight_priority, axis=1)
                                                       .format(format_dict),
                use_container_width=True,
                height=500
            )
            
            st.caption(f"Showing top 100 of {len(po_needed):,} SKUs needing orders")
            
            # INTERACTIVE VARIANT VIEWER - Works in BOTH modes!
            if "Parent" in view_mode:
                st.divider()
                st.markdown("### 🔍 Variant Details (Click to Expand)")
                st.caption("💡 Select any parent SKU below to see which specific sizes/colors need ordering")
            else:
                st.divider()
                st.markdown("### 🔍 Variant Size Breakdown")
                st.caption("💡 Click any SKU below to see detailed breakdown and related variants")
            
            # Create expandable sections for top SKUs
            display_limit = st.slider(
                "Number of SKUs to show details",
                min_value=5,
                max_value=50,
                value=10,
                help="Show detailed variant breakdown for top N SKUs"
            )
            
            top_po_skus = po_needed.head(display_limit)
            
            for idx, row in top_po_skus.iterrows():
                sku = row['OMS_SKU']
                po_amount = row['PO_Recommended']
                priority = row['Priority']
                
                # Get priority emoji
                priority_emoji = '🔴' if '🔴' in priority else '🟡' if '🟡' in priority else '🟢'
                
                if "Parent" in view_mode:
                    # Parent mode - show variants
                    variants = po_df[po_df['Parent_SKU'] == sku].copy()
                    variants = variants.sort_values('PO_Recommended', ascending=False)
                    variant_count = len(variants)
                    
                    with st.expander(
                        f"{priority_emoji} **{sku}** | PO: **{po_amount:,.0f} units** | {variant_count} variants",
                        expanded=False
                    ):
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total PO", f"{po_amount:,.0f}")
                        col2.metric("Variants", variant_count)
                        col3.metric("Current Stock", f"{variants['Total_Inventory'].sum():,.0f}")
                        col4.metric("Combined ADS", f"{variants['ADS'].sum():.2f}")
                        
                        st.markdown("---")
                        st.markdown("**📦 Order Breakdown by Variant:**")
                        
                        # Create visual breakdown
                        for _, variant_row in variants.iterrows():
                            var_sku = variant_row['OMS_SKU']
                            var_po = variant_row['PO_Recommended']
                            var_stock = variant_row['Total_Inventory']
                            var_ads = variant_row['ADS']
                            var_days = variant_row['Days_Left']
                            
                            # Extract size from full SKU
                            size = var_sku.replace(sku, '').strip('-_')
                            if not size:
                                size = var_sku
                            
                            # Priority indicator
                            if var_days < lead_time:
                                urgency = "🔴 URGENT"
                                urgency_color = "#fee2e2"
                            elif var_days < lead_time + 7:
                                urgency = "🟡 HIGH"
                                urgency_color = "#fef3c7"
                            else:
                                urgency = "🟢 OK"
                                urgency_color = "#d1fae5"
                            
                            # Create nice display
                            col_a, col_b, col_c, col_d, col_e = st.columns([2, 1, 1, 1, 2])
                            
                            with col_a:
                                st.markdown(f"**{size}**")
                            with col_b:
                                st.markdown(f"Stock: {var_stock:.0f}")
                            with col_c:
                                st.markdown(f"ADS: {var_ads:.2f}")
                            with col_d:
                                st.markdown(f"{var_days:.1f} days")
                            with col_e:
                                st.markdown(f"<span style='background-color: {urgency_color}; padding: 4px 8px; border-radius: 4px;'>**Order: {var_po:.0f}** | {urgency}</span>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Size distribution chart
                        if len(variants) > 1:
                            st.markdown("**📊 PO Distribution by Size:**")
                            chart_data = variants[['OMS_SKU', 'PO_Recommended']].copy()
                            chart_data['Size'] = chart_data['OMS_SKU'].apply(lambda x: x.replace(sku, '').strip('-_') or x)
                            
                            fig = px.bar(
                                chart_data,
                                x='Size',
                                y='PO_Recommended',
                                title=f'Order Quantity by Size - {sku}',
                                labels={'PO_Recommended': 'Units to Order', 'Size': 'Size/Variant'}
                            )
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download variant-specific PO
                        variant_csv = variants[['OMS_SKU', 'Total_Inventory', 'ADS', 'Days_Left', 'PO_Recommended']].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 Download {sku} Variants",
                            variant_csv,
                            f"po_variants_{sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                            key=f"download_{sku}"
                        )
                
                else:
                    # Variant mode - show related variants and details
                    parent = get_parent_sku(sku)
                    related = po_df[po_df['Parent_SKU'] == parent].copy()
                    related = related.sort_values('PO_Recommended', ascending=False)
                    
                    with st.expander(
                        f"{priority_emoji} **{sku}** | PO: **{po_amount:,.0f} units**",
                        expanded=False
                    ):
                        # Current variant details
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("PO Needed", f"{po_amount:,.0f}")
                        col2.metric("Current Stock", f"{row['Total_Inventory']:,.0f}")
                        col3.metric("ADS", f"{row['ADS']:.2f}")
                        col4.metric("Days Left", f"{row['Days_Left']:.1f}")
                        
                        # Show calculation breakdown
                        st.markdown("---")
                        st.markdown("**🧮 Calculation Breakdown:**")
                        
                        calc_col1, calc_col2 = st.columns(2)
                        
                        with calc_col1:
                            st.markdown(f"""
                            **Requirements:**
                            - Lead Time Stock: {row['Lead_Time_Demand']:.0f} units
                            - Target Buffer: {row['Target_Stock']:.0f} units
                            - Safety Stock: {row['Safety_Stock']:.0f} units
                            - **Total Need**: {row['Total_Required']:.0f} units
                            """)
                        
                        with calc_col2:
                            st.markdown(f"""
                            **Current Status:**
                            - Current Stock: {row['Total_Inventory']:.0f} units
                            - Days Coverage: {row['Days_Left']:.1f} days
                            - **Shortfall**: {po_amount:.0f} units
                            - Priority: {priority}
                            """)
                        
                        # Show related variants if any
                        if len(related) > 1:
                            st.markdown("---")
                            st.markdown(f"**👕 Related Variants from Parent '{parent}':**")
                            
                            for _, rel_row in related.iterrows():
                                if rel_row['OMS_SKU'] == sku:
                                    continue  # Skip current SKU
                                
                                rel_sku = rel_row['OMS_SKU']
                                rel_po = rel_row['PO_Recommended']
                                rel_stock = rel_row['Total_Inventory']
                                
                                size = rel_sku.replace(parent, '').strip('-_') or rel_sku
                                
                                col_x, col_y, col_z = st.columns([3, 2, 2])
                                with col_x:
                                    st.markdown(f"• {size}")
                                with col_y:
                                    st.markdown(f"Stock: {rel_stock:.0f}")
                                with col_z:
                                    st.markdown(f"PO: {rel_po:.0f}")
                            
                            # Total for parent
                            total_parent_po = related['PO_Recommended'].sum()
                            st.info(f"💡 **Total PO for '{parent}' style: {total_parent_po:.0f} units across {len(related)} variants**")
            
            # Quick action buttons
            st.divider()
            st.markdown("### ⚡ Quick Actions")
            
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("🎯 Show Only Urgent (🔴)", use_container_width=True):
                    st.info("💡 Tip: Use the Priority filter at the top to show only urgent items")
            
            with action_col2:
                if "Parent" in view_mode:
                    if st.button("🔄 Switch to Variant View", use_container_width=True):
                        st.info("💡 Toggle the 'View Mode' radio button at the top to switch views")
                else:
                    if st.button("🔄 Switch to Parent View", use_container_width=True):
                        st.info("💡 Toggle the 'View Mode' radio button at the top to switch views")
            
            with action_col3:
                st.markdown("**📊 Total PO Value:**")
                total_po_display = po_needed['PO_Recommended'].sum()
                st.markdown(f"### {total_po_display:,.0f} units")
            
            # Download
            st.divider()
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv = po_needed[display_cols].to_csv(index=False).encode('utf-8')
                filename_suffix = "parent_sku" if "Parent" in view_mode else "variants"
                st.download_button(
                    "📥 Download PO List (CSV)",
                    csv,
                    f"po_recommendations_{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download with formulas visible
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    po_needed[display_cols].to_excel(writer, sheet_name='PO_Recommendations', index=False)
                    
                    # Add variant detail sheet if in parent mode
                    if "Parent" in view_mode and len(po_needed) > 0:
                        variant_detail_cols = ['Parent_SKU', 'OMS_SKU', 'Total_Inventory', 'ADS', 
                                              'PO_Recommended', 'Priority']
                        variant_detail = po_df[po_df['Parent_SKU'].isin(po_needed['OMS_SKU'])]
                        if not variant_detail.empty:
                            variant_detail[variant_detail_cols].to_excel(writer, sheet_name='Variant_Details', index=False)
                
                st.download_button(
                    "📥 Download PO List (Excel)",
                    excel_buffer.getvalue(),
                    f"po_recommendations_{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
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
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Movements", f"{len(transfer_df):,}")
        c2.metric("Units Transferred", f"{transfer_df['Quantity'].sum():,.0f}")
        c3.metric("FC Transfers", f"{len(transfer_df[transfer_df['Transaction Type'] == 'FC_TRANSFER']):,}")
        c4.metric("Unique Routes", f"{len(transfer_df.groupby(['Ship From Fc', 'Ship To Fc'])):,}")
        
        st.divider()
        
        # Top routes
        st.markdown("### 🔝 Top Transfer Routes")
        routes = transfer_df.groupby(['Ship From Fc', 'Ship To Fc']).agg({
            'Quantity': ['sum', 'count']
        }).reset_index()
        routes.columns = ['From FC', 'To FC', 'Units', 'Transfers']
        routes = routes.sort_values('Units', ascending=False).head(20)
        
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
        sku = st.selectbox("Select SKU", [""] + sorted(sales['Sku'].dropna().unique().tolist()))
        days = st.slider("Forecast Days", 7, 90, 30)
        
        if sku:
            subset = sales[sales['Sku'] == sku].copy()
            subset['ds'] = pd.to_datetime(subset['TxnDate']).dt.date
            daily = subset.groupby('ds')['Units_Effective'].sum().reset_index()
            daily.columns = ['ds', 'y']
            daily['ds'] = pd.to_datetime(daily['ds'])
            
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
                    fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y'], name='Actual'))
                    future_only = forecast[forecast['ds'] > daily['ds'].max()]
                    fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], 
                                            name='Forecast', line=dict(dash='dash')))
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
        # Search and period controls
        colA, colB, colC = st.columns([3, 2, 1])
        
        with colA:
            search = st.text_input("Enter SKU", placeholder="e.g., 1065YK", key="drill_search")
        
        with colB:
            period = st.selectbox(
                "Period", 
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"], 
                index=1,
                key="drill_period"
            )
        
        with colC:
            grace = st.number_input(
                "Grace Days",
                min_value=0,
                max_value=14,
                value=7,
                help="Buffer for late transactions",
                key="drill_grace"
            )
        
        # Filter by period with grace
        fdf = df.copy()
        fdf['TxnDate'] = pd.to_datetime(fdf['TxnDate'], errors='coerce')
        max_d = fdf['TxnDate'].max()
        
        if period != "All Time" and not pd.isna(max_d):
            # Extract base days from period
            base_days = 7 if "7" in period else 30 if "30" in period else 90
            
            # Add grace period
            total_days = base_days + grace
            cutoff = max_d - timedelta(days=total_days)
            
            fdf = fdf[fdf['TxnDate'] >= cutoff]
            
            # Show actual date range
            actual_min = fdf['TxnDate'].min()
            date_range_text = f"{actual_min.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')} " \
                            f"({base_days} days + {grace} day grace)"
        else:
            date_range_text = f"All Time: {fdf['TxnDate'].min().strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"
        
        # Display date range and transaction count
        st.info(f"📅 **Period:** {date_range_text} | **Transactions:** {len(fdf):,}")
        
        if search:
            matches = fdf[fdf['Sku'].str.contains(search, case=False, na=False)].copy()
            
            if matches.empty:
                st.warning("No matching SKUs found.")
            else:
                # Show number of variants found
                unique_skus = matches['Sku'].nunique()
                st.success(f"✅ Found **{unique_skus}** SKU variant(s) matching '{search}'")
                
                st.divider()
                
                # Metrics
                sold = matches[matches['Transaction Type'] == 'Shipment']['Quantity'].sum()
                ret = matches[matches['Transaction Type'] == 'Refund']['Quantity'].sum()
                net = matches['Units_Effective'].sum()
                rate = (ret / sold * 100) if sold > 0 else 0
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sold Pieces", f"{int(sold):,}")
                m2.metric("Returns", f"{int(ret):,}")
                m3.metric("Net Units", f"{int(net):,}")
                m4.metric("Return %", f"{rate:.1f}%")
                
                st.divider()
                
                # Panel breakdown
                st.markdown("### 🏪 Marketplace Breakdown (Panel-wise)")
                sold_p, ret_p, net_p = build_panel_pivots(matches)
                
                combined = pd.concat([sold_p, ret_p, net_p], axis=1).fillna(0).reset_index()
                
                if not combined.empty:
                    st.dataframe(combined, use_container_width=True, height=400)
                    
                    # Panel chart
                    st.markdown("### 📊 Sales by Marketplace")
                    if not sold_p.empty:
                        # Create chart data from sold pivot
                        chart_data = []
                        for sku in sold_p.index:
                            for col in sold_p.columns:
                                panel = col.replace(' | Sold', '')
                                qty = sold_p.loc[sku, col]
                                if qty > 0:
                                    chart_data.append({
                                        'SKU': sku,
                                        'Marketplace': panel,
                                        'Sold Pieces': qty
                                    })
                        
                        if chart_data:
                            chart_df = pd.DataFrame(chart_data)
                            
                            # If multiple SKUs, show stacked by SKU
                            if unique_skus > 1:
                                fig = px.bar(
                                    chart_df, 
                                    x='Marketplace', 
                                    y='Sold Pieces',
                                    color='SKU',
                                    title=f'Sales Distribution Across Marketplaces ({unique_skus} SKUs)'
                                )
                            else:
                                # Single SKU - simple bar chart
                                fig = px.bar(
                                    chart_df,
                                    x='Marketplace',
                                    y='Sold Pieces',
                                    title='Sales by Marketplace'
                                )
                            
                            fig.update_xaxes(tickangle=-25)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No panel breakdown available for selected period.")
                
                st.markdown("---")
                
                # Variant breakdown (if multiple SKUs found)
                if unique_skus > 1:
                    st.markdown("### 📋 Variant Comparison")
                    
                    variant_summary = matches.groupby('Sku').agg({
                        'Quantity': lambda x: x[matches.loc[x.index, 'Transaction Type'] == 'Shipment'].sum(),
                        'Units_Effective': 'sum'
                    }).reset_index()
                    variant_summary.columns = ['SKU', 'Sold Pieces', 'Net Units']
                    variant_summary = variant_summary.sort_values('Sold Pieces', ascending=False)
                    
                    # Add percentage
                    total_sold = variant_summary['Sold Pieces'].sum()
                    variant_summary['% of Total'] = (variant_summary['Sold Pieces'] / total_sold * 100).round(1)
                    
                    st.dataframe(variant_summary, use_container_width=True, hide_index=True)
                    
                    # Variant chart
                    fig = px.bar(
                        variant_summary.head(10),
                        x='SKU',
                        y='Sold Pieces',
                        title='Top 10 Variants by Sales',
                        text='% of Total'
                    )
                    fig.update_traces(texttemplate='%{text}%', textposition='outside')
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                
                # Recent transactions
                st.markdown("### 📜 Recent Transactions (Latest 100)")
                cols = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Source']
                if 'OrderId' in matches.columns:
                    cols.append('OrderId')
                
                # Add units effective
                if 'Units_Effective' in matches.columns:
                    cols.append('Units_Effective')
                
                display_txns = matches.sort_values('TxnDate', ascending=False).head(100)[cols].copy()
                
                # Format date
                display_txns['TxnDate'] = display_txns['TxnDate'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(display_txns, use_container_width=True, height=400)
                
                # Download option
                st.markdown("---")
                csv = matches[cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Transaction History",
                    csv,
                    f"sku_drilldown_{search}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        else:
            st.info("💡 **Tip:** Type a SKU (or partial SKU) in the search box above to start analysis.")
            
            # Show sample of available SKUs
            st.markdown("### 📦 Sample SKUs Available")
            sample_skus = fdf['Sku'].value_counts().head(20).reset_index()
            sample_skus.columns = ['SKU', 'Transaction Count']
            st.dataframe(sample_skus, use_container_width=True, height=300)

st.divider()
st.caption("💡 Yash Gallery ERP v3.0 | All data in OMS SKU format | Refactored & Production Ready")

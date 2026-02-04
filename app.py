#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System
- Inventory consolidation with OMS SKU mapping
- Sales analytics from all marketplaces (ZIP support)
- PO/Reorder recommendations
- AI Demand Forecasting
- Multi-warehouse management
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Yash Gallery ERP",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Yash Gallery Complete ERP System")
st.caption("Inventory + Sales + PO Recommendations with OMS SKU Mapping")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'sku_mapping' not in st.session_state:
    st.session_state.sku_mapping = {}
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()

# ==========================================
# HELPER FUNCTIONS - SKU MAPPING
# ==========================================

def load_sku_mapping(mapping_file) -> dict:
    """Load SKU mapping from Excel with multiple sheets."""
    mapping_dict = {}
    
    try:
        xls = pd.ExcelFile(mapping_file)
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(mapping_file, sheet_name=sheet_name)
            
            # Find columns
            seller_col = None
            oms_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['seller', 'myntra', 'messho', 'snapdeal', 'sku id']):
                    if 'sku' in col_lower:
                        seller_col = col
                elif 'oms' in col_lower and 'sku' in col_lower:
                    oms_col = col
            
            if seller_col is None and len(df.columns) > 1:
                seller_col = df.columns[1]
            if oms_col is None:
                oms_col = df.columns[-1]
            
            if seller_col and oms_col:
                for _, row in df.iterrows():
                    seller_sku = str(row[seller_col]).strip()
                    oms_sku = str(row[oms_col]).strip()
                    
                    if seller_sku and oms_sku and seller_sku != 'nan' and oms_sku != 'nan':
                        mapping_dict[seller_sku] = oms_sku
        
        return mapping_dict
        
    except Exception as e:
        st.error(f"Error loading SKU mapping: {e}")
        return {}


def map_to_oms_sku(seller_sku, mapping_dict):
    """Map seller SKU to OMS SKU with cleaning."""
    if pd.isna(seller_sku):
        return None
    
    # Clean SKU
    clean_sku = str(seller_sku).strip().replace('"""', '').replace('SKU:', '').strip()
    return mapping_dict.get(clean_sku, clean_sku)


# ==========================================
# FILE READING HELPERS
# ==========================================

def read_zip_or_csv(file_obj, is_zip=False):
    """Read CSV from ZIP or direct file."""
    try:
        if is_zip:
            with zipfile.ZipFile(file_obj, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    with zip_ref.open(csv_files[0]) as f:
                        return pd.read_csv(f)
        else:
            file_obj.seek(0)
            return pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return pd.DataFrame()


# ==========================================
# SALES DATA LOADERS
# ==========================================

def load_amazon_sales(file_obj, sku_mapping, source_name):
    """Load Amazon sales from ZIP."""
    try:
        df = read_zip_or_csv(file_obj, is_zip=True)
        
        if df.empty or 'Sku' not in df.columns:
            return pd.DataFrame()
        
        # Map to OMS SKU
        df['OMS_SKU'] = df['Sku'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        df['Source'] = source_name
        
        # Transaction type
        df['TxnType'] = df['Transaction Type'].apply(
            lambda x: 'Refund' if 'return' in str(x).lower() or 'refund' in str(x).lower()
            else 'Cancel' if 'cancel' in str(x).lower()
            else 'Shipment'
        )
        
        # Calculate units
        df['Units_Effective'] = df.apply(
            lambda row: -row['Quantity'] if row['TxnType'] == 'Refund' else row['Quantity'], axis=1
        )
        
        sales = df[['OMS_SKU', 'Order Date', 'TxnType', 'Quantity', 'Units_Effective', 'Source']].copy()
        sales.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source']
        
        st.success(f"✅ {source_name}: {len(sales):,} records")
        return sales
        
    except Exception as e:
        st.error(f"Error loading {source_name}: {e}")
        return pd.DataFrame()


def load_flipkart_sales(file_obj, sku_mapping):
    """Load Flipkart sales."""
    try:
        df = pd.read_excel(file_obj, sheet_name='Sales Report')
        
        # Clean and map SKU
        df['SKU_Clean'] = df['SKU'].astype(str).str.replace('"""', '').str.replace('SKU:', '').str.strip()
        df['OMS_SKU'] = df['SKU_Clean'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Item Quantity'], errors='coerce').fillna(0)
        df['Source'] = 'Flipkart'
        
        df['TxnType'] = df['Event Sub Type'].apply(
            lambda x: 'Refund' if 'return' in str(x).lower() else 'Shipment'
        )
        
        df['Units_Effective'] = df.apply(
            lambda row: -row['Quantity'] if row['TxnType'] == 'Refund' else row['Quantity'], axis=1
        )
        
        sales = df[['OMS_SKU', 'Order Date', 'TxnType', 'Quantity', 'Units_Effective', 'Source']].copy()
        sales.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source']
        
        st.success(f"✅ Flipkart: {len(sales):,} records")
        return sales
        
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}")
        return pd.DataFrame()


def load_meesho_sales(file_obj, sku_mapping):
    """Load Meesho sales from ZIP."""
    try:
        with zipfile.ZipFile(file_obj, 'r') as zip_ref:
            excel_files = [f for f in zip_ref.namelist() if 'tcs_sales.xlsx' in f and 'return' not in f.lower()]
            
            if excel_files:
                with zip_ref.open(excel_files[0]) as f:
                    df = pd.read_excel(f)
                    
                    df['OMS_SKU'] = df['identifier'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
                    df['Order Date'] = pd.to_datetime(df['order_date'], errors='coerce')
                    df['Quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
                    df['Source'] = 'Meesho'
                    df['TxnType'] = 'Shipment'
                    df['Units_Effective'] = df['Quantity']
                    
                    sales = df[['OMS_SKU', 'Order Date', 'TxnType', 'Quantity', 'Units_Effective', 'Source']].copy()
                    sales.columns = ['Sku', 'TxnDate', 'Transaction Type', 'Quantity', 'Units_Effective', 'Source']
                    
                    st.success(f"✅ Meesho: {len(sales):,} records")
                    return sales
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Meesho: {e}")
        return pd.DataFrame()


def load_stock_transfer(file_obj):
    """Load Amazon Stock Transfer data (FC movements)."""
    try:
        df = read_zip_or_csv(file_obj, is_zip=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Get relevant columns
        transfer_df = df[['Invoice Date', 'Ship From Fc', 'Ship To Fc', 'Quantity', 'Transaction Type']].copy()
        transfer_df['Invoice Date'] = pd.to_datetime(transfer_df['Invoice Date'], errors='coerce')
        transfer_df['Quantity'] = pd.to_numeric(transfer_df['Quantity'], errors='coerce').fillna(0)
        
        st.success(f"✅ Stock Transfer: {len(transfer_df):,} movements, {transfer_df['Quantity'].sum():,.0f} units")
        return transfer_df
        
    except Exception as e:
        st.error(f"Error loading Stock Transfer: {e}")
        return pd.DataFrame()



# ==========================================
# INVENTORY DATA LOADERS
# ==========================================

def load_oms_inventory(file_obj):
    """Load OMS inventory."""
    try:
        df = pd.read_csv(file_obj)
        inv = df[['Item SkuCode', 'Inventory']].copy()
        inv.columns = ['OMS_SKU', 'OMS_Inventory']
        inv = inv.drop_duplicates(subset=['OMS_SKU'], keep='first')
        st.success(f"✅ OMS: {len(inv):,} SKUs, {inv['OMS_Inventory'].sum():,.0f} units")
        return inv
    except Exception as e:
        st.error(f"Error loading OMS: {e}")
        return pd.DataFrame()


def load_flipkart_inventory(file_obj, sku_mapping):
    """Load Flipkart inventory."""
    try:
        df = pd.read_csv(file_obj)
        df['OMS_SKU'] = df['SKU'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        
        inv = df.groupby('OMS_SKU').agg({
            'Live on Website': 'sum'
        }).reset_index()
        inv.columns = ['OMS_SKU', 'Flipkart_Live']
        
        st.success(f"✅ Flipkart Inv: {len(inv):,} SKUs, {inv['Flipkart_Live'].sum():,.0f} live")
        return inv
    except Exception as e:
        st.error(f"Error loading Flipkart inventory: {e}")
        return pd.DataFrame()


def load_myntra_inventory(file_obj, sku_mapping):
    """Load Myntra inventory."""
    try:
        df = pd.read_csv(file_obj)
        
        if 'seller sku code' in df.columns:
            df['OMS_SKU'] = df['seller sku code'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        else:
            df['OMS_SKU'] = df['sku code'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        
        inv = df.groupby('OMS_SKU').agg({
            'sellable inventory count': 'sum'
        }).reset_index()
        inv.columns = ['OMS_SKU', 'Myntra_Inventory']
        
        st.success(f"✅ Myntra Inv: {len(inv):,} SKUs, {inv['Myntra_Inventory'].sum():,.0f} units")
        return inv
    except Exception as e:
        st.error(f"Error loading Myntra inventory: {e}")
        return pd.DataFrame()


def load_amazon_inventory(file_obj, sku_mapping):
    """Load Amazon inventory (excluding ZNNE to avoid OMS duplicates)."""
    try:
        df = pd.read_csv(file_obj)
        
        # CRITICAL: Exclude ZNNE location - this is already in OMS inventory (VDN/TTP)
        # ZNNE = Your own warehouse, not Amazon FC
        if 'Location' in df.columns:
            original_count = len(df)
            df = df[df['Location'] != 'ZNNE']
            excluded_count = original_count - len(df)
            if excluded_count > 0:
                st.info(f"ℹ️ Excluded {excluded_count} ZNNE records (already in OMS inventory)")
        
        df['OMS_SKU'] = df['MSKU'].apply(lambda x: map_to_oms_sku(x, sku_mapping))
        
        inv = df.groupby('OMS_SKU').agg({
            'Ending Warehouse Balance': 'sum'
        }).reset_index()
        inv.columns = ['OMS_SKU', 'Amazon_Inventory']
        
        st.success(f"✅ Amazon Inv: {len(inv):,} SKUs, {inv['Amazon_Inventory'].sum():,.0f} units (FCs only, excluding ZNNE)")
        return inv
    except Exception as e:
        st.error(f"Error loading Amazon inventory: {e}")
        return pd.DataFrame()


# ==========================================
# SIDEBAR - FILE UPLOADS
# ==========================================

st.sidebar.header("📂 File Uploads")

# SKU Mapping
st.sidebar.markdown("### 1️⃣ SKU Mapping (Required)")
mapping_file = st.sidebar.file_uploader(
    "SKU Mapping Excel",
    type=['xlsx'],
    key='mapping',
    help="Copy_of_All_penal_replace_sku.xlsx"
)

if mapping_file:
    st.session_state.sku_mapping = load_sku_mapping(mapping_file)
    st.sidebar.success(f"✅ {len(st.session_state.sku_mapping):,} mappings")

st.sidebar.divider()

# Sales Files
st.sidebar.markdown("### 2️⃣ Sales Data")

b2c_sales = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=['zip'], key='b2c')
b2b_sales = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=['zip'], key='b2b')
transfer_sales = st.sidebar.file_uploader("Amazon Stock Transfer (ZIP)", type=['zip'], key='transfer', 
                                         help="FC-to-FC movement tracking (optional)")
flipkart_sales = st.sidebar.file_uploader("Flipkart Sales (Excel)", type=['xlsx'], key='fk_sales')
meesho_sales = st.sidebar.file_uploader("Meesho Sales (ZIP)", type=['zip'], key='meesho')

st.sidebar.divider()

# Inventory Files
st.sidebar.markdown("### 3️⃣ Inventory Data")

oms_inv = st.sidebar.file_uploader("OMS Inventory", type=['csv'], key='oms_inv')
flipkart_inv = st.sidebar.file_uploader("Flipkart Inventory", type=['csv'], key='fk_inv')
myntra_inv = st.sidebar.file_uploader("Myntra Inventory", type=['csv'], key='myntra_inv')
amazon_inv = st.sidebar.file_uploader("Amazon Inventory", type=['csv'], key='amz_inv')

# ==========================================
# MAIN APP LOGIC
# ==========================================

if not st.session_state.sku_mapping:
    st.info("""
    ### 👋 Welcome to Yash Gallery ERP
    
    **Step 1:** Upload SKU Mapping file (sidebar)
    **Step 2:** Upload Sales files (Amazon B2C/B2B, Flipkart, Meesho)
    **Step 3:** Upload Inventory files (OMS, marketplaces)
    **Step 4:** View analytics, generate POs, forecast demand!
    
    📄 Required: `Copy_of_All_penal_replace_sku.xlsx`
    """)
    st.stop()

st.divider()

# Process Sales Data
sales_dfs = []

if b2c_sales:
    sales_dfs.append(load_amazon_sales(b2c_sales, st.session_state.sku_mapping, "Amazon B2C"))
if b2b_sales:
    sales_dfs.append(load_amazon_sales(b2b_sales, st.session_state.sku_mapping, "Amazon B2B"))
if flipkart_sales:
    sales_dfs.append(load_flipkart_sales(flipkart_sales, st.session_state.sku_mapping))
if meesho_sales:
    sales_dfs.append(load_meesho_sales(meesho_sales, st.session_state.sku_mapping))

if sales_dfs:
    st.session_state.sales_df = pd.concat([df for df in sales_dfs if not df.empty], ignore_index=True)

# Process Stock Transfer Data (optional)
if 'transfer_df' not in st.session_state:
    st.session_state.transfer_df = pd.DataFrame()

if transfer_sales:
    st.session_state.transfer_df = load_stock_transfer(transfer_sales)

# Process Inventory Data
inv_dfs = []

if oms_inv:
    inv_dfs.append(load_oms_inventory(oms_inv))
if flipkart_inv and st.session_state.sku_mapping:
    inv_dfs.append(load_flipkart_inventory(flipkart_inv, st.session_state.sku_mapping))
if myntra_inv and st.session_state.sku_mapping:
    inv_dfs.append(load_myntra_inventory(myntra_inv, st.session_state.sku_mapping))
if amazon_inv and st.session_state.sku_mapping:
    inv_dfs.append(load_amazon_inventory(amazon_inv, st.session_state.sku_mapping))

if inv_dfs:
    # Merge all inventory
    consolidated_inv = inv_dfs[0]
    for df in inv_dfs[1:]:
        consolidated_inv = pd.merge(consolidated_inv, df, on='OMS_SKU', how='outer')
    
    # Fill NaN
    numeric_cols = consolidated_inv.select_dtypes(include=[np.number]).columns
    consolidated_inv[numeric_cols] = consolidated_inv[numeric_cols].fillna(0)
    
    # ===== PARENT SKU GROUPING =====
    # Remove marketplace suffixes to get parent SKU
    # Examples: 1001YKBEIGE-3XL_Myntra → 1001YKBEIGE-3XL
    #           1001PLYKBEIGE-3XL → 1001YKBEIGE-3XL (via mapping)
    
    def get_parent_sku(oms_sku):
        """Remove marketplace suffixes like _Myntra, _Flipkart, etc."""
        if pd.isna(oms_sku):
            return oms_sku
        sku_str = str(oms_sku)
        # Remove common suffixes
        for suffix in ['_Myntra', '_Flipkart', '_Amazon', '_Meesho', '_MYNTRA', '_FLIPKART']:
            if sku_str.endswith(suffix):
                sku_str = sku_str.replace(suffix, '')
                break
        return sku_str
    
    consolidated_inv['Parent_SKU'] = consolidated_inv['OMS_SKU'].apply(get_parent_sku)
    
    # Group by Parent SKU and sum all inventory
    inv_cols = [c for c in consolidated_inv.columns if 'Inventory' in c or 'Live' in c]
    
    agg_dict = {col: 'sum' for col in inv_cols}
    consolidated_inv_grouped = consolidated_inv.groupby('Parent_SKU').agg(agg_dict).reset_index()
    
    # Rename Parent_SKU back to OMS_SKU for consistency
    consolidated_inv_grouped = consolidated_inv_grouped.rename(columns={'Parent_SKU': 'OMS_SKU'})
    
    # Calculate totals
    consolidated_inv_grouped['Total_Inventory'] = consolidated_inv_grouped[inv_cols].sum(axis=1)
    
    # Remove rows where all inventory is 0
    consolidated_inv_grouped = consolidated_inv_grouped[consolidated_inv_grouped['Total_Inventory'] > 0]
    
    st.session_state.inventory_df = consolidated_inv_grouped
    
    st.success(f"✅ Consolidated to {len(consolidated_inv_grouped):,} parent SKUs (removed marketplace duplicates)")


# ==========================================
# TABS
# ==========================================

tab_dashboard, tab_inventory, tab_po, tab_logistics, tab_forecast, tab_sku = st.tabs([
    "📊 Sales Dashboard",
    "📦 Inventory View", 
    "🎯 PO Recommendations",
    "🚚 Logistics & Transfers",
    "🤖 AI Forecast",
    "🔍 SKU Drilldown"
])

# ==========================================
# TAB 1: SALES DASHBOARD
# ==========================================

with tab_dashboard:
    st.subheader("📊 Sales Analytics Dashboard")
    
    if st.session_state.sales_df.empty:
        st.warning("⚠️ No sales data loaded. Upload sales files in sidebar.")
    else:
        sales_df = st.session_state.sales_df
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        shipments = len(sales_df[sales_df['Transaction Type'] == 'Shipment'])
        returns = len(sales_df[sales_df['Transaction Type'] == 'Refund'])
        net_units = sales_df['Units_Effective'].sum()
        return_rate = (returns / shipments * 100) if shipments > 0 else 0
        
        col1.metric("📦 Total Transactions", f"{len(sales_df):,}")
        col2.metric("✅ Shipments", f"{shipments:,}")
        col3.metric("↩️ Returns", f"{returns:,}")
        col4.metric("📊 Return Rate", f"{return_rate:.1f}%")
        
        st.divider()
        
        # Sales by marketplace
        st.markdown("### 📊 Sales by Marketplace")
        
        source_summary = sales_df.groupby('Source').agg({
            'Units_Effective': 'sum',
            'Sku': 'count'
        }).reset_index()
        source_summary.columns = ['Marketplace', 'Net Units', 'Orders']
        source_summary = source_summary.sort_values('Orders', ascending=False)
        
        col_chart, col_table = st.columns([1, 1])
        
        with col_chart:
            fig = px.pie(source_summary, values='Orders', names='Marketplace',
                        title='Orders Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.dataframe(source_summary, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Top SKUs
        st.markdown("### 🏆 Top 20 Selling SKUs")
        
        top_skus = sales_df[sales_df['Transaction Type'] == 'Shipment'].groupby('Sku').agg({
            'Units_Effective': 'sum'
        }).reset_index().sort_values('Units_Effective', ascending=False).head(20)
        
        fig = px.bar(top_skus, x='Sku', y='Units_Effective', title='Top Sellers')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: INVENTORY VIEW
# ==========================================

with tab_inventory:
    st.subheader("📦 Consolidated Inventory")
    
    if st.session_state.inventory_df.empty:
        st.warning("⚠️ No inventory data loaded. Upload inventory files in sidebar.")
    else:
        inv_df = st.session_state.inventory_df
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("📦 Total SKUs", f"{len(inv_df):,}")
        col2.metric("📊 Total Units", f"{inv_df['Total_Inventory'].sum():,.0f}")
        
        if 'OMS_Inventory' in inv_df.columns:
            col3.metric("🏢 OMS Warehouse", f"{inv_df['OMS_Inventory'].sum():,.0f}")
        
        marketplace_inv = inv_df['Total_Inventory'].sum() - (inv_df['OMS_Inventory'].sum() if 'OMS_Inventory' in inv_df.columns else 0)
        col4.metric("🌐 Marketplace Total", f"{marketplace_inv:,.0f}")
        
        st.divider()
        
        # Search
        search_sku = st.text_input("🔍 Search SKU", placeholder="e.g., 1065YKBLUE")
        
        display_df = inv_df.copy()
        if search_sku:
            display_df = display_df[display_df['OMS_SKU'].str.contains(search_sku, case=False, na=False)]
        
        st.dataframe(display_df, use_container_width=True, height=500)

# ==========================================
# TAB 3: PO RECOMMENDATIONS
# ==========================================

with tab_po:
    st.subheader("🎯 Purchase Order Recommendations")
    
    if st.session_state.sales_df.empty or st.session_state.inventory_df.empty:
        st.error("""
        ### ⚠️ Missing Required Data
        
        PO calculations require:
        - ✅ **Sales data** (to calculate demand velocity)
        - ✅ **Inventory data** (to know current stock)
        
        📤 Upload both in the sidebar to generate PO recommendations.
        """)
    else:
        # Configuration
        st.markdown("### ⚙️ Configuration")
        
        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
        
        with col_cfg1:
            velocity_period = st.selectbox(
                "Velocity Calculation",
                ["Last 30 Days", "Last 60 Days", "Last 90 Days"],
                help="Period to calculate Average Daily Sales (ADS)"
            )
            days_back = 30 if "30" in velocity_period else (60 if "60" in velocity_period else 90)
        
        with col_cfg2:
            target_days = st.number_input(
                "Target Stock (Days)",
                min_value=30,
                max_value=180,
                value=60,
                help="How many days of stock to maintain"
            )
        
        with col_cfg3:
            lead_time = st.number_input(
                "Lead Time (Days)",
                min_value=1,
                max_value=90,
                value=15,
                help="Days from order to receiving stock"
            )
        
        with col_cfg4:
            safety_factor = st.slider(
                "Safety Stock %",
                min_value=0,
                max_value=50,
                value=20,
                help="Extra buffer stock percentage"
            )
        
        st.divider()
        
        # Calculate demand
        sales_df = st.session_state.sales_df
        inv_df = st.session_state.inventory_df
        
        max_date = sales_df['TxnDate'].max()
        cutoff_date = max_date - timedelta(days=days_back)
        
        # Filter to velocity period
        recent_sales = sales_df[sales_df['TxnDate'] >= cutoff_date].copy()
        
        # ===== SMART ADS CALCULATION WITH STOCKOUT DETECTION =====
        
        # Step 1: Get sales summary
        recent_sales['Date'] = recent_sales['TxnDate'].dt.date
        
        sales_summary = recent_sales.groupby('Sku').agg({
            'Units_Effective': 'sum',
            'Quantity': 'count',
            'Date': 'nunique'  # Days with sales
        }).reset_index()
        sales_summary.columns = ['OMS_SKU', 'Total_Units_Sold', 'Transactions', 'Days_With_Sales']
        
        # Step 2: Calculate running inventory to detect stockouts
        # For each SKU, simulate daily inventory backwards from current stock
        
        def calculate_available_days(row):
            """
            Calculate how many days product was actually available for sale.
            Works backwards from current inventory using daily sales.
            """
            sku = row['OMS_SKU']
            current_stock = row.get('Total_Inventory', 0)
            
            if current_stock <= 0:
                # Currently out of stock - check if there were any sales
                sku_sales = recent_sales[recent_sales['Sku'] == sku].copy()
                if len(sku_sales) > 0:
                    # Had sales before, now stockout
                    days_with_sales = len(sku_sales['Date'].unique())
                    return days_with_sales, 0, days_back - days_with_sales
                else:
                    # No sales and no stock - was out of stock entire period
                    return 0, 0, days_back
            
            # Get daily sales for this SKU
            sku_sales = recent_sales[recent_sales['Sku'] == sku].copy()
            
            if len(sku_sales) == 0:
                # Has stock but no sales in period - available all days
                return days_back, 0, 0
            
            # Group by date and sum sales
            daily_sales = sku_sales.groupby('Date')['Units_Effective'].sum().sort_index(ascending=False)
            
            # Simulate backwards from today
            running_stock = current_stock
            available_days = 0
            stockout_days = 0
            
            # Create date range for entire period
            date_range = pd.date_range(end=max_date, periods=days_back, freq='D')
            
            for date in date_range[::-1]:  # Go backwards from most recent
                date_key = date.date()
                daily_sale = daily_sales.get(date_key, 0)
                
                # Add back the sales that happened on this day
                running_stock += daily_sale
                
                if running_stock > 0:
                    available_days += 1
                else:
                    stockout_days += 1
            
            no_sales_days = days_back - available_days - stockout_days
            return available_days, stockout_days, no_sales_days
        
        # Merge with inventory
        po_df = pd.merge(inv_df, sales_summary, on='OMS_SKU', how='left')
        po_df['Total_Units_Sold'] = po_df['Total_Units_Sold'].fillna(0)
        po_df['Transactions'] = po_df['Transactions'].fillna(0)
        po_df['Days_With_Sales'] = po_df['Days_With_Sales'].fillna(0)
        
        # Calculate available days with stockout detection
        availability_data = po_df.apply(calculate_available_days, axis=1, result_type='expand')
        availability_data.columns = ['Days_Available', 'Days_Stockout', 'Days_No_Activity']
        
        po_df = pd.concat([po_df, availability_data], axis=1)
        
        # Calculate smart ADS based on available days only
        # Use whichever is greater: days with sales or calculated available days
        po_df['Active_Days'] = po_df[['Days_With_Sales', 'Days_Available']].max(axis=1)
        
        # Minimum 7 days to avoid inflated ADS from sporadic sales
        po_df['Active_Days'] = po_df['Active_Days'].clip(lower=7)
        
        # Calculate ADS = Total Sales / Active Available Days
        po_df['ADS'] = np.where(
            po_df['Active_Days'] > 0,
            po_df['Total_Units_Sold'] / po_df['Active_Days'],
            0
        )
        
        # Add period column and stockout flag
        po_df['ADS_Period'] = po_df['Active_Days'].astype(int)
        po_df['Stockout_Flag'] = po_df['Days_Stockout'].apply(
            lambda x: f'⚠️ {int(x)}d OOS' if x > 0 else ''
        )
        
        # Calculate days left
        po_df['Days_Left'] = np.where(
            po_df['ADS'] > 0,
            po_df['Total_Inventory'] / po_df['ADS'],
            999
        )
        
        # Calculate requirements
        po_df['Lead_Time_Demand'] = po_df['ADS'] * lead_time
        po_df['Safety_Stock'] = po_df['Lead_Time_Demand'] * (safety_factor / 100)
        po_df['Target_Stock'] = po_df['ADS'] * target_days
        po_df['Total_Required'] = po_df['Target_Stock'] + po_df['Safety_Stock']
        po_df['PO_Needed'] = (po_df['Total_Required'] - po_df['Total_Inventory']).clip(lower=0)
        
        # Round PO to nearest 5 or 10
        po_df['PO_Recommended'] = (np.ceil(po_df['PO_Needed'] / 5) * 5).astype(int)
        
        # Priority classification
        def get_priority(row):
            if row['Days_Left'] < lead_time and row['PO_Recommended'] > 0:
                return '🔴 URGENT'
            elif row['Days_Left'] < lead_time + 7 and row['PO_Recommended'] > 0:
                return '🟡 HIGH'
            elif row['PO_Recommended'] > 0:
                return '🟢 MEDIUM'
            else:
                return '⚪ OK'
        
        po_df['Priority'] = po_df.apply(get_priority, axis=1)
        
        # Filter to SKUs needing orders
        po_needed = po_df[po_df['PO_Recommended'] > 0].copy()
        po_needed = po_needed.sort_values(['Priority', 'Days_Left'])
        
        # Summary metrics
        st.markdown("### 📊 Summary")
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        urgent = len(po_needed[po_needed['Priority'] == '🔴 URGENT'])
        high = len(po_needed[po_needed['Priority'] == '🟡 HIGH'])
        medium = len(po_needed[po_needed['Priority'] == '🟢 MEDIUM'])
        total_po_units = po_needed['PO_Recommended'].sum()
        
        sum_col1.metric("🔴 Urgent Orders", urgent)
        sum_col2.metric("🟡 High Priority", high)
        sum_col3.metric("🟢 Medium Priority", medium)
        sum_col4.metric("📦 Total Units to Order", f"{total_po_units:,.0f}")
        
        st.divider()
        
        # Filters
        st.markdown("### 🔍 Filter SKUs")
        
        filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
        
        with filter_col1:
            sku_search = st.text_input(
                "Search SKU",
                placeholder="e.g., 1309YK or RED or -XL",
                help="Search by SKU code, color, size"
            )
        
        with filter_col2:
            priority_filter = st.multiselect(
                "Priority",
                ['🔴 URGENT', '🟡 HIGH', '🟢 MEDIUM'],
                default=['🔴 URGENT', '🟡 HIGH', '🟢 MEDIUM']
            )
        
        with filter_col3:
            warehouse_filter = st.selectbox(
                "Warehouse View",
                ["All", "OMS Low Stock", "Marketplace Low Stock"]
            )
        
        # Apply filters
        display_po = po_needed.copy()
        
        if sku_search:
            display_po = display_po[display_po['OMS_SKU'].str.contains(sku_search, case=False, na=False)]
        
        if priority_filter:
            display_po = display_po[display_po['Priority'].isin(priority_filter)]
        
        if warehouse_filter == "OMS Low Stock" and 'OMS_Inventory' in display_po.columns:
            display_po = display_po[display_po['OMS_Inventory'] < display_po['ADS'] * lead_time]
        elif warehouse_filter == "Marketplace Low Stock":
            marketplace_cols = [c for c in display_po.columns if any(x in c for x in ['Flipkart', 'Myntra', 'Amazon']) and 'Inventory' in c]
            if marketplace_cols:
                display_po['Marketplace_Stock'] = display_po[marketplace_cols].sum(axis=1)
                display_po = display_po[display_po['Marketplace_Stock'] < display_po['ADS'] * 7]
        
        # Display PO table
        st.markdown(f"### 📋 Purchase Orders Needed: {len(display_po):,} SKUs")
        
        # Info about smart ADS calculation
        st.info("""
        **📊 Smart ADS Calculation with Stockout Detection:**
        - ADS calculated based on **days product was available** for sale (had inventory)
        - System works backwards from current stock to detect stockout periods
        - **⚠️ Stockout Flag**: Shows days product was out of stock (e.g., "⚠️ 5d OOS" = out of stock 5 days)
        - **ADS_Period**: Shows actual available days used for calculation
        - Products with stockouts get higher priority as demand may be suppressed
        - Example: If out of stock 10 days, ADS based on remaining 20 available days
        """)
        
        # Select columns to display
        display_cols = ['Priority', 'OMS_SKU', 'Total_Inventory']
        
        # Add warehouse columns if available
        if 'OMS_Inventory' in display_po.columns:
            display_cols.append('OMS_Inventory')
        if 'Flipkart_Live' in display_po.columns:
            display_cols.append('Flipkart_Live')
        if 'Myntra_Inventory' in display_po.columns:
            display_cols.append('Myntra_Inventory')
        if 'Amazon_Inventory' in display_po.columns:
            display_cols.append('Amazon_Inventory')
        
        display_cols.extend([
            'Total_Units_Sold',
            'ADS',
            'ADS_Period',  # Show how many days used for ADS calculation
            'Stockout_Flag',  # Show stockout warning
            'Days_Left',
            'Target_Stock',
            'Safety_Stock',
            'PO_Recommended'
        ])
        
        # Filter to available columns
        display_cols = [c for c in display_cols if c in display_po.columns]
        
        # Style function
        def highlight_priority(row):
            colors = []
            for col in row.index:
                if col == 'Priority':
                    if '🔴' in str(row[col]):
                        colors.append('background-color: #ffcccc; font-weight: bold')
                    elif '🟡' in str(row[col]):
                        colors.append('background-color: #fff3cd')
                    else:
                        colors.append('background-color: #d4edda')
                elif col == 'Stockout_Flag':
                    if '⚠️' in str(row[col]):
                        colors.append('background-color: #ffcccc; font-weight: bold; color: #dc3545')
                    else:
                        colors.append('')
                elif col == 'Days_Left':
                    if row[col] < lead_time:
                        colors.append('background-color: #ffcccc; font-weight: bold')
                    elif row[col] < lead_time + 7:
                        colors.append('background-color: #fff3cd')
                    else:
                        colors.append('')
                elif col == 'PO_Recommended':
                    colors.append('background-color: #e7f3ff; font-weight: bold')
                else:
                    colors.append('')
            return colors
        
        # Display table
        st.dataframe(
            display_po[display_cols].head(100).style.apply(highlight_priority, axis=1)
                                                   .format({
                                                       'ADS': '{:.2f}',
                                                       'ADS_Period': '{:.0f} days',
                                                       'Days_Left': '{:.1f}',
                                                       'Target_Stock': '{:.0f}',
                                                       'Safety_Stock': '{:.0f}',
                                                       'PO_Recommended': '{:.0f}',
                                                       'Total_Inventory': '{:.0f}',
                                                       'OMS_Inventory': '{:.0f}',
                                                       'Total_Units_Sold': '{:.0f}'
                                                   }),
            use_container_width=True,
            height=600
        )
        
        st.caption(f"Showing top 100 of {len(display_po):,} SKUs needing reorder")
        
        # Warehouse breakdown
        if len(display_po) > 0:
            st.divider()
            st.markdown("### 🏢 Warehouse-wise Analysis")
            
            warehouse_summary = []
            
            if 'OMS_Inventory' in display_po.columns:
                oms_low = len(display_po[display_po['OMS_Inventory'] < display_po['Lead_Time_Demand']])
                warehouse_summary.append({
                    'Warehouse': '🏢 OMS (VDN + TTP)',
                    'SKUs Low Stock': oms_low,
                    'Total Stock': display_po['OMS_Inventory'].sum(),
                    'Recommended Action': f'Replenish {oms_low} SKUs'
                })
            
            if 'Flipkart_Live' in display_po.columns:
                fk_stock = display_po['Flipkart_Live'].sum()
                warehouse_summary.append({
                    'Warehouse': '📦 Flipkart FCs',
                    'SKUs Low Stock': '-',
                    'Total Stock': fk_stock,
                    'Recommended Action': f'{fk_stock:,.0f} units at FCs'
                })
            
            if 'Myntra_Inventory' in display_po.columns:
                myntra_stock = display_po['Myntra_Inventory'].sum()
                warehouse_summary.append({
                    'Warehouse': '👔 Myntra',
                    'SKUs Low Stock': '-',
                    'Total Stock': myntra_stock,
                    'Recommended Action': f'{myntra_stock:,.0f} units at warehouse'
                })
            
            if 'Amazon_Inventory' in display_po.columns:
                amz_stock = display_po['Amazon_Inventory'].sum()
                warehouse_summary.append({
                    'Warehouse': '📦 Amazon FCs',
                    'SKUs Low Stock': '-',
                    'Total Stock': amz_stock,
                    'Recommended Action': f'{amz_stock:,.0f} units at FCs (excl. ZNNE)'
                })
            
            if warehouse_summary:
                st.dataframe(pd.DataFrame(warehouse_summary), use_container_width=True, hide_index=True)
        
        # Download options
        st.divider()
        st.markdown("### ⬇️ Download PO Recommendations")
        
        col_down1, col_down2 = st.columns(2)
        
        with col_down1:
            # CSV download
            csv_buffer = io.StringIO()
            display_po[display_cols].to_csv(csv_buffer, index=False)
            
            st.download_button(
                "📥 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"PO_Recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_down2:
            # Excel download with multiple sheets
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                display_po[display_cols].to_excel(writer, sheet_name='PO_Recommendations', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Urgent SKUs', 'High Priority', 'Medium Priority', 'Total Units to Order'],
                    'Value': [urgent, high, medium, total_po_units]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                "📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"PO_Recommendations_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ==========================================
# TAB 4: LOGISTICS & TRANSFERS
# ==========================================

with tab_logistics:
    st.subheader("🚚 Amazon Stock Transfers & FC Movements")
    
    if st.session_state.transfer_df.empty:
        st.info("""
        ### 📦 Stock Transfer Analytics
        
        Upload **Amazon Stock Transfer Report** in the sidebar to view:
        - FC-to-FC movement tracking
        - Transfer volumes by route
        - Most active fulfillment centers
        - Monthly movement trends
        
        💡 This helps understand Amazon's inventory distribution network.
        """)
    else:
        transfer_df = st.session_state.transfer_df
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_movements = len(transfer_df)
        total_units = transfer_df['Quantity'].sum()
        unique_routes = len(transfer_df.groupby(['Ship From Fc', 'Ship To Fc']))
        fc_transfers = len(transfer_df[transfer_df['Transaction Type'] == 'FC_TRANSFER'])
        
        col1.metric("📦 Total Movements", f"{total_movements:,}")
        col2.metric("📊 Units Transferred", f"{total_units:,.0f}")
        col3.metric("🔄 Unique Routes", unique_routes)
        col4.metric("✅ FC Transfers", f"{fc_transfers:,}")
        
        st.divider()
        
        # Top routes
        st.markdown("### 🔝 Top Transfer Routes")
        
        route_summary = transfer_df.groupby(['Ship From Fc', 'Ship To Fc']).agg({
            'Quantity': ['sum', 'count']
        }).reset_index()
        route_summary.columns = ['From FC', 'To FC', 'Total Units', 'Transfers']
        route_summary = route_summary.sort_values('Total Units', ascending=False).head(20)
        
        col_chart, col_table = st.columns([1, 1])
        
        with col_chart:
            route_summary['Route'] = route_summary['From FC'] + ' → ' + route_summary['To FC']
            fig = px.bar(route_summary.head(10), x='Route', y='Total Units',
                        title='Top 10 Transfer Routes by Volume')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.dataframe(route_summary, use_container_width=True, hide_index=True, height=400)
        
        st.divider()
        
        # Most active FCs
        st.markdown("### 🏢 Most Active Fulfillment Centers")
        
        col_from, col_to = st.columns(2)
        
        with col_from:
            st.markdown("**📤 Shipping From (Outbound)**")
            from_fc = transfer_df.groupby('Ship From Fc').agg({
                'Quantity': 'sum'
            }).reset_index().sort_values('Quantity', ascending=False).head(10)
            from_fc.columns = ['FC', 'Units Shipped']
            st.dataframe(from_fc, use_container_width=True, hide_index=True)
        
        with col_to:
            st.markdown("**📥 Shipping To (Inbound)**")
            to_fc = transfer_df.groupby('Ship To Fc').agg({
                'Quantity': 'sum'
            }).reset_index().sort_values('Quantity', ascending=False).head(10)
            to_fc.columns = ['FC', 'Units Received']
            st.dataframe(to_fc, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Transaction types
        st.markdown("### 📋 Transfer Types")
        
        txn_types = transfer_df['Transaction Type'].value_counts().reset_index()
        txn_types.columns = ['Type', 'Count']
        
        fig = px.pie(txn_types, values='Count', names='Type', title='Transfer Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Download
        st.divider()
        csv_buffer = io.StringIO()
        transfer_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📥 Download Transfer Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"stock_transfers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 5: FORECAST
# ==========================================

with tab_forecast:
    st.subheader("🤖 AI Demand Forecasting")
    
    if st.session_state.sales_df.empty:
        st.warning("⚠️ Upload sales data to generate forecasts.")
    else:
        st.info("AI forecasting will be added here using Prophet...")

# ==========================================
# TAB 6: SKU DRILLDOWN
# ==========================================

with tab_sku:
    st.subheader("🔍 SKU Drilldown Analysis")
    
    if st.session_state.sales_df.empty:
        st.warning("⚠️ Upload sales data for SKU analysis.")
    else:
        search_sku = st.text_input("Search SKU", placeholder="1065YKBLUE", key="sku_search")
        
        if search_sku:
            filtered = st.session_state.sales_df[
                st.session_state.sales_df['Sku'].str.contains(search_sku, case=False, na=False)
            ]
            
            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True)
            else:
                st.warning(f"No data found for '{search_sku}'")

st.divider()
st.caption("💡 All data displayed in OMS SKU format | Yash Gallery ERP v2.0")

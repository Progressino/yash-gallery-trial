# app.py
# ======================================================================================
# Yash Gallery ERP & Planner (Progressino) - Full App (Updated & Fixed)
# - Pivot sales ingestion (Amazon/Flipkart/Meesho/Myntra)
# - OMS inventory mapping with date/disposition selection
# - PO science: service level, spike cap, hide non-urgent, per-SKU LeadTime/Pack/MOQ
# - Forecasting with weekly-branch fix
# - FIXES: Current stock persistence, date picker for inventory, better PO clarity
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import io
from datetime import date, timedelta
import math
import re

# ===========================
# 1. APP CONFIGURATION
# ===========================
st.set_page_config(page_title="Yash Gallery ERP - Progressino", layout="wide", page_icon="🧵")

st.title("🧵 Yash Gallery: ERP & Planner")
st.markdown("""
<style>
    .main-metric { font-size: 24px; font-weight: bold; }
    .sub-metric { font-size: 14px; color: #555; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 8px; border-left: 5px solid #28a745; }
    .warning-box { padding: 15px; background-color: #fff3cd; color: #856404; border-radius: 8px; border-left: 5px solid #ffc107; }
    .info-box { padding: 15px; background-color: #d1ecf1; color: #0c5460; border-radius: 8px; border-left: 5px solid #17a2b8; }
    .danger-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 8px; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ===========================
# 2. HELPER FUNCTIONS
# ===========================
SALES_REQUIRED = {"transaction type", "sku", "quantity"}
TRANSFER_REQUIRED = {"transaction type", "sku", "quantity", "ship from fc", "ship to fc"}

DATE_CANDIDATES = [
    "Shipment Date", "Order Date", "Invoice Date", "Posted Date",
    "shipment date", "order date", "invoice date", "posted date",
    "date", "txn date", "transaction date"
]

TYPE_ALIASES = {
    "shipment": "Shipment", "shipped": "Shipment", "ship": "Shipment",
    "refund": "Refund", "return": "Refund", "customer return": "Refund",
    "freereplacement": "FreeReplacement", "free replacement": "FreeReplacement", "replacement": "FreeReplacement"
}

REVENUE_CANDIDATES = [
    "Invoice Amount", "Total Amount", "Order Amount", "Order Total",
    "Item Total", "Item Price", "Price", "Amount"
]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_report_type(df: pd.DataFrame) -> str:
    df = _norm_cols(df)
    cols = {c.lower().strip() for c in df.columns}
    if TRANSFER_REQUIRED.issubset(cols):
        return "transfer"
    if SALES_REQUIRED.issubset(cols):
        return "sales"
    return "unknown"

def pick_best_date(df: pd.DataFrame):
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
        for dc in df.columns:
            if dc.lower().strip() == c.lower().strip():
                return dc
    return None

def _canon_txn(t: str) -> str:
    t = (t or "").strip()
    key = t.lower().replace("_", " ").strip()
    return TYPE_ALIASES.get(key, t)

def _derive_revenue_row(row: pd.Series) -> float:
    for c in REVENUE_CANDIDATES:
        if c in row and pd.notna(row[c]):
            val = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(val):
                return float(val)
    price_cols = ["Item Price", "Price", "Unit Price"]
    for pc in price_cols:
        if pc in row and pd.notna(row[pc]):
            p = pd.to_numeric(row[pc], errors="coerce")
            q = pd.to_numeric(row.get("Quantity", 0), errors="coerce")
            if pd.notna(p) and pd.notna(q):
                if _canon_txn(str(row.get("Transaction Type", ""))) in ("Shipment", "FreeReplacement"):
                    return float(p * q)
                else:
                    return 0.0
    return 0.0

def normalize_sales(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    df = _norm_cols(df)
    df["Source"] = source_label
    date_col = pick_best_date(df)
    df["TxnDate"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["Transaction Type"] = df["Transaction Type"].astype(str).map(_canon_txn)
    df["Sku"] = df["Sku"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    if not any(c in df.columns for c in REVENUE_CANDIDATES):
        df["Invoice Amount"] = 0.0
    df["Revenue_Effective"] = df.apply(_derive_revenue_row, axis=1)
    df.loc[df["Transaction Type"] == "FreeReplacement", "Revenue_Effective"] = 0.0
    t = df["Transaction Type"]; q = df["Quantity"]
    df["Units_Effective"] = np.where(t == "Shipment", q, np.where(t == "Refund", -q, np.where(t == "FreeReplacement", q, 0)))
    df["Units_Gross"] = np.where(t.isin(["Shipment", "FreeReplacement"]), q, 0)
    df["Units_Return"] = np.where(t == "Refund", q, 0)
    df["Transaction Type"] = df["Transaction Type"].astype("category")
    df["Sku"] = df["Sku"].astype("category")
    return df

def normalize_transfer(df: pd.DataFrame) -> pd.DataFrame:
    df = _norm_cols(df)
    df["Transaction Type"] = df["Transaction Type"].astype(str).str.strip()
    df["Sku"] = df["Sku"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Transaction Type"] = df["Transaction Type"].astype("category")
    df["Sku"] = df["Sku"].astype("category")
    return df

@st.cache_data(show_spinner=False)
def read_file_auto_cached(content: bytes, name: str) -> pd.DataFrame:
    try:
        nl = name.lower()
        if nl.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content), low_memory=False)
        elif nl.endswith('.xlsx'):
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading {name}: {e}")
        return pd.DataFrame()
    return pd.DataFrame()

def read_streamlit_file(file):
    return read_file_auto_cached(file.getvalue(), file.name)

# ===========================
# MEESHO-SPECIFIC FILE READERS
# ===========================

def read_meesho_payment_report(file_obj, filename: str) -> pd.DataFrame:
    """Parse Meesho Payment Report - Order Payments sheet with multi-level headers."""
    try:
        # Read Order Payments sheet with header on row 1, skip row 2 (empty)
        df = pd.read_excel(file_obj, sheet_name='Order Payments', engine='openpyxl', header=1, skiprows=[2])
        
        # Clean and normalize
        df = df.dropna(how='all')
        df.columns = [str(c).strip() for c in df.columns]
        
        # Debug: Show what columns we found
        # st.write(f"DEBUG - Columns found: {df.columns.tolist()[:10]}")
        
        # Map to standard schema
        col_map = {
            'Order Date': 'TxnDate',
            'Supplier SKU': 'Sku',
            'Quantity': 'Quantity',
            'Live Order Status': 'Status',
            'Total Payout': 'Revenue'
        }
        
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        if 'Sku' not in df.columns or 'Quantity' not in df.columns:
            st.warning(f"⚠️ {filename}: Required columns (SKU/Quantity) not found")
            st.write(f"Available columns: {df.columns.tolist()[:10]}")
            return pd.DataFrame()
        
        # Convert columns carefully
        df['TxnDate'] = pd.to_datetime(df['TxnDate'], errors='coerce') if 'TxnDate' in df.columns else pd.NaT
        df['Sku'] = df['Sku'].astype(str).str.strip()
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        
        # Determine transaction type from status
        def get_txn_type(status):
            s = str(status).upper()
            if 'RETURN' in s or 'REFUND' in s:
                return 'Refund'
            elif 'CANCEL' in s:
                return 'Cancel'
            else:
                return 'Shipment'
        
        # Handle Status column - check if it exists
        if 'Status' in df.columns:
            df['Transaction Type'] = df['Status'].apply(get_txn_type)
        else:
            # Default to Shipment if Status column doesn't exist
            df['Transaction Type'] = 'Shipment'
        
        # Calculate effective units
        t = df['Transaction Type']
        q = df['Quantity']
        df['Units_Effective'] = np.where(t == 'Shipment', q, np.where(t == 'Refund', -q, 0))
        df['Units_Gross'] = np.where(t == 'Shipment', q, 0)
        df['Units_Return'] = np.where(t == 'Refund', q, 0)
        
        # Revenue - handle missing Revenue column
        if 'Revenue' in df.columns:
            df['Revenue_Effective'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
        else:
            df['Revenue_Effective'] = 0.0
        
        df['Source'] = 'Meesho'
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        
        return result
        
    except Exception as e:
        import traceback
        st.error(f"❌ Could not parse Meesho Payment Report ({filename})")
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()


def read_tcs_sales(file_obj, filename: str) -> pd.DataFrame:
    """Parse TCS Sales report (Meesho standard format)."""
    try:
        df = pd.read_excel(file_obj, engine='openpyxl')
        
        col_map = {
            'identifier': 'Sku',
            'order_date': 'TxnDate',
            'quantity': 'Quantity',
            'total_invoice_value': 'Revenue'
        }
        
        df = df.rename(columns=col_map)
        
        if 'Sku' not in df.columns or 'Quantity' not in df.columns:
            st.warning(f"⚠️ {filename}: Required columns not found")
            return pd.DataFrame()
        
        df['TxnDate'] = pd.to_datetime(df['TxnDate'], errors='coerce')
        df['Sku'] = df['Sku'].astype(str).str.strip()
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        df['Revenue_Effective'] = pd.to_numeric(df.get('Revenue', 0), errors='coerce').fillna(0)
        
        # TCS sales are shipments only
        df['Transaction Type'] = 'Shipment'
        df['Units_Effective'] = df['Quantity']
        df['Units_Gross'] = df['Quantity']
        df['Units_Return'] = 0
        df['Source'] = 'Meesho-TCS'
        
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        return df[[c for c in out_cols if c in df.columns]]
        
    except Exception as e:
        st.warning(f"❌ Could not parse TCS Sales ({filename}): {e}")
        return pd.DataFrame()

# ===========================
# MYNTRA-SPECIFIC FILE READERS
# Add these after the Meesho parsers in app_complete_with_meesho.py
# ===========================

def read_myntra_pg_forward(file_obj, filename: str) -> pd.DataFrame:
    """Parse Myntra PG_Forward_Settled (shipments with SKU details)."""
    try:
        df = pd.read_excel(file_obj, engine='openpyxl') if filename.endswith('.xlsx') else pd.read_csv(file_obj, low_memory=False)
        
        # Key columns: sku_code, delivery_date, packing_date
        if 'sku_code' not in df.columns:
            return pd.DataFrame()
        
        # Use delivery_date as primary, fallback to packing_date
        date_col = 'delivery_date' if 'delivery_date' in df.columns else 'packing_date'
        
        df['TxnDate'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Sku'] = df['sku_code'].astype(str).str.strip()
        df['Quantity'] = 1  # Each row is 1 item
        
        # Filter out rows where return_type is filled (those are returns in forward file)
        if 'return_type' in df.columns:
            df = df[df['return_type'].isna()]
        
        df['Transaction Type'] = 'Shipment'
        df['Units_Effective'] = df['Quantity']
        df['Units_Gross'] = df['Quantity']
        df['Units_Return'] = 0
        df['Source'] = 'Myntra'
        
        # Revenue - check various price columns
        revenue_col = None
        for col in ['item_price', 'selling_price', 'mrp', 'final_price']:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            df['Revenue_Effective'] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
        else:
            df['Revenue_Effective'] = 0.0
        
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        
        # Remove rows with no date
        result = result[result['TxnDate'].notna()]
        
        return result
        
    except Exception as e:
        st.warning(f"❌ Could not parse Myntra PG Forward ({filename}): {e}")
        return pd.DataFrame()


def read_myntra_pg_reverse(file_obj, filename: str) -> pd.DataFrame:
    """Parse Myntra PG_Reverse_Settled (returns/refunds with SKU details)."""
    try:
        df = pd.read_excel(file_obj, engine='openpyxl') if filename.endswith('.xlsx') else pd.read_csv(file_obj, low_memory=False)
        
        if 'sku_code' not in df.columns or 'return_date' not in df.columns:
            return pd.DataFrame()
        
        df['TxnDate'] = pd.to_datetime(df['return_date'], errors='coerce')
        df['Sku'] = df['sku_code'].astype(str).str.strip()
        df['Quantity'] = 1  # Each row is 1 item
        
        # Check return_type to distinguish refunds vs exchanges
        # return_refund = actual return, exchange = replacement
        if 'return_type' in df.columns:
            df['Transaction Type'] = df['return_type'].apply(
                lambda x: 'Refund' if str(x).lower() == 'return_refund' else 'FreeReplacement'
            )
        else:
            df['Transaction Type'] = 'Refund'
        
        # For refunds: negative units, for exchanges: treat as replacement
        df['Units_Effective'] = np.where(
            df['Transaction Type'] == 'Refund', -df['Quantity'], df['Quantity']
        )
        df['Units_Gross'] = 0
        df['Units_Return'] = np.where(df['Transaction Type'] == 'Refund', df['Quantity'], 0)
        df['Source'] = 'Myntra'
        
        # Revenue for returns (usually 0 or negative)
        df['Revenue_Effective'] = 0.0
        
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        
        # Remove rows with no date
        result = result[result['TxnDate'].notna()]
        
        return result
        
    except Exception as e:
        st.warning(f"❌ Could not parse Myntra PG Reverse ({filename}): {e}")
        return pd.DataFrame()


def read_myntra_gstr_packed(file_obj, filename: str) -> pd.DataFrame:
    """Parse Myntra GSTR Packed report (alternative shipment source)."""
    try:
        df = pd.read_csv(file_obj, low_memory=False)
        
        if 'sku_id' not in df.columns:
            return pd.DataFrame()
        
        # Use order_shipped_date or order_packed_date
        date_col = None
        for col in ['order_shipped_date', 'order_packed_date', 'order_created_date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return pd.DataFrame()
        
        df['TxnDate'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Sku'] = df['sku_id'].astype(str).str.strip()
        df['Quantity'] = 1
        
        # Filter by order_status: C=Completed/Delivered, SH=Shipped, F=?, PK=Packed, RTO=Return
        # Only include C and SH as shipments
        if 'order_status' in df.columns:
            df = df[df['order_status'].isin(['C', 'SH'])]
        
        df['Transaction Type'] = 'Shipment'
        df['Units_Effective'] = df['Quantity']
        df['Units_Gross'] = df['Quantity']
        df['Units_Return'] = 0
        df['Source'] = 'Myntra-GSTR'
        df['Revenue_Effective'] = 0.0
        
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        result = result[result['TxnDate'].notna()]
        
        return result
        
    except Exception as e:
        st.warning(f"❌ Could not parse Myntra GSTR Packed ({filename}): {e}")
        return pd.DataFrame()





# ---------- Pivot sales ingestion (Flipkart/Meesho/Myntra/Amazon) ----------

# ===========================
# FLIPKART-SPECIFIC FILE READERS
# ===========================

def read_flipkart_sales_report(file_obj, filename: str) -> pd.DataFrame:
    """Parse Flipkart Sales Report - Sales Report sheet."""
    try:
        # Read "Sales Report" sheet with header at row 0
        df = pd.read_excel(file_obj, sheet_name='Sales Report', engine='openpyxl', header=0)
        
        # Key columns: SKU (or FSN), Order Date, Event Type, Event Sub Type
        if 'SKU' not in df.columns and 'FSN' not in df.columns:
            st.warning(f"⚠️ {filename}: No SKU or FSN column found")
            return pd.DataFrame()
        
        # Use SKU if available, otherwise FSN
        sku_col = 'SKU' if 'SKU' in df.columns else 'FSN'
        
        # Clean SKU - remove triple quotes
        df['Sku'] = df[sku_col].astype(str).str.replace('"""', '').str.strip()
        
        # Use Order Date as primary date
        if 'Order Date' in df.columns:
            df['TxnDate'] = pd.to_datetime(df['Order Date'], errors='coerce')
        else:
            return pd.DataFrame()
        
        df['Quantity'] = 1  # Each row is 1 item
        
        # Determine transaction type from Event Type and Event Sub Type
        # Event Type: Settled, Reversed
        # Event Sub Type: Sale, Return, Cancellation, etc.
        
        def get_txn_type(row):
            event_type = str(row.get('Event Type', '')).upper()
            event_sub = str(row.get('Event Sub Type', '')).upper()
            
            if 'RETURN' in event_sub or 'RETURNED' in event_sub:
                return 'Refund'
            elif 'CANCEL' in event_sub:
                return 'Cancel'
            elif 'REPLACE' in event_sub or 'EXCHANGE' in event_sub:
                return 'FreeReplacement'
            elif 'SALE' in event_sub or 'SETTLED' in event_type:
                return 'Shipment'
            else:
                return 'Shipment'  # Default
        
        df['Transaction Type'] = df.apply(get_txn_type, axis=1)
        
        # Calculate effective units
        t = df['Transaction Type']
        q = df['Quantity']
        df['Units_Effective'] = np.where(
            t == 'Shipment', q,
            np.where(t == 'Refund', -q,
            np.where(t == 'FreeReplacement', q, 0))
        )
        df['Units_Gross'] = np.where(t.isin(['Shipment', 'FreeReplacement']), q, 0)
        df['Units_Return'] = np.where(t == 'Refund', q, 0)
        
        # Revenue - check for price columns
        revenue_col = None
        for col in ['Customer Sale Price', 'Sale Amount', 'Total Sale Price']:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            df['Revenue_Effective'] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
        else:
            df['Revenue_Effective'] = 0.0
        
        df['Source'] = 'Flipkart'
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        
        # Remove rows with no date
        result = result[result['TxnDate'].notna()]
        
        return result
        
    except Exception as e:
        import traceback
        st.error(f"❌ Could not parse Flipkart Sales Report ({filename})")
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()


def read_flipkart_payment_report(file_obj, filename: str) -> pd.DataFrame:
    """Parse Flipkart Payment Report - Orders sheet (has SKU details)."""
    try:
        # Read "Orders" sheet with header at row 1
        df = pd.read_excel(file_obj, sheet_name='Orders', engine='openpyxl', header=1)
        
        # Check for key columns
        if 'Seller SKU' not in df.columns or 'Order ID' not in df.columns:
            st.warning(f"⚠️ {filename}: Missing required columns")
            return pd.DataFrame()
        
        # Clean SKU
        df['Sku'] = df['Seller SKU'].astype(str).str.strip()
        
        # Use Order Date or Payment Date
        date_col = None
        for col in ['Order Date', ' Payment Date', 'Payment Date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            st.warning(f"⚠️ {filename}: No date column found")
            return pd.DataFrame()
        
        df['TxnDate'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Quantity'] = 1  # Each row is 1 item
        
        # Payment report typically contains settled shipments
        # Check if there's a return indicator
        if 'Return Sub Type' in df.columns:
            df['Transaction Type'] = df['Return Sub Type'].apply(
                lambda x: 'Refund' if pd.notna(x) else 'Shipment'
            )
        else:
            df['Transaction Type'] = 'Shipment'
        
        t = df['Transaction Type']
        q = df['Quantity']
        df['Units_Effective'] = np.where(t == 'Shipment', q, -q)
        df['Units_Gross'] = np.where(t == 'Shipment', q, 0)
        df['Units_Return'] = np.where(t == 'Refund', q, 0)
        
        # Revenue
        revenue_col = None
        for col in ['Sale Amount (Rs.)', 'Customer Paid Amount', 'Total Sale Price']:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            df['Revenue_Effective'] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)
        else:
            df['Revenue_Effective'] = 0.0
        
        df['Source'] = 'Flipkart-Payment'
        df['Transaction Type'] = df['Transaction Type'].astype('category')
        df['Sku'] = df['Sku'].astype('category')
        
        out_cols = ['TxnDate', 'Transaction Type', 'Sku', 'Quantity', 'Source',
                    'Revenue_Effective', 'Units_Effective', 'Units_Gross', 'Units_Return']
        result = df[[c for c in out_cols if c in df.columns]].copy()
        
        # Remove rows with no date or empty SKU
        result = result[result['TxnDate'].notna() & (result['Sku'] != '')]
        
        return result
        
    except Exception as e:
        import traceback
        st.error(f"❌ Could not parse Flipkart Payment Report ({filename})")
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

def _is_date_like(x: str) -> bool:
    try:
        pd.to_datetime(x, errors="raise")
        return True
    except Exception:
        return False

def detect_sku_column(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    # Direct hits
    for c in cols:
        lc = str(c).lower()
        if re.search(r'\bsku\b', lc) or 'msku' in lc or 'item sku' in lc or 'item skucode' in lc or 'product code' in lc:
            return c
    # heuristics: first non-date, non-empty col
    non_date = [c for c in cols if not _is_date_like(str(c))]
    return non_date[0] if non_date else None

def is_pivot_sales(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    df = _norm_cols(df)
    date_cols = [c for c in df.columns if _is_date_like(str(c))]
    sku_col = detect_sku_column(df)
    return bool(sku_col) and len(date_cols) >= 3

def source_from_filename(name: str) -> str:
    n = name.lower()
    if "flipkart" in n or "fk" in n: return "Flipkart"
    if "meesho" in n: return "Meesho"
    if "myntra" in n: return "Myntra"
    if "amazon" in n or "amz" in n: return "Amazon"
    return "Marketplace"

def melt_pivot_sales(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Return long format: Sku, TxnDate, Units_Effective (+units gross/return, revenue=0), Source."""
    df = _norm_cols(df).dropna(how="all").dropna(axis=1, how="all")
    sku_col = detect_sku_column(df)
    date_cols = [c for c in df.columns if _is_date_like(str(c))]
    if not sku_col or not date_cols:
        return pd.DataFrame()
    keep = [sku_col] + date_cols
    df = df[[c for c in keep if c in df.columns]].copy()
    df_long = df.melt(id_vars=[sku_col], value_vars=date_cols, var_name="TxnDate", value_name="Units")
    df_long["TxnDate"] = pd.to_datetime(df_long["TxnDate"], errors="coerce")
    df_long = df_long.dropna(subset=["TxnDate"])
    df_long["Sku"] = df_long[sku_col].astype(str).str.strip()
    df_long["Units"] = pd.to_numeric(df_long["Units"], errors="coerce").fillna(0)
    df_long = df_long.groupby(["Sku", "TxnDate"], as_index=False)["Units"].sum()
    # decorate to match normalized sales schema (shipments only)
    out = pd.DataFrame({
        "TxnDate": df_long["TxnDate"],
        "Transaction Type": "Shipment",
        "Sku": df_long["Sku"],
        "Quantity": df_long["Units"],
        "Source": source_from_filename(filename),
        "Revenue_Effective": 0.0,
        "Units_Effective": df_long["Units"],
        "Units_Gross": df_long["Units"],
        "Units_Return": 0.0
    })
    out["Sku"] = out["Sku"].astype("category")
    out["Transaction Type"] = out["Transaction Type"].astype("category")
    return out

# ---------- Forecast helpers ----------
def build_daily_series_for_sku(sales_df: pd.DataFrame, sku: str):
    df = sales_df[sales_df["Sku"] == sku].copy()
    df = df[df["TxnDate"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    daily = (df.assign(ds=df["TxnDate"].dt.date)
             .groupby("ds")["Units_Effective"].sum()
             .reset_index())
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds")
    full = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full).fillna(0.0).rename_axis("ds").reset_index()
    daily.columns = ["ds", "y"]
    return daily

def forecast_series(daily: pd.DataFrame, horizon_days: int = 60, use_weekly_if_sparse: bool = True):
    # Naive fallback
    if len(daily) < 15:
        level = daily["y"].tail(7).mean() if len(daily) >= 7 else daily["y"].mean()
        future = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        fc = pd.DataFrame({
            "ds": future,
            "yhat": level,
            "yhat_lower": np.maximum(0, level * 0.7),
            "yhat_upper": level * 1.3
        })
        return fc

    use_weekly = use_weekly_if_sparse and daily["y"].astype(bool).sum() < 30
    if use_weekly:
        w = daily.copy()
        w["ds"] = w["ds"].dt.to_period("W").apply(lambda r: r.start_time)
        w = w.groupby("ds", as_index=False)["y"].sum()
        m = Prophet(seasonality_mode="multiplicative",
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1)
        m.fit(w.rename(columns={"ds": "ds", "y": "y"}))
        weekly_periods = max(1, math.ceil(horizon_days / 7))
        future = m.make_future_dataframe(periods=weekly_periods, freq="W")
        f = m.predict(future)
        f_week = f[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        f_week["week"] = f_week["ds"].dt.to_period("W")
        f_week = f_week.drop_duplicates(subset=["week"], keep="last")
        wk_map = (f_week.set_index("week")[["yhat", "yhat_lower", "yhat_upper"]]
                         .to_dict(orient="index"))
        out_days = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1),
                                 periods=horizon_days, freq="D")
        rows = []
        for d in out_days:
            wk = d.to_period("W")
            val = wk_map.get(wk)
            if val is None:
                rows.append({"ds": d, "yhat": 0.0, "yhat_lower": 0.0, "yhat_upper": 0.0})
                continue
            rows.append({
                "ds": d,
                "yhat": max(0, val["yhat"] / 7.0),
                "yhat_lower": max(0, val["yhat_lower"] / 7.0),
                "yhat_upper": max(0, val["yhat_upper"] / 7.0)
            })
        return pd.DataFrame(rows)

    m = Prophet(seasonality_mode="multiplicative",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1)
    m.fit(daily.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    f = m.predict(future)
    fc = f[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[c] = fc[c].clip(lower=0)
    return fc.tail(horizon_days)

def plot_prophet_forecast(daily, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="lines", name="Actual Sales",
                             line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="AI Prediction",
                             line=dict(color='#0068C9', width=2)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                             fill=None, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                             fill="tonexty", line=dict(width=0), name="Uncertainty Range",
                             fillcolor="rgba(0, 104, 201, 0.2)"))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Date", yaxis_title="Units Sold", hovermode="x unified")
    return fig

# ===========================
# 3. SIDEBAR & DATA LOADING
# ===========================
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.write("Progressino")
st.sidebar.markdown("<h4 style='text-align: center;'>Developed by Progressino</h4>", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.header("1. Data Import")
# New: pivot marketplace files
marketplace_files = st.sidebar.file_uploader(
    "Upload Marketplace Pivot Sales (Amazon/Flipkart/Meesho/Myntra) - CSV/XLSX",
    type=["csv", "xlsx"], accept_multiple_files=True
)
# keep your original inputs as well
b2c_files = st.sidebar.file_uploader("Upload B2C Sales (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)
b2b_files = st.sidebar.file_uploader("Upload B2B Sales (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)
transfer_files = st.sidebar.file_uploader("Upload Stock Transfers", type=["csv", "xlsx"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("Optional: OMS Inventory (XLSX/CSV)", type=["xlsx", "csv"])
vendor_file = st.sidebar.file_uploader("Optional: Vendor Master (per-SKU LeadTime/Pack/MOQ)", type=["xlsx", "csv"])

st.sidebar.divider()
st.sidebar.header("2. Manufacturing & Policy Settings")
lead_time_days = st.sidebar.number_input("Default Lead Time (Days)", min_value=1, max_value=180, value=30)
safety_stock_pct = st.sidebar.slider("Safety Stock Buffer % (floor)", 0, 100, 20)
service_level = st.sidebar.selectbox("Service Level Target", ["90%", "95%", "99%"], index=0)
apply_spike_cap = st.sidebar.toggle("Cap ADS spikes at 1.5× baseline", value=True)
hide_non_urgent = st.sidebar.toggle("Hide non-urgent SKUs (Days Left > Lead Time + 7)", value=False)

# Map service level to z
z_map = {"90%": 1.28, "95%": 1.645, "99%": 2.33}
z_value = z_map[service_level]

# Load Data
sales_frames = []
transfer_frames = []

# 3.1 Marketplace pivot/snapshot files + Meesho-specific handlers
if marketplace_files:
    st.sidebar.markdown("**📤 Processing Files...**")
    for f in marketplace_files:
        fname_lower = f.name.lower()
        long_df = pd.DataFrame()
        
        # Meesho-specific file detection by name
        if 'meesho' in fname_lower and 'payment' in fname_lower:
            with st.sidebar:
                with st.spinner(f"Parsing {f.name}..."):
                    long_df = read_meesho_payment_report(io.BytesIO(f.getvalue()), f.name)
                    if not long_df.empty:
                        st.success(f"✅ {f.name} ({len(long_df)} records)")
                    
        elif 'tcs' in fname_lower and 'sales' in fname_lower:
            # TCS sales file has aggregate data without real SKUs - skip it
            st.sidebar.warning(f"⚠️ {f.name} - Aggregated data (no SKU details), skipping")
            continue
        
        # Myntra-specific file detection
        elif 'myntra' in fname_lower:
            if 'pg_forward' in fname_lower or 'forward_settled' in fname_lower:
                with st.sidebar:
                    with st.spinner(f"Parsing {f.name}..."):
                        long_df = read_myntra_pg_forward(io.BytesIO(f.getvalue()), f.name)
                        if not long_df.empty:
                            st.success(f"✅ {f.name} ({len(long_df)} shipments)")
            
            elif 'pg_reverse' in fname_lower or 'reverse_settled' in fname_lower:
                with st.sidebar:
                    with st.spinner(f"Parsing {f.name}..."):
                        long_df = read_myntra_pg_reverse(io.BytesIO(f.getvalue()), f.name)
                        if not long_df.empty:
                            st.success(f"✅ {f.name} ({len(long_df)} returns)")
            
            elif 'gstr' in fname_lower and 'packed' in fname_lower:
                with st.sidebar:
                    with st.spinner(f"Parsing {f.name}..."):
                        long_df = read_myntra_gstr_packed(io.BytesIO(f.getvalue()), f.name)
                        if not long_df.empty:
                            st.success(f"✅ {f.name} ({len(long_df)} records)")
            
            elif 'gstr' in fname_lower and ('rt' in fname_lower or 'rto' in fname_lower):
                # Skip GSTR RT and RTO files - data already in PG_Reverse
                st.sidebar.info(f"ℹ️ {f.name} - Duplicate data (use PG_Reverse instead), skipping")
                continue
        
        # Flipkart-specific file detection
        elif 'flipkart' in fname_lower:
            if 'sales' in fname_lower and 'report' in fname_lower:
                with st.sidebar:
                    with st.spinner(f"Parsing {f.name}..."):
                        long_df = read_flipkart_sales_report(io.BytesIO(f.getvalue()), f.name)
                        if not long_df.empty:
                            st.success(f"✅ {f.name} ({len(long_df)} records)")
            
            elif 'payment' in fname_lower and 'report' in fname_lower:
                with st.sidebar:
                    with st.spinner(f"Parsing {f.name}..."):
                        long_df = read_flipkart_payment_report(io.BytesIO(f.getvalue()), f.name)
                        if not long_df.empty:
                            st.success(f"✅ {f.name} ({len(long_df)} records)")
        
        else:
            # Try existing pivot/snapshot detection for other marketplaces
            df_raw = read_streamlit_file(f)
            # try primary sheet if xlsx has multiple
            if df_raw.empty and f.name.lower().endswith(".xlsx"):
                try:
                    xls = pd.ExcelFile(io.BytesIO(f.getvalue()), engine="openpyxl")
                    for sh in xls.sheet_names:
                        trial = xls.parse(sh)
                        if not trial.empty:
                            df_raw = trial
                            break
                except Exception:
                    pass
            if not df_raw.empty and is_pivot_sales(df_raw):
                long_df = melt_pivot_sales(df_raw, f.name)
                if not long_df.empty:
                    st.sidebar.success(f"✅ {f.name} ({len(long_df)} records)")
        
        if not long_df.empty:
            sales_frames.append(long_df)

# 3.2 Keep your B2C/B2B normalizer for transactional reports
if b2c_files:
    for f in b2c_files:
        df = read_streamlit_file(f)
        if detect_report_type(df) == "sales":
            sales_frames.append(normalize_sales(df, "Amazon B2C"))
if b2b_files:
    for f in b2b_files:
        df = read_streamlit_file(f)
        if detect_report_type(df) == "sales":
            sales_frames.append(normalize_sales(df, "Amazon B2B"))

# 3.3 Transfers
if transfer_files:
    for f in transfer_files:
        df = read_streamlit_file(f)
        if detect_report_type(df) == "transfer":
            transfer_frames.append(normalize_transfer(df))

raw_sales_df = pd.concat(sales_frames, ignore_index=True) if len(sales_frames) else pd.DataFrame()
transfer_df = pd.concat(transfer_frames, ignore_index=True) if len(transfer_frames) else None

# 3.4 Global Date Filter (applies to all tabs)
if not raw_sales_df.empty and raw_sales_df["TxnDate"].notna().any():
    min_d = raw_sales_df["TxnDate"].min().date()
    max_d = raw_sales_df["TxnDate"].max().date()
    st.sidebar.header("3. Global Date Filter")
    date_range = st.sidebar.date_input("Filter Range", (min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        sales_df = raw_sales_df[(raw_sales_df["TxnDate"] >= pd.to_datetime(start_d)) &
                                (raw_sales_df["TxnDate"] <= pd.to_datetime(end_d))]
    else:
        sales_df = raw_sales_df.copy()
else:
    sales_df = raw_sales_df.copy()

# Helper: IMPROVED inventory mapping UI + extraction
def map_inventory_current_stock(inv_file_obj):
    """Returns: (stock_series, selected_date_value, metadata_dict)"""
    if inv_file_obj is None:
        return pd.Series(dtype=float), None, None

    inv_raw = read_streamlit_file(inv_file_obj)
    if inv_raw.empty:
        return pd.Series(dtype=float), None, None

    st.markdown("**🗂️ Inventory File Mapping**")
    cols = inv_raw.columns.tolist()

    # SKU column detection
    sku_guess = detect_sku_column(inv_raw) or cols[0]
    col_sku = st.selectbox("📦 SKU Column", cols, index=(cols.index(sku_guess) if sku_guess in cols else 0))

    # Detect actual date columns (parse as dates)
    actual_date_cols = []
    for c in cols:
        # Try first 5 non-null values
        sample = inv_raw[c].dropna().head(5)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().all():
                    actual_date_cols.append(c)
            except:
                pass

    # Stock quantity column selection
    if actual_date_cols:
        st.markdown("**📅 Inventory has date-based stock columns**")
        
        # Parse all date columns to find latest
        date_map = {}
        for dc in actual_date_cols:
            try:
                # Get the actual date value (first non-null)
                sample_date = pd.to_datetime(inv_raw[dc].dropna().iloc[0])
                date_map[dc] = sample_date
            except:
                pass
        
        if date_map:
            # Sort by actual date
            sorted_dates = sorted(date_map.items(), key=lambda x: x[1], reverse=True)
            latest_col = sorted_dates[0][0]
            latest_date = sorted_dates[0][1]
            
            # NEW: User-friendly calendar date picker
            st.markdown("**Select Inventory Date:**")
            
            # Get available dates
            available_dates = sorted([d for d in date_map.values()], reverse=True)
            min_date = min(available_dates).date()
            max_date = max(available_dates).date()
            
            # Calendar date input with latest date as default
            selected_date_input = st.date_input(
                "📅 Pick Inventory Date",
                value=latest_date.date(),
                min_value=min_date,
                max_value=max_date,
                help=f"Available range: {min_date} to {max_date}. Latest date selected by default."
            )
            
            # Find the column with the closest matching date
            selected_date_dt = pd.to_datetime(selected_date_input)
            closest_col = min(date_map.keys(), 
                            key=lambda c: abs((date_map[c] - selected_date_dt).total_seconds()))
            
            col_stock = closest_col
            selected_date_val = date_map[col_stock]
            
            # Show which date was actually selected
            if selected_date_val.date() != selected_date_input:
                st.info(f"📌 Using closest available date: **{selected_date_val.strftime('%Y-%m-%d')}**")
            else:
                st.success(f"✅ Using selected date: **{selected_date_val.strftime('%Y-%m-%d')}**")
        else:
            # Fallback if parsing failed
            col_stock = st.selectbox("📊 Stock Column", actual_date_cols)
            selected_date_val = None
    else:
        # No date columns - show all numeric columns
        numeric_cols = inv_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = cols
        col_stock = st.selectbox(
            "📊 Stock Quantity Column", 
            numeric_cols,
            help="No date columns detected - showing numeric columns"
        )
        selected_date_val = None

    # Disposition filtering (optional)
    col_disp = st.selectbox("🏷️ Disposition/Status Column (optional)", ["(none)"] + cols, index=0)
    disp_filter = None
    tmp_inv = inv_raw.copy()
    if col_disp != "(none)":
        disp_values = sorted(tmp_inv[col_disp].dropna().astype(str).unique().tolist())
        if disp_values:
            disp_filter = st.multiselect(
                "✅ Keep Disposition(s)", 
                disp_values,
                default=[v for v in disp_values if v.lower() in ["sellable", "available", "good"]],
                help="Select stock statuses to include"
            )
            if disp_filter:
                tmp_inv = tmp_inv[tmp_inv[col_disp].astype(str).isin(disp_filter)]

    # Build stock series
    if col_stock not in tmp_inv.columns or col_sku not in tmp_inv.columns:
        st.warning("⚠️ Selected columns are not valid for this file.")
        return pd.Series(dtype=float), None, None

    tmp = tmp_inv[[col_sku, col_stock]].copy()
    tmp.columns = ["Sku", "Stock"]
    tmp["Sku"] = tmp["Sku"].astype(str).str.strip()
    tmp["Stock"] = pd.to_numeric(tmp["Stock"], errors="coerce").fillna(0)
    stock_map = tmp.groupby("Sku")["Stock"].sum()
    
    # Show summary
    total_stock = stock_map.sum()
    stock_date_str = selected_date_val.strftime('%Y-%m-%d') if selected_date_val else "N/A"
    st.success(f"✅ Loaded {len(stock_map)} SKUs | Total Stock: {total_stock:,.0f} units | Date: {stock_date_str}")
    
    inv_meta = {
        "sku_col": col_sku, 
        "stock_col": col_stock, 
        "disposition": disp_filter,
        "stock_date": stock_date_str
    }
    return stock_map, selected_date_val, inv_meta

# Optional vendor master
def read_vendor_master(file_obj):
    if file_obj is None:
        return pd.DataFrame()
    df = read_streamlit_file(file_obj)
    if df.empty:
        return df
    df = _norm_cols(df)
    # normalize col names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("sku", "msku", "item sku", "item skucode"): rename_map[c] = "Sku"
        elif "lead" in lc and "time" in lc: rename_map[c] = "LeadTimeDays"
        elif "moq" in lc: rename_map[c] = "MOQ"
        elif "pack" in lc and "size" in lc: rename_map[c] = "PackSize"
    df = df.rename(columns=rename_map)
    for col in ["LeadTimeDays", "MOQ", "PackSize"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Sku" in df.columns:
        df["Sku"] = df["Sku"].astype(str).str.strip()
        df = df.groupby("Sku", as_index=False).agg(
            LeadTimeDays=("LeadTimeDays", "max"),
            MOQ=("MOQ", "max"),
            PackSize=("PackSize", "max"),
        )
    return df

vendor_df = read_vendor_master(vendor_file)

# ===========================
# 4. MAIN TABS
# ===========================
tab_dash, tab_plan, tab_item, tab_po, tab_forecast, tab_sku, tab_transfers = st.tabs([
    "📊 AI Sales Dashboard",
    "🏭 Production Planning",
    "👕 Item Master",
    "📦 Reorder / PO",
    "📈 AI Forecast",
    "🔎 SKU Drilldown",
    "🚚 Stock Transfers"
])

# ==========================================
# TAB 1: AI SALES DASHBOARD (ENHANCED)
# ==========================================
with tab_dash:
    st.subheader("📊 AI-Centric Sales Intelligence")

    # USE RAW_SALES_DF (all uploaded data) instead of filtered sales_df
    dashboard_df = raw_sales_df.copy()
    
    if not dashboard_df.empty:
        has_dates = dashboard_df["TxnDate"].notna().any()

        total_sales = float(dashboard_df["Units_Effective"].sum())
        total_rev = float(dashboard_df.get("Revenue_Effective", pd.Series(dtype=float)).sum()) if "Revenue_Effective" in dashboard_df.columns else 0.0
        total_refunds = float(dashboard_df.get("Units_Return", pd.Series(dtype=float)).sum()) if "Units_Return" in dashboard_df.columns else 0.0
        total_gross = float(dashboard_df.get("Units_Gross", pd.Series(dtype=float)).sum()) if "Units_Gross" in dashboard_df.columns else total_sales

        avg_order_value = total_rev / total_sales if total_sales > 0 else 0.0
        refund_rate = (total_refunds / total_gross * 100) if total_gross > 0 else 0.0

        # Show data sources
        if "Source" in dashboard_df.columns:
            sources = dashboard_df["Source"].value_counts()
            st.markdown(f"**📦 Data Sources:** {', '.join([f'{src} ({cnt:,} txns)' for src, cnt in sources.items()])}")
        
        st.divider()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Net Revenue", f"₹ {total_rev:,.0f}")
        k2.metric("Net Units Sold", f"{int(total_sales):,}")
        k3.metric("AOV (Avg Order Value)", f"₹ {avg_order_value:.0f}")
        k4.metric("Refund Rate", f"{refund_rate:.1f}%", delta_color="inverse")
        k5.metric("Total Returns", f"{int(total_refunds)} pcs")

        st.divider()

        # Smart Insights
        insights = []
        if has_dates:
            max_date = dashboard_df["TxnDate"].max()
            last_7 = dashboard_df[dashboard_df["TxnDate"] > (max_date - pd.Timedelta(days=7))]
            prev_7 = dashboard_df[(dashboard_df["TxnDate"] <= (max_date - pd.Timedelta(days=7))) &
                              (dashboard_df["TxnDate"] > (max_date - pd.Timedelta(days=14)))]
            s_curr = float(last_7["Units_Effective"].sum())
            s_prev = float(prev_7["Units_Effective"].sum())
            if s_prev > 0:
                growth = (s_curr - s_prev) / s_prev * 100
                if growth > 10:
                    insights.append(f"🚀 **Momentum:** +{growth:.1f}% vs last week.")
                elif growth < -10:
                    insights.append(f"📉 **Slowdown:** {growth:.1f}% vs last week.")

        sku_rev_sum = dashboard_df.groupby("Sku")["Units_Effective"].sum()
        if not sku_rev_sum.empty:
            top_sku_name = sku_rev_sum.idxmax()
            top_sku_qty = int(sku_rev_sum.max())
            insights.append(f"🏆 **Star Product:** **{top_sku_name}** ({top_sku_qty} units).")

        c_smart1, c_smart2 = st.columns([2, 1])
        with c_smart1:
            st.markdown("#### 🧠 Smart Narrative")
            if insights:
                for i in insights:
                    st.markdown(f"- {i}")
            else:
                st.caption("No significant trend detected in the selected range.")

            if refund_rate > 20:
                st.markdown('<div class="warning-box">⚠️ <b>High Return Rate Alert:</b> Your overall return rate is above 20%. Check sizing for top returned items.</div>',
                            unsafe_allow_html=True)
            elif total_sales > 1000 and refund_rate < 5:
                st.markdown('<div class="success-box">✅ <b>Quality Success:</b> Return rate is exceptionally low (&lt;5%).</div>',
                            unsafe_allow_html=True)

        with c_smart2:
            st.markdown("#### 📅 Weekly Heatmap")
            if has_dates:
                tmp = dashboard_df[dashboard_df["TxnDate"].notna()].copy()
                tmp["DayOfWeek"] = tmp["TxnDate"].dt.day_name()
                dow_data = tmp.groupby("DayOfWeek")["Units_Effective"].sum().reindex(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                ).fillna(0)
                st.bar_chart(dow_data, color="#0068C9")
            else:
                st.info("No dates available for weekly analysis.")

        st.divider()

        # Sales by Source breakdown
        if "Source" in dashboard_df.columns and dashboard_df["Source"].nunique() > 1:
            st.subheader("📊 Sales by Marketplace")
            source_breakdown = dashboard_df.groupby("Source").agg({
                "Units_Effective": "sum",
                "Revenue_Effective": "sum"
            }).reset_index()
            source_breakdown.columns = ["Marketplace", "Units Sold", "Revenue"]
            source_breakdown = source_breakdown.sort_values("Units Sold", ascending=False)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_source_units = px.pie(
                    source_breakdown, 
                    values="Units Sold", 
                    names="Marketplace",
                    title="Units by Marketplace",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_source_units, use_container_width=True)
            
            with col_chart2:
                fig_source_rev = px.bar(
                    source_breakdown,
                    x="Marketplace",
                    y="Revenue",
                    title="Revenue by Marketplace",
                    color="Marketplace",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_source_rev, use_container_width=True)
            
            st.dataframe(source_breakdown.style.format({"Units Sold": "{:,.0f}", "Revenue": "₹{:,.0f}"}), 
                        use_container_width=True)
            st.divider()

        # Pareto Analysis
        st.subheader("📐 Strategic Analysis (Pareto Principle)")
        c_pareto1, c_pareto2 = st.columns(2)
        with c_pareto1:
            sku_rev = dashboard_df.groupby("Sku")["Revenue_Effective"].sum().sort_values(ascending=False).reset_index() \
                      if "Revenue_Effective" in dashboard_df.columns else pd.DataFrame()
            if not sku_rev.empty and sku_rev["Revenue_Effective"].sum() > 0:
                sku_rev["Cumulative Revenue"] = sku_rev["Revenue_Effective"].cumsum()
                sku_rev["Cumulative %"] = (sku_rev["Cumulative Revenue"] / sku_rev["Revenue_Effective"].sum()) * 100

                def classify(pct):
                    if pct <= 80: return "A (Top 80% Rev)"
                    elif pct <= 95: return "B (Next 15% Rev)"
                    return "C (Bottom 5%)"

                sku_rev["Class"] = sku_rev["Cumulative %"].apply(classify)

                fig_pareto = px.bar(
                    sku_rev.head(20), x="Sku", y="Revenue_Effective", color="Class",
                    title="Top 20 SKUs by Revenue Contribution",
                    color_discrete_map={
                        "A (Top 80% Rev)": "#28a745",
                        "B (Next 15% Rev)": "#ffc107",
                        "C (Bottom 5%)": "#dc3545"
                    }
                )
                st.plotly_chart(fig_pareto, use_container_width=True)
            else:
                st.info("Insufficient revenue data for Pareto analysis (no revenue columns).")

        with c_pareto2:
            if 'Class' in (locals().get('sku_rev', pd.DataFrame()).columns if 'sku_rev' in locals() else []):
                class_counts = sku_rev["Class"].value_counts().reset_index()
                class_counts.columns = ["Class", "SKU Count"]
                class_rev = sku_rev.groupby("Class")["Revenue_Effective"].sum().reset_index()
                total_rev_sum = sku_rev["Revenue_Effective"].sum()
                pareto_summary = pd.merge(class_counts, class_rev, on="Class")
                pareto_summary["Revenue %"] = (pareto_summary["Revenue_Effective"] / total_rev_sum * 100).round(1)
                st.markdown("##### 📌 Inventory Classification")
                st.dataframe(pareto_summary, use_container_width=True)
                st.caption("**Insight:** Focus stock availability on **Class A** items; they drive ~80% of revenue.")

        # Daily Sales Velocity + 7d average
        st.subheader("📈 Daily Sales Velocity (All Sources)")
        if has_dates:
            daily_trend = dashboard_df.groupby(dashboard_df["TxnDate"].dt.date)["Units_Effective"].sum().rename("Units")
            roll7 = daily_trend.rolling(7, min_periods=1).mean().rename("7d Avg")
            st.line_chart(pd.concat([daily_trend, roll7], axis=1))
        else:
            st.info("Upload data with dates to view velocity trends.")
    else:
        st.info("👋 **Welcome to Yash Gallery AI ERP!** Upload Sales Reports (pivot marketplace, B2C/B2B) in the sidebar to activate the AI Dashboard.")

# ==========================================
# TAB 2: PRODUCTION PLANNING
# ==========================================
with tab_plan:
    st.subheader("🏭 Internal Sales Order & BOM Planning")
    st.caption("Plan requirements, check fabric availability, and optimize costs.")

    if 'mock_fabric_inventory' not in st.session_state:
        st.session_state['mock_fabric_inventory'] = pd.DataFrame({
            "Fabric Name": ["Cotton Cambric 60x60", "Cotton Cambric 60x60", "Cotton Cambric 60x60", "Rayon 140g", "Rayon 140g"],
            "Width": ["44 inch", "54 inch", "63 inch", "44 inch", "58 inch"],
            "Stock (Mtr)": [1200, 5000, 800, 200, 4500],
            "Rate (₹/mtr)": [85, 105, 120, 90, 110]
        })

    c_plan1, c_plan2 = st.columns([1, 2])

    with c_plan1:
        st.markdown("#### 1. Requirement Details")
        with st.form("planning_form"):
            selected_style = st.selectbox("Select Style / SKU", ["YG-ANARKALI-101", "YG-KURTI-205", "YG-SET-301"])
            order_type = st.radio("Order Type", ["New Style", "Repeat Order"], horizontal=True)
            qty_required = st.number_input("Total Quantity (Pcs)", min_value=1, value=500)
            merchant = st.selectbox("Assign Merchant", ["Merchant A (Riya)", "Merchant B (Amit)", "Merchant C (Priya)"])
            req_delivery = st.date_input("Required Delivery Date", value=date.today() + timedelta(days=45))
            st.markdown("---")
            st.markdown("**Standard BOM Def:**")
            fabric_type = st.selectbox("Fabric Type", ["Cotton Cambric 60x60", "Rayon 140g"])
            std_consumption = st.number_input("Std Consumption (Mtr) @ 44\"", value=2.5)
            check_bom = st.form_submit_button("🔍 Check BOM & Inventory")

    with c_plan2:
        if check_bom:
            st.markdown(f"#### 2. BOM Analysis for: **{selected_style}** ({qty_required} pcs)")
            inv_df = st.session_state['mock_fabric_inventory']
            fabric_variants = inv_df[inv_df["Fabric Name"] == fabric_type].copy()

            if fabric_variants.empty:
                st.error("Fabric type not found.")
            else:
                results = []
                base_width_inch = 44.0
                for _, row in fabric_variants.iterrows():
                    width_str = row['Width']
                    current_width_inch = float(width_str.split()[0])
                    adj_consumption = round(std_consumption * (base_width_inch / current_width_inch), 2)
                    total_mtr_needed = np.ceil(adj_consumption * qty_required)
                    stock_avail = row['Stock (Mtr)']
                    rate = row['Rate (₹/mtr)']
                    fabric_cost_per_pc = adj_consumption * rate
                    status = "✅ Available" if stock_avail >= total_mtr_needed else f"❌ Shortage ({stock_avail - total_mtr_needed} mtr)"

                    results.append({
                        "Fabric Option": f"{fabric_type} - {width_str}",
                        "Width": width_str,
                        "Cons. (Mtr)": adj_consumption,
                        "Total Req (Mtr)": total_mtr_needed,
                        "Stock (Mtr)": stock_avail,
                        "Rate (₹)": rate,
                        "Fabric Cost/Pc": round(fabric_cost_per_pc, 2),
                        "Status": status
                    })

                res_df = pd.DataFrame(results)
                st.markdown("##### 🧵 Fabric Options")
                avail_opts = res_df[res_df["Status"].str.contains("Available")]
                best_idx = avail_opts["Fabric Cost/Pc"].idxmin() if not avail_opts.empty else -1

                def highlight_best(row):
                    if row.name == best_idx:
                        return ['background-color: #d4edda; font-weight: bold'] * len(row)
                    elif "Shortage" in row["Status"]:
                        return ['background-color: #f8d7da'] * len(row)
                    return [''] * len(row)

                st.dataframe(res_df.style.apply(highlight_best, axis=1), use_container_width=True)

                if not avail_opts.empty and best_idx in res_df.index:
                    st.success(f"💡 Suggestion: Use **{res_df.loc[best_idx]['Fabric Option']}**.")
                    with st.expander("Confirm Sales Order", expanded=True):
                        final_choice = st.selectbox("Confirm Fabric Selection", res_df["Fabric Option"].tolist())
                        if st.button("✅ Generate Internal SO"):
                            st.balloons()
                            st.success("SO Generated Successfully!")
                else:
                    st.warning("⚠️ No sufficient stock. Raise Fabric PO.")
        else:
            st.info("👈 Fill details on the left.")

# ==========================================
# TAB 3: ITEM MASTER
# ==========================================
with tab_item:
    st.markdown("### 🛠️ Create New Finished Good (FG) Style")
    with st.form("fg_creation_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            style_no = st.text_input("Style No.", placeholder="YG-ANARKALI-101")
            sku_code_base = st.text_input("Base SKU", placeholder="YG-101")
        with c2:
            category = st.selectbox("Category", ["Women", "Men", "Kids"])
            item_desc = st.selectbox("Item Type", ["Kurti", "Kurti Set", "Bottom"])
        with c3:
            launch_date = st.date_input("Launch Date", value=date.today())
            season = st.selectbox("Season", ["SS-26", "AW-26", "Core"])
        with c4:
            merchant = st.text_input("Merchandiser")

        st.divider()
        cv1, cv2 = st.columns(2)
        with cv1:
            colors = st.multiselect("Color(s)", ["Indigo", "Red", "Mustard", "Pink"])
        with cv2:
            sizes = st.multiselect("Size Range", ["S", "M", "L", "XL", "2XL"])

        st.divider()
        ct1, ct2 = st.columns(2)
        with ct1:
            no_parts = st.number_input("No. of Parts", 1, 20, 4)
        with ct2:
            is_printed = st.checkbox("Is Printed Style?")

        st.divider()
        st.markdown("#### BOM Costing")
        bom_data = {"Component": ["Fabric", "Lining"], "Rate": [120, 40], "Cons.": [2.5, 0]}
        edited_bom = st.data_editor(pd.DataFrame(bom_data), num_rows="dynamic", use_container_width=True)

        submitted = st.form_submit_button("💾 Save FG Item")
        if submitted:
            st.success(f"Item {style_no} created!")

# ==========================================
# TAB 4: REORDER / PO (FIXED VERSION)
# ==========================================
with tab_po:
    st.subheader("📦 Purchase Order Recommendation Engine")
    
    # First, load inventory ONCE at the top (outside velocity mode selection)
    # This ensures stock data persists regardless of velocity mode changes
    current_stock_global = pd.Series(dtype=float)
    inv_date_used = None
    inv_meta = None
    
    if inv_file is not None:
        with st.expander("🗂️ Inventory Mapping", expanded=True):
            current_stock_global, inv_date_used, inv_meta = map_inventory_current_stock(inv_file)
    
    st.divider()
    
    # Now show velocity controls
    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        calc_basis = st.radio(
            "⚡ Velocity Calculation Mode:",
            ["Last 7 Days (Fast Response)", "Last 30 Days (Stable)", "Full History"],
            horizontal=True
        )

    if not sales_df.empty and sales_df["TxnDate"].notna().any():
        max_date = sales_df["TxnDate"].max()

        # Time window
        if "Last 7 Days" in calc_basis:
            start_date_calc = max_date - pd.Timedelta(days=7)
            period_label = "Last 7d"
            days_divisor = 7
        elif "Last 30 Days" in calc_basis:
            start_date_calc = max_date - pd.Timedelta(days=30)
            period_label = "Last 30d"
            days_divisor = 30
        else:
            start_date_calc = sales_df["TxnDate"].min()
            period_label = "Total"
            days_divisor = (max_date - start_date_calc).days + 1

        # Filtered recent window (inclusive)
        recent_sales = sales_df[sales_df["TxnDate"] >= start_date_calc].copy()

        # Gross/Returns/Net for display
        demand_group = recent_sales.groupby("Sku")
        gross_series = demand_group["Units_Gross"].sum().rename(f"Gross ({period_label})") if "Units_Gross" in recent_sales.columns else demand_group["Units_Effective"].sum().rename(f"Gross ({period_label})")
        return_series = demand_group["Units_Return"].sum().rename(f"Returns ({period_label})") if "Units_Return" in recent_sales.columns else pd.Series(0, index=gross_series.index, name=f"Returns ({period_label})")
        net_series = demand_group["Units_Effective"].sum().rename(f"Net ({period_label})")

        # ADS & Std Dev from daily aggregation in the window
        recent_non_na = recent_sales[recent_sales["TxnDate"].notna()].copy()
        recent_non_na["ds"] = recent_non_na["TxnDate"].dt.date
        daily_by_sku = (recent_non_na
                        .groupby(["Sku", "ds"])["Units_Effective"]
                        .sum()
                        .reset_index())

        ads_series = daily_by_sku.groupby("Sku")["Units_Effective"].mean().rename("ADS")
        std_series = daily_by_sku.groupby("Sku")["Units_Effective"].std(ddof=0).fillna(0.0).rename("Sigma_d")

        # Build reorder DF
        all_index = sorted(set(daily_by_sku["Sku"].unique()) | set(current_stock_global.index))
        reorder = pd.DataFrame(index=all_index)
        reorder = reorder.join(ads_series, how="left").fillna(0)
        reorder = reorder.join(std_series, how="left").fillna(0)
        reorder[f"Gross ({period_label})"] = gross_series.reindex(reorder.index).fillna(0)
        reorder[f"Returns ({period_label})"] = return_series.reindex(reorder.index).fillna(0)
        reorder[f"Net ({period_label})"] = net_series.reindex(reorder.index).fillna(0)
        
        # USE THE GLOBAL STOCK (persists across velocity changes)
        reorder["Current Stock"] = pd.to_numeric(current_stock_global.reindex(reorder.index)).fillna(0)

        # Trend baseline (30d)
        start_30 = max_date - pd.Timedelta(days=30)
        baseline_sales = (sales_df[sales_df["TxnDate"] > start_30]
                          .groupby("Sku")["Units_Effective"].sum() / 30)
        reorder["Baseline_ADS"] = baseline_sales.reindex(reorder.index).fillna(0)

        # Spike cap (optional)
        if apply_spike_cap:
            reorder["ADS_used"] = np.where(
                reorder["Baseline_ADS"] > 0,
                np.minimum(reorder["ADS"], reorder["Baseline_ADS"] * 1.5),
                reorder["ADS"]
            )
        else:
            reorder["ADS_used"] = reorder["ADS"]

        # Per-SKU lead time, pack, MOQ (from vendor master if provided)
        if not vendor_df.empty:
            vm = vendor_df.set_index("Sku")
            reorder["LeadTimeDays_SKU"] = vm.reindex(reorder.index)["LeadTimeDays"].fillna(np.nan)
            reorder["PackSize_SKU"] = vm.reindex(reorder.index)["PackSize"].fillna(np.nan)
            reorder["MOQ_SKU"] = vm.reindex(reorder.index)["MOQ"].fillna(np.nan)
        else:
            reorder["LeadTimeDays_SKU"] = np.nan
            reorder["PackSize_SKU"] = np.nan
            reorder["MOQ_SKU"] = np.nan

        # Effective parameters per SKU
        L_eff = reorder["LeadTimeDays_SKU"].fillna(lead_time_days)
        pack_eff = reorder["PackSize_SKU"].fillna(10).astype(int)
        moq_eff = reorder["MOQ_SKU"].fillna(0).astype(int)

        # Safety stock + Total Required
        z = z_value
        L = L_eff.astype(float)
        reorder["Lead Time Demand"] = reorder["ADS_used"] * L
        reorder["Safety Stock (stat)"] = z * reorder["Sigma_d"] * np.sqrt(L)
        reorder["Safety Stock (pct)"] = reorder["Lead Time Demand"] * (safety_stock_pct / 100.0)
        reorder["Safety Stock"] = np.maximum(reorder["Safety Stock (stat)"], reorder["Safety Stock (pct)"])
        reorder["Total Required"] = reorder["Lead Time Demand"] + reorder["Safety Stock"]

        def round_to_pack_moq(req, pack, moq):
            if req <= 0:
                return 0
            # apply moq first then pack rounding
            x = max(req, moq)
            if pack <= 1:
                return int(math.ceil(x))
            return int(math.ceil(x / pack) * pack)

        reorder["Recommended PO Raw"] = (reorder["Total Required"] - reorder["Current Stock"]).clip(lower=0)
        reorder["Recommended PO"] = [
            round_to_pack_moq(req, pack, moq)
            for req, pack, moq in zip(reorder["Recommended PO Raw"], pack_eff, moq_eff)
        ]

        # Days left = stock cover (use ADS_used)
        reorder["Days Left"] = np.where(
            reorder["ADS_used"] > 0.1,
            (reorder["Current Stock"] / reorder["ADS_used"]).round(1),
            999
        )
        
        # NEW: Stock Status indicator
        reorder["Stock Status"] = np.where(
            reorder["Current Stock"] >= reorder["Total Required"],
            "✅ Sufficient",
            np.where(
                reorder["Current Stock"] > 0,
                "⚠️ Low Stock",
                "❌ Out of Stock"
            )
        )

        # KPIs
        display_master = reorder.copy()
        if hide_non_urgent:
            display_master = display_master[(display_master["Days Left"] <= (L + 7)) | (display_master["Recommended PO"] > 0)]

        total_units = int(display_master["Recommended PO"].sum())
        urgent_skus = int((display_master["Days Left"] < L).sum())
        out_of_stock = int((display_master["Current Stock"] == 0).sum())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Units to Order", f"{total_units:,}")
        kpi2.metric("Critical SKUs", f"{urgent_skus}", delta="Stockout Risk", delta_color="inverse")
        kpi3.metric("Out of Stock", f"{out_of_stock}", delta_color="inverse")
        kpi4.metric("Stock Date", inv_meta["stock_date"] if inv_meta else "N/A")

        # Display
        # ===== SKU SEARCH & FILTERING =====
        st.markdown("---")
        st.markdown("### 🔍 Search & Filter SKUs")
        
        search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
        
        with search_col1:
            sku_search = st.text_input(
                "Search SKU (supports partial match)",
                placeholder="e.g., 1065YKBLUE or 1065YK or BLUE-L",
                help="Enter full SKU or partial text to find matching products",
                key="po_search"
            )
        
        with search_col2:
            search_type_po = st.radio(
                "Search Mode",
                ["All SKUs", "Exact Match", "Contains"],
                horizontal=False,
                key="po_search_mode"
            )
        
        with search_col3:
            filter_urgent = st.checkbox("Show Only Urgent POs", value=False, help="Show only SKUs needing immediate orders")
        
        # Apply search filter
        if sku_search and search_type_po != "All SKUs":
            if search_type_po == "Contains":
                matching_indices = [idx for idx in display_master.index if sku_search.upper() in str(idx).upper()]
            else:  # Exact Match
                matching_indices = [idx for idx in display_master.index if sku_search.upper() == str(idx).upper()]
            
            if matching_indices:
                display_master = display_master.loc[matching_indices]
                st.success(f"✅ Found {len(matching_indices)} matching SKU(s)")
            else:
                st.warning(f"❌ No SKUs found matching '{sku_search}'")
                st.stop()
        
        # Apply urgent filter
        if filter_urgent:
            display_master = display_master[display_master["Recommended PO"] > 0]
        
        # ===== VARIANT GROUPING ANALYSIS =====
        if sku_search and search_type_po == "Contains" and len(display_master) > 1:
            st.markdown("---")
            st.markdown(f"### 📦 Size Variant Analysis for '{sku_search}'")
            
            variant_summary = display_master.copy()
            variant_summary["SKU"] = variant_summary.index
            
            # Aggregate metrics
            agg_col1, agg_col2, agg_col3, agg_col4, agg_col5 = st.columns(5)
            
            total_variants = len(variant_summary)
            total_po_needed = int(variant_summary["Recommended PO"].sum())
            total_current_stock = int(variant_summary["Current Stock"].sum())
            total_demand = int(variant_summary[f"Net ({period_label})"].sum())
            variants_needing_po = int((variant_summary["Recommended PO"] > 0).sum())
            
            agg_col1.metric("Total Variants", total_variants)
            agg_col2.metric("Combined PO Needed", f"{total_po_needed:,} units")
            agg_col3.metric("Current Stock (All)", f"{total_current_stock:,} units")
            agg_col4.metric("Total Demand", f"{total_demand:,} units")
            agg_col5.metric("Variants Needing PO", f"{variants_needing_po}/{total_variants}")
            
            st.markdown("#### 📊 Variant Comparison")
            
            # Create comparison table
            comparison_cols = [
                "SKU",
                f"Net ({period_label})",
                "Current Stock",
                "Stock Status",
                "Days Left",
                "Recommended PO"
            ]
            
            comparison_df = variant_summary[[c for c in comparison_cols if c in variant_summary.columns]].copy()
            comparison_df = comparison_df.sort_values("Recommended PO", ascending=False)
            
            # Add priority flag
            comparison_df["Priority"] = comparison_df.apply(
                lambda row: "🔴 URGENT" if row["Days Left"] < lead_time_days and row["Recommended PO"] > 0
                else "🟡 MEDIUM" if row["Recommended PO"] > 0
                else "🟢 OK",
                axis=1
            )
            
            # Reorder columns
            display_order = ["Priority", "SKU"] + [c for c in comparison_df.columns if c not in ["Priority", "SKU"]]
            comparison_df = comparison_df[display_order]
            
            # Style the table
            def highlight_variant_priority(row):
                colors = []
                for col in row.index:
                    if col == "Priority":
                        if "URGENT" in str(row[col]):
                            colors.append('background-color: #f8d7da; font-weight: bold')
                        elif "MEDIUM" in str(row[col]):
                            colors.append('background-color: #fff3cd')
                        else:
                            colors.append('background-color: #d4edda')
                    elif col == "Recommended PO":
                        if row[col] > 0:
                            colors.append('background-color: #fff3cd; font-weight: bold')
                        else:
                            colors.append('')
                    else:
                        colors.append('')
                return colors
            
            st.dataframe(
                comparison_df.style.apply(highlight_variant_priority, axis=1)
                                   .format({
                                       f"Net ({period_label})": "{:.0f}",
                                       "Current Stock": "{:.0f}",
                                       "Days Left": "{:.1f}",
                                       "Recommended PO": "{:.0f}"
                                   }),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary insights
            st.markdown("#### 💡 Ordering Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Which sizes need urgent ordering
                urgent_variants = comparison_df[comparison_df["Priority"] == "🔴 URGENT"]
                if not urgent_variants.empty:
                    st.error(f"**⚠️ Urgent:** {len(urgent_variants)} variant(s) need immediate ordering")
                    for _, row in urgent_variants.iterrows():
                        st.write(f"- {row['SKU']}: Order {int(row['Recommended PO'])} units (only {row['Days Left']:.0f} days left)")
                else:
                    st.success("✅ No urgent orders needed for this product family")
            
            with insight_col2:
                # Size distribution recommendation
                if total_po_needed > 0:
                    st.info("**📊 Suggested Order Distribution:**")
                    for _, row in comparison_df[comparison_df["Recommended PO"] > 0].iterrows():
                        pct = (row["Recommended PO"] / total_po_needed * 100) if total_po_needed > 0 else 0
                        st.write(f"- {row['SKU']}: {int(row['Recommended PO'])} units ({pct:.0f}%)")
                    
                    st.caption(f"**Total PO for {sku_search}: {total_po_needed:,} units**")
            
            st.markdown("---")
        
        # ===== MAIN PO TABLE =====
        st.markdown("### 📋 Purchase Order Recommendations")
        
        cols_to_show = [f"Gross ({period_label})", f"Returns ({period_label})", f"Net ({period_label})",
                        "ADS", "ADS_used", 
                        "Current Stock", "Stock Status", "Days Left", 
                        "Lead Time Demand", "Safety Stock", "Total Required", "Recommended PO"]
        display_df = display_master[cols_to_show].sort_values("Recommended PO", ascending=False)

        # Color coding for stock status
        def highlight_stock_status(row):
            colors = []
            for col in row.index:
                if col == "Stock Status":
                    if "Sufficient" in str(row[col]):
                        colors.append('background-color: #d4edda')
                    elif "Low Stock" in str(row[col]):
                        colors.append('background-color: #fff3cd')
                    else:
                        colors.append('background-color: #f8d7da')
                elif col == "Days Left":
                    val = row[col]
                    if val < lead_time_days:
                        colors.append('background-color: #f8d7da; font-weight: bold')
                    elif val < lead_time_days + 7:
                        colors.append('background-color: #fff3cd')
                    else:
                        colors.append('')
                else:
                    colors.append('')
            return colors

        st.dataframe(
            display_df.style.apply(highlight_stock_status, axis=1)
                              .format({
                                  "ADS": "{:.2f}", "ADS_used": "{:.2f}",
                                  "Recommended PO": "{:.0f}", f"Gross ({period_label})": "{:.0f}",
                                  f"Net ({period_label})": "{:.0f}", "Lead Time Demand": "{:.1f}",
                                  "Safety Stock": "{:.1f}", "Total Required": "{:.1f}",
                                  "Current Stock": "{:.0f}", "Days Left": "{:.1f}"
                              }),
            use_container_width=True,
            height=600
        )

        # Help text
        st.info("""
        **Understanding the Table:**
        - **Stock Status**: ✅ Sufficient (no order needed) | ⚠️ Low Stock (order recommended) | ❌ Out of Stock (urgent order)
        - **Days Left**: How many days current stock will last at current velocity
        - **Recommended PO**: 0 means you have enough stock; >0 means you need to order
        """)

        # Downloads
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer)
        st.download_button("⬇️ Download PO as CSV",
                           data=csv_buffer.getvalue(),
                           file_name=f"PO_Recommendation_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        # Excel multi-sheet (PO + raw components + params)
        xl_buffer = io.BytesIO()
        with pd.ExcelWriter(xl_buffer, engine="openpyxl") as writer:
            display_df.to_excel(writer, sheet_name="PO", index=True)
            base_cols = [f"Gross ({period_label})", f"Returns ({period_label})", f"Net ({period_label})",
                         "ADS", "ADS_used", "Current Stock", "Stock Status",
                         "Lead Time Demand", "Safety Stock", "Total Required", "Recommended PO", "Days Left"]
            extra_cols = ["LeadTimeDays_SKU", "PackSize_SKU", "MOQ_SKU"]
            export_cols = base_cols + [c for c in extra_cols if c in reorder.columns]
            reorder[export_cols].to_excel(writer, sheet_name="ReorderData", index=True)
            if not vendor_df.empty:
                vendor_df.to_excel(writer, sheet_name="VendorMaster", index=False)
            if inv_meta:
                pd.DataFrame([inv_meta]).to_excel(writer, sheet_name="InventoryMapping", index=False)
        st.download_button("⬇️ Download PO (Excel)",
                           data=xl_buffer.getvalue(),
                           file_name=f"PO_Recommendation_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Please upload Sales Reports to generate PO recommendations.")

# ==========================================
# TAB 5: AI FORECAST
# ==========================================
with tab_forecast:
    st.subheader("🤖 AI Demand Forecast")
    if not sales_df.empty:
        sku_list = sorted(sales_df["Sku"].unique())
        c_f1, c_f2 = st.columns([1, 2])
        with c_f1:
            selected_sku = st.selectbox("Select SKU to Forecast", sku_list)
            forecast_days = st.slider("Forecast Horizon", 30, 90, 60)

        if selected_sku:
            daily = build_daily_series_for_sku(sales_df, selected_sku)
            if len(daily) > 0:
                with st.spinner("Training AI..."):
                    fc = forecast_series(daily, horizon_days=forecast_days)
                next_30 = fc.head(30)["yhat"].sum()
                st.metric("Predicted Demand (Next 30 Days)", f"{int(next_30)} units")
                fig = plot_prophet_forecast(daily, fc)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for AI forecast.")
    else:
        st.info("Upload data to see forecasts.")

# ==========================================
# TAB 6: SKU DRILLDOWN
# ==========================================
with tab_sku:
    st.subheader("🔎 SKU Analysis & Marketplace Breakdown")
    if not sales_df.empty:
        st.markdown("**Search Options:**")
        search_col1, search_col2 = st.columns([2, 1])
        
        with search_col1:
            # Text search box for partial matching
            search_term = st.text_input(
                "🔍 Search SKU (supports partial match)",
                placeholder="e.g., 1065YKBLUE or 1065YK or BLUE-L",
                help="Enter full SKU or partial text to find matching products"
            )
        
        with search_col2:
            search_mode = st.radio(
                "Search Type",
                ["Exact Match", "Contains (Partial)"],
                horizontal=True,
                help="Exact: Find specific SKU | Contains: Find all matching SKUs"
            )
        
        # Filter SKUs based on search
        all_skus = sorted(sales_df["Sku"].unique())
        
        if search_term:
            if search_mode == "Contains (Partial)":
                matching_skus = [sku for sku in all_skus if search_term.upper() in str(sku).upper()]
            else:
                matching_skus = [sku for sku in all_skus if search_term.upper() == str(sku).upper()]
            
            if not matching_skus:
                st.warning(f"❌ No SKUs found matching '{search_term}'. Try partial search or check spelling.")
                st.info(f"💡 Tip: Try searching for just the style code (e.g., '1065YK' instead of '1065YKBLUE-L')")
            else:
                st.success(f"✅ Found {len(matching_skus)} matching SKU(s)")
                
                # If multiple matches, show variant analysis
                if len(matching_skus) > 1:
                    st.markdown("---")
                    st.markdown(f"### 📦 SKU Variants for '{search_term}'")
                    st.caption(f"Found {len(matching_skus)} size/color variants")
                    
                    # Aggregate data for all variants
                    all_variants = sales_df[sales_df["Sku"].isin(matching_skus)].copy()
                    
                    # Summary across all variants
                    var_col1, var_col2, var_col3, var_col4 = st.columns(4)
                    
                    total_variants_shipments = int(all_variants[all_variants["Transaction Type"] == "Shipment"]["Quantity"].sum())
                    total_variants_returns = int(all_variants[all_variants["Transaction Type"] == "Refund"]["Quantity"].sum())
                    total_variants_net = int(all_variants["Units_Effective"].sum())
                    total_variants_revenue = float(all_variants["Revenue_Effective"].sum())
                    
                    var_col1.metric("Total Variants", len(matching_skus))
                    var_col2.metric("Combined Shipments", f"{total_variants_shipments:,}")
                    var_col3.metric("Combined Net Units", f"{total_variants_net:,}")
                    var_col4.metric("Combined Revenue", f"₹{total_variants_revenue:,.0f}")
                    
                    # Breakdown by variant (size/color)
                    st.markdown("#### 📊 Performance by Variant")
                    
                    variant_analysis = all_variants.groupby("Sku").agg({
                        "Units_Effective": "sum",
                        "Quantity": "sum", 
                        "Revenue_Effective": "sum"
                    }).reset_index()
                    variant_analysis.columns = ["SKU", "Net Units", "Total Transactions", "Revenue"]
                    variant_analysis = variant_analysis.sort_values("Net Units", ascending=False)
                    
                    # Add percentage and rank
                    variant_analysis["% of Group"] = (
                        variant_analysis["Net Units"] / variant_analysis["Net Units"].sum() * 100
                    ).round(1)
                    variant_analysis["Rank"] = range(1, len(variant_analysis) + 1)
                    
                    # Reorder columns
                    variant_analysis = variant_analysis[["Rank", "SKU", "Net Units", "Total Transactions", "Revenue", "% of Group"]]
                    
                    # Display with highlighting
                    def highlight_top_variants(row):
                        if row["Rank"] == 1:
                            return ['background-color: #d4edda; font-weight: bold'] * len(row)
                        elif row["Rank"] <= 3:
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    st.dataframe(
                        variant_analysis.style.apply(highlight_top_variants, axis=1)
                                              .format({
                                                  "Net Units": "{:,.0f}",
                                                  "Total Transactions": "{:,.0f}",
                                                  "Revenue": "₹{:,.0f}",
                                                  "% of Group": "{:.1f}%"
                                              }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Top performer callout
                    top_variant = variant_analysis.iloc[0]
                    st.info(f"🏆 **Best Selling Variant:** {top_variant['SKU']} ({top_variant['% of Group']:.0f}% of group sales)")
                    
                    st.markdown("---")
                
                # Allow selection from matching SKUs
                drill_sku = st.selectbox(
                    "Select SKU for Detailed Analysis" if len(matching_skus) > 1 else "Analyzing SKU",
                    matching_skus,
                    key="drill"
                )
        else:
            # No search term - show dropdown with all SKUs
            drill_sku = st.selectbox("Or select from all SKUs:", all_skus, key="drill_all")
        
        # Individual SKU analysis (only if a specific SKU is selected)
        if 'drill_sku' in locals() or 'drill_sku' in globals():
            subset = sales_df[sales_df["Sku"] == drill_sku].copy()
            
            if not subset.empty:
                st.markdown("---")
                st.markdown(f"### 📋 Detailed Analysis: **{drill_sku}**")
                
                # Summary metrics for this SKU
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_shipments = int(subset[subset["Transaction Type"] == "Shipment"]["Quantity"].sum())
                total_returns = int(subset[subset["Transaction Type"] == "Refund"]["Quantity"].sum())
                net_units = int(subset["Units_Effective"].sum())
                total_revenue = float(subset["Revenue_Effective"].sum())
                return_rate = (total_returns / total_shipments * 100) if total_shipments > 0 else 0
                
                col1.metric("Total Shipments", f"{total_shipments:,}")
                col2.metric("Total Returns", f"{total_returns:,}")
                col3.metric("Net Units Sold", f"{net_units:,}")
                col4.metric("Total Revenue", f"₹{total_revenue:,.0f}")
                col5.metric("Return Rate", f"{return_rate:.1f}%", delta_color="inverse")
                
                st.divider()
                
                # Marketplace breakdown
                if "Source" in subset.columns:
                    st.markdown("#### 📊 Marketplace Performance")
                    
                    marketplace_summary = subset.groupby("Source").agg({
                        "Units_Effective": "sum",
                        "Quantity": "sum",
                        "Revenue_Effective": "sum"
                    }).reset_index()
                    marketplace_summary.columns = ["Marketplace", "Net Units", "Total Transactions", "Revenue"]
                    marketplace_summary = marketplace_summary.sort_values("Net Units", ascending=False)
                    
                    # Add percentage
                    marketplace_summary["% of Total"] = (
                        marketplace_summary["Net Units"] / marketplace_summary["Net Units"].sum() * 100
                    ).round(1)
                    
                    col_chart, col_table = st.columns([1, 1])
                    
                    with col_chart:
                        # Pie chart showing marketplace distribution
                        fig_marketplace = px.pie(
                            marketplace_summary,
                            values="Net Units",
                            names="Marketplace",
                            title=f"Where {drill_sku} is Sold",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_marketplace, use_container_width=True)
                    
                    with col_table:
                        st.markdown("**Breakdown by Channel:**")
                        st.dataframe(
                            marketplace_summary.style.format({
                                "Net Units": "{:,.0f}",
                                "Total Transactions": "{:,.0f}",
                                "Revenue": "₹{:,.0f}",
                                "% of Total": "{:.1f}%"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Highlight dominant marketplace
                        dominant = marketplace_summary.iloc[0]
                        st.info(f"💡 **Primary Channel:** {dominant['Marketplace']} ({dominant['% of Total']:.0f}% of sales)")
                    
                    st.divider()
                
                # Transaction history with Source column
                st.markdown("#### 📋 Transaction History (Last 50)")
                
                display_cols = ["TxnDate", "Source", "Transaction Type", "Quantity", "Revenue_Effective"]
                display_cols = [c for c in display_cols if c in subset.columns]
                
                history_df = subset[display_cols].sort_values("TxnDate", ascending=False).head(50)
                
                # Apply styling
                def highlight_txn_type(row):
                    colors = []
                    for col in row.index:
                        if col == "Transaction Type":
                            if row[col] == "Shipment":
                                colors.append('background-color: #d4edda')
                            elif row[col] == "Refund":
                                colors.append('background-color: #f8d7da')
                            else:
                                colors.append('')
                        else:
                            colors.append('')
                    return colors
                
                st.dataframe(
                    history_df.style.apply(highlight_txn_type, axis=1)
                                    .format({
                                        "TxnDate": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "",
                                        "Revenue_Effective": "₹{:.2f}",
                                        "Quantity": "{:.0f}"
                                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Download option
                csv_buffer = io.StringIO()
                subset[display_cols].to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download Full Transaction History",
                    data=csv_buffer.getvalue(),
                    file_name=f"SKU_{drill_sku}_transactions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No transactions found for this SKU.")
    else:
        st.info("Upload sales data to drill down.")

# ==========================================
# TAB 7: TRANSFERS
# ==========================================
with tab_transfers:
    st.subheader("🚚 Inventory Transfers")
    if transfer_df is not None and not transfer_df.empty:
        st.dataframe(transfer_df, use_container_width=True)
    else:
        st.info("Upload Transfer report to see data.")
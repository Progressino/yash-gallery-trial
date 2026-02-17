#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System — app.py (MTR-INTEGRATED VERSION)

✅ What's new vs previous version:
─────────────────────────────────
1) FULL MTR (Merchant Tax Report) Integration
   - Dedicated sidebar uploaders for Amazon B2B & B2C MTR ZIPs/CSVs
   - Handles both B2B (89 cols) and B2C (78 cols) column schemas
   - load_mtr_reports() correctly parses Invoice Amount, Tax, State, Payment Method
   - Supports multi-file upload: upload all months at once
   - Strips leading/trailing whitespace from B2C column headers automatically

2) NEW "📑 MTR Analytics" tab
   - Revenue KPIs: Gross, Net, Tax Collected, Avg Order Value
   - B2B vs B2C side-by-side comparison
   - Monthly revenue trend chart (B2B vs B2C)
   - State-wise revenue heatmap
   - Transaction type breakdown (Shipment / Refund / Cancel)
   - Payment method distribution
   - Top SKUs by revenue
   - Downloadable MTR summary (CSV + Excel)

3) PO Engine, Inventory, Dashboard, Forecast, Deep Dive — all retained & unchanged.

4) ADS fix retained: stable period denominator, not "days with sales".
5) Inventory fix retained: variant vs parent grouping.
"""

import io
import re
import zipfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Prophet is optional — imported lazily inside the Forecast tab
# so a missing install only breaks forecasting, not the whole app
try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 1) PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Yash Gallery ERP",
    page_icon="🚀",
    layout="wide"
)

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
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 5px solid #002B5B;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"]  { color:#6B7280; font-size:0.9rem; font-weight:600; }
    div[data-testid="stMetricValue"]  { color:#111827; font-size:1.8rem !important; font-weight:700; }
    div.stButton > button {
        background: linear-gradient(135deg,#002B5B 0%,#1e40af 100%);
        color: white; border: none;
        padding: 0.5rem 1.2rem;
        border-radius: 8px; font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(30,64,175,0.3);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white; border-radius: 6px;
        color: #4B5563; font-weight: 600;
        border: 1px solid #E5E7EB; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #002B5B !important;
        color: white !important;
        border: 1px solid #002B5B;
    }
    h1, h2, h3 { color: #002B5B !important; }
    .mtr-kpi-card {
        background: white; border-radius: 12px;
        padding: 16px; border: 1px solid #E5E7EB;
        border-left: 5px solid #002B5B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Yash Gallery Command Center")
st.caption("Complete ERP: Sales Analytics • Inventory • PO Engine • MTR Analytics • AI Forecasting")

# ══════════════════════════════════════════════════════════════
# 2) SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "sku_mapping":           {},
        "sales_df":              pd.DataFrame(),
        "inventory_df_variant":  pd.DataFrame(),
        "inventory_df_parent":   pd.DataFrame(),
        "transfer_df":           pd.DataFrame(),
        "mtr_df":                pd.DataFrame(),   # ✅ NEW: combined MTR data
        "amazon_date_basis":     "Shipment Date",
        "include_replacements":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ══════════════════════════════════════════════════════════════
# 3) CONFIGURATION DATACLASS
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SalesConfig:
    date_basis: str = "Shipment Date"
    include_replacements: bool = False

# ══════════════════════════════════════════════════════════════
# 4) UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════
def clean_sku(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip()

def get_parent_sku(oms_sku) -> str:
    if pd.isna(oms_sku):
        return oms_sku
    s = str(oms_sku).strip()
    marketplace_suffixes = ["_Myntra","_Flipkart","_Amazon","_Meesho",
                            "_MYNTRA","_FLIPKART","_AMAZON","_MEESHO"]
    for suf in marketplace_suffixes:
        if s.endswith(suf):
            s = s.replace(suf, "")
            break
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2:
            last = parts[-1].upper()
            size_patterns = {"XS","S","M","L","XL","XXL","XXXL","2XL","3XL","4XL","5XL","6XL"}
            common_colors = {
                "RED","BLUE","GREEN","BLACK","WHITE","YELLOW","PINK","PURPLE","ORANGE","BROWN",
                "GREY","GRAY","NAVY","MAROON","BEIGE","CREAM","GOLD","SILVER","TAN","KHAKI",
                "OLIVE","TEAL","CORAL","PEACH"
            }
            is_size  = (last in size_patterns or last.endswith("XL") or last.isdigit()
                        or (len(last) <= 4 and any(c in last for c in ["S","M","L","X"])))
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

def fmt_inr(val: float) -> str:
    """Format number as Indian Rupees with ₹ symbol and commas."""
    if abs(val) >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.2f} Cr"
    elif abs(val) >= 1_00_000:
        return f"₹{val/1_00_000:.2f} L"
    return f"₹{val:,.0f}"

# ══════════════════════════════════════════════════════════════
# 5) SKU MAPPING LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_sku_mapping(mapping_file) -> Dict[str, str]:
    mapping_dict = {}
    try:
        xls = pd.ExcelFile(mapping_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(mapping_file, sheet_name=sheet_name)
            if df.empty or len(df.columns) < 2:
                continue
            seller_col, oms_col = None, None
            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["seller","myntra","meesho","snapdeal","sku id"]) and "sku" in col_lower:
                    seller_col = col
                if "oms" in col_lower and "sku" in col_lower:
                    oms_col = col
            if seller_col is None and len(df.columns) > 1:
                seller_col = df.columns[1]
            if oms_col is None:
                oms_col = df.columns[-1]
            if seller_col and oms_col:
                for _, row in df.iterrows():
                    s = clean_sku(row.get(seller_col, ""))
                    o = clean_sku(row.get(oms_col, ""))
                    if s and o and s != "nan" and o != "nan":
                        mapping_dict[s] = o
        return mapping_dict
    except Exception as e:
        st.error(f"Error loading SKU mapping: {e}")
        return {}

# ══════════════════════════════════════════════════════════════
# 6) MTR LOADER  ✅ SINGLE-ZIP RECURSIVE VERSION
# ══════════════════════════════════════════════════════════════

# Month name → zero-padded number
_MONTH_MAP = {
    "JANUARY":"01","FEBRUARY":"02","MARCH":"03","APRIL":"04",
    "MAY":"05","JUNE":"06","JULY":"07","AUGUST":"08",
    "SEPTEMBER":"09","OCTOBER":"10","NOVEMBER":"11","DECEMBER":"12",
}

def _detect_report_type(filename: str) -> str:
    """Return 'B2B', 'B2C', or 'UNKNOWN' from the CSV/zip filename."""
    n = filename.upper()
    if "B2B" in n:
        return "B2B"
    if "B2C" in n:
        return "B2C"
    return "UNKNOWN"

def _detect_period(filename: str):
    """
    Return (period_str, label) e.g. ('2024-04', 'April 2024').
    Looks for MONTHNAME + 4-digit year anywhere in the filename.
    Falls back to None, None if not found.
    """
    n = filename.upper()
    for month_name, month_num in _MONTH_MAP.items():
        if month_name in n:
            m = re.search(r"(20\d{2})", n)
            if m:
                year = m.group(1)
                return f"{year}-{month_num}", f"{month_name.title()} {year}"
    return None, None


def _parse_mtr_csv(csv_bytes: bytes, report_type: str, source_file: str) -> pd.DataFrame:
    """
    Parse raw CSV bytes into a normalised MTR DataFrame.
    Handles both B2B (89-col) and B2C (78-col) schemas transparently.
    """
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False)
    except Exception as e:
        return pd.DataFrame()

    raw.columns = raw.columns.str.strip()          # ← fixes B2C trailing spaces
    if raw.empty:
        return pd.DataFrame()

    # ── date ────────────────────────────────────────────────
    if "Shipment Date" in raw.columns:
        raw["_Date"] = pd.to_datetime(raw["Shipment Date"], errors="coerce")
    elif "Invoice Date" in raw.columns:
        raw["_Date"] = pd.to_datetime(raw["Invoice Date"], errors="coerce")
    else:
        raw["_Date"] = pd.NaT

    # ── numeric money cols ──────────────────────────────────
    for col in ["Invoice Amount","Tax Exclusive Gross","Total Tax Amount",
                "Principal Amount","Cgst Tax","Sgst Tax","Igst Tax","Utgst Tax",
                "Shipping Amount","Item Promo Discount",
                "Tcs Igst Amount","Tcs Cgst Amount","Tcs Sgst Amount",
                "Compensatory Cess Tax"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

    raw["Quantity"] = pd.to_numeric(raw.get("Quantity", 0), errors="coerce").fillna(0)

    # ── helper: safe column access ──────────────────────────
    def g(*names):
        for n in names:
            if n in raw.columns:
                return raw[n].fillna("").astype(str)
        return pd.Series("", index=raw.index)

    def gn(*names):
        for n in names:
            if n in raw.columns:
                return pd.to_numeric(raw[n], errors="coerce").fillna(0)
        return pd.Series(0.0, index=raw.index)

    # ── build output ────────────────────────────────────────
    out = pd.DataFrame({
        "Date":             raw["_Date"],
        "Report_Type":      report_type,
        "Transaction_Type": g("Transaction Type").str.strip(),
        "SKU":              g("Sku").str.strip(),
        "ASIN":             g("Asin").str.strip(),
        "Description":      g("Item Description").str.strip(),
        "Quantity":         raw["Quantity"],
        "Invoice_Amount":   gn("Invoice Amount"),
        "Tax_Excl_Gross":   gn("Tax Exclusive Gross"),
        "Total_Tax":        gn("Total Tax Amount"),
        "Principal_Amt":    gn("Principal Amount"),
        "CGST":             gn("Cgst Tax"),
        "SGST":             gn("Sgst Tax"),
        "IGST":             gn("Igst Tax"),
        "IGST_Rate":        gn("Igst Rate"),
        "Ship_To_State":    g("Ship To State").str.strip().str.upper(),
        "Ship_To_City":     g("Ship To City").str.strip().str.title(),
        "Warehouse_Id":     g("Warehouse Id").str.strip(),
        "Fulfillment":      g("Fulfillment Channel").str.strip(),
        "Payment_Method":   g("Payment Method Code").str.strip(),
        "Order_Id":         g("Order Id").str.strip(),
        "Invoice_Number":   g("Invoice Number").str.strip(),
        "Credit_Note_No":   g("Credit Note No").str.strip(),
        # B2B-only (empty string for B2C rows)
        "Buyer_Name":       g("Buyer Name").str.strip(),
        "Bill_To_State":    g("Bill To State").str.strip().str.upper(),
        "Buyer_GSTIN":      g("Customer Bill To Gstid").str.strip(),
        "IRN_Status":       g("Irn Filing Status").str.strip(),
        "Source_File":      source_file,
    })

    out = out.dropna(subset=["Date"])
    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    return out


def load_mtr_from_main_zip(main_zip_file) -> pd.DataFrame:
    """
    Single-entry-point loader.

    Accepts ONE main ZIP file (uploaded via Streamlit).
    Structure handled:
        main.zip
        └── April-2024.zip   (sub-zip, named by month)
            └── MTR_B2B-APRIL-2024-XXXX.csv   (one CSV per sub-zip)
        └── April-2024.zip
            └── MTR_B2C-APRIL-2024-XXXX.csv
        └── May-2024.zip
            └── MTR_B2B-MAY-2024-XXXX.csv
        ...

    B2B vs B2C auto-detected from the CSV filename (must contain 'B2B' or 'B2C').
    Also works if CSVs are nested deeper (3+ levels) or if the main zip
    directly contains CSVs (no sub-zips).

    Returns deduplicated, normalised MTR DataFrame.
    """
    parts      = []
    skipped    = []
    csv_count  = 0

    def _walk(zf: zipfile.ZipFile, depth: int = 0):
        nonlocal csv_count
        for item_name in zf.namelist():
            base = Path(item_name).name
            if not base:
                continue
            try:
                data = zf.read(item_name)
            except Exception:
                continue

            if base.lower().endswith(".zip"):
                # Recurse into sub-zip
                try:
                    sub_zf = zipfile.ZipFile(io.BytesIO(data))
                    _walk(sub_zf, depth + 1)
                except Exception as e:
                    skipped.append(f"{base}: {e}")

            elif base.lower().endswith(".csv"):
                rtype = _detect_report_type(base)
                if rtype == "UNKNOWN":
                    # Try parent zip name for type hint
                    rtype = _detect_report_type(item_name)
                if rtype == "UNKNOWN":
                    skipped.append(f"{base}: cannot determine B2B/B2C")
                    continue

                df = _parse_mtr_csv(data, rtype, base)
                if not df.empty:
                    csv_count += 1
                    parts.append(df)
                else:
                    skipped.append(f"{base}: empty after parse")

    # ── open & walk ─────────────────────────────────────────
    try:
        main_zip_file.seek(0)
        root_zf = zipfile.ZipFile(main_zip_file)
    except Exception as e:
        st.error(f"Cannot open main ZIP: {e}")
        return pd.DataFrame(), 0

    _walk(root_zf)

    if skipped:
        st.sidebar.warning(f"⚠️ Skipped {len(skipped)} file(s):\n" + "\n".join(skipped[:5]))

    if not parts:
        return pd.DataFrame(), 0

    combined = pd.concat(parts, ignore_index=True)

    # ── deduplicate across overlapping monthly uploads ───────
    combined = combined.drop_duplicates(
        subset=["Order_Id", "SKU", "Transaction_Type", "Date"], keep="first"
    )

    return combined, csv_count

# ══════════════════════════════════════════════════════════════
# 7) SALES DATA LOADERS (unchanged)
# ══════════════════════════════════════════════════════════════
def load_amazon_sales(zip_file, mapping: Dict[str, str], source: str, config: SalesConfig) -> pd.DataFrame:
    df = read_zip_csv(zip_file)
    if df.empty or "Sku" not in df.columns:
        return pd.DataFrame()

    df["OMS_SKU"] = df["Sku"].apply(lambda x: map_to_oms_sku(x, mapping))
    date_col = config.date_basis
    if date_col not in df.columns:
        date_col = ("Shipment Date" if "Shipment Date" in df.columns else
                    "Invoice Date"  if "Invoice Date"  in df.columns else
                    "Order Date"    if "Order Date"    in df.columns else df.columns[0])

    df["TxnDate"]  = pd.to_datetime(df[date_col], errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)

    def classify_txn(t):
        s = str(t).lower()
        if "refund" in s or "return" in s:   return "Refund"
        if "cancel" in s:                    return "Cancel"
        if "freereplacement" in s or "replacement" in s: return "FreeReplacement"
        return "Shipment"

    df["TxnType"] = df.get("Transaction Type", "").apply(classify_txn)
    if not config.include_replacements:
        df.loc[df["TxnType"] == "FreeReplacement", "Quantity"] = 0

    df["Units_Effective"] = np.where(
        df["TxnType"] == "Refund",  -df["Quantity"],
        np.where(df["TxnType"] == "Cancel", 0, df["Quantity"])
    )
    df["Source"] = source

    order_col = next((c for c in df.columns if "order" in c.lower() and "id" in c.lower()), None)
    df["OrderId"] = df[order_col] if order_col else np.nan

    result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
    result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
    return result.dropna(subset=["TxnDate"])


def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_file, sheet_name="Sales Report")
        if df.empty: return pd.DataFrame()
        df["OMS_SKU"] = df["SKU"].apply(clean_sku).apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"]  = pd.to_datetime(df.get("Order Date"), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("Item Quantity", 0), errors="coerce").fillna(0)
        df["Source"]   = "Flipkart"
        df["TxnType"]  = df.get("Event Sub Type","").apply(
            lambda x: "Refund" if "return" in str(x).lower() else "Shipment")
        df["Units_Effective"] = np.where(df["TxnType"] == "Refund", -df["Quantity"], df["Quantity"])
        df["OrderId"] = df.get("Order ID", df.get("Order Id", np.nan))
        result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
        result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}"); return pd.DataFrame()


def load_meesho_sales(zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            excel_files = [f for f in z.namelist()
                           if "tcs_sales" in f.lower() and f.lower().endswith(".xlsx")
                           and "return" not in f.lower()]
            if not excel_files: return pd.DataFrame()
            with z.open(excel_files[0]) as f:
                df = pd.read_excel(f)
        if df.empty: return pd.DataFrame()
        df["OMS_SKU"]        = df.get("identifier").apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"]        = pd.to_datetime(df.get("order_date"), errors="coerce")
        df["Quantity"]       = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
        df["Source"]         = "Meesho"
        df["TxnType"]        = "Shipment"
        df["Units_Effective"]= df["Quantity"]
        df["OrderId"]        = df.get("sub_order_num", np.nan)
        result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
        result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Meesho: {e}"); return pd.DataFrame()

# ══════════════════════════════════════════════════════════════
# 8) INVENTORY LOADERS (unchanged from fixed version)
# ══════════════════════════════════════════════════════════════
def load_inventory_consolidated(
    oms_file, fk_file, myntra_file, amz_file,
    mapping: Dict[str, str],
    group_by_parent: bool = False
) -> pd.DataFrame:
    inv_dfs = []

    if oms_file:
        df = read_csv_safe(oms_file)
        if not df.empty and {"Item SkuCode","Inventory"}.issubset(df.columns):
            df = df.rename(columns={"Item SkuCode":"OMS_SKU","Inventory":"OMS_Inventory"})
            df["OMS_SKU"]       = df["OMS_SKU"].astype(str)
            df["OMS_Inventory"] = pd.to_numeric(df["OMS_Inventory"], errors="coerce").fillna(0)
            inv_dfs.append(df[["OMS_SKU","OMS_Inventory"]].groupby("OMS_SKU").sum().reset_index())

    if fk_file:
        df = read_csv_safe(fk_file)
        if not df.empty and {"SKU","Live on Website"}.issubset(df.columns):
            df["OMS_SKU"]      = df["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Flipkart_Live"]= pd.to_numeric(df["Live on Website"], errors="coerce").fillna(0)
            inv_dfs.append(df.groupby("OMS_SKU")["Flipkart_Live"].sum().reset_index())

    if myntra_file:
        df = read_csv_safe(myntra_file)
        if not df.empty:
            sku_col = next((c for c in df.columns if "seller sku code" in c.lower() or "sku code" in c.lower()), None)
            inv_col = next((c for c in df.columns if "sellable inventory count" in c.lower()), None)
            if sku_col and inv_col:
                df["OMS_SKU"]          = df[sku_col].apply(lambda x: map_to_oms_sku(x, mapping))
                df["Myntra_Inventory"] = pd.to_numeric(df[inv_col], errors="coerce").fillna(0)
                inv_dfs.append(df.groupby("OMS_SKU")["Myntra_Inventory"].sum().reset_index())

    if amz_file:
        df = read_csv_safe(amz_file)
        if not df.empty and {"MSKU","Ending Warehouse Balance"}.issubset(df.columns):
            if "Location" in df.columns:
                excl = len(df[df["Location"] == "ZNNE"])
                df   = df[df["Location"] != "ZNNE"]
                if excl > 0:
                    st.sidebar.info(f"ℹ️ Excluded {excl:,} ZNNE records")
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
        consolidated = (consolidated
                        .groupby("Parent_SKU")[inv_cols + ["Marketplace_Total","Total_Inventory"]]
                        .sum().reset_index()
                        .rename(columns={"Parent_SKU":"OMS_SKU"}))

    return consolidated[consolidated["Total_Inventory"] > 0]

# ══════════════════════════════════════════════════════════════
# 9) STOCK TRANSFER LOADER (unchanged)
# ══════════════════════════════════════════════════════════════
def load_stock_transfer(zip_file) -> pd.DataFrame:
    df = read_zip_csv(zip_file)
    if df.empty: return pd.DataFrame()
    required = ["Invoice Date","Ship From Fc","Ship To Fc","Quantity","Transaction Type"]
    if not all(c in df.columns for c in required): return pd.DataFrame()
    result = df[required].copy()
    result["Invoice Date"] = pd.to_datetime(result["Invoice Date"], errors="coerce")
    result["Quantity"]     = pd.to_numeric(result["Quantity"], errors="coerce").fillna(0)
    return result

# ══════════════════════════════════════════════════════════════
# 10) PANEL PIVOT BUILDER (unchanged)
# ══════════════════════════════════════════════════════════════
def build_panel_pivots(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    w = df.copy()
    w["Panel"] = w["Source"].astype(str)
    sold = (w[w["Transaction Type"]=="Shipment"]
            .groupby(["Sku","Panel"])["Quantity"].sum()
            .unstack(fill_value=0))
    sold.columns = [f"{c} | Sold" for c in sold.columns]
    ret  = (w[w["Transaction Type"]=="Refund"]
            .groupby(["Sku","Panel"])["Quantity"].sum()
            .unstack(fill_value=0))
    ret.columns  = [f"{c} | Return" for c in ret.columns]
    net  = (w.groupby(["Sku","Panel"])["Units_Effective"].sum()
             .unstack(fill_value=0))
    net.columns  = [f"{c} | Net" for c in net.columns]
    return sold, ret, net

# ══════════════════════════════════════════════════════════════
# 11) PO BASE CALCULATOR (unchanged / fixed ADS)
# ══════════════════════════════════════════════════════════════
def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7
) -> pd.DataFrame:
    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])

    max_date = df["TxnDate"].max()
    cutoff   = max_date - timedelta(days=period_days)
    recent   = df[df["TxnDate"] >= cutoff].copy()

    sold    = recent[recent["Transaction Type"]=="Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU","Sold_Units"]
    returns = recent[recent["Transaction Type"]=="Refund"].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU","Return_Units"]
    net     = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU","Net_Units"]

    summary = (sold.merge(returns, on="OMS_SKU", how="outer")
                   .merge(net,     on="OMS_SKU", how="outer")
                   .fillna(0))

    po_df = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna(
        {"Sold_Units":0,"Return_Units":0,"Net_Units":0})

    denom = max(period_days, min_denominator)
    demand_units = po_df["Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["Sold_Units"]
    po_df["ADS"] = (demand_units / denom).fillna(0)

    po_df["Stockout_Flag"] = ""
    po_df.loc[(po_df["ADS"] > 0) & (po_df["Total_Inventory"] <= 0), "Stockout_Flag"] = "⚠️ OOS"
    return po_df

# ══════════════════════════════════════════════════════════════
# 12) SIDEBAR — FILE UPLOADS
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📂 Data Upload")
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.markdown("### Yash Gallery")
st.sidebar.divider()

map_file = st.sidebar.file_uploader(
    "1️⃣ SKU Mapping (Required)", type=["xlsx"],
    help="Copy_of_All_penal_replace_sku.xlsx")

st.sidebar.markdown("### ⚙️ Amazon Settings")
st.session_state.amazon_date_basis = st.sidebar.selectbox(
    "Date Basis", ["Shipment Date","Invoice Date","Order Date"], index=0)
st.session_state.include_replacements = st.sidebar.checkbox("Include FreeReplacement", value=False)

st.sidebar.divider()

# ── MTR Reports — single main ZIP ───────────────────────────
st.sidebar.markdown("### 2️⃣ MTR Reports (Amazon Tax)")
st.sidebar.caption(
    "Upload the **one main ZIP** that contains all monthly sub-zips.\n\n"
    "Structure: `main.zip → April-2024.zip → MTR_B2B/B2C-APRIL-2024-xxx.csv`\n\n"
    "B2B / B2C auto-detected from CSV filenames."
)
mtr_main_zip = st.sidebar.file_uploader(
    "MTR — Main ZIP (all months)",
    type=["zip"], key="mtr_main_zip"
)

st.sidebar.divider()

st.sidebar.markdown("### 3️⃣ Sales Data (Units)")
f_b2c      = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=["zip"], key="b2c")
f_b2b      = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=["zip"], key="b2b")
f_transfer = st.sidebar.file_uploader("Stock Transfer (ZIP)", type=["zip"], key="transfer")
f_fk       = st.sidebar.file_uploader("Flipkart (Excel)", type=["xlsx"], key="fk")
f_meesho   = st.sidebar.file_uploader("Meesho (ZIP)", type=["zip"], key="meesho")

st.sidebar.divider()

st.sidebar.markdown("### 4️⃣ Inventory Data")
i_oms    = st.sidebar.file_uploader("OMS (CSV)",     type=["csv"],  key="oms")
i_fk     = st.sidebar.file_uploader("Flipkart (CSV)",type=["csv"],  key="fk_inv")
i_myntra = st.sidebar.file_uploader("Myntra (CSV)",  type=["csv"],  key="myntra")
i_amz    = st.sidebar.file_uploader("Amazon (CSV)",  type=["csv"],  key="amz")

st.sidebar.divider()

if st.sidebar.button("🚀 Load All Data", use_container_width=True):
    if not map_file:
        st.sidebar.error("SKU Mapping required!")
    else:
        with st.spinner("Loading data…"):
            # SKU Mapping
            st.session_state.sku_mapping = load_sku_mapping(map_file)
            if st.session_state.sku_mapping:
                st.sidebar.success(f"✅ Mapping: {len(st.session_state.sku_mapping):,} SKUs")

            config = SalesConfig(
                date_basis=st.session_state.amazon_date_basis,
                include_replacements=st.session_state.include_replacements,
            )

            # ── MTR ──────────────────────────────────────────
            if mtr_main_zip:
                result = load_mtr_from_main_zip(mtr_main_zip)
                if isinstance(result, tuple):
                    mtr_combined, csv_count = result
                else:
                    mtr_combined, csv_count = result, 0

                st.session_state.mtr_df = mtr_combined

                if not mtr_combined.empty:
                    months    = mtr_combined["Month"].nunique()
                    b2b_count = (mtr_combined["Report_Type"] == "B2B").sum()
                    b2c_count = (mtr_combined["Report_Type"] == "B2C").sum()
                    st.sidebar.success(
                        f"✅ MTR: {len(mtr_combined):,} records | "
                        f"{csv_count} CSVs | {months} months\n"
                        f"B2B: {b2b_count:,} | B2C: {b2c_count:,}"
                    )
                else:
                    st.sidebar.warning("⚠️ MTR ZIP loaded but no valid records found. "
                                       "Check that CSV filenames contain 'B2B' or 'B2C'.")

            # ── Sales ─────────────────────────────────────────
            sales_parts = []
            if f_b2c:    sales_parts.append(load_amazon_sales(f_b2c,   st.session_state.sku_mapping, "Amazon B2C", config))
            if f_b2b:    sales_parts.append(load_amazon_sales(f_b2b,   st.session_state.sku_mapping, "Amazon B2B", config))
            if f_fk:     sales_parts.append(load_flipkart_sales(f_fk,  st.session_state.sku_mapping))
            if f_meesho: sales_parts.append(load_meesho_sales(f_meesho,st.session_state.sku_mapping))
            if sales_parts:
                st.session_state.sales_df = pd.concat(
                    [d for d in sales_parts if not d.empty], ignore_index=True)
                st.sidebar.success(f"✅ Sales: {len(st.session_state.sales_df):,} records")
            else:
                st.session_state.sales_df = pd.DataFrame()

            # ── Inventory ─────────────────────────────────────
            st.session_state.inventory_df_variant = load_inventory_consolidated(
                i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=False)
            st.session_state.inventory_df_parent  = load_inventory_consolidated(
                i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=True)
            if not st.session_state.inventory_df_variant.empty:
                st.sidebar.success(f"✅ Inventory (Variant): {len(st.session_state.inventory_df_variant):,} SKUs")
            if not st.session_state.inventory_df_parent.empty:
                st.sidebar.success(f"✅ Inventory (Parent): {len(st.session_state.inventory_df_parent):,} Styles")

            # ── Transfers ─────────────────────────────────────
            if f_transfer:
                st.session_state.transfer_df = load_stock_transfer(f_transfer)
                if not st.session_state.transfer_df.empty:
                    st.sidebar.success(f"✅ Transfers: {len(st.session_state.transfer_df):,} records")

        st.rerun()

# ══════════════════════════════════════════════════════════════
# 13) GUARD
# ══════════════════════════════════════════════════════════════
if not st.session_state.sku_mapping:
    st.info("👋 **Welcome!** Upload SKU Mapping and click **Load All Data** to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# 14) MAIN TABS  (MTR tab added)
# ══════════════════════════════════════════════════════════════
tab_dash, tab_mtr, tab_inv, tab_po, tab_logistics, tab_forecast, tab_drill = st.tabs([
    "📊 Dashboard",
    "📑 MTR Analytics",
    "📦 Inventory",
    "🎯 PO Engine",
    "🚚 Logistics",
    "📈 AI Forecast",
    "🔍 Deep Dive",
])

# ──────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ──────────────────────────────────────────────────────────────
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
                ["Last 7 Days","Last 30 Days","Last 60 Days","Last 90 Days","All Time"],
                index=1, key="dash_period")
        with col_grace:
            grace_days = st.number_input("Grace Period (Days)", 0, 14, 7,
                help="Extra days to capture late transactions")

        df = df.copy()
        df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
        max_date = df["TxnDate"].max()

        if period_option == "All Time":
            filtered_df = df
            date_range_text = f"All Time: {df['TxnDate'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            base_days  = 7 if "7" in period_option else 30 if "30" in period_option else 60 if "60" in period_option else 90
            total_days = base_days + grace_days
            filtered_df  = df[df["TxnDate"] >= (max_date - timedelta(days=total_days))]
            date_range_text = (f"Period: {filtered_df['TxnDate'].min().strftime('%Y-%m-%d')} "
                               f"to {max_date.strftime('%Y-%m-%d')} ({base_days}+{grace_days})")

        st.info(f"📅 **{date_range_text}** | Transactions: {len(filtered_df):,}")

        filtered_df = filtered_df.copy()
        filtered_df["Quantity"]       = pd.to_numeric(filtered_df["Quantity"], errors="coerce").fillna(0)
        filtered_df["Units_Effective"]= pd.to_numeric(filtered_df["Units_Effective"], errors="coerce").fillna(0)

        sold_pcs   = filtered_df[filtered_df["Transaction Type"]=="Shipment"]["Quantity"].sum()
        ret_pcs    = filtered_df[filtered_df["Transaction Type"]=="Refund"]["Quantity"].sum()
        net_units  = filtered_df["Units_Effective"].sum()
        orders     = (filtered_df[filtered_df["Transaction Type"]=="Shipment"]["OrderId"].nunique()
                      if "OrderId" in filtered_df.columns
                      else len(filtered_df[filtered_df["Transaction Type"]=="Shipment"]))
        return_rate= (ret_pcs / sold_pcs * 100) if sold_pcs > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛒 Orders",       f"{orders:,}")
        c2.metric("✅ Sold Pieces",  f"{int(sold_pcs):,}")
        c3.metric("↩️ Returns",      f"{int(ret_pcs):,}")
        c4.metric("📊 Return Rate",  f"{return_rate:.1f}%")
        c5.metric("📦 Net Units",    f"{int(net_units):,}")

        st.info(f"**Settings:** Amazon Date: {st.session_state.amazon_date_basis} | "
                f"Include Replacements: {st.session_state.include_replacements}")
        st.divider()

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### 🏆 Top 20 Selling SKUs")
            top = (filtered_df[filtered_df["Transaction Type"]=="Shipment"]
                   .groupby("Sku")["Quantity"].sum()
                   .sort_values(ascending=False).head(20).reset_index())
            fig = px.bar(top, x="Sku", y="Quantity", title="Top Sellers (Pieces)")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📊 Marketplace Split")
            source_summary = filtered_df.groupby("Source")["Quantity"].sum().reset_index()
            fig = px.pie(source_summary, values="Quantity", names="Source", hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2: MTR ANALYTICS  ✅ NEW
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# TAB 2: MTR ANALYTICS
# ──────────────────────────────────────────────────────────────
with tab_mtr:
    st.subheader("📑 MTR Analytics — Amazon Tax Report")
    mtr = st.session_state.mtr_df

    if mtr.empty:
        st.info(
            "📂 **No MTR data loaded yet.**\n\n"
            "Upload your main ZIP in the sidebar under **2️⃣ MTR Reports** "
            "and click **Load All Data**.\n\n"
            "Structure: `main.zip → April-2024.zip → MTR_B2B/B2C-APRIL-2024.csv`"
        )
    else:
        try:
            # ── Filters ──────────────────────────────────────────
            with st.expander("🔧 Filters", expanded=True):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    all_months = sorted(mtr["Month"].dropna().unique())
                    sel_months = st.multiselect("Months", all_months, default=all_months, key="mtr_months")
                with fc2:
                    sel_rtype = st.multiselect("Report Type", ["B2B", "B2C"],
                                               default=["B2B", "B2C"], key="mtr_rtype")
                with fc3:
                    all_txn = sorted(mtr["Transaction_Type"].dropna().unique())
                    default_txn = [t for t in ["Shipment", "Refund"] if t in all_txn]
                    sel_txn = st.multiselect("Transaction Types", all_txn,
                                             default=default_txn, key="mtr_txn")

            mf = mtr[
                mtr["Month"].isin(sel_months) &
                mtr["Report_Type"].isin(sel_rtype) &
                mtr["Transaction_Type"].isin(sel_txn)
            ].copy()

            if mf.empty:
                st.warning("No data for selected filters.")
            else:
                # ── masks ─────────────────────────────────────────
                shipped  = mf["Transaction_Type"] == "Shipment"
                refunded = mf["Transaction_Type"] == "Refund"

                gross_rev  = mf.loc[shipped,  "Invoice_Amount"].sum()
                refund_amt = mf.loc[refunded, "Invoice_Amount"].abs().sum()
                net_rev    = gross_rev - refund_amt
                total_tax  = mf.loc[shipped,  "Total_Tax"].sum()
                units_sold = mf.loc[shipped,  "Quantity"].sum()
                units_ret  = mf.loc[refunded, "Quantity"].abs().sum()
                order_cnt  = mf.loc[shipped,  "Order_Id"].nunique()
                aov        = gross_rev / order_cnt if order_cnt else 0

                # ── KPIs ──────────────────────────────────────────
                st.markdown("### 💰 Revenue KPIs")
                k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
                k1.metric("💵 Gross Revenue",   fmt_inr(gross_rev))
                k2.metric("↩️ Refunds",         fmt_inr(refund_amt))
                k3.metric("✅ Net Revenue",     fmt_inr(net_rev))
                k4.metric("🏛️ Tax Collected",   fmt_inr(total_tax))
                k5.metric("📦 Units Sold",      f"{int(units_sold):,}")
                k6.metric("🛒 Orders",          f"{order_cnt:,}")
                k7.metric("💳 AOV",             fmt_inr(aov))

                st.divider()

                # ── B2B vs B2C comparison ─────────────────────────
                st.markdown("### 🔀 B2B vs B2C Comparison")
                comp_rows = []
                for rt in ["B2B", "B2C"]:
                    sub  = mf[mf["Report_Type"] == rt]
                    sh   = sub["Transaction_Type"] == "Shipment"
                    rf   = sub["Transaction_Type"] == "Refund"
                    gr   = sub.loc[sh, "Invoice_Amount"].sum()
                    ref  = sub.loc[rf, "Invoice_Amount"].abs().sum()
                    ord_ = sub.loc[sh, "Order_Id"].nunique()
                    us   = sub.loc[sh, "Quantity"].sum()
                    comp_rows.append({
                        "Type":          rt,
                        "Gross Revenue": fmt_inr(gr),
                        "Refunds":       fmt_inr(ref),
                        "Net Revenue":   fmt_inr(gr - ref),
                        "Tax":           fmt_inr(sub.loc[sh, "Total_Tax"].sum()),
                        "Orders":        f"{ord_:,}",
                        "Units Sold":    f"{int(us):,}",
                        "AOV":           fmt_inr(gr / ord_) if ord_ else "₹0",
                    })
                st.dataframe(pd.DataFrame(comp_rows).set_index("Type"), use_container_width=True)

                st.divider()

                # ── Monthly revenue trend ─────────────────────────
                st.markdown("### 📈 Monthly Revenue Trend — B2B vs B2C")
                monthly = (mf[shipped]
                           .groupby(["Month", "Report_Type"])["Invoice_Amount"]
                           .sum().reset_index()
                           .sort_values("Month"))
                monthly.columns = ["Month", "Report_Type", "Gross_Revenue"]
                fig = px.line(monthly, x="Month", y="Gross_Revenue", color="Report_Type",
                              markers=True,
                              color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                              title="Monthly Gross Revenue",
                              labels={"Gross_Revenue": "Revenue (₹)", "Month": "Month"})
                fig.update_layout(hovermode="x unified", height=400)
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)

                # refund % bar
                monthly_ref = (mf[refunded]
                               .groupby(["Month", "Report_Type"])["Invoice_Amount"]
                               .sum().abs().reset_index())
                monthly_ref.columns = ["Month", "Report_Type", "Refund_Amt"]
                monthly_comb = monthly.merge(monthly_ref, on=["Month", "Report_Type"], how="left").fillna(0)
                monthly_comb["Refund_%"] = (
                    monthly_comb["Refund_Amt"] / monthly_comb["Gross_Revenue"].replace(0, np.nan) * 100
                ).fillna(0).round(2)
                fig2 = px.bar(monthly_comb, x="Month", y="Refund_%", color="Report_Type",
                              barmode="group",
                              color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                              title="Monthly Refund %",
                              labels={"Refund_%": "Refund Rate (%)"})
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

                st.divider()

                # ── State-wise revenue ────────────────────────────
                st.markdown("### 🗺️ State-wise Revenue")
                sc1, sc2 = st.columns([1, 3])
                with sc1:
                    state_rt = st.radio("Report Type", ["Both", "B2B", "B2C"],
                                        horizontal=False, key="mtr_state_rt")
                    top_n    = st.slider("Top N States", 5, 25, 15, key="mtr_topn")
                state_src = mf[shipped].copy()
                if state_rt != "Both":
                    state_src = state_src[state_src["Report_Type"] == state_rt]
                state_rev = (state_src.groupby("Ship_To_State")["Invoice_Amount"]
                             .sum().sort_values(ascending=False).head(top_n).reset_index())
                state_rev.columns = ["State", "Revenue"]
                with sc2:
                    fig3 = px.bar(state_rev, x="Revenue", y="State", orientation="h",
                                  color="Revenue", color_continuous_scale="Blues",
                                  title=f"Top {top_n} States by Revenue")
                    fig3.update_layout(height=max(300, top_n * 28), yaxis=dict(autorange="reversed"))
                    fig3.update_xaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig3, use_container_width=True)

                st.divider()

                # ── State heatmap ─────────────────────────────────
                st.markdown("### 🔥 State Revenue Heatmap (Top 12 × Month)")
                top12 = (mf[shipped].groupby("Ship_To_State")["Invoice_Amount"]
                         .sum().nlargest(12).index.tolist())
                heat_src = (mf[shipped & mf["Ship_To_State"].isin(top12)]
                            .groupby(["Ship_To_State", "Month"])["Invoice_Amount"]
                            .sum().reset_index()
                            .pivot(index="Ship_To_State", columns="Month", values="Invoice_Amount")
                            .fillna(0))
                if not heat_src.empty:
                    fig4 = px.imshow(heat_src / 1000, color_continuous_scale="YlOrRd",
                                     labels=dict(color="Revenue (₹K)"),
                                     title="Revenue Heatmap (₹ Thousands)", aspect="auto")
                    fig4.update_layout(height=420)
                    st.plotly_chart(fig4, use_container_width=True)

                st.divider()

                # ── Payment methods ───────────────────────────────
                st.markdown("### 💳 Payment Method Distribution")
                pm1, pm2 = st.columns(2)
                with pm1:
                    pm_df = (mf[shipped].groupby(["Payment_Method", "Report_Type"])["Invoice_Amount"]
                             .sum().reset_index())
                    fig5 = px.bar(pm_df, x="Payment_Method", y="Invoice_Amount",
                                  color="Report_Type", barmode="group",
                                  color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                                  title="Payment Methods by Revenue")
                    fig5.update_xaxes(tickangle=-30)
                    fig5.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig5, use_container_width=True)
                with pm2:
                    pm_units = (mf[shipped].groupby("Payment_Method")["Quantity"]
                                .sum().sort_values(ascending=False).head(10).reset_index())
                    pm_units.columns = ["Method", "Units"]
                    fig6 = px.pie(pm_units, values="Units", names="Method",
                                  title="Payment Split (Units)", hole=0.4)
                    fig6.update_layout(height=340)
                    st.plotly_chart(fig6, use_container_width=True)

                st.divider()

                # ── Transaction type breakdown ────────────────────
                st.markdown("### 📋 Transaction Type Breakdown")
                txn_rev = (mf.groupby(["Transaction_Type", "Report_Type"])["Invoice_Amount"]
                           .sum().reset_index())
                fig7 = px.bar(txn_rev, x="Transaction_Type", y="Invoice_Amount",
                              color="Report_Type", barmode="group",
                              color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                              title="Revenue by Transaction Type")
                fig7.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig7, use_container_width=True)

                st.divider()

                # ── Top SKUs ──────────────────────────────────────
                st.markdown("### 🏆 Top 20 SKUs by Gross Revenue")
                sku_rev = (mf[shipped].groupby(["SKU", "Report_Type"])["Invoice_Amount"]
                           .sum().reset_index()
                           .sort_values("Invoice_Amount", ascending=False).head(20))
                fig8 = px.bar(sku_rev, x="SKU", y="Invoice_Amount", color="Report_Type",
                              color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                              title="Top 20 SKUs by Revenue")
                fig8.update_xaxes(tickangle=-45)
                fig8.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig8, use_container_width=True)

                st.divider()

                # ── Warehouse ─────────────────────────────────────
                st.markdown("### 🏭 Warehouse / FC Revenue")
                wh_df = (mf[shipped].groupby(["Warehouse_Id", "Report_Type"])["Invoice_Amount"]
                         .sum().reset_index().sort_values("Invoice_Amount", ascending=False))
                fig9 = px.bar(wh_df, x="Warehouse_Id", y="Invoice_Amount", color="Report_Type",
                              barmode="group",
                              color_discrete_map={"B2B": "#002B5B", "B2C": "#E63946"},
                              title="Revenue by Warehouse / FC")
                fig9.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig9, use_container_width=True)

                st.divider()

                # ── Raw data viewer ───────────────────────────────
                st.markdown("### 🔍 Raw MTR Data Viewer")
                search_sku = st.text_input("Search by SKU / ASIN / Buyer Name", key="mtr_search")
                view_df = mf.copy()
                if search_sku:
                    mask = (
                        view_df["SKU"].str.contains(search_sku, case=False, na=False) |
                        view_df["ASIN"].str.contains(search_sku, case=False, na=False) |
                        view_df["Buyer_Name"].str.contains(search_sku, case=False, na=False)
                    )
                    view_df = view_df[mask]
                    st.caption(f"Showing {len(view_df):,} matches")

                show_cols = [c for c in [
                    "Date", "Report_Type", "Transaction_Type", "SKU", "Description",
                    "Quantity", "Invoice_Amount", "Total_Tax", "CGST", "SGST", "IGST",
                    "Ship_To_State", "Payment_Method", "Warehouse_Id",
                    "Order_Id", "Invoice_Number", "Buyer_Name", "IRN_Status", "Month"
                ] if c in view_df.columns]

                st.dataframe(
                    view_df[show_cols].sort_values("Date", ascending=False).head(500),
                    use_container_width=True, height=400
                )
                st.caption(f"Showing up to 500 of {len(view_df):,} records")

                st.divider()

                # ── Downloads ─────────────────────────────────────
                st.markdown("### 📥 Download MTR Summary")
                dl1, dl2, dl3 = st.columns(3)

                monthly_summary = (
                    mf[shipped]
                    .groupby(["Month", "Report_Type"])
                    .agg(Gross_Revenue=("Invoice_Amount", "sum"),
                         Total_Tax=("Total_Tax", "sum"),
                         Units_Sold=("Quantity", "sum"),
                         Orders=("Order_Id", "nunique"))
                    .reset_index()
                    .merge(
                        mf[refunded].groupby(["Month", "Report_Type"])
                        .agg(Refunds=("Invoice_Amount", lambda x: x.abs().sum()))
                        .reset_index(),
                        on=["Month", "Report_Type"], how="left"
                    )
                    .fillna(0)
                )
                monthly_summary["Net_Revenue"] = (
                    monthly_summary["Gross_Revenue"] - monthly_summary["Refunds"])
                monthly_summary["AOV"] = (
                    monthly_summary["Gross_Revenue"] / monthly_summary["Orders"].replace(0, np.nan)
                ).fillna(0).round(2)

                with dl1:
                    st.download_button(
                        "📥 Full MTR Data (CSV)",
                        mf[show_cols].to_csv(index=False).encode("utf-8"),
                        f"mtr_full_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True)
                with dl2:
                    st.download_button(
                        "📥 Monthly Summary (CSV)",
                        monthly_summary.to_csv(index=False).encode("utf-8"),
                        f"mtr_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True)
                with dl3:
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        mf[show_cols].to_excel(writer, sheet_name="MTR_Full", index=False)
                        monthly_summary.to_excel(writer, sheet_name="Monthly_Summary", index=False)
                        mf[mf["Report_Type"] == "B2B"][show_cols].to_excel(writer, sheet_name="B2B", index=False)
                        mf[mf["Report_Type"] == "B2C"][show_cols].to_excel(writer, sheet_name="B2C", index=False)
                        state_sum = (mf[shipped]
                                     .groupby(["Ship_To_State", "Report_Type"])
                                     .agg(Revenue=("Invoice_Amount", "sum"),
                                          Units=("Quantity", "sum"))
                                     .reset_index()
                                     .sort_values("Revenue", ascending=False))
                        state_sum.to_excel(writer, sheet_name="State_Summary", index=False)
                    st.download_button(
                        "📥 Full Excel Report",
                        excel_buf.getvalue(),
                        f"mtr_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)

        except Exception as e:
            st.error(f"MTR Analytics error: {e}")
            import traceback
            st.code(traceback.format_exc())


with tab_inv:
    st.subheader("📦 Consolidated Inventory")
    mode = st.radio("Inventory View", ["Variant (Size/Color)","Parent (Style Only)"], horizontal=True)
    inv  = (st.session_state.inventory_df_variant if "Variant" in mode
            else st.session_state.inventory_df_parent)

    if inv.empty:
        st.warning("⚠️ No inventory data loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{len(inv):,}")
        c2.metric("Total Units",   f"{inv['Total_Inventory'].sum():,.0f}")
        if "OMS_Inventory" in inv.columns:
            c3.metric("OMS Warehouse", f"{inv['OMS_Inventory'].sum():,.0f}")
        c4.metric("Marketplaces",  f"{inv['Marketplace_Total'].sum():,.0f}" if "Marketplace_Total" in inv.columns else "0")
        st.divider()
        search = st.text_input("🔍 Search SKU", placeholder="Type to filter…")
        display = inv[inv["OMS_SKU"].str.contains(search, case=False, na=False)] if search else inv
        st.dataframe(display, use_container_width=True, height=500)

# ──────────────────────────────────────────────────────────────
# TAB 4: PO ENGINE
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# TAB 4: PO ENGINE
# ──────────────────────────────────────────────────────────────
with tab_po:
    st.subheader("🎯 Purchase Order Recommendations")

    if st.session_state.sales_df.empty or (
        st.session_state.inventory_df_variant.empty and
        st.session_state.inventory_df_parent.empty
    ):
        st.warning("⚠️ Please load Sales data and Inventory data first, then click Load All Data.")
    else:
        try:
            col_view, col_info = st.columns([1, 3])
            with col_view:
                view_mode = st.radio("Group By",
                    ["By Variant (Size/Color)", "By Parent SKU (Style Only)"], key="po_view_mode")
            with col_info:
                st.info("✅ Variant mode = size/color level PO\n\n"
                        "✅ Parent mode = style level PO (all sizes combined)")

            st.divider()
            st.markdown("### ⚙️ PO Parameters")
            c1, c2, c3, c4, c5 = st.columns(5)
            velocity    = c1.selectbox("Velocity Period",
                            ["Last 7 Days", "Last 30 Days", "Last 60 Days", "Last 90 Days"],
                            key="po_velocity")
            base_days   = 7 if "7" in velocity else 30 if "30" in velocity else 60 if "60" in velocity else 90
            grace_days  = c2.number_input("Grace Days", 0, 14, 7)
            lead_time   = c3.number_input("Lead Time (Days)", 1, 180, 15)
            target_days = c4.number_input("Target Stock (Days)", 0, 180, 60)
            safety_pct  = c5.slider("Safety Stock %", 0, 100, 20)

            st.divider()
            demand_basis = st.selectbox("Demand Basis", ["Sold", "Net"], index=0,
                help="Sold = shipments only (recommended). Net = shipments − returns.")
            min_den = st.number_input("Min ADS Denominator", 1, 60, 7)
            total_period = int(base_days + grace_days)

            if "Parent" in view_mode:
                inv_for_po   = st.session_state.inventory_df_parent.copy()
                sales_for_po = st.session_state.sales_df.copy()
                sales_for_po["Sku"] = sales_for_po["Sku"].apply(get_parent_sku)
            else:
                inv_for_po   = st.session_state.inventory_df_variant.copy()
                sales_for_po = st.session_state.sales_df.copy()

            po_df = calculate_po_base(sales_for_po, inv_for_po, total_period,
                                      demand_basis, int(min_den))

            if po_df.empty:
                st.warning("No PO calculations available. Check that sales and inventory data overlap.")
            else:
                po_df["Days_Left"]        = np.where(po_df["ADS"] > 0,
                                                      po_df["Total_Inventory"] / po_df["ADS"], 999)
                po_df["Lead_Time_Demand"] = po_df["ADS"] * lead_time
                po_df["Target_Stock"]     = po_df["ADS"] * target_days
                po_df["Base_Requirement"] = po_df["Lead_Time_Demand"] + po_df["Target_Stock"]
                po_df["Safety_Stock"]     = po_df["Base_Requirement"] * (safety_pct / 100)
                po_df["Total_Required"]   = po_df["Base_Requirement"] + po_df["Safety_Stock"]
                po_df["PO_Recommended"]   = (
                    np.ceil((po_df["Total_Required"] - po_df["Total_Inventory"])
                            .clip(lower=0) / 5) * 5
                ).astype(int)

                def get_priority(row):
                    if row["Days_Left"] < lead_time         and row["PO_Recommended"] > 0: return "🔴 URGENT"
                    if row["Days_Left"] < lead_time + 7     and row["PO_Recommended"] > 0: return "🟡 HIGH"
                    if row["PO_Recommended"] > 0:                                           return "🟢 MEDIUM"
                    return "⚪ OK"

                po_df["Priority"] = po_df.apply(get_priority, axis=1)
                po_needed = po_df[po_df["PO_Recommended"] > 0].sort_values(["Priority", "Days_Left"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔴 Urgent", len(po_needed[po_needed["Priority"] == "🔴 URGENT"]))
                m2.metric("🟡 High",   len(po_needed[po_needed["Priority"] == "🟡 HIGH"]))
                m3.metric("🟢 Medium", len(po_needed[po_needed["Priority"] == "🟢 MEDIUM"]))
                m4.metric("📦 Total Units", f"{po_needed['PO_Recommended'].sum():,}")

                st.divider()
                search = st.text_input("🔍 Search SKU", key="po_search")
                if search:
                    po_needed = po_needed[
                        po_needed["OMS_SKU"].astype(str).str.contains(search, case=False, na=False)]

                display_cols = [c for c in [
                    "Priority", "OMS_SKU", "Total_Inventory", "Sold_Units",
                    "Return_Units", "Net_Units", "ADS", "Days_Left",
                    "Lead_Time_Demand", "Target_Stock", "Safety_Stock",
                    "Total_Required", "PO_Recommended", "Stockout_Flag"
                ] if c in po_needed.columns]

                def highlight_priority(row):
                    result = []
                    for col in row.index:
                        if col == "Priority":
                            if "🔴" in str(row[col]):   result.append("background-color:#fee2e2;font-weight:bold")
                            elif "🟡" in str(row[col]): result.append("background-color:#fef3c7")
                            else:                        result.append("background-color:#d1fae5")
                        elif col == "PO_Recommended":
                            result.append("background-color:#dbeafe;font-weight:bold")
                        elif col == "Days_Left" and float(row[col]) < float(lead_time):
                            result.append("background-color:#fee2e2;font-weight:bold")
                        else:
                            result.append("")
                    return result

                fmt_dict = {c: "{:.3f}" if c == "ADS" else "{:.1f}" if c == "Days_Left" else "{:.0f}"
                            for c in display_cols if c not in ["Priority", "OMS_SKU", "Stockout_Flag"]}

                st.dataframe(
                    po_needed[display_cols].head(200).style
                    .apply(highlight_priority, axis=1).format(fmt_dict),
                    use_container_width=True, height=520)
                st.caption(f"Showing top 200 of {len(po_needed):,} SKUs needing orders")

                st.divider()
                suffix = "parent" if "Parent" in view_mode else "variant"
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button("📥 Download PO (CSV)",
                        po_needed[display_cols].to_csv(index=False).encode("utf-8"),
                        f"po_{suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True)
                with c_dl2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w:
                        po_needed[display_cols].to_excel(w, sheet_name="PO_Recommendations", index=False)
                    st.download_button("📥 Download PO (Excel)", buf.getvalue(),
                        f"po_{suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
        except Exception as e:
            st.error(f"PO Engine error: {e}")
            import traceback
            st.code(traceback.format_exc())


with tab_logistics:
    st.subheader("🚚 Stock Transfers & FC Movements")
    transfer_df = st.session_state.transfer_df
    if transfer_df.empty:
        st.info("📦 Upload Amazon Stock Transfer file to view logistics data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Movements",  f"{len(transfer_df):,}")
        c2.metric("Units Transferred",f"{transfer_df['Quantity'].sum():,.0f}")
        c3.metric("FC Transfers",     f"{len(transfer_df[transfer_df['Transaction Type']=='FC_TRANSFER']):,}")
        c4.metric("Unique Routes",    f"{len(transfer_df.groupby(['Ship From Fc','Ship To Fc'])):,}")
        st.divider()
        st.markdown("### 🔝 Top Transfer Routes")
        routes = transfer_df.groupby(["Ship From Fc","Ship To Fc"]).agg(
            {"Quantity":["sum","count"]}).reset_index()
        routes.columns = ["From FC","To FC","Units","Transfers"]
        st.dataframe(routes.sort_values("Units", ascending=False).head(20), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 6: AI FORECAST
# ──────────────────────────────────────────────────────────────
with tab_forecast:
    st.subheader("📈 AI Demand Forecasting")
    sales = st.session_state.sales_df
    if not _PROPHET_AVAILABLE:
        st.warning("⚠️ Prophet not installed. Add `prophet` to your `requirements.txt` to enable forecasting.")
    elif sales.empty:
        st.warning("⚠️ Upload sales data for forecasting.")
    else:
        sku  = st.selectbox("Select SKU", [""]+sorted(sales["Sku"].dropna().unique().tolist()))
        days = st.slider("Forecast Days", 7, 90, 30)
        if sku:
            subset = sales[sales["Sku"] == sku].copy()
            subset["ds"] = pd.to_datetime(subset["TxnDate"]).dt.date
            daily  = subset.groupby("ds")["Units_Effective"].sum().reset_index()
            daily.columns = ["ds","y"]
            daily["ds"] = pd.to_datetime(daily["ds"])
            if len(daily) < 14:
                st.warning("Need at least 14 days of data.")
            else:
                try:
                    with st.spinner("Forecasting…"):
                        m       = Prophet(daily_seasonality=False, weekly_seasonality=True)
                        m.fit(daily)
                        future  = m.make_future_dataframe(periods=days)
                        forecast= m.predict(future)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual"))
                    fut = forecast[forecast["ds"] > daily["ds"].max()]
                    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast",
                                             line=dict(dash="dash")))
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"🤖 Predicted demand (next {days} days): **{int(fut['yhat'].sum())} units**")
                except Exception as e:
                    st.error(f"Forecast error: {e}")

# ──────────────────────────────────────────────────────────────
# TAB 7: DEEP DIVE
# ──────────────────────────────────────────────────────────────
with tab_drill:
    st.subheader("🔍 Deep Dive & Panel Analysis")
    df = st.session_state.sales_df
    if df.empty:
        st.warning("⚠️ Upload sales data for deep dive.")
    else:
        colA, colB, colC = st.columns([3, 2, 1])
        with colA: search = st.text_input("Enter SKU", placeholder="e.g., 1065YK", key="drill_search")
        with colB: period = st.selectbox("Period", ["Last 7 Days","Last 30 Days","Last 90 Days","All Time"], index=1, key="drill_period")
        with colC: grace  = st.number_input("Grace Days", 0, 14, 7, key="drill_grace")

        fdf = df.copy()
        fdf["TxnDate"] = pd.to_datetime(fdf["TxnDate"], errors="coerce")
        max_d = fdf["TxnDate"].max()

        if period != "All Time" and not pd.isna(max_d):
            base   = 7 if "7" in period else 30 if "30" in period else 90
            cutoff = max_d - timedelta(days=base + grace)
            fdf    = fdf[fdf["TxnDate"] >= cutoff]
            date_range_text = (f"{fdf['TxnDate'].min().strftime('%Y-%m-%d')} "
                               f"to {max_d.strftime('%Y-%m-%d')} ({base}+{grace})")
        else:
            date_range_text = f"All Time: {fdf['TxnDate'].min().strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}"

        st.info(f"📅 **Period:** {date_range_text} | **Transactions:** {len(fdf):,}")

        if search:
            matches = fdf[fdf["Sku"].astype(str).str.contains(search, case=False, na=False)].copy()
            if matches.empty:
                st.warning("No matching SKUs found.")
            else:
                st.success(f"✅ Found **{matches['Sku'].nunique()}** SKU variant(s) matching '{search}'")
                st.divider()
                sold  = matches[matches["Transaction Type"]=="Shipment"]["Quantity"].sum()
                ret   = matches[matches["Transaction Type"]=="Refund"]["Quantity"].sum()
                net   = matches["Units_Effective"].sum()
                rate  = (ret / sold * 100) if sold > 0 else 0
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sold Pieces", f"{int(sold):,}")
                m2.metric("Returns",     f"{int(ret):,}")
                m3.metric("Net Units",   f"{int(net):,}")
                m4.metric("Return %",    f"{rate:.1f}%")
                st.divider()
                st.markdown("### 🏪 Marketplace Breakdown (Panel-wise)")
                sold_p, ret_p, net_p = build_panel_pivots(matches)
                combined = pd.concat([sold_p, ret_p, net_p], axis=1).fillna(0).reset_index()
                if not combined.empty:
                    st.dataframe(combined, use_container_width=True, height=400)
                st.markdown("### 📜 Recent Transactions (Latest 100)")
                cols = ["Sku","TxnDate","Transaction Type","Quantity","Source"]
                if "OrderId"        in matches.columns: cols.append("OrderId")
                if "Units_Effective" in matches.columns: cols.append("Units_Effective")
                display_txns = matches.sort_values("TxnDate", ascending=False).head(100)[cols].copy()
                display_txns["TxnDate"] = pd.to_datetime(display_txns["TxnDate"]).dt.strftime("%Y-%m-%d")
                st.dataframe(display_txns, use_container_width=True, height=400)
                st.download_button(
                    "📥 Download Full Transaction History",
                    matches[cols].to_csv(index=False).encode("utf-8"),
                    f"sku_drilldown_{search}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.info("💡 Type a SKU (or partial) in the search box above.")
            st.markdown("### 📦 Sample SKUs Available")
            sample = fdf["Sku"].value_counts().head(20).reset_index()
            sample.columns = ["SKU","Transaction Count"]
            st.dataframe(sample, use_container_width=True, height=300)

st.divider()
st.caption("💡 Yash Gallery ERP | MTR-Integrated Version | "
           "PO engine fixed (inventory alignment + ADS logic) | "
           "MTR Analytics: Revenue • Tax • State • Payment • Trends")

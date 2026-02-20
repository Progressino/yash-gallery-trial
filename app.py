#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System — app.py
(Bulletproof MTR Loader + Seasonal PO + Prophet Debug)
Fixed: MTR date parsing, 2023 ghost data, deduplication on Invoice_Number
Fixed: Meesho grace period, return date parsing for accurate Dec 2025 counts
"""

import gc
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

# Prophet is optional - Catch explicit errors if it fails to load
try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
    _PROPHET_ERR = ""
except Exception as e:
    _PROPHET_AVAILABLE = False
    _PROPHET_ERR = str(e)

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
        "mtr_df":                pd.DataFrame(),
        "myntra_df":             pd.DataFrame(),
        "meesho_df":             pd.DataFrame(),
        "flipkart_df":           pd.DataFrame(),
        "daily_orders_df":       pd.DataFrame(),   # combined daily orders from all platforms
        "daily_detect_log":      [],               # list of "filename → platform" strings
        "amazon_date_basis":     "Shipment Date",
        "include_replacements":  False,
        "daily_sales_sources":   [],
        "daily_sales_rows":      0,
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
    if abs(val) >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.2f} Cr"
    elif abs(val) >= 1_00_000:
        return f"₹{val/1_00_000:.2f} L"
    return f"₹{val:,.0f}"

def fillna_numeric(df, value=0):
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(value)
    return df

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
# 6) BULLETPROOF MTR LOADER — Memory-Efficient Streaming
# ══════════════════════════════════════════════════════════════
_CAT_COLS = {"Ship_To_State", "Warehouse_Id", "Fulfillment", "Payment_Method"}

# Minimal columns to keep — drop Description, ASIN, Credit_Note_No to save RAM
_MTR_KEEP_COLS = [
    "Date", "Report_Type", "Transaction_Type", "SKU", "Quantity",
    "Invoice_Amount", "Total_Tax", "CGST", "SGST", "IGST",
    "Ship_To_State", "Warehouse_Id", "Fulfillment", "Payment_Method",
    "Order_Id", "Invoice_Number", "Buyer_Name", "IRN_Status",
    "Month", "Month_Label",
]

def _parse_date_flexible(series: pd.Series) -> pd.Series:
    """Try explicit date formats in priority order to prevent ghost-year misparses."""
    priority_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
        "%d-%b-%Y", "%d/%b/%Y", "%m/%d/%Y",
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
    ]
    non_null  = series.dropna()
    threshold = max(int(len(non_null) * 0.70), 1) if len(non_null) > 0 else 1
    for fmt in priority_formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() >= threshold:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _downcast_mtr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressively shrink a parsed MTR DataFrame in-place.
    Converts string columns → category, floats → float32, date → date32.
    Typical saving: 60-75% vs default pandas dtypes.
    """
    cat_cols = ["Report_Type", "Transaction_Type", "Ship_To_State",
                "Warehouse_Id", "Fulfillment", "Payment_Method",
                "IRN_Status", "Month", "Month_Label"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    float_cols = ["Quantity", "Invoice_Amount", "Total_Tax", "CGST", "SGST", "IGST"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    return df


def _parse_mtr_csv(csv_bytes: bytes, source_file: str):
    try:
        raw = pd.read_csv(
            io.BytesIO(csv_bytes), dtype=str, low_memory=False,
            encoding="utf-8", on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        try:
            raw = pd.read_csv(
                io.BytesIO(csv_bytes), dtype=str, low_memory=False,
                encoding="ISO-8859-1", on_bad_lines="skip"
            )
        except Exception:
            return pd.DataFrame(), "Encoding Error"
    except Exception as e:
        return pd.DataFrame(), f"Parse Error: {e}"

    if raw.empty:
        return pd.DataFrame(), "Empty file"

    raw.columns = raw.columns.astype(str).str.strip().str.lower()

    is_b2b = "buyer name" in raw.columns or "customer bill to gstid" in raw.columns
    report_type = "B2B" if is_b2b else "B2C"

    want_b2c = {
        "shipment date", "invoice date", "transaction type", "sku",
        "quantity", "invoice amount", "total tax amount",
        "cgst tax", "sgst tax", "igst tax", "ship to state", "warehouse id",
        "fulfillment channel", "payment method code", "order id",
        "invoice number",
    }
    want_b2b = want_b2c | {"buyer name", "irn filing status"}
    want = want_b2b if is_b2b else want_b2c

    raw = raw[[c for c in raw.columns if c in want]]

    date_col = next(
        (d for d in ["shipment date", "invoice date", "transaction date", "order date"]
         if d in raw.columns), None
    )
    raw["_Date"] = _parse_date_flexible(raw[date_col]) if date_col else pd.NaT

    initial_len = len(raw)
    raw = raw.dropna(subset=["_Date"])
    dropped_dates = initial_len - len(raw)
    if raw.empty:
        return pd.DataFrame(), f"All {initial_len} rows had invalid/missing dates."

    current_year = datetime.now().year
    valid_mask = raw["_Date"].dt.year.between(2018, current_year + 1)
    ghost_rows = (~valid_mask).sum()
    raw = raw[valid_mask]
    if raw.empty:
        return pd.DataFrame(), f"All rows had out-of-range years."

    def g(name):
        return raw[name].fillna("").astype(str).str.strip() if name in raw.columns \
               else pd.Series("", index=raw.index, dtype=str)

    def gn(name):
        return pd.to_numeric(raw[name], errors="coerce").fillna(0).astype("float32") \
               if name in raw.columns \
               else pd.Series(0.0, index=raw.index, dtype="float32")

    txn_raw = g("transaction type")
    txn = txn_raw.str.lower()
    txn_std = pd.Series("Shipment", index=raw.index, dtype=str)
    txn_std[txn.str.contains("return|refund", na=False)] = "Refund"
    txn_std[txn.str.contains("cancel", na=False)]        = "Cancel"

    out = pd.DataFrame({
        "Date":             raw["_Date"],
        "Report_Type":      report_type,
        "Transaction_Type": txn_std,
        "SKU":              g("sku"),
        "Quantity":         gn("quantity"),
        "Invoice_Amount":   gn("invoice amount"),
        "Total_Tax":        gn("total tax amount"),
        "CGST":             gn("cgst tax"),
        "SGST":             gn("sgst tax"),
        "IGST":             gn("igst tax"),
        "Ship_To_State":    g("ship to state").str.upper(),
        "Warehouse_Id":     g("warehouse id"),
        "Fulfillment":      g("fulfillment channel"),
        "Payment_Method":   g("payment method code"),
        "Order_Id":         g("order id"),
        "Invoice_Number":   g("invoice number"),
        "Buyer_Name":       g("buyer name"),
        "IRN_Status":       g("irn filing status"),
    })

    del raw

    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")

    out = _downcast_mtr(out)

    msgs = []
    if dropped_dates: msgs.append(f"Dropped {dropped_dates} rows missing dates.")
    if ghost_rows:    msgs.append(f"Dropped {ghost_rows} rows with out-of-range years.")
    return out, ("OK" if not msgs else " | ".join(msgs))


def _collect_csv_entries(main_zip_file):
    """Walk nested ZIPs and collect all CSV entries without reading their bytes yet."""
    entries = []
    skipped = []

    def _walk(zf, depth=0):
        for item_name in zf.namelist():
            base = Path(item_name).name
            if not base:
                continue
            if base.lower().endswith(".zip") and depth < 3:
                try:
                    data   = zf.read(item_name)
                    sub_zf = zipfile.ZipFile(io.BytesIO(data))
                    _walk(sub_zf, depth + 1)
                    del data
                except Exception as e:
                    skipped.append(f"{base}: Zip extraction error {e}")
            elif base.lower().endswith(".csv"):
                entries.append((zf, item_name, base))

    _walk(main_zip_file)
    return entries, skipped


def load_mtr_from_main_zip(main_zip_file):
    """
    Memory-efficient MTR loader.
    Processes one CSV at a time, frees each after parsing,
    and does a single pd.concat at the end on already-downcasted frames.
    """
    import gc
    skipped   = []
    csv_count = 0
    dfs: List[pd.DataFrame] = []

    try:
        main_zip_file.seek(0)
        root_zf = zipfile.ZipFile(main_zip_file)
    except Exception as e:
        st.error(f"Cannot open main ZIP: {e}")
        return pd.DataFrame(), 0, []

    entries, skipped = _collect_csv_entries(root_zf)
    if not entries:
        return pd.DataFrame(), 0, skipped

    prog  = st.sidebar.progress(0, text="Loading MTR files…")
    total = len(entries)

    for idx, (zf, item_name, base) in enumerate(entries):
        try:
            data = zf.read(item_name)
            df, msg = _parse_mtr_csv(data, base)
            del data
            gc.collect()

            if df.empty:
                skipped.append(f"{base}: {msg}")
            else:
                dfs.append(df)
                csv_count += 1
                if msg != "OK":
                    skipped.append(f"{base}: Partial Load ({msg})")
        except Exception as e:
            skipped.append(f"{base}: Critical Error - {e}")

        prog.progress((idx + 1) / total, text=f"MTR {idx + 1}/{total}: {base}")

    prog.empty()

    if not dfs:
        return pd.DataFrame(), 0, skipped

    combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    has_inv = combined["Invoice_Number"].str.strip() != ""
    dedup_a = combined[has_inv].drop_duplicates(
        subset=["Invoice_Number", "SKU", "Transaction_Type", "Date"], keep="first"
    )
    dedup_b = combined[~has_inv].drop_duplicates(
        subset=["Order_Id", "SKU", "Transaction_Type", "Date"], keep="first"
    )
    combined = pd.concat([dedup_a, dedup_b], ignore_index=True)
    del dedup_a, dedup_b
    gc.collect()

    combined = _downcast_mtr(combined)

    return combined, csv_count, skipped


# ══════════════════════════════════════════════════════════════
# 7) SALES DATA LOADERS
# ══════════════════════════════════════════════════════════════
def _downcast_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink the unified sales_df: category for Source/TxnType, float32 for Quantity."""
    for c in ["Transaction Type", "Source"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in ["Quantity", "Units_Effective"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    return df


def merge_daily_into_sales(
    base_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    if daily_df.empty:
        return base_df
    if base_df.empty:
        return daily_df

    base  = base_df.copy()
    daily = daily_df.copy()

    base["_date"]  = pd.to_datetime(base["TxnDate"],  errors="coerce").dt.normalize()
    daily["_date"] = pd.to_datetime(daily["TxnDate"], errors="coerce").dt.normalize()

    daily_keys = set(
        zip(daily["Sku"].astype(str),
            daily["_date"].astype(str),
            daily["Source"].astype(str))
    )

    overlap_mask = base.apply(
        lambda r: (str(r["Sku"]), str(r["_date"]), str(r["Source"])) in daily_keys,
        axis=1,
    )
    base_clean = base[~overlap_mask].drop(columns=["_date"])
    daily_clean = daily.drop(columns=["_date"])

    merged = pd.concat([base_clean, daily_clean], ignore_index=True)
    del base, daily, base_clean, daily_clean
    return _downcast_sales(merged)


# ══════════════════════════════════════════════════════════════
# DAILY REPORT PARSERS  (new platform-specific formats)
# Each returns a unified DataFrame with columns:
#   Date, SKU, Platform, TxnType, Quantity, Revenue, State, OrderId
# ══════════════════════════════════════════════════════════════

def _std_daily(date, sku, platform, txn_type, qty, revenue, state="", order_id=""):
    """Helper: create one standardised daily-order row as a dict."""
    return {
        "Date":      pd.to_datetime(date, errors="coerce"),
        "SKU":       str(sku).strip(),
        "Platform":  platform,
        "TxnType":   txn_type,
        "Quantity":  float(qty),
        "Revenue":   float(revenue),
        "State":     str(state).strip().title(),
        "OrderId":   str(order_id),
    }


def detect_daily_platform(file_obj) -> str:
    """
    Auto-detect which marketplace a daily report belongs to by inspecting
    its column names — no filename matching needed.

    Fingerprints (unique columns per platform):
      Amazon          → 'merchant sku'  +  'amazon order id'
      Flipkart/Myntra → 'seller sku code'  +  'store order id'
      Meesho          → 'reason for credit entry'  +  'sub order no'
      Myntra Earn More→ 'sku id'  +  'final sale units'  +  'gross units'

    Returns one of: 'amazon' | 'flipkart' | 'meesho' | 'myntra' | 'unknown'
    """
    try:
        file_obj.seek(0)
        name = getattr(file_obj, "name", "").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file_obj, nrows=1)
        else:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, nrows=1, encoding=enc)
                    break
                except Exception:
                    continue
            else:
                return "unknown"

        cols = set(c.strip().lower() for c in df.columns)
        file_obj.seek(0)  # reset after reading

        if "merchant sku" in cols and "amazon order id" in cols:
            return "amazon"
        if "seller sku code" in cols and "store order id" in cols:
            return "flipkart"
        if "reason for credit entry" in cols and "sub order no" in cols:
            return "meesho"
        if "sku id" in cols and "final sale units" in cols and "gross units" in cols:
            return "myntra"
        return "unknown"
    except Exception:
        return "unknown"


def parse_daily_amazon_csv(file_obj, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Parse Amazon daily shipment CSV (e.g. 937788020504.csv).
    Columns: Customer Shipment Date, Merchant SKU, Quantity, Product Amount,
             Amazon Order Id, Shipment To State, ...
    Every row = 1 shipped unit. No returns in this report format.
    """
    try:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)
        if df.empty:
            return pd.DataFrame()

        rows = []
        for _, r in df.iterrows():
            raw_sku = str(r.get("Merchant SKU", "")).strip()
            oms_sku = map_to_oms_sku(raw_sku, mapping)
            rows.append(_std_daily(
                date     = r.get("Customer Shipment Date"),
                sku      = oms_sku,
                platform = "Amazon",
                txn_type = "Shipment",
                qty      = pd.to_numeric(r.get("Quantity", 1), errors="coerce") or 1,
                revenue  = pd.to_numeric(r.get("Product Amount", 0), errors="coerce") or 0,
                state    = r.get("Shipment To State", ""),
                order_id = r.get("Amazon Order Id", ""),
            ))
        out = pd.DataFrame(rows)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out.dropna(subset=["Date"])
    except Exception as e:
        st.warning(f"Amazon daily parse error: {e}")
        return pd.DataFrame()


def parse_daily_flipkart_csv(file_obj, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Parse Flipkart/Myntra PPMP daily seller orders CSV.
    Columns: seller sku code, order status, final amount, seller price, state, created on, ...
    Status codes: WP=Waiting for Pickup, PK=Packed, SH=Shipped, F=Failed/Cancelled
    WP + PK + SH = active orders that need stock; F = cancelled, excluded.
    """
    try:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.strip().str.lower()

        STATUS_TXNTYPE = {
            "wp": "Shipment",   # waiting for pickup — stock committed
            "pk": "Shipment",   # packed
            "sh": "Shipment",   # shipped
            "f":  "Cancel",     # failed / cancelled
        }
        rows = []
        for _, r in df.iterrows():
            status  = str(r.get("order status", "")).strip().lower()
            txn     = STATUS_TXNTYPE.get(status, "Shipment")
            raw_sku = str(r.get("seller sku code", "")).strip()
            oms_sku = map_to_oms_sku(raw_sku, mapping)
            rows.append(_std_daily(
                date     = r.get("created on"),
                sku      = oms_sku,
                platform = "Flipkart/Myntra",
                txn_type = txn,
                qty      = 1,
                revenue  = pd.to_numeric(r.get("seller price", r.get("final amount", 0)), errors="coerce") or 0,
                state    = r.get("state", ""),
                order_id = str(r.get("store order id", r.get("order release id", ""))),
            ))
        out = pd.DataFrame(rows)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out.dropna(subset=["Date"])
    except Exception as e:
        st.warning(f"Flipkart daily parse error: {e}")
        return pd.DataFrame()


def parse_daily_meesho_csv(file_obj, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Parse Meesho daily orders CSV.
    Columns: Reason for Credit Entry, Sub Order No, SKU, Size, Quantity, Order Date,
             Supplier Discounted Price, Customer State, ...
    Reasons: SHIPPED, READY_TO_SHIP → active; CANCELLED → excluded; HOLD/PENDING → pending.
    Meesho puts SKU + Size separately — combine to get full variant SKU (e.g. 1485PLYKMUSTARD-XL).
    """
    try:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)
        if df.empty:
            return pd.DataFrame()

        REASON_TXNTYPE = {
            "shipped":        "Shipment",
            "ready_to_ship":  "Shipment",
            "cancelled":      "Cancel",
            "hold":           "Pending",
            "pending":        "Pending",
        }
        rows = []
        for _, r in df.iterrows():
            reason  = str(r.get("Reason for Credit Entry", "")).strip().lower()
            txn     = REASON_TXNTYPE.get(reason, "Shipment")
            raw_sku = str(r.get("SKU", "")).strip()
            size    = str(r.get("Size", "")).strip()
            # Build full variant SKU (same format as monthly files)
            full_sku = f"{raw_sku}-{size}" if size and size.lower() not in ("nan","") else raw_sku
            oms_sku  = map_to_oms_sku(full_sku, mapping)
            qty      = pd.to_numeric(r.get("Quantity", 1), errors="coerce") or 1
            price    = pd.to_numeric(r.get("Supplier Discounted Price (Incl GST and Commision)", 0), errors="coerce") or 0
            rows.append(_std_daily(
                date     = r.get("Order Date"),
                sku      = oms_sku,
                platform = "Meesho",
                txn_type = txn,
                qty      = qty,
                revenue  = price * qty,
                state    = r.get("Customer State", ""),
                order_id = str(r.get("Sub Order No", "")),
            ))
        out = pd.DataFrame(rows)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out.dropna(subset=["Date"])
    except Exception as e:
        st.warning(f"Meesho daily parse error: {e}")
        return pd.DataFrame()


def parse_daily_myntra_xlsx(file_obj, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Parse Myntra Earn More / daily summary report (.xlsx).
    Columns: SKU ID, Order Date, Gross Units, Final Sale Units, Cancellation Units,
             Return Units, Final Sale Amount, ...
    This is an aggregated report (one row per SKU per date), not order-level.
    We expand into Shipment / Cancel / Refund rows so the standard sales pipeline
    treats it the same as other sources.
    """
    try:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.strip()

        rows = []
        for _, r in df.iterrows():
            raw_sku = str(r.get("SKU ID", "")).strip()
            oms_sku = map_to_oms_sku(raw_sku, mapping)
            date    = r.get("Order Date")

            gross   = int(pd.to_numeric(r.get("Gross Units", 0),        errors="coerce") or 0)
            sold    = int(pd.to_numeric(r.get("Final Sale Units", 0),   errors="coerce") or 0)
            cancel  = int(pd.to_numeric(r.get("Cancellation Units", 0), errors="coerce") or 0)
            ret     = int(pd.to_numeric(r.get("Return Units", 0),       errors="coerce") or 0)
            revenue = pd.to_numeric(r.get("Final Sale Amount", 0),      errors="coerce") or 0

            if sold > 0:
                rows.append(_std_daily(date, oms_sku, "Myntra", "Shipment", sold, revenue))
            if cancel > 0:
                rows.append(_std_daily(date, oms_sku, "Myntra", "Cancel",   cancel, 0))
            if ret > 0:
                rows.append(_std_daily(date, oms_sku, "Myntra", "Refund",   ret, 0))

        out = pd.DataFrame(rows) if rows else pd.DataFrame()
        if out.empty:
            return pd.DataFrame()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out.dropna(subset=["Date"])
    except Exception as e:
        st.warning(f"Myntra daily parse error: {e}")
        return pd.DataFrame()


def combine_daily_orders(*dfs) -> pd.DataFrame:
    """Concatenate parsed daily DataFrames and deduplicate by (SKU, Platform, OrderId, TxnType)."""
    parts = [d for d in dfs if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["Date"]     = pd.to_datetime(combined["Date"], errors="coerce")
    combined["Quantity"] = pd.to_numeric(combined["Quantity"], errors="coerce").fillna(0)
    combined["Revenue"]  = pd.to_numeric(combined["Revenue"],  errors="coerce").fillna(0)
    # Dedup: same order can't appear twice (protects against re-upload)
    combined = combined.drop_duplicates(
        subset=["SKU", "Platform", "OrderId", "TxnType"], keep="first"
    )
    return combined


def daily_to_sales_rows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the unified daily_orders_df into the standard sales_df schema
    (Sku, TxnDate, Transaction Type, Quantity, Units_Effective, Source, OrderId)
    so it feeds into merge_daily_into_sales and the PO Engine.
    """
    if daily_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              daily_df["SKU"],
        "TxnDate":          daily_df["Date"],
        "Transaction Type": daily_df["TxnType"],
        "Quantity":         daily_df["Quantity"],
        "Units_Effective":  np.where(daily_df["TxnType"] == "Refund",  -daily_df["Quantity"],
                            np.where(daily_df["TxnType"] == "Cancel",   0,
                            np.where(daily_df["TxnType"] == "Pending",  0,
                                     daily_df["Quantity"]))),
        "Source":           daily_df["Platform"],
        "OrderId":          daily_df["OrderId"],
    })
    return _downcast_sales(out)


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
        if "refund" in s or "return" in s:                      return "Refund"
        if "cancel" in s:                                        return "Cancel"
        if "freereplacement" in s or "replacement" in s:        return "FreeReplacement"
        return "Shipment"

    df["TxnType"] = df.get("Transaction Type", "").apply(classify_txn)
    if not config.include_replacements:
        df.loc[df["TxnType"] == "FreeReplacement", "Quantity"] = 0

    df["Units_Effective"] = np.where(df["TxnType"] == "Refund",  -df["Quantity"],
                            np.where(df["TxnType"] == "Cancel",   0, df["Quantity"]))
    df["Source"] = source

    order_col = next((c for c in df.columns if "order" in c.lower() and "id" in c.lower()), None)
    df["OrderId"] = df[order_col] if order_col else np.nan

    result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
    result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
    del df
    return _downcast_sales(result.dropna(subset=["TxnDate"]))


def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """Legacy single-file loader used by Tier-2 monthly upload and daily snapshot."""
    try:
        df = pd.read_excel(xlsx_file, sheet_name="Sales Report")
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.strip()
        df["OMS_SKU"] = df["SKU"].apply(clean_sku).apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"]  = pd.to_datetime(df.get("Buyer Invoice Date", df.get("Order Date")), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("Item Quantity", 0), errors="coerce").fillna(0)
        df["Source"]   = "Flipkart"

        def _fk_txn(event):
            e = str(event).strip()
            if e == "Sale":                return "Shipment"
            if e == "Return":             return "Refund"
            if e == "Cancellation":       return "Cancel"
            if e == "Return Cancellation":return "ReturnCancel"
            return "Shipment"
        df["TxnType"] = df.get("Event Sub Type", pd.Series(["Sale"]*len(df))).apply(_fk_txn)
        df["Units_Effective"] = np.where(df["TxnType"]=="Refund",      -df["Quantity"],
                                np.where(df["TxnType"]=="Cancel",        0,
                                np.where(df["TxnType"]=="ReturnCancel",  df["Quantity"],
                                         df["Quantity"])))
        df["OrderId"] = df.get("Order ID", df.get("Order Id", np.nan))
        result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
        result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
        del df
        return _downcast_sales(result.dropna(subset=["TxnDate"]))
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}")
        return pd.DataFrame()


# ── Month extraction from Flipkart filename ────────────────────
def _fk_month_from_filename(fname: str):
    """
    Parse YYYY-MM from Flipkart monthly filenames.
    Handles: DEC-2025.xlsx, Jan-2025.xlsx, JAN-2026.xlsx,
             March-2025.xlsx, July-2025.xlsx, feb-2025.xlsx, etc.
    Returns e.g. '2025-12' or None if unparseable.
    """
    base = Path(fname).stem.upper()
    _MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
            "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12,
            "MARCH":3,"APRIL":4,"JUNE":6,"JULY":7,"SEPT":9,"AUGUST":8}
    parts = re.split(r"[-_\s]", base)
    for i, p in enumerate(parts):
        mon_num = _MON.get(p[:5]) or _MON.get(p[:4]) or _MON.get(p[:3])
        if mon_num:
            # Look for a 4-digit year in same or adjacent part
            for q in parts:
                if re.fullmatch(r"20\d{2}", q):
                    return f"{q}-{mon_num:02d}"
    return None


def _parse_flipkart_xlsx(file_bytes: bytes, fname: str, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Parse one Flipkart monthly Excel file (Sales Report sheet).

    Month bucketing uses Buyer Invoice Date — Flipkart guarantees ALL rows
    in a monthly file have a Buyer Invoice Date within that calendar month,
    exactly analogous to Meesho's month_number field. This means we NEVER
    need a grace period for Flipkart: the file boundary IS the month boundary.

    Event Sub Types handled:
      Sale              → TxnType='Shipment', qty positive
      Return            → TxnType='Refund',   qty positive (units returned)
      Cancellation      → TxnType='Cancel',   qty counted (for reporting) but Units_Effective=0
      Return Cancellation → TxnType='ReturnCancel', reverses a prior return

    Dedup key: Buyer Invoice ID — unique per transaction event in Flipkart system.

    Revenue: uses Buyer Invoice Amount (correctly signed by Flipkart:
      positive for Sale/ReturnCancel, negative for Return/Cancellation).
    """
    try:
        xl   = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Sales Report")
        if xl.empty:
            return pd.DataFrame()

        # Strip trailing spaces that vary between monthly exports
        xl.columns = xl.columns.str.strip()

        # ── Month: Buyer Invoice Date (always in the file's calendar month) ──
        # Fall back to Order Date only if Buyer Invoice Date missing
        date_col = "Buyer Invoice Date" if "Buyer Invoice Date" in xl.columns else "Order Date"
        xl["Date"] = pd.to_datetime(xl[date_col], errors="coerce")

        # Month from filename is the authoritative source (like Meesho month_number)
        file_month = _fk_month_from_filename(fname)
        if file_month:
            xl["Month"] = file_month
        else:
            xl["Month"] = xl["Date"].dt.to_period("M").astype(str)

        # ── Transaction type ──
        def _fk_txn(event):
            e = str(event).strip()
            if e == "Sale":                return "Shipment"
            if e == "Return":             return "Refund"
            if e == "Cancellation":       return "Cancel"
            if e == "Return Cancellation":return "ReturnCancel"
            return "Shipment"
        xl["TxnType"] = xl["Event Sub Type"].apply(_fk_txn)

        # ── Quantity & revenue ──
        xl["Quantity"]       = pd.to_numeric(xl.get("Item Quantity", 1), errors="coerce").fillna(0).astype("float32")
        xl["Invoice_Amount"] = pd.to_numeric(xl.get("Buyer Invoice Amount", 0), errors="coerce").fillna(0).astype("float32")

        # ── OMS SKU (strip Flipkart triple-quote format) ──
        xl["OMS_SKU"] = xl["SKU"].apply(clean_sku).apply(lambda x: map_to_oms_sku(x, mapping))

        # ── State for geography charts ──
        state_col = next((c for c in xl.columns if "Delivery State" in c), None)
        xl["State"] = xl[state_col].fillna("").astype(str).str.upper().str.strip() if state_col else ""

        # ── Order / Invoice IDs for dedup ──
        xl["OrderId"]         = xl.get("Order ID",         xl.get("Order Id",         "")).astype(str)
        xl["BuyerInvoiceId"]  = xl.get("Buyer Invoice ID", xl.get("Buyer Invoice Id", "")).astype(str)

        out = xl[["Date","Month","TxnType","Quantity","Invoice_Amount",
                  "OMS_SKU","State","OrderId","BuyerInvoiceId"]].copy()
        return out.dropna(subset=["Date"])

    except Exception as e:
        return pd.DataFrame()  # caller logs the skip


def load_flipkart_full(main_zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Load ALL Flipkart monthly Excel files from a master ZIP.

    Each .xlsx inside the ZIP is a monthly Flipkart Sales Report.
    Month is sourced from the filename (DEC-2025.xlsx → 2025-12), which
    mirrors how we use Meesho's month_number field — immune to cross-month
    order dates that would otherwise require a grace period.

    Dedup: full-row drop_duplicates() to remove identical rows from re-uploads
    while preserving legitimate duplicate Order Item IDs (Return + Return
    Cancellation pairs that cancel each other out in revenue but both count).
    """
    dfs     = []
    skipped = []
    try:
        main_zip_file.seek(0)
        root_zf = zipfile.ZipFile(main_zip_file)
    except Exception as e:
        st.error(f"Cannot open Flipkart ZIP: {e}")
        return pd.DataFrame()

    xlsx_items = [n for n in root_zf.namelist() if n.lower().endswith(".xlsx")]
    prog = st.sidebar.progress(0, text="Loading Flipkart files…")

    for idx, item_name in enumerate(xlsx_items):
        base = Path(item_name).name
        prog.progress((idx + 1) / max(len(xlsx_items), 1), text=f"Flipkart {idx+1}/{len(xlsx_items)}: {base}")
        try:
            file_bytes = root_zf.read(item_name)
            df = _parse_flipkart_xlsx(file_bytes, base, mapping)
            if df.empty:
                skipped.append(f"{base}: no data / unrecognised format")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{base}: {e}")

    prog.empty()
    if skipped:
        with st.sidebar.expander(f"⚠️ Flipkart: {len(skipped)} files skipped"):
            for s in skipped:
                st.write(s)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Full-row dedup: keeps Return+ReturnCancel pairs (differ in TxnType/Invoice_Amount)
    # while removing exact duplicates from accidental re-uploads of the same file.
    combined = combined.drop_duplicates(keep="first")
    return combined


def flipkart_to_sales_rows(fk_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Flipkart analytics df into the standard sales_df format."""
    if fk_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              fk_df["OMS_SKU"],
        "TxnDate":          fk_df["Date"],
        "Transaction Type": fk_df["TxnType"],
        "Quantity":         fk_df["Quantity"],
        "Units_Effective":  np.where(fk_df["TxnType"]=="Refund",       -fk_df["Quantity"],
                            np.where(fk_df["TxnType"]=="Cancel",         0,
                            np.where(fk_df["TxnType"]=="ReturnCancel",   fk_df["Quantity"],
                                     fk_df["Quantity"]))),
        "Source":           "Flipkart",
        "OrderId":          fk_df["OrderId"],
    })
    return out


def _parse_meesho_inner_zip(inner_zf) -> pd.DataFrame:
    """
    Parse one Meesho monthly inner ZIP.

    KEY FIX: For RETURNS (Refund type rows), use the actual return/refund date
    column (return_date, return_created_date, return_pickup_date) rather than
    order_date. This ensures return quantities land in the correct calendar month
    and aren't lost when filtering by month.

    Two formats exist:
      A) Non-GST: ForwardReports.xlsx + Reverse.xlsx
      B) GST:     tcs_sales.xlsx + tcs_sales_return.xlsx
    """
    files = {f.lower(): f for f in inner_zf.namelist()}
    rows  = []

    def _best_date_col(df, prefer_return: bool = False) -> str:
        """
        Pick the best available date column.
        For return rows: prefer return_date > return_created_date > return_pickup_date > order_date.
        For forward rows: prefer order_date > created_date.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        if prefer_return:
            for candidate in ["return_date", "return_created_date", "return_pickup_date",
                               "pickup_date", "reverse_pickup_date"]:
                if candidate in cols_lower:
                    return cols_lower[candidate]
        # Fall back to order date
        for candidate in ["order_date", "created_date", "order_created_date"]:
            if candidate in cols_lower:
                return cols_lower[candidate]
        return None

    # ── Format B: GST zips ──
    if "tcs_sales.xlsx" in files:
        with inner_zf.open(files["tcs_sales.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            # Use order_date as the transaction Date
            date_col = _best_date_col(df, prefer_return=False)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("total_invoice_value", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state_new", "")
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Shipment"
            # KEY FIX 1: Build Month from Meesho's own financial_year + month_number
            # columns instead of deriving it from order_date. This guarantees every
            # row lands in the exact month Meesho intends, regardless of order_date
            # edge-cases (e.g. Dec orders placed in Nov due to timezone).
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None  # will fall back to date-derived month below
            rows.append(df[["_Date","_TxnType","_Qty","_Rev","_State","_OrderId","_Month"]])

    if "tcs_sales_return.xlsx" in files:
        with inner_zf.open(files["tcs_sales_return.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            # KEY FIX 2: Use cancel_return_date (not order_date) as the transaction
            # Date so returns are time-stamped to when they actually came back,
            # not when the original order was placed (which can be months earlier).
            date_col = _best_date_col(df, prefer_return=True)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("total_invoice_value", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state_new", "")
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Refund"
            # KEY FIX 1 (same as above): use Meesho's month_number for Month bucketing
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None
            rows.append(df[["_Date","_TxnType","_Qty","_Rev","_State","_OrderId","_Month"]])

    # ── Format A: Non-GST zips ──
    if "forwardreports.xlsx" in files and not rows:
        with inner_zf.open(files["forwardreports.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=False)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("meesho_price", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state", df.get("state", ""))
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            def _meesho_txn(s):
                s = str(s).lower()
                if "return" in s or "rto" in s: return "Refund"
                if "cancel" in s:               return "Cancel"
                return "Shipment"
            df["_TxnType"] = df.get("order_status", "").apply(_meesho_txn)
            df["_Month"]   = None  # Non-GST has no month_number; will use date
            rows.append(df[["_Date","_TxnType","_Qty","_Rev","_State","_OrderId","_Month"]])

    if "reverse.xlsx" in files and not any(
        (r["_TxnType"] == "Refund").any() for r in rows if not r.empty
    ):
        with inner_zf.open(files["reverse.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            # Use return date for reverse rows
            date_col = _best_date_col(df, prefer_return=True)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("meesho_price", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state", df.get("state", ""))
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Refund"
            df["_Month"]   = None
            rows.append(df[["_Date","_TxnType","_Qty","_Rev","_State","_OrderId","_Month"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.columns = ["Date","TxnType","Quantity","Invoice_Amount","State","OrderId","_Month"]
    out["Date"]           = pd.to_datetime(out["Date"], errors="coerce")
    out["Quantity"]       = out["Quantity"].astype("float32")
    out["Invoice_Amount"] = out["Invoice_Amount"].astype("float32")
    out["State"]          = out["State"].astype(str).str.upper().str.strip()

    # KEY FIX 1: Use Meesho's own month_number/financial_year where available
    # (GST format). Fall back to date-derived period for non-GST format rows.
    out["Month"] = out["_Month"].where(
        out["_Month"].notna(),
        out["Date"].dt.to_period("M").astype(str)
    )
    out = out.drop(columns=["_Month"])
    return out.dropna(subset=["Date"])


def load_meesho_full(main_zip_file) -> pd.DataFrame:
    """
    Load ALL Meesho monthly zips (both GST and non-GST formats) from the master ZIP.
    Returns a combined Meesho analytics DataFrame (no per-SKU — totals only).
    """
    dfs     = []
    skipped = []
    try:
        main_zip_file.seek(0)
        root_zf = zipfile.ZipFile(main_zip_file)
    except Exception as e:
        st.error(f"Cannot open Meesho ZIP: {e}")
        return pd.DataFrame()

    prog  = st.sidebar.progress(0, text="Loading Meesho files…")
    items = [n for n in root_zf.namelist() if n.lower().endswith(".zip")]

    for idx, item_name in enumerate(items):
        base = Path(item_name).name
        try:
            data = root_zf.read(item_name)
            with zipfile.ZipFile(io.BytesIO(data)) as inner_zf:
                df = _parse_meesho_inner_zip(inner_zf)
            if df.empty:
                skipped.append(f"{base}: No recognised data files")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{base}: {e}")
        prog.progress((idx + 1) / max(len(items), 1), text=f"Meesho {idx+1}/{len(items)}: {base}")

    prog.empty()
    if skipped:
        with st.sidebar.expander(f"⚠️ Meesho: {len(skipped)} files skipped"):
            for s in skipped: st.write(s)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # KEY FIX 3: Use full-row dedup instead of (OrderId, TxnType, Date).
    #
    # Why: Meesho's tcs_sales.xlsx contains "adjustment pairs" — two rows with
    # the same sub_order_num, same date, same TxnType, but different invoice
    # amounts (one positive + one small negative correction). The old 3-column
    # dedup was collapsing these pairs into one row, dropping the correction
    # row and silently losing quantity units. Full-row dedup keeps both rows of
    # a pair (since Invoice_Amount differs) while still removing truly identical
    # duplicate rows that can appear when the same monthly ZIP is loaded twice.
    combined = combined.drop_duplicates(keep="first")
    return combined


def meesho_to_sales_rows(meesho_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Meesho analytics df into the standard sales_df format.
    Since there's no per-SKU data, all units go under Sku='MEESHO_TOTAL'.
    """
    if meesho_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              "MEESHO_TOTAL",
        "TxnDate":          meesho_df["Date"],
        "Transaction Type": meesho_df["TxnType"],
        "Quantity":         meesho_df["Quantity"],
        "Units_Effective":  np.where(meesho_df["TxnType"]=="Refund", -meesho_df["Quantity"],
                            np.where(meesho_df["TxnType"]=="Cancel",  0, meesho_df["Quantity"])),
        "Source":           "Meesho",
        "OrderId":          meesho_df["OrderId"],
    })
    return out


# ──────────────────────────────────────────────────────────────
# MYNTRA PPMP LOADER
# ──────────────────────────────────────────────────────────────
def _parse_myntra_csv(csv_bytes: bytes, filename: str, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False, on_bad_lines="skip")
    except Exception as e:
        return pd.DataFrame(), f"CSV parse error: {e}"

    if df.empty:
        return pd.DataFrame(), "Empty file"

    df.columns = df.columns.str.strip().str.lower()

    date_col = next((c for c in df.columns if "order_created_date" in c or "order_date" in c), None)
    if not date_col:
        return pd.DataFrame(), "No date column found"

    df["_Date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame(), "All dates invalid"

    sku_col = next((c for c in df.columns if c in ["sku_id", "skuid", "sku"]), None)
    if not sku_col:
        return pd.DataFrame(), "No SKU column"
    df["_OMS_SKU"] = df[sku_col].apply(lambda x: map_to_oms_sku(str(x).strip(), mapping))

    qty_col = next((c for c in df.columns if c == "quantity"), None)
    df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1) if qty_col else 1.0

    rev_col = next((c for c in df.columns if c in ["invoiceamount", "invoice_amount", "net_amount", "shipment_value"]), None)
    df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0) if rev_col else 0.0

    status_col = next((c for c in df.columns if "order_status" in c), None)
    def _myntra_txn(s):
        s = str(s).strip().upper()
        if s in ("RTO",):                    return "Refund"
        if s in ("F", "IC", "FAILED"):       return "Cancel"
        if s in ("C", "SH", "PK", "SHIPPED","CONFIRMED","DELIVERED"): return "Shipment"
        return "Shipment"
    df["_TxnType"] = df[status_col].apply(_myntra_txn) if status_col else "Shipment"

    state_col   = next((c for c in df.columns if c in ["state", "customer_delivery_state_code"]), None)
    pm_col      = next((c for c in df.columns if "payment_method" in c), None)
    wh_col      = next((c for c in df.columns if "warehouse_id" in c), None)
    order_col   = next((c for c in df.columns if c in ["order_id", "packet_id"]), None)

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "OMS_SKU":        df["_OMS_SKU"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"].astype("float32"),
        "Invoice_Amount": df["_Rev"].astype("float32"),
        "State":          df[state_col].fillna("").str.upper().str.strip() if state_col else "",
        "Payment_Method": df[pm_col].fillna("") if pm_col else "",
        "Warehouse_Id":   df[wh_col].fillna("") if wh_col else "",
        "OrderId":        df[order_col].fillna("") if order_col else "",
    })
    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    return out, "OK"


def load_myntra_full(main_zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    dfs     = []
    skipped = []
    try:
        main_zip_file.seek(0)
        root_zf = zipfile.ZipFile(main_zip_file)
    except Exception as e:
        st.error(f"Cannot open Myntra ZIP: {e}")
        return pd.DataFrame()

    csv_items = [n for n in root_zf.namelist() if n.lower().endswith(".csv")]
    prog = st.sidebar.progress(0, text="Loading Myntra files…")

    for idx, item_name in enumerate(csv_items):
        base = Path(item_name).name
        try:
            data = root_zf.read(item_name)
            df, msg = _parse_myntra_csv(data, base, mapping)
            if df.empty:
                skipped.append(f"{base}: {msg}")
            else:
                dfs.append(df)
                if msg != "OK":
                    skipped.append(f"{base}: Partial ({msg})")
        except Exception as e:
            skipped.append(f"{base}: {e}")
        prog.progress((idx + 1) / max(len(csv_items), 1), text=f"Myntra {idx+1}/{len(csv_items)}: {base}")

    prog.empty()
    if skipped:
        with st.sidebar.expander(f"⚠️ Myntra: {len(skipped)} files had issues"):
            for s in skipped: st.write(s)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["OrderId","OMS_SKU","TxnType","Date"], keep="first")
    return combined


def myntra_to_sales_rows(myntra_df: pd.DataFrame) -> pd.DataFrame:
    if myntra_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              myntra_df["OMS_SKU"],
        "TxnDate":          myntra_df["Date"],
        "Transaction Type": myntra_df["TxnType"],
        "Quantity":         myntra_df["Quantity"],
        "Units_Effective":  np.where(myntra_df["TxnType"]=="Refund", -myntra_df["Quantity"],
                            np.where(myntra_df["TxnType"]=="Cancel",  0, myntra_df["Quantity"])),
        "Source":           "Myntra",
        "OrderId":          myntra_df["OrderId"],
    })
    return out

# ══════════════════════════════════════════════════════════════
# 8) INVENTORY LOADERS
# ══════════════════════════════════════════════════════════════
def load_inventory_consolidated(oms_file, fk_file, myntra_file, amz_file, mapping: Dict[str, str], group_by_parent: bool = False) -> pd.DataFrame:
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
            df["OMS_SKU"]       = df["SKU"].apply(lambda x: map_to_oms_sku(x, mapping))
            df["Flipkart_Live"] = pd.to_numeric(df["Live on Website"], errors="coerce").fillna(0)
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
        consolidated = (consolidated.groupby("Parent_SKU")[inv_cols + ["Marketplace_Total","Total_Inventory"]]
                        .sum().reset_index().rename(columns={"Parent_SKU":"OMS_SKU"}))

    return consolidated[consolidated["Total_Inventory"] > 0]


def load_stock_transfer(zip_file) -> pd.DataFrame:
    df = read_zip_csv(zip_file)
    if df.empty:
        return pd.DataFrame()
    required = ["Invoice Date","Ship From Fc","Ship To Fc","Quantity","Transaction Type"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    result = df[required].copy()
    result["Invoice Date"] = pd.to_datetime(result["Invoice Date"], errors="coerce")
    result["Quantity"]     = pd.to_numeric(result["Quantity"], errors="coerce").fillna(0)
    return result

# ══════════════════════════════════════════════════════════════
# 11) PO BASE CALCULATOR (SEASONAL INTEGRATED)
# ══════════════════════════════════════════════════════════════

def _mtr_to_sales_df(mtr_df: pd.DataFrame, sku_mapping: Dict[str, str], group_by_parent: bool = False) -> pd.DataFrame:
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


def get_indian_fy_quarter(date: pd.Timestamp) -> tuple:
    """
    Return (FY_label, Q_label) for an Indian fiscal year date.
    Indian FY runs Apr-Mar: FY2025 = Apr 2024 – Mar 2025.
    Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar
    """
    m = date.month
    y = date.year
    if m >= 4:
        fy = y + 1          # Apr 2024 → FY2025
        q  = 1 if m <= 6 else 2 if m <= 9 else 3
    else:
        fy = y              # Jan 2025 → FY2025
        q  = 4
    return fy, q


_Q_LABELS = {
    1: "Apr–Jun",
    2: "Jul–Sep",
    3: "Oct–Dec",
    4: "Jan–Mar",
}


def quarter_col_name(fy: int, q: int) -> str:
    """Return human-readable column label using calendar start year.
    e.g. FY2026 Q1 (Apr-Jun 2025) → 'Apr–Jun 2025'
         FY2026 Q4 (Jan-Mar 2026) → 'Jan–Mar 2026'
    """
    cal_year = fy - 1 if q in (1, 2, 3) else fy
    return f"{_Q_LABELS[q]} {cal_year}"


def calculate_quarterly_history(
    sales_df: pd.DataFrame,
    mtr_df: pd.DataFrame = None,
    myntra_df: pd.DataFrame = None,
    sku_mapping: Dict[str, str] = None,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> pd.DataFrame:
    """
    Build a per-SKU quarterly sales history table (Indian fiscal year quarters).

    Data sources used — in priority order:
      1. sales_df  — the merged monthly/daily sales DataFrame (Amazon + Flipkart + Meesho already inside)
      2. mtr_df    — Amazon MTR (adds historical depth if sales_df doesn't go back far enough)
      3. myntra_df — Myntra historical (same reason)

    Returns a wide DataFrame:
        OMS_SKU | Apr–Jun 2024 | Jul–Sep 2024 | … | Avg_Monthly | Units_30d | ADS | Status

    Status thresholds (ADS = avg daily sales over last 90 days):
        🚀 Fast Moving  : ADS ≥ 1.0  (≥30 units/month)
        ✅ Moderate     : ADS ≥ 0.33 (≥10 units/month)
        🐢 Slow Selling : ADS ≥ 0.10 (≥ 3 units/month)
        ❌ Not Moving   : ADS < 0.10
    """
    parts = []

    # ── 1. sales_df (already has Amazon + Flipkart + Meesho merged) ──
    if not sales_df.empty and "Sku" in sales_df.columns:
        txn_col = "Transaction Type"
        tmp = sales_df[["Sku", "TxnDate", "Quantity", txn_col]].copy()
        tmp.columns = ["SKU", "Date", "Qty", "TxnType"]
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["Qty"]  = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
        parts.append(tmp.dropna(subset=["Date"]))

    # ── 2. MTR (adds deeper Amazon history) ──────────────────────────
    if mtr_df is not None and not mtr_df.empty:
        mtr_sku_col  = next((c for c in mtr_df.columns if c in ["SKU","Sku","OMS_SKU"]), None)
        mtr_date_col = next((c for c in mtr_df.columns if c in ["Date","TxnDate"]),        None)
        mtr_qty_col  = next((c for c in mtr_df.columns if c in ["Quantity","Qty"]),        None)
        mtr_txn_col  = next((c for c in mtr_df.columns if c in ["Transaction_Type","Transaction Type","TxnType"]), None)
        if mtr_sku_col and mtr_date_col and mtr_qty_col:
            tmp = mtr_df[[mtr_sku_col, mtr_date_col, mtr_qty_col]].copy()
            tmp.columns = ["SKU", "Date", "Qty"]
            tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
            tmp["TxnType"] = mtr_df[mtr_txn_col].values if mtr_txn_col else "Shipment"
            if sku_mapping:
                tmp["SKU"] = tmp["SKU"].apply(lambda x: map_to_oms_sku(x, sku_mapping))
            parts.append(tmp.dropna(subset=["Date"]))

    # ── 3. Myntra historical (adds deeper Myntra history) ────────────
    if myntra_df is not None and not myntra_df.empty:
        myn_sku_col  = next((c for c in myntra_df.columns if c in ["OMS_SKU","Sku","SKU"]),  None)
        myn_date_col = next((c for c in myntra_df.columns if c in ["Date","TxnDate"]),         None)
        myn_qty_col  = next((c for c in myntra_df.columns if c in ["Quantity","Qty"]),         None)
        myn_txn_col  = next((c for c in myntra_df.columns if c in ["TxnType","Transaction Type"]), None)
        if myn_sku_col and myn_date_col and myn_qty_col:
            tmp = myntra_df[[myn_sku_col, myn_date_col, myn_qty_col]].copy()
            tmp.columns = ["SKU", "Date", "Qty"]
            tmp["Date"]    = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp["Qty"]     = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0)
            tmp["TxnType"] = myntra_df[myn_txn_col].values if myn_txn_col else "Shipment"
            parts.append(tmp.dropna(subset=["Date"]))

    if not parts:
        return pd.DataFrame()

    hist = pd.concat(parts, ignore_index=True)
    hist = hist[hist["TxnType"].astype(str).str.strip() == "Shipment"]
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"])
    hist["Qty"] = pd.to_numeric(hist["Qty"], errors="coerce").fillna(0)
    hist = hist[hist["Qty"] > 0]

    if hist.empty:
        return pd.DataFrame()

    if group_by_parent:
        hist["SKU"] = hist["SKU"].apply(get_parent_sku)

    # ── Tag each row with (FY, Q) using vectorised apply ─────────────
    fy_q = hist["Date"].apply(get_indian_fy_quarter)
    hist["FY"] = fy_q.apply(lambda x: x[0])
    hist["QN"] = fy_q.apply(lambda x: x[1])

    # ── Determine which n_quarters to display (oldest → newest) ──────
    today          = pd.Timestamp.today()
    cur_fy, cur_q  = get_indian_fy_quarter(today)
    quarter_seq    = []
    fy_i, q_i      = cur_fy, cur_q
    for _ in range(n_quarters):
        quarter_seq.append((fy_i, q_i))
        q_i -= 1
        if q_i == 0:
            q_i = 4
            fy_i -= 1
    quarter_seq = list(reversed(quarter_seq))   # oldest first

    # ── Pivot: SKU × quarter_col ──────────────────────────────────────
    hist["col"] = hist.apply(
        lambda r: quarter_col_name(int(r["FY"]), int(r["QN"])), axis=1
    )
    grp   = hist.groupby(["SKU", "col"])["Qty"].sum().reset_index()
    pivot = grp.pivot_table(index="SKU", columns="col", values="Qty",
                            aggfunc="sum", fill_value=0).reset_index()
    pivot = pivot.rename(columns={"SKU": "OMS_SKU"})
    pivot.columns.name = None

    # Ensure every quarter column exists, in order, even if no sales
    ordered_q_cols = []
    for fy_j, q_j in quarter_seq:
        col = quarter_col_name(fy_j, q_j)
        ordered_q_cols.append(col)
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["OMS_SKU"] + ordered_q_cols]

    # ── Summary metrics ───────────────────────────────────────────────
    # Avg Monthly = mean of last 4 full quarters ÷ 3 (quarter→month)
    last4 = ordered_q_cols[-4:]
    pivot["Avg_Monthly"] = (pivot[last4].mean(axis=1) / 3).round(1)

    # ADS based on last 90 days actual data
    cutoff_90 = today - timedelta(days=90)
    r90 = hist[hist["Date"] >= cutoff_90].groupby("SKU")["Qty"].sum().reset_index()
    r90.columns = ["OMS_SKU", "Units_90d"]
    pivot = pivot.merge(r90, on="OMS_SKU", how="left").fillna({"Units_90d": 0})
    pivot["ADS"] = (pivot["Units_90d"] / 90).round(3)

    # Last 30d units
    cutoff_30 = today - timedelta(days=30)
    r30 = hist[hist["Date"] >= cutoff_30].groupby("SKU")["Qty"].sum().reset_index()
    r30.columns = ["OMS_SKU", "Units_30d"]
    pivot = pivot.merge(r30, on="OMS_SKU", how="left").fillna({"Units_30d": 0})

    # Frequency = distinct sale days in last 30d (how many days had ≥1 unit sold)
    f30 = (
        hist[hist["Date"] >= cutoff_30]
        .assign(_day=lambda d: d["Date"].dt.normalize())
        .groupby("SKU")["_day"].nunique()
        .reset_index()
    )
    f30.columns = ["OMS_SKU", "Freq_30d"]
    pivot = pivot.merge(f30, on="OMS_SKU", how="left").fillna({"Freq_30d": 0})
    pivot["Freq_30d"] = pivot["Freq_30d"].astype(int)

    # Status classification
    def _status(ads):
        if ads >= 1.0:  return "🚀 Fast Moving"
        if ads >= 0.33: return "✅ Moderate"
        if ads >= 0.10: return "🐢 Slow Selling"
        return "❌ Not Moving"

    pivot["Status"]   = pivot["ADS"].apply(_status)
    pivot["Units_90d"] = pivot["Units_90d"].astype(int)
    pivot["Units_30d"] = pivot["Units_30d"].astype(int)
    return pivot


def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    lead_time: int,
    target_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7,
    use_seasonality: bool = False,
    seasonal_weight: float = 0.5,
    mtr_df: pd.DataFrame = None,
    myntra_df: pd.DataFrame = None,
    sku_mapping: Dict[str, str] = None,
    group_by_parent: bool = False,
) -> pd.DataFrame:

    if sales_df.empty or inv_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
    df = df.dropna(subset=["TxnDate"])

    max_date = df["TxnDate"].max()
    cutoff   = max_date - timedelta(days=period_days)
    recent   = df[df["TxnDate"] >= cutoff].copy()

    sold = recent[recent["Transaction Type"]=="Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
    sold.columns = ["OMS_SKU", "Sold_Units"]
    returns = recent[recent["Transaction Type"]=="Refund"].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU", "Return_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU", "Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)
    po_df = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna({"Sold_Units":0,"Return_Units":0,"Net_Units":0})

    denom = max(period_days, min_denominator)
    demand_units = po_df["Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["Sold_Units"]
    po_df["Recent_ADS"] = (demand_units / denom).fillna(0)

    if use_seasonality:
        hist_parts  = [df]
        source_tags = []

        if mtr_df is not None and not mtr_df.empty and sku_mapping is not None:
            mtr_sales = _mtr_to_sales_df(mtr_df, sku_mapping, group_by_parent=group_by_parent)
            if not mtr_sales.empty:
                hist_parts.append(mtr_sales)
                source_tags.append("MTR")

        if myntra_df is not None and not myntra_df.empty:
            myn_sales = myntra_to_sales_rows(myntra_df)
            if not myn_sales.empty:
                myn_sales["TxnDate"] = pd.to_datetime(myn_sales["TxnDate"], errors="coerce")
                if group_by_parent:
                    myn_sales["Sku"] = myn_sales["Sku"].apply(get_parent_sku)
                hist_parts.append(myn_sales)
                source_tags.append("Myntra")

        ly_source_label = "+".join(source_tags) if source_tags else "Sales"

        if len(hist_parts) > 1:
            hist_df = pd.concat(hist_parts, ignore_index=True)
            hist_df = hist_df.drop_duplicates(
                subset=["Sku", "TxnDate", "Transaction Type"], keep="last"
            )
        else:
            hist_df = df

        ly_trailing_end   = max_date - timedelta(days=365)
        ly_trailing_start = ly_trailing_end - timedelta(days=period_days)

        ly_fwd_start = (max_date + timedelta(days=lead_time)) - timedelta(days=365)
        ly_fwd_end   = (max_date + timedelta(days=lead_time + max(target_days, period_days))) - timedelta(days=365)

        ly_sales_trailing = hist_df[(hist_df["TxnDate"] >= ly_trailing_start) & (hist_df["TxnDate"] < ly_trailing_end)].copy()
        ly_sales_fwd      = hist_df[(hist_df["TxnDate"] >= ly_fwd_start)      & (hist_df["TxnDate"] < ly_fwd_end)].copy()

        if not ly_sales_trailing.empty:
            ly_sales      = ly_sales_trailing
            ly_days_count = max((ly_trailing_end - ly_trailing_start).days, min_denominator)
        elif not ly_sales_fwd.empty:
            ly_sales      = ly_sales_fwd
            ly_days_count = max((ly_fwd_end - ly_fwd_start).days, min_denominator)
        else:
            ly_broad_start = max_date - timedelta(days=730)
            ly_broad_end   = max_date - timedelta(days=365)
            ly_sales       = hist_df[(hist_df["TxnDate"] >= ly_broad_start) & (hist_df["TxnDate"] < ly_broad_end)].copy()
            ly_days_count  = max((ly_broad_end - ly_broad_start).days, min_denominator)

        po_df.attrs["ly_source"]        = ly_source_label
        po_df.attrs["ly_window_start"]  = ly_trailing_start
        po_df.attrs["ly_window_end"]    = ly_trailing_end
        po_df.attrs["hist_date_min"]    = hist_df["TxnDate"].min() if not hist_df.empty else pd.NaT
        po_df.attrs["hist_date_max"]    = hist_df["TxnDate"].max() if not hist_df.empty else pd.NaT

        if not ly_sales.empty:
            ly_sold = (ly_sales[ly_sales["Transaction Type"] == "Shipment"]
                       .groupby("Sku")["Quantity"].sum().reset_index())
            ly_sold.columns = ["OMS_SKU", "LY_Sold_Units"]

            ly_net = ly_sales.groupby("Sku")["Units_Effective"].sum().reset_index()
            ly_net.columns = ["OMS_SKU", "LY_Net_Units"]

            ly_summary = ly_sold.merge(ly_net, on="OMS_SKU", how="outer").fillna(0)
            po_df = pd.merge(po_df, ly_summary, on="OMS_SKU", how="left").fillna(0)

            ly_demand = po_df["LY_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["LY_Sold_Units"]
            po_df["LY_ADS"] = (ly_demand / ly_days_count).round(3)

            po_df["ADS"] = np.where(
                po_df["LY_ADS"] > 0,
                (po_df["Recent_ADS"] * (1 - seasonal_weight)) + (po_df["LY_ADS"] * seasonal_weight),
                po_df["Recent_ADS"]
            )
        else:
            po_df["ADS"]    = po_df["Recent_ADS"]
            po_df["LY_ADS"] = 0.0
    else:
        po_df["ADS"]    = po_df["Recent_ADS"]
        po_df["LY_ADS"] = 0

    po_df["Stockout_Flag"] = ""
    po_df.loc[(po_df["ADS"] > 0) & (po_df["Total_Inventory"] <= 0), "Stockout_Flag"] = "⚠️ OOS"
    return po_df

# ══════════════════════════════════════════════════════════════
# 12) SIDEBAR — FILE UPLOADS  (3-Tier: Historical / Monthly / Daily)
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📂 Data Upload")

# ── Always Required ──────────────────────────────────────────
map_file = st.sidebar.file_uploader(
    "1️⃣ SKU Mapping (Required)",
    type=["xlsx"],
    help="Master SKU mapping table. Required before loading any other data."
)

st.sidebar.markdown("### ⚙️ Amazon Settings")
st.session_state.amazon_date_basis    = st.sidebar.selectbox(
    "Date Basis", ["Shipment Date", "Invoice Date", "Order Date"], index=0
)
st.session_state.include_replacements = st.sidebar.checkbox("Include FreeReplacement", value=False)
st.sidebar.divider()

with st.sidebar.expander("📚 Tier 1 — Historical Data (Multi-Year)", expanded=True):
    st.caption(
        "Upload your full archive files here. These power **YoY seasonality** in the PO Engine "
        "and the **MTR Analytics** tab. Upload once — they persist across sessions until you reload."
    )

    mtr_main_zip = st.file_uploader(
        "Amazon MTR — Master ZIP (all months/years)",
        type=["zip"], key="mtr_main_zip",
        help="ZIP containing all monthly Amazon MTR CSVs (B2B + B2C). "
             "Covers 2+ years. Used for tax analytics AND as the LY source for PO seasonality."
    )
    f_meesho = st.file_uploader(
        "Meesho — Master ZIP (all months/years)",
        type=["zip"], key="meesho",
        help="Master ZIP containing all Meesho monthly ZIPs. Both GST and non-GST formats supported."
    )
    f_myntra = st.file_uploader(
        "Myntra PPMP — Master ZIP (all months/years)",
        type=["zip"], key="myntra_sales",
        help="Master ZIP containing all Myntra PPMP monthly CSVs."
    )
    f_flipkart_zip = st.file_uploader(
        "Flipkart — Master ZIP (all months/years)",
        type=["zip"], key="flipkart_zip",
        help="ZIP containing all Flipkart monthly Sales Report Excel files (e.g. DEC-2025.xlsx). "
             "Month is detected from the filename. Each file must have a 'Sales Report' sheet."
    )

st.sidebar.divider()

with st.sidebar.expander("📅 Tier 2 — Monthly Sales (Recent Velocity)", expanded=True):
    st.caption(
        "Upload this month's (or last month's) sales exports. These drive the **Recent ADS** "
        "in the PO Engine and the **Sales Dashboard**. Replace with fresh files each month."
    )

    f_b2c = st.file_uploader(
        "Amazon B2C Sales (ZIP)",
        type=["zip"], key="b2c",
        help="Amazon B2C order-level ZIP export for the current/last month."
    )
    f_b2b = st.file_uploader(
        "Amazon B2B Sales (ZIP)",
        type=["zip"], key="b2b",
        help="Amazon B2B order-level ZIP export for the current/last month."
    )
    f_fk = st.file_uploader(
        "Flipkart Sales (Excel)",
        type=["xlsx"], key="fk",
        help="Flipkart Sales Report Excel (sheet name: 'Sales Report')."
    )
    f_transfer = st.file_uploader(
        "Amazon Stock Transfer (ZIP)",
        type=["zip"], key="transfer",
        help="Amazon inter-FC stock transfer report for the Logistics tab."
    )

st.sidebar.divider()

with st.sidebar.expander("📦 Tier 3 — Daily Snapshot (Inventory + Today's Sales)", expanded=True):
    st.caption(
        "Today's inventory snapshots **and** latest daily sales files. "
        "Daily sales are merged on top of your monthly files — "
        "no double-counting: if a day already exists in monthly data, "
        "the daily file takes priority for that day."
    )

    st.markdown("**📦 Inventory (refresh daily)**")
    i_oms = st.file_uploader(
        "OMS Inventory (CSV)",
        type=["csv"], key="oms",
        help="OMS warehouse inventory export. Columns: Item SkuCode, Inventory."
    )
    i_fk = st.file_uploader(
        "Flipkart Inventory (CSV)",
        type=["csv"], key="fk_inv",
        help="Flipkart listing inventory CSV. Columns: SKU, Live on Website."
    )
    i_myntra = st.file_uploader(
        "Myntra Inventory (CSV)",
        type=["csv"], key="myntra",
        help="Myntra sellable inventory CSV. Columns: Seller SKU Code, Sellable Inventory Count."
    )
    i_amz = st.file_uploader(
        "Amazon Inventory (CSV)",
        type=["csv"], key="amz",
        help="Amazon inventory ledger CSV. Columns: MSKU, Ending Warehouse Balance."
    )

    st.markdown("**🗓️ Daily Orders — PO Check**")
    st.caption(
        "Drop any mix of today's daily reports here — Amazon, Flipkart, Meesho, Myntra. "
        "The platform is detected automatically from the file contents."
    )
    d_daily_files = st.file_uploader(
        "Daily Order Reports (any platform, any order)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="daily_auto",
        help="Accepted formats:\n"
             "• Amazon: CSV with 'Merchant SKU' + 'Amazon Order Id'\n"
             "• Flipkart/Myntra PPMP: CSV with 'seller sku code' + 'store order id'\n"
             "• Meesho: CSV with 'Reason for Credit Entry' + 'Sub Order No'\n"
             "• Myntra Earn More: XLSX with 'SKU ID' + 'Final Sale Units' + 'Gross Units'"
    )

st.sidebar.divider()

# ── Data Coverage Summary + RAM monitor ──────────────────────
def _get_ram_mb() -> float:
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    except Exception:
        return 0.0


def _df_mb(df: pd.DataFrame) -> float:
    try:
        return df.memory_usage(deep=True).sum() / 1_048_576
    except Exception:
        return 0.0


def _show_data_coverage():
    rows = []
    ss   = st.session_state

    def _date_range(df, col="TxnDate"):
        try:
            d = pd.to_datetime(df[col], errors="coerce").dropna()
            return f"{d.min().strftime('%b %y')} → {d.max().strftime('%b %y')}" if len(d) else "no dates"
        except Exception:
            return "loaded"

    total_mb = 0.0
    if not ss.mtr_df.empty:
        mb = _df_mb(ss.mtr_df); total_mb += mb
        rows.append(f"📑 MTR: {len(ss.mtr_df):,} rows | {_date_range(ss.mtr_df, 'Date')} | {mb:.0f} MB")
    if not ss.sales_df.empty:
        mb = _df_mb(ss.sales_df); total_mb += mb
        rows.append(f"📊 Sales: {len(ss.sales_df):,} rows | {_date_range(ss.sales_df)} | {mb:.0f} MB")
    meesho_df = ss.get("meesho_df", pd.DataFrame())
    if not meesho_df.empty:
        mb = _df_mb(meesho_df); total_mb += mb
        rows.append(f"🛒 Meesho: {len(meesho_df):,} rows | {_date_range(meesho_df, 'Date')} | {mb:.0f} MB")
    myntra_df = ss.get("myntra_df", pd.DataFrame())
    if not myntra_df.empty:
        mb = _df_mb(myntra_df); total_mb += mb
        rows.append(f"🛍️ Myntra: {len(myntra_df):,} rows | {_date_range(myntra_df, 'Date')} | {mb:.0f} MB")
    if not ss.inventory_df_variant.empty:
        mb = _df_mb(ss.inventory_df_variant); total_mb += mb
        rows.append(f"📦 Inventory: {len(ss.inventory_df_variant):,} SKUs | {mb:.0f} MB")

    if rows:
        ram = _get_ram_mb()
        ram_color = "🟢" if ram < 400 else "🟡" if ram < 700 else "🔴"
        st.sidebar.markdown(f"**📊 Loaded Data** {ram_color} `{ram:.0f} MB` process RAM")
        for r in rows:
            st.sidebar.caption(r)
        st.sidebar.caption(f"DataFrame total: {total_mb:.0f} MB")

        daily_sources = ss.get("daily_sales_sources", [])
        daily_rows    = ss.get("daily_sales_rows", 0)
        if daily_sources:
            st.sidebar.caption(
                f"🗓️ Daily merged: **{', '.join(daily_sources)}** "
                f"({daily_rows:,} rows — overlapping dates replaced)"
            )

        if st.sidebar.button("🗑️ Clear All Data (Free RAM)", use_container_width=True):
            for key in ["mtr_df", "sales_df", "meesho_df", "myntra_df",
                        "inventory_df_variant", "inventory_df_parent",
                        "transfer_df", "sku_mapping",
                        "daily_sales_sources", "daily_sales_rows"]:
                if key in st.session_state:
                    del st.session_state[key]
            gc.collect()
            st.rerun()

if st.sidebar.button("🚀 Load All Data", use_container_width=True):
    if not map_file:
        st.sidebar.error("SKU Mapping required!")
    else:
        _load_error = None
        with st.spinner("Loading data…"):
            try:
                st.session_state.sku_mapping = load_sku_mapping(map_file)
                config = SalesConfig(
                    date_basis=st.session_state.amazon_date_basis,
                    include_replacements=st.session_state.include_replacements
                )

                if mtr_main_zip:
                    mtr_combined, csv_count, mtr_skipped = load_mtr_from_main_zip(mtr_main_zip)
                    st.session_state.mtr_df = mtr_combined
                    del mtr_combined
                    gc.collect()
                    if mtr_skipped:
                        with st.sidebar.expander(f"⚠️ MTR: {len(mtr_skipped)} files had issues"):
                            for s in mtr_skipped:
                                st.write(s)

                if f_meesho:
                    meesho_combined = load_meesho_full(f_meesho)
                    st.session_state.meesho_df = meesho_combined
                    del meesho_combined
                    gc.collect()

                if f_myntra:
                    myntra_combined = load_myntra_full(f_myntra, st.session_state.sku_mapping)
                    st.session_state.myntra_df = myntra_combined
                    del myntra_combined
                    gc.collect()

                if f_flipkart_zip:
                    fk_combined = load_flipkart_full(f_flipkart_zip, st.session_state.sku_mapping)
                    st.session_state.flipkart_df = fk_combined
                    del fk_combined
                    gc.collect()

                sales_parts = []
                if f_b2c:
                    sales_parts.append(load_amazon_sales(f_b2c, st.session_state.sku_mapping, "Amazon B2C", config))
                    gc.collect()
                if f_b2b:
                    sales_parts.append(load_amazon_sales(f_b2b, st.session_state.sku_mapping, "Amazon B2B", config))
                    gc.collect()
                if f_fk:
                    sales_parts.append(load_flipkart_sales(f_fk, st.session_state.sku_mapping))
                    gc.collect()

                meesho_df_ss  = st.session_state.get("meesho_df",   pd.DataFrame())
                myntra_df_ss  = st.session_state.get("myntra_df",   pd.DataFrame())
                flipkart_df_ss = st.session_state.get("flipkart_df", pd.DataFrame())
                if not meesho_df_ss.empty:
                    sales_parts.append(meesho_to_sales_rows(meesho_df_ss))
                if not myntra_df_ss.empty:
                    sales_parts.append(myntra_to_sales_rows(myntra_df_ss))
                if not flipkart_df_ss.empty:
                    sales_parts.append(flipkart_to_sales_rows(flipkart_df_ss))

                if sales_parts:
                    combined_sales = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)
                    combined_sales = _downcast_sales(combined_sales)
                    st.session_state.sales_df = combined_sales
                    del sales_parts, combined_sales
                    gc.collect()

                # ── Daily orders: auto-detect platform from file contents ──
                _daily_parts   = []
                _daily_detected = {}   # fname → detected platform (for UI feedback)
                for _f in (d_daily_files or []):
                    _platform = detect_daily_platform(_f)
                    _daily_detected[_f.name] = _platform
                    try:
                        if _platform == "amazon":
                            _d = parse_daily_amazon_csv(_f, st.session_state.sku_mapping)
                        elif _platform == "flipkart":
                            _d = parse_daily_flipkart_csv(_f, st.session_state.sku_mapping)
                        elif _platform == "meesho":
                            _d = parse_daily_meesho_csv(_f, st.session_state.sku_mapping)
                        elif _platform == "myntra":
                            _d = parse_daily_myntra_xlsx(_f, st.session_state.sku_mapping)
                        else:
                            st.sidebar.warning(f"⚠️ Could not detect platform for **{_f.name}** — skipped.")
                            continue
                        if not _d.empty:
                            _daily_parts.append(_d)
                    except Exception as _pe:
                        st.sidebar.warning(f"⚠️ Error parsing {_f.name}: {_pe}")
                    gc.collect()

                if _daily_detected:
                    _icons = {"amazon":"🟠","flipkart":"🟡","meesho":"🔴","myntra":"🟣","unknown":"⚪"}
                    _lines = [f"{_icons.get(p,'⚪')} **{fn}** → {p.title()}"
                              for fn, p in _daily_detected.items()]
                    st.session_state["daily_detect_log"] = _lines

                if _daily_parts:
                    _daily_combined = combine_daily_orders(*_daily_parts)
                    st.session_state.daily_orders_df = _daily_combined

                    # Also feed into sales_df for PO Engine velocity calculation
                    _daily_sales = daily_to_sales_rows(_daily_combined)
                    if not _daily_sales.empty:
                        base = st.session_state.sales_df
                        base = merge_daily_into_sales(base, _daily_sales)
                        st.session_state.sales_df = base
                        st.session_state.daily_sales_sources = _daily_combined["Platform"].unique().tolist()
                        st.session_state.daily_sales_rows    = len(_daily_combined)
                        del base
                    del _daily_parts, _daily_combined
                    gc.collect()

                st.session_state.inventory_df_variant = load_inventory_consolidated(
                    i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=False
                )
                st.session_state.inventory_df_parent = load_inventory_consolidated(
                    i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=True
                )
                gc.collect()

                if f_transfer:
                    st.session_state.transfer_df = load_stock_transfer(f_transfer)

            except Exception as _load_err:
                import traceback as _tb
                _load_error = _tb.format_exc()
                st.sidebar.error(f"❌ Load failed: {_load_err}")

        if _load_error:
            st.error("**Loading failed — full traceback:**")
            st.code(_load_error)
        else:
            st.rerun()

_show_data_coverage()

if not st.session_state.sku_mapping:
    st.info("👋 **Welcome!** Upload SKU Mapping and click **Load All Data** to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# 14) MAIN TABS
# ══════════════════════════════════════════════════════════════
tab_dash, tab_mtr, tab_myntra, tab_meesho, tab_flipkart, tab_daily, tab_inv, tab_po, tab_logistics, tab_forecast, tab_drill = st.tabs([
    "📊 Dashboard", "📑 MTR Analytics", "🛍️ Myntra", "🛒 Meesho", "🟡 Flipkart",
    "📋 Daily Orders", "📦 Inventory", "🎯 PO Engine", "🚚 Logistics", "📈 AI Forecast", "🔍 Deep Dive",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("📊 Sales Analytics Dashboard")
    df = st.session_state.sales_df
    if df.empty:
        st.warning("⚠️ No sales data loaded. Upload sales files and click Load Data.")
    else:
        col_period, col_grace = st.columns([3, 1])
        with col_period:
            period_option = st.selectbox("Analysis Period", ["Last 7 Days","Last 30 Days","Last 60 Days","Last 90 Days","All Time"], index=1, key="dash_period")
        with col_grace:
            grace_days = st.number_input("Grace Period (Days)", 0, 14, 7, help="Extends the window back to properly capture both lagging sales and subsequent refunds.")

        df = df.copy()
        df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
        max_date = df["TxnDate"].max()

        if period_option == "All Time":
            filtered_df     = df
            total_days      = (max_date - df["TxnDate"].min()).days
            date_range_text = f"All Time: {df['TxnDate'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            base_days   = 7 if "7" in period_option else 30 if "30" in period_option else 60 if "60" in period_option else 90
            total_days  = base_days + grace_days
            filtered_df = df[df["TxnDate"] >= (max_date - timedelta(days=total_days))]
            date_range_text = f"Period: {filtered_df['TxnDate'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({base_days} Days + {grace_days} Grace Days)"

        st.info(f"📅 **{date_range_text}** | Transactions: {len(filtered_df):,}")
        st.caption(f"*(Note: Both Sales **and Returns** are calculated tightly within this window to ensure metric parity)*")

        filtered_df["Quantity"]        = pd.to_numeric(filtered_df["Quantity"], errors="coerce").fillna(0)
        filtered_df["Units_Effective"] = pd.to_numeric(filtered_df["Units_Effective"], errors="coerce").fillna(0)

        sold_pcs    = filtered_df[filtered_df["Transaction Type"]=="Shipment"]["Quantity"].sum()
        ret_pcs     = filtered_df[filtered_df["Transaction Type"]=="Refund"]["Quantity"].sum()
        net_units   = filtered_df["Units_Effective"].sum()
        orders      = (filtered_df[filtered_df["Transaction Type"]=="Shipment"]["OrderId"].nunique()
                       if "OrderId" in filtered_df.columns
                       else len(filtered_df[filtered_df["Transaction Type"]=="Shipment"]))
        return_rate = (ret_pcs / sold_pcs * 100) if sold_pcs > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛒 Orders",      f"{orders:,}")
        c2.metric("✅ Sold Pieces", f"{int(sold_pcs):,}")
        c3.metric("↩️ Returns",     f"{int(ret_pcs):,}")
        c4.metric("📊 Return Rate", f"{return_rate:.1f}%")
        c5.metric("📦 Net Units",   f"{int(net_units):,}")
        st.divider()

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### 🏆 Top 20 Selling SKUs")
            top = (filtered_df[filtered_df["Transaction Type"]=="Shipment"]
                   .groupby("Sku")["Quantity"].sum()
                   .sort_values(ascending=False).head(20).reset_index())
            fig = px.bar(top, x="Sku", y="Quantity", title="Top Sellers (Pieces)")
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.markdown("### 📊 Marketplace Split")
            source_summary = filtered_df.groupby("Source")["Quantity"].sum().reset_index()
            fig = px.pie(source_summary, values="Quantity", names="Source", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — MTR ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_mtr:
    st.subheader("📑 MTR Analytics — Amazon Tax Report")
    mtr = st.session_state.mtr_df

    if mtr.empty:
        st.info("📂 **No MTR data loaded yet.**")
    else:
        mtr_years = sorted(mtr["Date"].dt.year.dropna().unique().tolist())
        st.success(f"✅ MTR data covers years: **{', '.join(str(y) for y in mtr_years)}** | "
                   f"Total rows: **{len(mtr):,}** | "
                   f"Date range: **{mtr['Date'].min().strftime('%d %b %Y')}** → **{mtr['Date'].max().strftime('%d %b %Y')}**")

        with st.expander("🔧 Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            all_months = sorted(mtr["Month"].dropna().unique().tolist())
            with fc1:
                sel_months = st.multiselect("Months", all_months, default=all_months, key="mtr_months")
            with fc2:
                sel_rtype = st.multiselect("Report Type", ["B2B","B2C"], default=["B2B","B2C"], key="mtr_rtype")
            with fc3:
                avail_txn = sorted(mtr["Transaction_Type"].dropna().unique().tolist())
                default_txn = [t for t in ["Shipment","Refund"] if t in avail_txn]
                sel_txn = st.multiselect("Transaction Types", avail_txn, default=default_txn, key="mtr_txn")

        mtr_copy = mtr.copy()
        mtr_copy["Month"] = mtr_copy["Month"].astype(str)
        sel_months_str = [str(m) for m in sel_months]

        mf = mtr_copy[
            mtr_copy["Month"].isin(sel_months_str) &
            mtr_copy["Report_Type"].isin(sel_rtype) &
            mtr_copy["Transaction_Type"].isin(sel_txn)
        ].copy()

        if mf.empty:
            st.warning("No data for selected filters.")
        else:
            shipped  = mf["Transaction_Type"] == "Shipment"
            refunded = mf["Transaction_Type"] == "Refund"

            gross_rev  = mf.loc[shipped,  "Invoice_Amount"].sum()
            refund_amt = mf.loc[refunded, "Invoice_Amount"].abs().sum()
            net_rev    = gross_rev - refund_amt
            total_tax  = mf.loc[shipped,  "Total_Tax"].sum()
            units_sold = mf.loc[shipped,  "Quantity"].sum()
            order_cnt  = mf.loc[shipped,  "Order_Id"].nunique()
            aov        = gross_rev / order_cnt if order_cnt else 0

            st.markdown("### 💰 Revenue KPIs")
            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            k1.metric("💵 Gross Rev",     fmt_inr(gross_rev))
            k2.metric("↩️ Refunds",       fmt_inr(refund_amt))
            k3.metric("✅ Net Rev",       fmt_inr(net_rev))
            k4.metric("🏛️ Tax Collected", fmt_inr(total_tax))
            k5.metric("📦 Units Sold",    f"{int(units_sold):,}")
            k6.metric("🛒 Orders",        f"{order_cnt:,}")
            k7.metric("💳 AOV",           fmt_inr(aov))

            st.divider()

            _sh = mf["Transaction_Type"] == "Shipment"
            _rf = mf["Transaction_Type"] == "Refund"

            _comp_rows = []
            for rt in ["B2B", "B2C"]:
                _s  = mf[mf["Report_Type"] == rt]
                _ss = _s["Transaction_Type"] == "Shipment"
                _sr = _s["Transaction_Type"] == "Refund"
                _gr  = float(_s.loc[_ss, "Invoice_Amount"].sum())
                _ref = float(_s.loc[_sr, "Invoice_Amount"].abs().sum())
                _ord = int(_s.loc[_ss, "Order_Id"].nunique())
                _us  = float(_s.loc[_ss, "Quantity"].sum())
                _comp_rows.append({
                    "Type":          rt,
                    "Gross Revenue": fmt_inr(_gr),
                    "Refunds":       fmt_inr(_ref),
                    "Net Revenue":   fmt_inr(_gr - _ref),
                    "Tax":           fmt_inr(float(_s.loc[_ss, "Total_Tax"].sum())),
                    "Orders":        f"{_ord:,}",
                    "Units Sold":    f"{int(_us):,}",
                    "AOV":           fmt_inr(_gr / _ord) if _ord else "₹0",
                })
            _comp_df = pd.DataFrame(_comp_rows).set_index("Type")

            _monthly = (mf[_sh].groupby(["Month","Report_Type"])["Invoice_Amount"].sum()
                        .reset_index().sort_values("Month"))
            _monthly.columns = ["Month","Report_Type","Gross_Revenue"]

            _monthly_ref = (mf[_rf].groupby(["Month","Report_Type"])["Invoice_Amount"].sum().abs().reset_index())
            _monthly_ref.columns = ["Month","Report_Type","Refund_Amt"]

            _monthly_comb = _monthly.merge(_monthly_ref, on=["Month","Report_Type"], how="left")
            _monthly_comb["Refund_Amt"] = _monthly_comb["Refund_Amt"].fillna(0)
            _monthly_comb["Refund_%"]   = (
                _monthly_comb["Refund_Amt"] / _monthly_comb["Gross_Revenue"].replace(0, np.nan) * 100
            ).fillna(0.0).round(2)

            _state_rev = (mf[_sh].groupby("Ship_To_State")["Invoice_Amount"].sum()
                          .sort_values(ascending=False).head(20).reset_index())
            _state_rev.columns = ["State","Revenue"]

            _top12 = mf[_sh].groupby("Ship_To_State")["Invoice_Amount"].sum().nlargest(12).index.tolist()
            _heat  = (mf[_sh & mf["Ship_To_State"].isin(_top12)]
                      .groupby(["Ship_To_State","Month"])["Invoice_Amount"].sum()
                      .reset_index()
                      .pivot(index="Ship_To_State", columns="Month", values="Invoice_Amount")
                      .fillna(0))

            _pm_rev   = (mf[_sh].groupby(["Payment_Method","Report_Type"])["Invoice_Amount"].sum().reset_index())
            _pm_units = (mf[_sh].groupby("Payment_Method")["Quantity"].sum()
                         .sort_values(ascending=False).head(10).reset_index())
            _pm_units.columns = ["Method","Units"]

            _txn_rev = mf.groupby(["Transaction_Type","Report_Type"])["Invoice_Amount"].sum().reset_index()
            _sku_rev = (mf[_sh].groupby(["SKU","Report_Type"])["Invoice_Amount"].sum()
                        .reset_index().sort_values("Invoice_Amount", ascending=False).head(20))
            _wh_rev  = (mf[_sh].groupby(["Warehouse_Id","Report_Type"])["Invoice_Amount"].sum()
                        .reset_index().sort_values("Invoice_Amount", ascending=False))

            with st.expander("🔀 B2B vs B2C Comparison", expanded=True):
                st.dataframe(_comp_df, use_container_width=True)

            with st.expander("📈 Monthly Revenue Trend", expanded=False):
                fig = px.line(_monthly, x="Month", y="Gross_Revenue", color="Report_Type", markers=True,
                              color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                              title="Monthly Gross Revenue")
                fig.update_layout(hovermode="x unified", height=360)
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(_monthly_comb, x="Month", y="Refund_%", color="Report_Type", barmode="group",
                              color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                              title="Monthly Refund %")
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("🗺️ Top 20 States by Revenue", expanded=False):
                fig3 = px.bar(_state_rev, x="Revenue", y="State", orientation="h",
                              color="Revenue", color_continuous_scale="Blues",
                              title="Top 20 States by Revenue")
                fig3.update_layout(height=520, yaxis=dict(autorange="reversed"))
                fig3.update_xaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("🔥 State Revenue Heatmap", expanded=False):
                if not _heat.empty:
                    fig4 = px.imshow(_heat / 1000, color_continuous_scale="YlOrRd",
                                     labels=dict(color="Revenue (₹K)"),
                                     title="Revenue Heatmap (₹ Thousands)", aspect="auto")
                    fig4.update_layout(height=380)
                    st.plotly_chart(fig4, use_container_width=True)

            with st.expander("💳 Payment Methods", expanded=False):
                pm1, pm2 = st.columns(2)
                with pm1:
                    fig5 = px.bar(_pm_rev, x="Payment_Method", y="Invoice_Amount", color="Report_Type",
                                  barmode="group", color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                                  title="Payment Methods by Revenue")
                    fig5.update_xaxes(tickangle=-30)
                    fig5.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig5, use_container_width=True)
                with pm2:
                    fig6 = px.pie(_pm_units, values="Units", names="Method",
                                  title="Payment Split (Units)", hole=0.4)
                    fig6.update_layout(height=300)
                    st.plotly_chart(fig6, use_container_width=True)

            with st.expander("📋 Transactions / Top SKUs / Warehouse", expanded=False):
                fig7 = px.bar(_txn_rev, x="Transaction_Type", y="Invoice_Amount", color="Report_Type",
                              barmode="group", color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                              title="Revenue by Transaction Type")
                fig7.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig7, use_container_width=True)

                fig8 = px.bar(_sku_rev, x="SKU", y="Invoice_Amount", color="Report_Type",
                              color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                              title="Top 20 SKUs by Revenue")
                fig8.update_xaxes(tickangle=-45)
                fig8.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig8, use_container_width=True)

                fig9 = px.bar(_wh_rev, x="Warehouse_Id", y="Invoice_Amount", color="Report_Type",
                              barmode="group", color_discrete_map={"B2B":"#002B5B","B2C":"#E63946"},
                              title="Revenue by Warehouse / FC")
                fig9.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig9, use_container_width=True)

            with st.expander("🔍 Raw Data Viewer & Downloads", expanded=False):
                _show_cols = [c for c in ["Date","Report_Type","Transaction_Type","SKU","Quantity",
                                           "Invoice_Amount","Total_Tax","Ship_To_State","Payment_Method",
                                           "Warehouse_Id","Order_Id","Invoice_Number","Buyer_Name",
                                           "IRN_Status","Month"] if c in mtr.columns]
                _view = mf[_show_cols]
                search_sku = st.text_input("Search SKU / Buyer Name / Invoice No", key="mtr_search")
                if search_sku:
                    _q = search_sku.lower()
                    _view = _view[
                        _view["SKU"].str.lower().str.contains(_q, na=False) |
                        _view["Buyer_Name"].str.lower().str.contains(_q, na=False) |
                        _view["Invoice_Number"].str.lower().str.contains(_q, na=False)
                    ]

                st.dataframe(_view.sort_values("Date", ascending=False).head(300),
                             use_container_width=True, height=380)
                st.caption(f"Showing up to 300 of {len(_view):,} records")

                st.markdown("#### 📥 Downloads")
                _dl1, _dl2 = st.columns(2)
                with _dl1:
                    st.download_button(
                        "📥 Filtered Data (CSV)",
                        _view.to_csv(index=False).encode("utf-8"),
                        f"mtr_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True
                    )
                with _dl2:
                    st.download_button(
                        "📥 Monthly Summary (CSV)",
                        _monthly_comb.to_csv(index=False).encode("utf-8"),
                        f"mtr_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True
                    )

# ══════════════════════════════════════════════════════════════
# TAB 3 — MYNTRA ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_myntra:
    st.subheader("🛍️ Myntra PPMP Analytics")
    myn = st.session_state.myntra_df
    if myn.empty:
        st.info("📂 Upload Myntra PPMP master ZIP in the Sales Data section and click Load All Data.")
    else:
        myn_years = sorted(myn["Date"].dt.year.dropna().unique().tolist())
        st.success(
            f"✅ Myntra data covers: **{', '.join(str(y) for y in myn_years)}** | "
            f"Rows: **{len(myn):,}** | "
            f"Range: **{myn['Date'].min().strftime('%d %b %Y')}** → **{myn['Date'].max().strftime('%d %b %Y')}**"
        )

        with st.expander("🔧 Filters", expanded=True):
            mf1, mf2, mf3 = st.columns(3)
            _myn_months = sorted(myn["Month"].dropna().unique().tolist())
            with mf1:
                _sel_myn_months = st.multiselect("Months", _myn_months, default=_myn_months, key="myn_months")
            with mf2:
                _myn_txn_types  = sorted(myn["TxnType"].dropna().unique().tolist())
                _sel_myn_txn    = st.multiselect("Transaction Type", _myn_txn_types,
                                                  default=[t for t in ["Shipment","Refund"] if t in _myn_txn_types], key="myn_txn")
            with mf3:
                _myn_states = sorted(myn["State"].dropna().unique().tolist())
                _sel_states = st.multiselect("State", _myn_states, default=_myn_states, key="myn_states")

        mf = myn[
            myn["Month"].isin(_sel_myn_months) &
            myn["TxnType"].isin(_sel_myn_txn) &
            myn["State"].isin(_sel_states)
        ].copy()

        if mf.empty:
            st.warning("No data for selected filters.")
        else:
            _myn_sh = mf["TxnType"] == "Shipment"
            _myn_rf = mf["TxnType"] == "Refund"

            gross_rev  = mf.loc[_myn_sh, "Invoice_Amount"].sum()
            refund_amt = mf.loc[_myn_rf, "Invoice_Amount"].abs().sum()
            net_rev    = gross_rev - refund_amt
            units_sold = mf.loc[_myn_sh, "Quantity"].sum()
            orders     = mf.loc[_myn_sh, "OrderId"].nunique()
            ret_units  = mf.loc[_myn_rf, "Quantity"].sum()
            ret_rate   = (ret_units / units_sold * 100) if units_sold > 0 else 0
            aov        = gross_rev / orders if orders else 0

            st.markdown("### 💰 KPIs")
            k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
            k1.metric("💵 Gross Rev",   fmt_inr(gross_rev))
            k2.metric("↩️ Refunds",     fmt_inr(refund_amt))
            k3.metric("✅ Net Rev",     fmt_inr(net_rev))
            k4.metric("📦 Units Sold",  f"{int(units_sold):,}")
            k5.metric("↩️ Returns",     f"{int(ret_units):,}")
            k6.metric("📊 Return Rate", f"{ret_rate:.1f}%")
            k7.metric("💳 AOV",         fmt_inr(aov))
            st.divider()

            _myn_monthly = (mf[_myn_sh].groupby("Month")["Invoice_Amount"].sum().reset_index().sort_values("Month"))
            _myn_monthly_ref = (mf[_myn_rf].groupby("Month")["Invoice_Amount"].sum().abs().reset_index())
            _myn_monthly_ref.columns = ["Month","Refund_Amt"]
            _myn_monthly_comb = _myn_monthly.merge(_myn_monthly_ref, on="Month", how="left").fillna(0)
            _myn_monthly_comb["Refund_%"] = (
                _myn_monthly_comb["Refund_Amt"] / _myn_monthly_comb["Invoice_Amount"].replace(0,np.nan) * 100
            ).fillna(0).round(2)

            with st.expander("📈 Monthly Revenue Trend", expanded=True):
                fig = px.bar(_myn_monthly_comb, x="Month", y="Invoice_Amount",
                             title="Monthly Gross Revenue — Myntra", color_discrete_sequence=["#E63946"])
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(_myn_monthly_comb, x="Month", y="Refund_%",
                              title="Monthly Refund % — Myntra", color_discrete_sequence=["#F4A261"])
                fig2.update_layout(height=280)
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("🗺️ Top States by Revenue", expanded=False):
                _myn_state = (mf[_myn_sh].groupby("State")["Invoice_Amount"].sum()
                              .sort_values(ascending=False).head(20).reset_index())
                _myn_state.columns = ["State","Revenue"]
                fig3 = px.bar(_myn_state, x="Revenue", y="State", orientation="h",
                              color="Revenue", color_continuous_scale="Reds",
                              title="Top 20 States — Myntra")
                fig3.update_layout(height=520, yaxis=dict(autorange="reversed"))
                fig3.update_xaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("🏆 Top SKUs by Revenue", expanded=False):
                _myn_sku = (mf[_myn_sh].groupby("OMS_SKU")["Invoice_Amount"].sum()
                            .sort_values(ascending=False).head(30).reset_index())
                fig4 = px.bar(_myn_sku, x="OMS_SKU", y="Invoice_Amount",
                              title="Top 30 SKUs — Myntra Revenue")
                fig4.update_xaxes(tickangle=-45)
                fig4.update_yaxes(tickprefix="₹", tickformat=",.0f")
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)

            with st.expander("💳 Payment Methods", expanded=False):
                _myn_pm = (mf[_myn_sh].groupby("Payment_Method")["Invoice_Amount"].sum()
                           .sort_values(ascending=False).reset_index())
                fig5 = px.pie(_myn_pm, values="Invoice_Amount", names="Payment_Method",
                              title="Payment Split — Myntra", hole=0.4)
                st.plotly_chart(fig5, use_container_width=True)

            with st.expander("🔍 Raw Data & Download", expanded=False):
                _myn_search = st.text_input("Search SKU / State", key="myn_search")
                _myn_view = mf.copy()
                if _myn_search:
                    _q = _myn_search.lower()
                    _myn_view = _myn_view[
                        _myn_view["OMS_SKU"].str.lower().str.contains(_q, na=False) |
                        _myn_view["State"].str.lower().str.contains(_q, na=False)
                    ]
                st.dataframe(_myn_view.sort_values("Date", ascending=False).head(300),
                             use_container_width=True, height=360)
                st.caption(f"Showing up to 300 of {len(_myn_view):,} records")
                st.download_button("📥 Download Filtered (CSV)",
                    _myn_view.to_csv(index=False).encode("utf-8"),
                    f"myntra_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — MEESHO ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_meesho:
    st.subheader("🛒 Meesho Analytics")
    mee = st.session_state.meesho_df
    if mee.empty:
        st.info("📂 Upload Meesho master ZIP in the Sales Data section and click Load All Data.")
    else:
        mee_years = sorted(mee["Date"].dt.year.dropna().unique().tolist())
        st.success(
            f"✅ Meesho data covers: **{', '.join(str(y) for y in mee_years)}** | "
            f"Rows: **{len(mee):,}** | "
            f"Range: **{mee['Date'].min().strftime('%d %b %Y')}** → **{mee['Date'].max().strftime('%d %b %Y')}**"
        )
        st.caption(
            "ℹ️ Meesho does not provide per-product SKU data. Analytics shown are store-level totals. "
            "Month bucketing uses Meesho's own `month_number` field (GST format) so sales and returns "
            "always land in the exact month Meesho intends, matching the numbers shown in Meesho portal."
        )

        # Simple 2-column filter — no grace period needed for GST format because
        # Month is now sourced from financial_year+month_number (always correct).
        # For non-GST (ForwardReports) the date-derived Month is used as before.
        with st.expander("🔧 Filters", expanded=True):
            _mee_months_all = sorted(mee["Month"].dropna().astype(str).unique().tolist())
            _mee_txn_types  = sorted(mee["TxnType"].dropna().unique().tolist())

            ef1, ef2 = st.columns(2)
            with ef1:
                _sel_mee_months = st.multiselect(
                    "Months", _mee_months_all,
                    default=_mee_months_all, key="mee_months"
                )
            with ef2:
                _sel_mee_txn = st.multiselect(
                    "Transaction Type", _mee_txn_types,
                    default=[t for t in ["Shipment","Refund"] if t in _mee_txn_types],
                    key="mee_txn"
                )

        # Straight Month filter — correct because parser now uses month_number
        if _sel_mee_months:
            ef = mee[
                mee["Month"].astype(str).isin(_sel_mee_months) &
                mee["TxnType"].isin(_sel_mee_txn)
            ].copy()
        else:
            ef = pd.DataFrame()

        if ef.empty:
            st.warning("No data for selected filters.")
        else:
            _mee_sh = ef["TxnType"] == "Shipment"
            _mee_rf = ef["TxnType"] == "Refund"

            gross_rev  = ef.loc[_mee_sh, "Invoice_Amount"].sum()
            refund_amt = ef.loc[_mee_rf, "Invoice_Amount"].abs().sum()
            net_rev    = gross_rev - refund_amt
            units_sold = ef.loc[_mee_sh, "Quantity"].sum()
            orders     = ef.loc[_mee_sh, "OrderId"].nunique()
            ret_units  = ef.loc[_mee_rf, "Quantity"].sum()
            ret_rate   = (ret_units / units_sold * 100) if units_sold > 0 else 0
            aov        = gross_rev / orders if orders else 0

            st.markdown("### 💰 KPIs")
            k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
            k1.metric("💵 Gross Rev",   fmt_inr(gross_rev))
            k2.metric("↩️ Refunds",     fmt_inr(refund_amt))
            k3.metric("✅ Net Rev",     fmt_inr(net_rev))
            k4.metric("📦 Units Sold",  f"{int(units_sold):,}")
            k5.metric("↩️ Returns",     f"{int(ret_units):,}")
            k6.metric("📊 Return Rate", f"{ret_rate:.1f}%")
            k7.metric("💳 AOV",         fmt_inr(aov))
            st.divider()

            # Monthly aggregation uses the transaction's own Month for bucketing
            _mee_monthly = (ef[_mee_sh].groupby("Month")["Invoice_Amount"].sum().reset_index().sort_values("Month"))
            _mee_monthly_ref = (ef[_mee_rf].groupby("Month")["Invoice_Amount"].sum().abs().reset_index())
            _mee_monthly_ref.columns = ["Month","Refund_Amt"]
            _mee_monthly_comb = _mee_monthly.merge(_mee_monthly_ref, on="Month", how="left").fillna(0)
            _mee_monthly_comb["Refund_%"] = (
                _mee_monthly_comb["Refund_Amt"] / _mee_monthly_comb["Invoice_Amount"].replace(0,np.nan) * 100
            ).fillna(0).round(2)
            _mee_units_monthly = (ef[_mee_sh].groupby("Month")["Quantity"].sum().reset_index().sort_values("Month"))

            with st.expander("📈 Monthly Revenue & Units Trend", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.bar(_mee_monthly_comb, x="Month", y="Invoice_Amount",
                                 title="Monthly Gross Revenue — Meesho",
                                 color_discrete_sequence=["#7B2D8B"])
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig2 = px.bar(_mee_units_monthly, x="Month", y="Quantity",
                                  title="Monthly Units Sold — Meesho",
                                  color_discrete_sequence=["#A855F7"])
                    fig2.update_layout(height=320)
                    st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.bar(_mee_monthly_comb, x="Month", y="Refund_%",
                              title="Monthly Refund % — Meesho",
                              color_discrete_sequence=["#F4A261"])
                fig3.update_layout(height=260)
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("🗺️ Top States by Revenue", expanded=False):
                _mee_state = (ef[_mee_sh].groupby("State")["Invoice_Amount"].sum()
                              .sort_values(ascending=False).head(20).reset_index())
                _mee_state.columns = ["State","Revenue"]
                if not _mee_state.empty and _mee_state["State"].str.strip().ne("").any():
                    fig4 = px.bar(_mee_state, x="Revenue", y="State", orientation="h",
                                  color="Revenue", color_continuous_scale="Purples",
                                  title="Top 20 States — Meesho")
                    fig4.update_layout(height=520, yaxis=dict(autorange="reversed"))
                    fig4.update_xaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("State data not available in this Meesho format.")

            with st.expander("🔍 Raw Data & Download", expanded=False):
                st.dataframe(ef.sort_values("Date", ascending=False).head(300),
                             use_container_width=True, height=360)
                st.caption(f"Showing up to 300 of {len(ef):,} records")
                st.download_button("📥 Download Filtered (CSV)",
                    ef.to_csv(index=False).encode("utf-8"),
                    f"meesho_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — FLIPKART ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_flipkart:
    st.subheader("🟡 Flipkart Analytics")
    fk = st.session_state.flipkart_df
    if fk.empty:
        st.info("📂 Upload the Flipkart master ZIP in the **Tier 1 — Historical Data** section of the sidebar and click **Load All Data**.")
    else:
        fk_years = sorted(fk["Date"].dt.year.dropna().unique().tolist())
        st.success(
            f"✅ Flipkart data covers: **{', '.join(str(y) for y in fk_years)}** | "
            f"Rows: **{len(fk):,}** | "
            f"Range: **{fk['Date'].min().strftime('%d %b %Y')}** → **{fk['Date'].max().strftime('%d %b %Y')}**"
        )
        st.caption(
            "ℹ️ Month bucketing uses **Buyer Invoice Date** — Flipkart guarantees all rows in a monthly "
            "file carry an invoice date within that calendar month, so numbers always match the Flipkart portal exactly."
        )

        with st.expander("🔧 Filters", expanded=True):
            _fk_months_all = sorted(fk["Month"].dropna().astype(str).unique().tolist())
            _fk_txn_types  = sorted(fk["TxnType"].dropna().unique().tolist())
            _fk_states_all = sorted(fk["State"].dropna().unique().tolist())
            _fk_states_all = [s for s in _fk_states_all if s]  # remove empty

            ff1, ff2, ff3 = st.columns(3)
            with ff1:
                _sel_fk_months = st.multiselect(
                    "Months", _fk_months_all,
                    default=_fk_months_all, key="fk_months"
                )
            with ff2:
                _sel_fk_txn = st.multiselect(
                    "Transaction Type", _fk_txn_types,
                    default=[t for t in ["Shipment","Refund","ReturnCancel"] if t in _fk_txn_types],
                    key="fk_txn"
                )
            with ff3:
                _sel_fk_states = st.multiselect(
                    "State", _fk_states_all,
                    default=_fk_states_all, key="fk_states"
                )

        # Apply filters
        fkf = fk[
            fk["Month"].astype(str).isin(_sel_fk_months) &
            fk["TxnType"].isin(_sel_fk_txn)
        ].copy()
        if _sel_fk_states:
            fkf = fkf[fkf["State"].isin(_sel_fk_states) | (fkf["State"] == "")]

        if fkf.empty:
            st.warning("No data for selected filters.")
        else:
            _fk_sh  = fkf["TxnType"] == "Shipment"
            _fk_rf  = fkf["TxnType"] == "Refund"
            _fk_rc  = fkf["TxnType"] == "ReturnCancel"
            _fk_cn  = fkf["TxnType"] == "Cancel"

            gross_rev   = fkf.loc[_fk_sh,  "Invoice_Amount"].sum()
            refund_amt  = fkf.loc[_fk_rf,  "Invoice_Amount"].abs().sum()
            rc_rev      = fkf.loc[_fk_rc,  "Invoice_Amount"].sum()   # positive — reverses returns
            net_rev     = gross_rev - refund_amt + rc_rev
            units_sold  = fkf.loc[_fk_sh,  "Quantity"].sum()
            ret_units   = fkf.loc[_fk_rf,  "Quantity"].sum()
            rc_units    = fkf.loc[_fk_rc,  "Quantity"].sum()         # return-cancellations
            cancel_units= fkf.loc[_fk_cn,  "Quantity"].sum()
            orders      = fkf.loc[_fk_sh,  "OrderId"].nunique()
            ret_rate    = (ret_units / units_sold * 100) if units_sold > 0 else 0
            aov         = gross_rev / orders if orders > 0 else 0

            st.markdown("### 💰 KPIs")
            k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
            k1.metric("💵 Gross Rev",       fmt_inr(gross_rev))
            k2.metric("↩️ Refunds",          fmt_inr(refund_amt))
            k3.metric("✅ Net Rev",          fmt_inr(net_rev))
            k4.metric("📦 Units Sold",       f"{int(units_sold):,}")
            k5.metric("↩️ Returns",          f"{int(ret_units):,}")
            k6.metric("🔄 Return Cancels",   f"{int(rc_units):,}")
            k7.metric("📊 Return Rate",      f"{ret_rate:.1f}%")
            k8.metric("💳 AOV",              fmt_inr(aov))
            st.divider()

            # ── Monthly Revenue & Units Trend ──────────────────
            with st.expander("📈 Monthly Revenue & Units Trend", expanded=True):
                _fk_monthly_sale = (
                    fkf[_fk_sh].groupby("Month")
                    .agg(Revenue=("Invoice_Amount","sum"), Units=("Quantity","sum"))
                    .reset_index().sort_values("Month")
                )
                _fk_monthly_ret = (
                    fkf[_fk_rf].groupby("Month")["Quantity"].sum()
                    .reset_index().rename(columns={"Quantity":"Returns"})
                )
                _fk_monthly = _fk_monthly_sale.merge(_fk_monthly_ret, on="Month", how="left").fillna(0)
                _fk_monthly["Return_%"] = (
                    _fk_monthly["Returns"] / _fk_monthly["Units"].replace(0, np.nan) * 100
                ).fillna(0).round(1)

                fig_fk_rev = px.bar(
                    _fk_monthly, x="Month", y="Revenue",
                    title="Monthly Gross Revenue — Flipkart",
                    color_discrete_sequence=["#F7C243"]
                )
                fig_fk_rev.update_yaxes(tickprefix="₹", tickformat=",.0f")
                fig_fk_rev.update_layout(height=320)
                st.plotly_chart(fig_fk_rev, use_container_width=True)

                col_u, col_r = st.columns(2)
                with col_u:
                    fig_fk_units = px.bar(
                        _fk_monthly, x="Month", y=["Units","Returns"],
                        barmode="group", title="Units Sold vs Returns",
                        color_discrete_sequence=["#F7C243","#E63946"]
                    )
                    fig_fk_units.update_layout(height=280)
                    st.plotly_chart(fig_fk_units, use_container_width=True)
                with col_r:
                    fig_fk_rr = px.line(
                        _fk_monthly, x="Month", y="Return_%",
                        title="Return Rate % by Month",
                        markers=True, color_discrete_sequence=["#E63946"]
                    )
                    fig_fk_rr.update_layout(height=280)
                    st.plotly_chart(fig_fk_rr, use_container_width=True)

            # ── State-wise breakdown ────────────────────────────
            with st.expander("🗺️ State-wise Sales Distribution", expanded=False):
                _fk_state = (
                    fkf[_fk_sh & (fkf["State"] != "")]
                    .groupby("State")
                    .agg(Units=("Quantity","sum"), Revenue=("Invoice_Amount","sum"))
                    .reset_index().sort_values("Units", ascending=False).head(20)
                )
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    fig_st_u = px.bar(
                        _fk_state, x="Units", y="State", orientation="h",
                        title="Top 20 States by Units", color_discrete_sequence=["#F7C243"]
                    )
                    fig_st_u.update_layout(height=420, yaxis={"categoryorder":"total ascending"})
                    st.plotly_chart(fig_st_u, use_container_width=True)
                with col_s2:
                    fig_st_r = px.bar(
                        _fk_state, x="Revenue", y="State", orientation="h",
                        title="Top 20 States by Revenue", color_discrete_sequence=["#1e40af"]
                    )
                    fig_st_r.update_xaxes(tickprefix="₹", tickformat=",.0f")
                    fig_st_r.update_layout(height=420, yaxis={"categoryorder":"total ascending"})
                    st.plotly_chart(fig_st_r, use_container_width=True)

            # ── Event type breakdown ────────────────────────────
            with st.expander("📋 Transaction Type Breakdown", expanded=False):
                _fk_events = (
                    fkf.groupby("TxnType")
                    .agg(Rows=("Quantity","count"), Units=("Quantity","sum"),
                         Revenue=("Invoice_Amount","sum"))
                    .reset_index()
                )
                _fk_events["Revenue"] = _fk_events["Revenue"].apply(fmt_inr)
                st.dataframe(_fk_events, use_container_width=True, hide_index=True)

            # ── SKU-level performance ────────────────────────────
            with st.expander("🏷️ SKU Performance", expanded=False):
                if "OMS_SKU" in fkf.columns:
                    _fk_sku = (
                        fkf[_fk_sh].groupby("OMS_SKU")
                        .agg(Units=("Quantity","sum"), Revenue=("Invoice_Amount","sum"))
                        .reset_index().sort_values("Units", ascending=False)
                    )
                    _fk_sku_ret = (
                        fkf[_fk_rf].groupby("OMS_SKU")["Quantity"].sum()
                        .reset_index().rename(columns={"Quantity":"Returns"})
                    )
                    _fk_sku = _fk_sku.merge(_fk_sku_ret, on="OMS_SKU", how="left").fillna(0)
                    _fk_sku["Return_%"] = (_fk_sku["Returns"] / _fk_sku["Units"].replace(0,np.nan)*100).fillna(0).round(1)
                    _fk_sku["Revenue_fmt"] = _fk_sku["Revenue"].apply(fmt_inr)

                    top_n = st.slider("Top N SKUs", 10, 100, 30, key="fk_sku_n")
                    st.dataframe(
                        _fk_sku.head(top_n).reset_index(drop=True),
                        use_container_width=True, hide_index=True
                    )

                    fig_sku = px.bar(
                        _fk_sku.head(20), x="OMS_SKU", y="Units",
                        title="Top 20 SKUs by Units Sold",
                        color_discrete_sequence=["#F7C243"]
                    )
                    fig_sku.update_layout(height=320, xaxis_tickangle=-45)
                    st.plotly_chart(fig_sku, use_container_width=True)
                else:
                    st.info("SKU data not available in this dataset.")

            # ── Download ────────────────────────────────────────
            st.divider()
            st.download_button(
                "⬇️ Download Filtered Data (CSV)",
                fkf.to_csv(index=False).encode("utf-8"),
                f"flipkart_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True
            )

# ══════════════════════════════════════════════════════════════
# TAB — DAILY ORDERS vs PO CHECK
# ══════════════════════════════════════════════════════════════
with tab_daily:
    st.subheader("📋 Daily Orders — PO Check")

    daily = st.session_state.daily_orders_df
    if daily.empty:
        st.info(
            "📂 Upload today's daily order reports in the **Tier 3 — Daily Snapshot** section "
            "of the sidebar (Amazon CSV, Flipkart/Myntra CSV, Meesho CSV, Myntra XLSX), "
            "then click **Load All Data**."
        )
        st.stop()

    # ── Auto-detection log ───────────────────────────────────
    detect_log = st.session_state.get("daily_detect_log", [])
    if detect_log:
        with st.expander("🔍 Auto-detected platforms", expanded=True):
            for line in detect_log:
                st.markdown(line)

    # ── Guard: ensure Date column is datetime with at least one valid value ──
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.dropna(subset=["Date"])
    if daily.empty:
        st.warning("⚠️ Daily data loaded but all rows had unparseable dates. Check files and reload.")
        st.stop()

    # ── Header summary ────────────────────────────────────────
    d_active  = daily[daily["TxnType"] == "Shipment"]
    d_cancel  = daily[daily["TxnType"] == "Cancel"]
    d_pending = daily[daily["TxnType"] == "Pending"]
    d_refund  = daily[daily["TxnType"] == "Refund"]

    _valid_dates = daily["Date"].dropna()
    date_label   = _valid_dates.dt.date.min() if len(_valid_dates) else "Unknown"
    platforms    = sorted(daily["Platform"].dropna().unique().tolist())

    st.success(
        f"✅ **{date_label}** | Platforms: **{', '.join(platforms) if platforms else 'None'}** | "
        f"Total rows: **{len(daily):,}**"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Active Orders",   f"{int(d_active['Quantity'].sum()):,}")
    c2.metric("❌ Cancelled",        f"{int(d_cancel['Quantity'].sum()):,}")
    c3.metric("⏳ Pending / Hold",   f"{int(d_pending['Quantity'].sum()):,}")
    c4.metric("↩️ Returns",          f"{int(d_refund['Quantity'].sum()):,}")
    c5.metric("💰 Revenue (Active)", fmt_inr(d_active["Revenue"].sum()))
    st.divider()

    # ── Filters ───────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            _sel_plat = st.multiselect(
                "Platform", platforms, default=platforms, key="daily_plat"
            )
        with fc2:
            _txn_opts = sorted(daily["TxnType"].unique().tolist())
            _sel_txn  = st.multiselect(
                "Status", _txn_opts,
                default=[t for t in ["Shipment","Pending"] if t in _txn_opts],
                key="daily_txn"
            )
        with fc3:
            _sku_search = st.text_input("Search SKU", "", key="daily_sku_search",
                                         placeholder="e.g. 1685YK, PLYK...")

    df_f = daily[daily["Platform"].isin(_sel_plat) & daily["TxnType"].isin(_sel_txn)]
    if _sku_search:
        df_f = df_f[df_f["SKU"].str.contains(_sku_search.strip(), case=False, na=False)]

    # ══════════════════════════════════════════════════════════
    # SECTION A — SKU-level velocity table with PO comparison
    # ══════════════════════════════════════════════════════════
    st.markdown("### 📊 Today's Velocity by SKU")
    st.caption(
        "Units ordered today per SKU, broken down by platform. "
        "**PO Suggest** = units the PO Engine recommends ordering based on ADS × lead time. "
        "**⚠️ Alert** flags SKUs selling today that have zero or low PO recommendation — "
        "these may need a manual PO review."
    )

    # Aggregate active orders by SKU + Platform
    sku_pivot = (
        df_f[df_f["TxnType"].isin(["Shipment","Pending"])]
        .groupby(["SKU", "Platform"])["Quantity"]
        .sum()
        .reset_index()
        .pivot_table(index="SKU", columns="Platform", values="Quantity", fill_value=0)
        .reset_index()
    )
    # Total column
    plat_cols = [c for c in sku_pivot.columns if c != "SKU"]
    sku_pivot["Today_Total"] = sku_pivot[plat_cols].sum(axis=1)

    # Revenue per SKU (active only)
    rev_by_sku = (
        df_f[df_f["TxnType"]=="Shipment"]
        .groupby("SKU")["Revenue"].sum().reset_index()
        .rename(columns={"Revenue":"Today_Rev"})
    )
    sku_pivot = sku_pivot.merge(rev_by_sku, on="SKU", how="left").fillna({"Today_Rev": 0})
    sku_pivot["Today_Rev"] = sku_pivot["Today_Rev"].apply(fmt_inr)

    # Pull PO recommendation if available
    po_available = not st.session_state.sales_df.empty and not (
        st.session_state.inventory_df_variant.empty and
        st.session_state.inventory_df_parent.empty
    )
    if po_available:
        try:
            _po_quick = calculate_po_base(
                sales_df     = st.session_state.sales_df,
                inv_df       = st.session_state.inventory_df_variant,
                period_days  = 37,       # 30 day + 7 grace
                lead_time    = 15,
                target_days  = 60,
                demand_basis = "Sold",
                min_denominator = 7,
                use_seasonality = False,
            )
            if not _po_quick.empty:
                _po_cols = [c for c in ["OMS_SKU","PO_Qty","ADS","Total_Inventory","Stockout_Flag"]
                            if c in _po_quick.columns]
                _po_quick = _po_quick[_po_cols].copy()
                _po_quick = _po_quick.rename(columns={
                    "OMS_SKU":         "SKU",
                    "PO_Qty":          "PO_Suggest",
                    "ADS":             "ADS_30d",
                    "Total_Inventory": "Stock",
                    "Stockout_Flag":   "OOS",
                })
                sku_pivot = sku_pivot.merge(_po_quick, on="SKU", how="left")
                sku_pivot["PO_Suggest"]  = sku_pivot["PO_Suggest"].fillna(0).astype(int)
                sku_pivot["Stock"]       = sku_pivot.get("Stock", pd.Series(dtype=float)).fillna(0).astype(int)
                sku_pivot["ADS_30d"]     = sku_pivot.get("ADS_30d", pd.Series(dtype=float)).fillna(0).round(2)
                sku_pivot["OOS"]         = sku_pivot.get("OOS", pd.Series(dtype=str)).fillna("")

                # Alert: selling today but PO_Suggest = 0 AND Stock <= Today_Total*3
                sku_pivot["Alert"] = ""
                alert_mask = (
                    (sku_pivot["Today_Total"] > 0) &
                    (sku_pivot["PO_Suggest"] == 0) &
                    (sku_pivot.get("Stock", 999) <= sku_pivot["Today_Total"] * 3)
                )
                sku_pivot.loc[alert_mask, "Alert"] = "⚠️ Review PO"
                sku_pivot.loc[sku_pivot["OOS"] != "", "Alert"] = "🔴 OOS"
        except Exception:
            pass  # PO calc optional — tab still works without it

    sku_pivot = sku_pivot.sort_values("Today_Total", ascending=False).reset_index(drop=True)

    # Highlight rows with alerts
    def _highlight_alert(row):
        if "🔴" in str(row.get("Alert", "")):
            return ["background-color: #fee2e2"] * len(row)
        if "⚠️" in str(row.get("Alert", "")):
            return ["background-color: #fef9c3"] * len(row)
        return [""] * len(row)

    st.dataframe(
        sku_pivot.style.apply(_highlight_alert, axis=1),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    # ══════════════════════════════════════════════════════════
    # SECTION B — Platform breakdown
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 🏪 Platform Breakdown")
    plat_summary = (
        daily[daily["Platform"].isin(_sel_plat)]
        .groupby(["Platform","TxnType"])
        .agg(Units=("Quantity","sum"), Revenue=("Revenue","sum"), SKUs=("SKU","nunique"))
        .reset_index()
    )
    plat_summary["Revenue"] = plat_summary["Revenue"].apply(fmt_inr)

    col_p, col_c = st.columns([2, 1])
    with col_p:
        _active_plat = (
            daily[(daily["Platform"].isin(_sel_plat)) & (daily["TxnType"]=="Shipment")]
            .groupby("Platform")["Quantity"].sum().reset_index()
        )
        if not _active_plat.empty:
            fig_p = px.bar(
                _active_plat, x="Platform", y="Quantity",
                title="Active Orders by Platform",
                color="Platform",
                color_discrete_sequence=["#F7C243","#4F46E5","#10B981","#EF4444"],
            )
            fig_p.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
    with col_c:
        st.dataframe(plat_summary, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # SECTION C — Top selling SKUs today (chart)
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 🏆 Top SKUs Today")
    _top_n = st.slider("Show top N SKUs", 10, 50, 20, key="daily_top_n")
    _top_skus = (
        df_f[df_f["TxnType"].isin(["Shipment","Pending"])]
        .groupby(["SKU","Platform"])["Quantity"].sum()
        .reset_index().sort_values("Quantity", ascending=False)
        .head(_top_n)
    )
    if not _top_skus.empty:
        fig_top = px.bar(
            _top_skus, x="SKU", y="Quantity", color="Platform",
            title=f"Top {_top_n} SKUs by Units Ordered Today",
            color_discrete_sequence=["#F7C243","#4F46E5","#10B981","#EF4444"],
        )
        fig_top.update_layout(height=340, xaxis_tickangle=-45)
        st.plotly_chart(fig_top, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # SECTION D — State distribution
    # ══════════════════════════════════════════════════════════
    with st.expander("🗺️ State Distribution", expanded=False):
        _state_df = (
            df_f[df_f["TxnType"]=="Shipment"]
            .groupby("State")["Quantity"].sum()
            .reset_index().sort_values("Quantity", ascending=False)
            .head(20)
        )
        _state_df = _state_df[_state_df["State"].str.strip() != ""]
        if not _state_df.empty:
            fig_st = px.bar(
                _state_df, x="Quantity", y="State", orientation="h",
                title="Top 20 States — Today's Active Orders",
                color_discrete_sequence=["#4F46E5"],
            )
            fig_st.update_layout(height=400, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig_st, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # SECTION E — Raw order detail
    # ══════════════════════════════════════════════════════════
    with st.expander("📄 Raw Order Detail", expanded=False):
        st.dataframe(
            df_f.sort_values(["Platform","SKU"]).reset_index(drop=True),
            use_container_width=True, height=400, hide_index=True
        )

    # Download
    st.divider()
    st.download_button(
        "⬇️ Download Daily Orders (CSV)",
        df_f.to_csv(index=False).encode("utf-8"),
        f"daily_orders_{date_label}.csv",
        "text/csv", use_container_width=True
    )

# ══════════════════════════════════════════════════════════════
# TAB 6 — INVENTORY
# ══════════════════════════════════════════════════════════════
with tab_inv:
    st.subheader("📦 Consolidated Inventory")
    mode = st.radio("Inventory View", ["Variant (Size/Color)","Parent (Style Only)"], horizontal=True)
    inv  = (st.session_state.inventory_df_variant if "Variant" in mode else st.session_state.inventory_df_parent)
    if inv.empty:
        st.warning("⚠️ No inventory data loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",   f"{len(inv):,}")
        c2.metric("Total Units",  f"{inv['Total_Inventory'].sum():,.0f}")
        if "OMS_Inventory" in inv.columns:
            c3.metric("OMS Warehouse", f"{inv['OMS_Inventory'].sum():,.0f}")
        c4.metric("Marketplaces", f"{inv['Marketplace_Total'].sum():,.0f}" if "Marketplace_Total" in inv.columns else "0")
        st.dataframe(inv, use_container_width=True, height=500)

# ══════════════════════════════════════════════════════════════
# TAB 6 — PO ENGINE
# ══════════════════════════════════════════════════════════════
with tab_po:
    st.subheader("🎯 Purchase Order Engine")
    if st.session_state.sales_df.empty or (
        st.session_state.inventory_df_variant.empty and
        st.session_state.inventory_df_parent.empty
    ):
        st.warning("⚠️ Please load Sales data and Inventory data first, then click Load All Data.")
    else:
        # ── View mode + parameters ────────────────────────────
        view_mode = st.radio(
            "Group By",
            ["By Variant (Size/Color)", "By Parent SKU (Style Only)"],
            key="po_view_mode", horizontal=True
        )
        is_parent = "Parent" in view_mode
        st.divider()

        # Two-column layout: parameters left, seasonal right
        col_params, col_seasonal = st.columns([3, 2])
        with col_params:
            st.markdown("#### ⚙️ PO Parameters")
            p1, p2, p3, p4, p5 = st.columns(5)
            velocity    = p1.selectbox("Velocity Window", ["Last 7 Days","Last 30 Days","Last 60 Days","Last 90 Days"], key="po_velocity", index=1)
            base_days   = 7 if "7" in velocity else 30 if "30" in velocity else 60 if "60" in velocity else 90
            grace_days  = p2.number_input("Grace Days", 0, 14, 7)
            lead_time   = p3.number_input("Lead Time (Days)", 1, 180, 15)
            target_days = p4.number_input("Target Stock (Days)", 0, 180, 60)
            safety_pct  = p5.slider("Safety Stock %", 0, 100, 20)
            demand_basis = st.selectbox("Demand Basis", ["Sold","Net"], index=0)
            min_den      = st.number_input("Min ADS Denominator", 1, 60, 7)

        with col_seasonal:
            st.markdown("#### 📅 Quarterly History Settings")
            n_quarters = st.slider("Quarters of history to show", 4, 12, 8,
                                   help="How many Indian fiscal year quarters to display (Q1=Apr–Jun, Q2=Jul–Sep, Q3=Oct–Dec, Q4=Jan–Mar)")
            use_seasonality  = st.checkbox("Blend with Last Year's seasonality?", value=False)
            seasonal_weight  = st.slider("Historical Weight %", 0, 100, 50) if use_seasonality else 0

        total_period = int(base_days + grace_days)

        # ── Inventory & sales prep ────────────────────────────
        if is_parent:
            inv_for_po   = st.session_state.inventory_df_parent.copy()
            sales_for_po = st.session_state.sales_df.copy()
            sales_for_po["Sku"] = sales_for_po["Sku"].apply(get_parent_sku)
        else:
            inv_for_po   = st.session_state.inventory_df_variant.copy()
            sales_for_po = st.session_state.sales_df.copy()

        # ── Build quarterly history ───────────────────────────
        with st.spinner("Building quarterly history…"):
            qhist = calculate_quarterly_history(
                sales_df        = sales_for_po,
                mtr_df          = st.session_state.mtr_df    if not st.session_state.mtr_df.empty    else None,
                myntra_df       = st.session_state.myntra_df if not st.session_state.myntra_df.empty else None,
                sku_mapping     = st.session_state.sku_mapping,
                group_by_parent = is_parent,
                n_quarters      = n_quarters,
            )

        # ── PO calculation ────────────────────────────────────
        po_df = calculate_po_base(
            sales_df        = sales_for_po,
            inv_df          = inv_for_po,
            period_days     = total_period,
            lead_time       = lead_time,
            target_days     = target_days,
            demand_basis    = demand_basis,
            min_denominator = int(min_den),
            use_seasonality = use_seasonality,
            seasonal_weight = seasonal_weight / 100.0,
            mtr_df          = st.session_state.mtr_df      if not st.session_state.mtr_df.empty      else None,
            myntra_df       = st.session_state.myntra_df   if not st.session_state.myntra_df.empty   else None,
            sku_mapping     = st.session_state.sku_mapping,
            group_by_parent = is_parent,
        )

        if po_df.empty:
            st.warning("No PO calculations available. Check that sales and inventory data overlap.")
        else:
            # ── Compute final PO columns ──────────────────────
            po_df["Days_Left"]        = np.where(po_df["ADS"] > 0, po_df["Total_Inventory"] / po_df["ADS"], 999)
            po_df["Lead_Time_Demand"] = po_df["ADS"] * lead_time
            po_df["Target_Stock"]     = po_df["ADS"] * target_days
            po_df["Base_Requirement"] = po_df["Lead_Time_Demand"] + po_df["Target_Stock"]
            po_df["Safety_Stock"]     = po_df["Base_Requirement"] * (safety_pct / 100)
            po_df["Total_Required"]   = po_df["Base_Requirement"] + po_df["Safety_Stock"]
            po_df["PO_Qty"]           = (
                np.ceil((po_df["Total_Required"] - po_df["Total_Inventory"]).clip(lower=0) / 5) * 5
            ).astype(int)

            def _priority(row):
                if row["Days_Left"] < lead_time      and row["PO_Qty"] > 0: return "🔴 URGENT"
                if row["Days_Left"] < lead_time + 7  and row["PO_Qty"] > 0: return "🟡 HIGH"
                if row["PO_Qty"] > 0:                                        return "🟢 MEDIUM"
                return "⚪ OK"
            po_df["Priority"] = po_df.apply(_priority, axis=1)

            # ── Seasonality banner ────────────────────────────
            if use_seasonality:
                ly_source    = po_df.attrs.get("ly_source", "Sales")
                ly_win_start = po_df.attrs.get("ly_window_start", None)
                ly_win_end   = po_df.attrs.get("ly_window_end",   None)
                skus_ly      = int((po_df["LY_ADS"] > 0).sum()) if "LY_ADS" in po_df.columns else 0
                win_str = (f"{ly_win_start.strftime('%d %b %Y')} → {ly_win_end.strftime('%d %b %Y')}"
                           if ly_win_start and ly_win_end else "unknown")
                if skus_ly == 0:
                    st.error(f"⚠️ LY_ADS = 0 for all SKUs. Source: {ly_source} | Window: {win_str}")
                else:
                    st.success(
                        f"✅ Seasonality active | Source: **{ly_source}** | Window: {win_str} | "
                        f"SKUs with LY data: **{skus_ly:,}/{len(po_df):,}** | "
                        f"Blend: {100-int(seasonal_weight)}% Recent + {int(seasonal_weight)}% LY"
                    )

            # ── KPI bar ───────────────────────────────────────
            urgent = len(po_df[po_df["Priority"]=="🔴 URGENT"])
            high   = len(po_df[po_df["Priority"]=="🟡 HIGH"])
            med    = len(po_df[po_df["Priority"]=="🟢 MEDIUM"])
            total_po_units = po_df["PO_Qty"].sum()
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("🔴 Urgent",       urgent)
            k2.metric("🟡 High",         high)
            k3.metric("🟢 Medium",       med)
            k4.metric("📦 Total PO Units", f"{total_po_units:,}")
            k5.metric("📊 SKUs Analysed",  f"{len(po_df):,}")
            st.divider()

            # ══════════════════════════════════════════════════
            # MAIN TABLE — exact layout matching your sheet
            # ══════════════════════════════════════════════════

            # Join quarterly history onto PO df
            if not qhist.empty:
                master = po_df.merge(qhist, on="OMS_SKU", how="left")
            else:
                master = po_df.copy()
                master["Avg_Monthly"] = 0
                master["Units_30d"]   = 0
                master["Freq_30d"]    = 0
                master["Status"]      = "—"

            # Resolve ADS/Status column collisions from merge
            if "ADS_x" in master.columns:
                master = master.rename(columns={"ADS_x": "ADS_po", "ADS_y": "ADS_all"})
                master["ADS"] = master["ADS_all"].fillna(master["ADS_po"])
            if "Status_x" in master.columns:
                master = master.rename(columns={"Status_x": "Status_po", "Status_y": "Status"})
            if "Status" not in master.columns:
                def _st(ads):
                    if ads >= 1.0:  return "🚀 Fast Moving"
                    if ads >= 0.33: return "✅ Moderate"
                    if ads >= 0.10: return "🐢 Slow Selling"
                    return "❌ Not Moving"
                master["Status"] = master["ADS"].apply(_st)

            master["Parent_SKU"] = master["OMS_SKU"].apply(get_parent_sku)
            master["Days_Left"] = np.where(
                master["ADS"] > 0, master["Total_Inventory"] / master["ADS"], 999
            ).round(1)
            master["Running_Days"] = master["Days_Left"].round(0).astype(int)
            master["PO_Qty"] = (
                np.ceil((
                    (master["ADS"] * lead_time + master["ADS"] * target_days) * (1 + safety_pct / 100)
                    - master["Total_Inventory"]
                ).clip(lower=0) / 5) * 5
            ).astype(int)
            master["Priority"] = master.apply(_priority, axis=1)

            # ── New derived columns matching the sheet ────────
            master["PO_Plus_Avail"]   = (master["PO_Qty"] + master["Total_Inventory"].fillna(0)).astype(int)
            master["OMS_Stock"]       = master["OMS_Inventory"].fillna(0).astype(int) if "OMS_Inventory" in master.columns else 0
            master["Avail_Inventory"] = master["Total_Inventory"].fillna(0).astype(int)

            for _c in ["Units_30d","Freq_30d","Avg_Monthly","Net_Units","Sold_Units","Return_Units"]:
                if _c in master.columns:
                    master[_c] = pd.to_numeric(master[_c], errors="coerce").fillna(0)

            # ── Quarter columns ───────────────────────────────
            q_cols = [c for c in master.columns if "–" in c and len(c) > 6 and c[-4:].isdigit()]

            # ── Filter controls ───────────────────────────────
            st.markdown("#### 🔍 Filter & Search")
            fa, fb, fc, fd = st.columns(4)
            with fa:
                _statuses = sorted(master["Status"].dropna().unique().tolist())
                sel_status = st.multiselect("Status", _statuses, default=_statuses, key="po_status_filter")
            with fb:
                sel_priority = st.multiselect("Priority", ["🔴 URGENT","🟡 HIGH","🟢 MEDIUM","⚪ OK"],
                                               default=["🔴 URGENT","🟡 HIGH","🟢 MEDIUM"], key="po_prio_filter")
            with fc:
                po_only = st.checkbox("Show only SKUs needing PO", value=True, key="po_only_flag")
            with fd:
                sku_search = st.text_input("Search SKU / Parent", "", key="po_sku_search", placeholder="e.g. 1457YK")

            filtered = master.copy()
            if sel_status:     filtered = filtered[filtered["Status"].isin(sel_status)]
            if sel_priority:   filtered = filtered[filtered["Priority"].isin(sel_priority)]
            if po_only:        filtered = filtered[filtered["PO_Qty"] > 0]
            if sku_search.strip():
                _mask = (
                    filtered["OMS_SKU"].str.contains(sku_search.strip(), case=False, na=False) |
                    filtered["Parent_SKU"].str.contains(sku_search.strip(), case=False, na=False)
                )
                filtered = filtered[_mask]
            filtered = filtered.sort_values(["Priority", "Days_Left"]).reset_index(drop=True)

            # ── COLUMN ORDER — matches screenshot left → right ─
            base_left  = ["OMS_SKU","Status","Parent_SKU","Units_30d","Freq_30d","Running_Days"]
            base_right = []
            if "Avg_Monthly" in filtered.columns: base_right.append("Avg_Monthly")
            base_right.append("ADS")
            if "Net_Units" in filtered.columns:   base_right.append("Net_Units")
            base_right += ["PO_Qty","PO_Plus_Avail","OMS_Stock","Avail_Inventory"]
            if use_seasonality and "LY_ADS" in filtered.columns: base_right.append("LY_ADS")
            base_right.append("Priority")

            disp_cols = base_left + q_cols + base_right
            disp_cols = [c for c in disp_cols if c in filtered.columns]

            rename_map = {
                "OMS_SKU":"SKU", "Parent_SKU":"Parent",
                "Units_30d":"1-Month Sale", "Freq_30d":"Freq (Days)",
                "Running_Days":"Running Days", "Avg_Monthly":"Avg Monthly",
                "Net_Units":"Net Sales", "PO_Qty":"PO Qty",
                "PO_Plus_Avail":"PO + Available", "OMS_Stock":"OMS Stock",
                "Avail_Inventory":"Available Inventory",
                "LY_ADS":"LY ADS", "ADS":"Daily Sales (ADS)",
            }
            disp_df = filtered[disp_cols].rename(columns=rename_map)

            _Q_COLOUR = {"Apr":"#fef9c3","Jul":"#dcfce7","Oct":"#ffedd5","Jan":"#dbeafe"}

            def _style_po_table(row):
                styles = []
                for col in row.index:
                    if col == "Priority":
                        if   "🔴" in str(row[col]): styles.append("background-color:#fee2e2;font-weight:bold")
                        elif "🟡" in str(row[col]): styles.append("background-color:#fef3c7;font-weight:bold")
                        elif "🟢" in str(row[col]): styles.append("background-color:#d1fae5")
                        else:                        styles.append("")
                    elif col == "PO Qty":
                        styles.append("background-color:#dbeafe;font-weight:bold;color:#1e40af")
                    elif col == "PO + Available":
                        styles.append("background-color:#e0e7ff;font-weight:bold")
                    elif col == "OMS Stock":
                        styles.append("background-color:#fef3c7")
                    elif col == "Available Inventory":
                        styles.append("background-color:#d1fae5")
                    elif col == "Status":
                        if "Fast"  in str(row[col]): styles.append("background-color:#d1fae5;font-weight:bold")
                        elif "Mod" in str(row[col]): styles.append("background-color:#fef9c3")
                        elif "Slow"in str(row[col]): styles.append("background-color:#fde68a")
                        elif "Not" in str(row[col]): styles.append("background-color:#fee2e2")
                        else:                         styles.append("")
                    elif col in q_cols:
                        val = float(row[col]) if pd.notna(row[col]) else 0
                        tint = _Q_COLOUR.get(col[:3], "#f3f4f6")
                        if   val == 0:  styles.append("background-color:#f3f4f6;color:#9ca3af")
                        elif val <= 5:  styles.append(f"background-color:{tint}")
                        elif val <= 15: styles.append("background-color:#fde68a")
                        elif val <= 30: styles.append("background-color:#fbbf24")
                        else:           styles.append("background-color:#f59e0b;font-weight:bold")
                    elif col == "Running Days":
                        val = float(row[col]) if pd.notna(row[col]) else 999
                        if   val < lead_time:      styles.append("background-color:#fee2e2;font-weight:bold")
                        elif val < lead_time + 14: styles.append("background-color:#fef3c7")
                        else:                      styles.append("background-color:#d1fae5")
                    elif col == "Daily Sales (ADS)":
                        styles.append("background-color:#f0fdf4;color:#166534")
                    else:
                        styles.append("")
                return styles

            fmt_dict = {}
            for c in disp_df.columns:
                if   c in ("Daily Sales (ADS)","LY ADS"): fmt_dict[c] = "{:.3f}"
                elif c == "Avg Monthly":                   fmt_dict[c] = "{:.1f}"
                elif c in ("PO Qty","PO + Available","OMS Stock","Available Inventory",
                           "Running Days","1-Month Sale","Net Sales","Freq (Days)"):
                    fmt_dict[c] = "{:.0f}"
                elif c in q_cols:                          fmt_dict[c] = "{:.0f}"

            st.markdown(
                f"**{len(filtered):,} SKUs** | "
                f"**{len(q_cols)} quarters** of history | "
                f"{'All platforms combined' if not qhist.empty else 'No quarterly history loaded'}"
            )
            st.caption(
                "🟡 Q1 Apr–Jun  |  🟢 Q2 Jul–Sep  |  🟠 Q3 Oct–Dec  |  🔵 Q4 Jan–Mar  "
                "| Cell shade = units sold: light = low, amber = mid, orange bold = high"
            )
            st.dataframe(
                disp_df.style.apply(_style_po_table, axis=1).format(fmt_dict, na_rep="—"),
                use_container_width=True,
                height=620,
                hide_index=True,
                column_config={
                    "SKU":                 st.column_config.TextColumn("SKU", width="medium"),
                    "Parent":              st.column_config.TextColumn("Parent SKU", width="small"),
                    "Status":              st.column_config.TextColumn("Status", width="medium"),
                    "1-Month Sale":        st.column_config.NumberColumn("1-Month Sale",   help="Units sold last 30 days"),
                    "Freq (Days)":         st.column_config.NumberColumn("Freq",           help="Days with ≥1 sale in last 30 days"),
                    "Running Days":        st.column_config.NumberColumn("Running Days",   help="Days of stock left at current ADS"),
                    "Avg Monthly":         st.column_config.NumberColumn("Avg Monthly",    help="Avg monthly sales from last 4 quarters ÷ 3"),
                    "Daily Sales (ADS)":   st.column_config.NumberColumn("ADS",            help="Avg Daily Sales — last 90 days, all platforms"),
                    "Net Sales":           st.column_config.NumberColumn("Net Sales",      help=f"Net units sold in last {total_period} days"),
                    "PO Qty":              st.column_config.NumberColumn("PO Qty",         help="Recommended PO qty (rounded to nearest 5)"),
                    "PO + Available":      st.column_config.NumberColumn("PO + Available", help="PO Qty + current total inventory"),
                    "OMS Stock":           st.column_config.NumberColumn("OMS Stock",      help="OMS warehouse stock only"),
                    "Available Inventory": st.column_config.NumberColumn("Available Inv.", help="Total stock: OMS + all marketplaces"),
                },
            )

            # ── Quarterly trend chart ─────────────────────────
            if q_cols:
                with st.expander("📈 Quarterly Sales Trend", expanded=True):
                    top_skus = (
                        master.nlargest(15, "Total_Inventory")["OMS_SKU"].tolist()
                        if "Total_Inventory" in master.columns else master["OMS_SKU"].head(15).tolist()
                    )
                    chart_skus = st.multiselect(
                        "Select SKUs to chart", master["OMS_SKU"].tolist(),
                        default=top_skus[:8], key="po_chart_skus"
                    )
                    if chart_skus:
                        chart_df = master[master["OMS_SKU"].isin(chart_skus)][["OMS_SKU"] + q_cols].copy()
                        chart_melted = chart_df.melt(id_vars="OMS_SKU", var_name="Quarter", value_name="Units")
                        # Shorten quarter labels for readability
                        chart_melted["Quarter"] = chart_melted["Quarter"].str.replace(r"^Q\d_", "", regex=True)
                        fig_q = px.line(
                            chart_melted, x="Quarter", y="Units", color="OMS_SKU",
                            title="Quarterly Sales History by SKU",
                            markers=True,
                        )
                        fig_q.update_layout(height=380, xaxis_tickangle=-30)
                        st.plotly_chart(fig_q, use_container_width=True)

            # ── Downloads ─────────────────────────────────────
            st.divider()
            suffix = "parent" if is_parent else "variant"
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "📥 Download Full PO Table (CSV)",
                    disp_df.to_csv(index=False).encode("utf-8"),
                    f"po_quarterly_{suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", use_container_width=True
                )
            with c_dl2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    disp_df.to_excel(w, sheet_name="PO_Quarterly", index=False)
                    # Second sheet: all SKUs (not filtered)
                    master[[c for c in disp_cols if c in master.columns]].to_excel(
                        w, sheet_name="All_SKUs", index=False
                    )
                st.download_button(
                    "📥 Download PO Table (Excel)",
                    buf.getvalue(),
                    f"po_quarterly_{suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# ══════════════════════════════════════════════════════════════
# TAB 7 — LOGISTICS
# ══════════════════════════════════════════════════════════════
with tab_logistics:
    st.subheader("🚚 Logistics Information")
    transfer_df = st.session_state.transfer_df
    if transfer_df.empty:
        st.info("📦 Upload Amazon Stock Transfer file to view logistics data.")
    else:
        st.dataframe(transfer_df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 8 — AI FORECAST
# ══════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("📈 AI Demand Forecasting")
    if not _PROPHET_AVAILABLE:
        st.error(f"⚠️ Prophet import failed. Python returned this error: `{_PROPHET_ERR}`")
        st.info("💡 **Why is this happening?** Even if you ran `pip install prophet`, Streamlit might be executing from a different global/virtual Python environment, or Prophet's underlying C++ dependencies (`cmdstanpy`) are failing to compile. Check your terminal logs or reinstall prophet in the same environment as Streamlit.")
    elif st.session_state.sales_df.empty:
        st.warning("⚠️ Upload sales data for forecasting.")
    else:
        sales = st.session_state.sales_df
        sku   = st.selectbox("Select SKU", [""]+sorted(sales["Sku"].dropna().unique().tolist()))
        days  = st.slider("Forecast Days", 7, 90, 30)

        if sku:
            subset = sales[sales["Sku"] == sku].copy()
            subset["ds"] = pd.to_datetime(subset["TxnDate"]).dt.date
            daily  = subset.groupby("ds")["Units_Effective"].sum().reset_index()
            daily.columns = ["ds","y"]
            daily["ds"] = pd.to_datetime(daily["ds"])

            if len(daily) < 14:
                st.warning("Need at least 14 days of historical data for this SKU to forecast.")
            else:
                try:
                    with st.spinner("Forecasting…"):
                        m        = Prophet(daily_seasonality=False, weekly_seasonality=True)
                        m.fit(daily)
                        future   = m.make_future_dataframe(periods=days)
                        forecast = m.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual"))
                    fut = forecast[forecast["ds"] > daily["ds"].max()]
                    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast", line=dict(dict(dash="dash"))))
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"🤖 Predicted demand (next {days} days): **{int(fut['yhat'].sum())} units**")
                except Exception as e:
                    st.error(f"Forecast error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 9 — DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab_drill:
    st.subheader("🔍 Deep Dive — SKU Panel Analysis")

    sales_df = st.session_state.sales_df
    mtr_df   = st.session_state.mtr_df
    inv_df   = st.session_state.inventory_df_variant

    if sales_df.empty and mtr_df.empty:
        st.warning("⚠️ Load sales or MTR data first to use Deep Dive.")
        st.stop()

    all_skus = set()
    if not sales_df.empty:
        all_skus.update(sales_df["Sku"].dropna().astype(str).unique())
    if not mtr_df.empty:
        all_skus.update(mtr_df["SKU"].dropna().astype(str).unique())
    all_skus = sorted(s for s in all_skus if s.strip())

    dd_col1, dd_col2 = st.columns([3, 1])
    with dd_col1:
        selected_sku = st.selectbox("🔎 Select SKU", [""] + all_skus, key="drill_sku")
    with dd_col2:
        drill_period = st.selectbox(
            "Period", ["Last 30 Days", "Last 60 Days", "Last 90 Days", "All Time"],
            index=2, key="drill_period"
        )

    if not selected_sku:
        st.info("👆 Select a SKU above to see its full performance panel.")
        st.stop()

    sales_df = sales_df.copy()
    sales_df["TxnDate"] = pd.to_datetime(sales_df["TxnDate"], errors="coerce")

    sku_sales = sales_df[sales_df["Sku"] == selected_sku].copy() if not sales_df.empty else pd.DataFrame()
    sku_mtr   = mtr_df[mtr_df["SKU"]  == selected_sku].copy()   if not mtr_df.empty   else pd.DataFrame()

    all_dates = []
    if not sku_sales.empty: all_dates.extend(sku_sales["TxnDate"].dropna().tolist())
    if not sku_mtr.empty:   all_dates.extend(pd.to_datetime(sku_mtr["Date"], errors="coerce").dropna().tolist())

    if not all_dates:
        st.warning(f"No data found for SKU **{selected_sku}**.")
        st.stop()

    max_date = max(all_dates)
    if drill_period == "All Time":
        cutoff = min(all_dates)
    else:
        days_map = {"Last 30 Days": 30, "Last 60 Days": 60, "Last 90 Days": 90}
        cutoff   = max_date - timedelta(days=days_map[drill_period])

    if not sku_sales.empty:
        sku_sales_p = sku_sales[sku_sales["TxnDate"] >= cutoff]
    else:
        sku_sales_p = pd.DataFrame()

    if not sku_mtr.empty:
        sku_mtr["Date"] = pd.to_datetime(sku_mtr["Date"], errors="coerce")
        sku_mtr_p = sku_mtr[sku_mtr["Date"] >= cutoff]
    else:
        sku_mtr_p = pd.DataFrame()

    period_days = max((max_date - cutoff).days, 1)

    def _decat(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if hasattr(out[col], "cat"):
                if out[col].cat.categories.dtype == "object":
                    out[col] = out[col].astype(str)
                else:
                    out[col] = out[col].astype(out[col].cat.categories.dtype)
        return out

    if not sku_mtr.empty:
        sku_mtr   = _decat(sku_mtr)
        sku_mtr_p = _decat(sku_mtr_p)
    if not sku_sales.empty:
        sku_sales   = _decat(sku_sales)
        sku_sales_p = _decat(sku_sales_p)

    sold_units   = 0
    return_units = 0
    net_units    = 0
    if not sku_sales_p.empty:
        sku_sales_p["Quantity"]       = pd.to_numeric(sku_sales_p["Quantity"],       errors="coerce").fillna(0)
        sku_sales_p["Units_Effective"]= pd.to_numeric(sku_sales_p["Units_Effective"],errors="coerce").fillna(0)
        sold_units   = sku_sales_p[sku_sales_p["Transaction Type"] == "Shipment"]["Quantity"].sum()
        return_units = sku_sales_p[sku_sales_p["Transaction Type"] == "Refund"]["Quantity"].sum()
        net_units    = sku_sales_p["Units_Effective"].sum()

    gross_rev  = 0.0
    refund_rev = 0.0
    net_rev    = 0.0
    if not sku_mtr_p.empty:
        sku_mtr_p["Invoice_Amount"] = pd.to_numeric(sku_mtr_p["Invoice_Amount"], errors="coerce").fillna(0)
        gross_rev  = sku_mtr_p[sku_mtr_p["Transaction_Type"] == "Shipment"]["Invoice_Amount"].sum()
        refund_rev = sku_mtr_p[sku_mtr_p["Transaction_Type"] == "Refund"]["Invoice_Amount"].abs().sum()
        net_rev    = gross_rev - refund_rev

    ads          = net_units / period_days if period_days > 0 else 0
    return_rate  = (return_units / sold_units * 100) if sold_units > 0 else 0
    asp          = gross_rev / sold_units if sold_units > 0 else 0

    curr_inv = 0
    inv_row  = pd.DataFrame()
    if not inv_df.empty and "OMS_SKU" in inv_df.columns:
        inv_row = inv_df[inv_df["OMS_SKU"] == selected_sku]
        if not inv_row.empty:
            curr_inv = inv_row["Total_Inventory"].iloc[0]

    days_cover = curr_inv / ads if ads > 0 else 999

    st.markdown(f"### 📦 `{selected_sku}` — {drill_period}")
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("✅ Sold Units",   f"{int(sold_units):,}")
    k2.metric("↩️ Returns",      f"{int(return_units):,}")
    k3.metric("📊 Return Rate",  f"{return_rate:.1f}%")
    k4.metric("📦 Net Units",    f"{int(net_units):,}")
    k5.metric("📈 Daily ADS",    f"{ads:.2f}")
    k6.metric("💰 Net Revenue",  fmt_inr(net_rev))
    k7.metric("💳 Avg Price",    fmt_inr(asp))
    st.divider()

    inv1, inv2, inv3, inv4 = st.columns(4)
    inv1.metric("🏭 Current Stock",  f"{int(curr_inv):,}")
    inv2.metric("📅 Days Cover",
                f"{int(days_cover) if days_cover < 999 else '∞'}",
                delta="⚠️ OOS Risk" if days_cover < 15 and ads > 0 else None,
                delta_color="inverse")
    if not inv_row.empty:
        if "OMS_Inventory" in inv_row.columns:
            inv3.metric("🏢 OMS Warehouse",   f"{int(inv_row['OMS_Inventory'].iloc[0]):,}")
        if "Marketplace_Total" in inv_row.columns:
            inv4.metric("🛒 Marketplace Stock",f"{int(inv_row['Marketplace_Total'].iloc[0]):,}")
    st.divider()

    ch1, ch2 = st.columns([3, 1])

    with ch1:
        st.markdown("#### 📈 Daily Sales Trend")
        if not sku_sales_p.empty:
            daily = (sku_sales_p[sku_sales_p["Transaction Type"] == "Shipment"]
                     .groupby(sku_sales_p["TxnDate"].dt.date)["Quantity"]
                     .sum().reset_index())
            daily.columns = ["Date", "Units"]
            daily["Date"] = pd.to_datetime(daily["Date"])

            daily = daily.sort_values("Date")
            daily["Rolling7"] = daily["Units"].rolling(7, min_periods=1).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=daily["Date"], y=daily["Units"],
                name="Daily Units", marker_color="#93c5fd", opacity=0.6
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily["Date"], y=daily["Rolling7"],
                name="7-Day Avg", line=dict(color="#002B5B", width=2)
            ))
            fig_trend.update_layout(height=300, margin=dict(t=10, b=10),
                                    legend=dict(orientation="h"),
                                    hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No unit-level sales data for this SKU in the selected period.")

    with ch2:
        st.markdown("#### 🏪 By Marketplace")
        if not sku_sales_p.empty and "Source" in sku_sales_p.columns:
            mkt = (sku_sales_p[sku_sales_p["Transaction Type"] == "Shipment"]
                   .groupby("Source")["Quantity"].sum().reset_index())
            if not mkt.empty:
                fig_mkt = px.pie(mkt, values="Quantity", names="Source", hole=0.45,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_mkt.update_layout(height=300, margin=dict(t=10, b=10),
                                      showlegend=True,
                                      legend=dict(orientation="v", x=0.0))
                fig_mkt.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_mkt, use_container_width=True)
            else:
                st.info("No marketplace data.")
        else:
            st.info("No marketplace data.")

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("#### 💰 Monthly Revenue (MTR)")
        if not sku_mtr.empty:
            sku_mtr["Invoice_Amount"] = pd.to_numeric(sku_mtr["Invoice_Amount"], errors="coerce").fillna(0)
            monthly_rev = (sku_mtr[sku_mtr["Transaction_Type"] == "Shipment"]
                           .groupby("Month")["Invoice_Amount"].sum()
                           .reset_index().sort_values("Month"))
            monthly_ref = (sku_mtr[sku_mtr["Transaction_Type"] == "Refund"]
                           .groupby("Month")["Invoice_Amount"].sum().abs()
                           .reset_index().sort_values("Month"))
            monthly_ref.columns = ["Month", "Refund_Amount"]
            monthly_comb = monthly_rev.merge(monthly_ref, on="Month", how="left").fillna(0)
            monthly_comb["Net_Revenue"] = monthly_comb["Invoice_Amount"] - monthly_comb["Refund_Amount"]

            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(
                x=monthly_comb["Month"], y=monthly_comb["Invoice_Amount"],
                name="Gross", marker_color="#002B5B"
            ))
            fig_rev.add_trace(go.Bar(
                x=monthly_comb["Month"], y=monthly_comb["Refund_Amount"],
                name="Refunds", marker_color="#E63946"
            ))
            fig_rev.add_trace(go.Scatter(
                x=monthly_comb["Month"], y=monthly_comb["Net_Revenue"],
                name="Net", line=dict(color="#10b981", width=2), mode="lines+markers"
            ))
            fig_rev.update_layout(barmode="overlay", height=300,
                                  margin=dict(t=10, b=10),
                                  hovermode="x unified",
                                  legend=dict(orientation="h"))
            fig_rev.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No MTR revenue data for this SKU.")

    with ch4:
        st.markdown("#### ↩️ Monthly Return Rate")
        if not sku_sales.empty:
            sku_sales["Quantity"] = pd.to_numeric(sku_sales["Quantity"], errors="coerce").fillna(0)
            sku_sales["Month"]    = sku_sales["TxnDate"].dt.to_period("M").astype(str)
            m_sold = (sku_sales[sku_sales["Transaction Type"] == "Shipment"]
                      .groupby("Month")["Quantity"].sum())
            m_ret  = (sku_sales[sku_sales["Transaction Type"] == "Refund"]
                      .groupby("Month")["Quantity"].sum())
            m_rate = (m_ret / m_sold.replace(0, np.nan) * 100).fillna(0).reset_index()
            m_rate.columns = ["Month", "Return_Rate_%"]
            m_rate = m_rate.sort_values("Month")

            fig_rr = px.bar(m_rate, x="Month", y="Return_Rate_%",
                            color="Return_Rate_%",
                            color_continuous_scale=["#10b981", "#f59e0b", "#E63946"],
                            range_color=[0, 30])
            fig_rr.update_layout(height=300, margin=dict(t=10, b=10),
                                 coloraxis_showscale=False)
            fig_rr.update_yaxes(ticksuffix="%")
            fig_rr.add_hline(y=10, line_dash="dash", line_color="orange",
                             annotation_text="10% threshold")
            st.plotly_chart(fig_rr, use_container_width=True)
        else:
            st.info("No sales data for return rate trend.")

    st.markdown("#### 📅 Year-on-Year Monthly Units")
    if not sku_sales.empty:
        sku_sales["Year"]  = sku_sales["TxnDate"].dt.year.astype(str)
        sku_sales["MonthN"]= sku_sales["TxnDate"].dt.month
        yoy = (sku_sales[sku_sales["Transaction Type"] == "Shipment"]
               .groupby(["Year", "MonthN"])["Quantity"].sum().reset_index())
        yoy["Month_Name"] = pd.to_datetime(yoy["MonthN"], format="%m").dt.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        yoy["Month_Name"] = pd.Categorical(yoy["Month_Name"], categories=month_order, ordered=True)
        yoy = yoy.sort_values(["Year", "MonthN"])

        fig_yoy = px.line(yoy, x="Month_Name", y="Quantity", color="Year",
                          markers=True,
                          color_discrete_sequence=["#93c5fd","#002B5B","#E63946","#10b981"],
                          labels={"Quantity": "Units Sold", "Month_Name": "Month"})
        fig_yoy.update_layout(height=320, hovermode="x unified",
                              legend=dict(orientation="h"),
                              margin=dict(t=10, b=10))
        st.plotly_chart(fig_yoy, use_container_width=True)
    else:
        st.info("No sales data for YoY comparison.")

    st.markdown("#### 🗺️ Top States (MTR)")
    if not sku_mtr_p.empty and "Ship_To_State" in sku_mtr_p.columns:
        state_data = (sku_mtr_p[sku_mtr_p["Transaction_Type"] == "Shipment"]
                      .groupby("Ship_To_State").agg(
                          Units=("Quantity", "sum"),
                          Revenue=("Invoice_Amount", "sum")
                      ).reset_index()
                      .sort_values("Revenue", ascending=False).head(15))
        state_data["Ship_To_State"] = state_data["Ship_To_State"].astype(str)

        fig_state = px.bar(state_data, x="Revenue", y="Ship_To_State",
                           orientation="h",
                           color="Units",
                           color_continuous_scale="Blues",
                           labels={"Ship_To_State": "State"},
                           text=state_data["Units"].astype(int).astype(str) + " units")
        fig_state.update_traces(textposition="outside")
        fig_state.update_layout(height=420, margin=dict(t=10, b=10),
                                yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False)
        fig_state.update_xaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("No MTR state data for this SKU in the selected period.")

    with st.expander("🗒️ Raw Transaction Log", expanded=False):
        tabs_raw = st.tabs(["Sales Transactions", "MTR Transactions"])
        with tabs_raw[0]:
            if not sku_sales_p.empty:
                show_cols = [c for c in ["TxnDate","Transaction Type","Quantity",
                                          "Units_Effective","Source","OrderId"]
                             if c in sku_sales_p.columns]
                st.dataframe(
                    sku_sales_p[show_cols].sort_values("TxnDate", ascending=False).head(500),
                    use_container_width=True, height=300
                )
                st.download_button(
                    "📥 Download Sales Log",
                    sku_sales_p[show_cols].to_csv(index=False).encode("utf-8"),
                    f"{selected_sku}_sales_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.info("No sales transactions.")

        with tabs_raw[1]:
            if not sku_mtr_p.empty:
                mtr_show = [c for c in ["Date","Report_Type","Transaction_Type","Quantity",
                                         "Invoice_Amount","Total_Tax","Ship_To_State",
                                         "Payment_Method","Invoice_Number","Buyer_Name"]
                            if c in sku_mtr_p.columns]
                st.dataframe(
                    sku_mtr_p[mtr_show].sort_values("Date", ascending=False).head(500),
                    use_container_width=True, height=300
                )
                st.download_button(
                    "📥 Download MTR Log",
                    sku_mtr_p[mtr_show].to_csv(index=False).encode("utf-8"),
                    f"{selected_sku}_mtr_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.info("No MTR transactions.")

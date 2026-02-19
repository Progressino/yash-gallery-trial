#!/usr/bin/env python3
"""
Yash Gallery Complete ERP System — app.py
(Bulletproof MTR Loader + Seasonal PO + Prophet Debug)
Fixed: MTR date parsing, 2023 ghost data, deduplication on Invoice_Number
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
# 6) BULLETPROOF MTR LOADER (Dynamic Column & B2B/B2C Match)
# ══════════════════════════════════════════════════════════════
_CAT_COLS = {"Ship_To_State", "Warehouse_Id", "Fulfillment", "Payment_Method"}

# ── FIX 1: Robust date parser that tries explicit formats before falling back ──
def _parse_date_flexible(series: pd.Series) -> pd.Series:
    """
    Try explicit date formats in priority order.
    Returns the first format that successfully parses > 70% of non-null values.
    Falls back to pandas inference as a last resort.
    Explicit formats prevent ambiguous day/month swaps that produce ghost years.
    """
    # Most Amazon MTR files use DD-MM-YYYY or DD/MM/YYYY
    priority_formats = [
        "%d-%m-%Y",    # 15-03-2024
        "%d/%m/%Y",    # 15/03/2024
        "%Y-%m-%d",    # 2024-03-15
        "%d-%b-%Y",    # 15-Mar-2024
        "%d/%b/%Y",    # 15/Mar/2024
        "%m/%d/%Y",    # 03/15/2024  (US format, lowest priority)
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    non_null = series.dropna()
    threshold = max(int(len(non_null) * 0.70), 1) if len(non_null) > 0 else 1

    for fmt in priority_formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() >= threshold:
                return parsed
        except Exception:
            continue

    # Last resort: pandas inference with dayfirst=True (can misparse ambiguous dates)
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _parse_mtr_csv(csv_bytes: bytes, source_file: str):
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False, encoding="utf-8", on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False, encoding="ISO-8859-1", on_bad_lines='skip')
        except Exception:
            return pd.DataFrame(), "Encoding Error"
    except Exception as e:
        return pd.DataFrame(), f"Parse Error: {e}"

    if raw.empty:
        return pd.DataFrame(), "Empty file"

    # Lowercase and strip everything to make it immune to Amazon column name changes
    raw.columns = raw.columns.astype(str).str.strip().str.lower()

    # Detect B2B dynamically from columns instead of fragile filenames
    is_b2b = "buyer name" in raw.columns or "customer bill to gstid" in raw.columns
    report_type = "B2B" if is_b2b else "B2C"

    want_b2c = {
        "shipment date", "invoice date", "transaction type", "sku", "asin",
        "item description", "quantity", "invoice amount", "total tax amount",
        "cgst tax", "sgst tax", "igst tax", "ship to state", "warehouse id",
        "fulfillment channel", "payment method code", "order id",
        "invoice number", "credit note no"
    }
    want_b2b = want_b2c.union({"buyer name", "bill to state", "customer bill to gstid", "irn filing status"})
    want = want_b2b if is_b2b else want_b2c

    # Keep only target columns
    keep_cols = [c for c in raw.columns if c in want]
    raw = raw[keep_cols]

    # Flexible Date Finder — prefer shipment date, then invoice date, etc.
    date_col = None
    for d in ["shipment date", "invoice date", "transaction date", "order date"]:
        if d in raw.columns:
            date_col = d
            break

    if date_col:
        # ── FIX 1 applied: use robust format-aware parser ──
        raw["_Date"] = _parse_date_flexible(raw[date_col])
    else:
        raw["_Date"] = pd.NaT

    initial_len = len(raw)
    raw = raw.dropna(subset=["_Date"])
    dropped_dates = initial_len - len(raw)

    if raw.empty:
        return pd.DataFrame(), f"All {initial_len} rows had invalid/missing dates."

    # ── FIX 2: Year sanity check — drop rows with implausible years ──
    current_year = datetime.now().year
    valid_year_mask = raw["_Date"].dt.year.between(2018, current_year + 1)
    ghost_rows = (~valid_year_mask).sum()
    if ghost_rows > 0:
        # Surface a warning but don't crash — just drop the bad rows
        raw = raw[valid_year_mask]
    if raw.empty:
        return pd.DataFrame(), f"All rows had out-of-range years (2018–{current_year+1}). Check source date format."

    for col in ["invoice amount", "total tax amount", "cgst tax", "sgst tax", "igst tax"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype("float32")

    if "quantity" in raw.columns:
        raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").fillna(0).astype("float32")
    else:
        raw["quantity"] = 0.0

    def g(name):
        return raw[name].fillna("").astype(str).str.strip() if name in raw.columns else pd.Series("", index=raw.index, dtype=str)

    def gn(name):
        return raw[name].astype("float32") if name in raw.columns else pd.Series(0.0, index=raw.index, dtype="float32")

    out = pd.DataFrame({
        "Date":             raw["_Date"],
        "Report_Type":      report_type,
        "Transaction_Type": g("transaction type"),
        "SKU":              g("sku"),
        "ASIN":             g("asin"),
        "Description":      g("item description"),
        "Quantity":         raw["quantity"],
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
        "Credit_Note_No":   g("credit note no"),
        "Buyer_Name":       g("buyer name"),
        "IRN_Status":       g("irn filing status"),
    })

    # Standardize Transaction Types in case Amazon renamed them
    out["Transaction_Type"] = out["Transaction_Type"].apply(
        lambda x: "Refund"   if "return" in str(x).lower() or "refund" in str(x).lower()
        else ("Cancel"       if "cancel" in str(x).lower()
        else "Shipment")
    )

    for col in _CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")

    msgs = []
    if dropped_dates > 0:
        msgs.append(f"Dropped {dropped_dates} rows missing dates.")
    if ghost_rows > 0:
        msgs.append(f"Dropped {ghost_rows} rows with out-of-range years.")
    msg = "OK" if not msgs else " | ".join(msgs)
    return out, msg


def _collect_csv_entries(main_zip_file):
    entries = []
    skipped = []

    def _walk(zf, depth=0):
        for item_name in zf.namelist():
            base = Path(item_name).name
            if not base:
                continue
            if base.lower().endswith(".zip"):
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
    skipped   = []
    csv_count = 0
    dfs       = []

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

            if df.empty:
                skipped.append(f"{base}: {msg}")
            else:
                dfs.append(df)
                csv_count += 1
                if msg != "OK":
                    skipped.append(f"{base}: Partial Load ({msg})")
        except Exception as e:
            skipped.append(f"{base}: Critical Error - {e}")

        prog.progress((idx + 1) / total, text=f"Loaded {idx + 1}/{total}: {base}")

    prog.empty()

    if not dfs:
        return pd.DataFrame(), 0, skipped

    combined = pd.concat(dfs, ignore_index=True)

    # ── FIX 3: Use Invoice_Number for deduplication instead of Order_Id ──
    # Order_Id can be empty string "" for many B2B rows, causing incorrect deduplication.
    # Invoice_Number is unique per line item and is far more reliable.
    dedup_subset = ["Invoice_Number", "SKU", "Transaction_Type", "Date"]
    # Only deduplicate on rows where Invoice_Number is non-empty to avoid collapsing valid rows
    has_invoice = combined["Invoice_Number"].str.strip() != ""
    deduped_with_inv = combined[has_invoice].drop_duplicates(subset=dedup_subset, keep="first")
    no_invoice       = combined[~has_invoice]
    # For rows without an Invoice_Number, fall back to Order_Id + SKU + Date
    deduped_no_inv   = no_invoice.drop_duplicates(
        subset=["Order_Id", "SKU", "Transaction_Type", "Date"], keep="first"
    )
    combined = pd.concat([deduped_with_inv, deduped_no_inv], ignore_index=True)

    for col in _CAT_COLS:
        if col in combined.columns:
            combined[col] = combined[col].astype("category")

    return combined, csv_count, skipped

# ══════════════════════════════════════════════════════════════
# 7) SALES DATA LOADERS
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
    return result.dropna(subset=["TxnDate"])


def load_flipkart_sales(xlsx_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_file, sheet_name="Sales Report")
        if df.empty:
            return pd.DataFrame()
        df["OMS_SKU"] = df["SKU"].apply(clean_sku).apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"]  = pd.to_datetime(df.get("Order Date"), errors="coerce")
        df["Quantity"] = pd.to_numeric(df.get("Item Quantity", 0), errors="coerce").fillna(0)
        df["Source"]   = "Flipkart"
        df["TxnType"]  = df.get("Event Sub Type","").apply(lambda x: "Refund" if "return" in str(x).lower() else "Shipment")
        df["Units_Effective"] = np.where(df["TxnType"] == "Refund", -df["Quantity"], df["Quantity"])
        df["OrderId"] = df.get("Order ID", df.get("Order Id", np.nan))
        result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
        result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Flipkart: {e}")
        return pd.DataFrame()


def load_meesho_sales(zip_file, mapping: Dict[str, str]) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            excel_files = [f for f in z.namelist() if "tcs_sales" in f.lower() and f.lower().endswith(".xlsx") and "return" not in f.lower()]
            if not excel_files:
                return pd.DataFrame()
            with z.open(excel_files[0]) as f:
                df = pd.read_excel(f)
        if df.empty:
            return pd.DataFrame()
        df["OMS_SKU"]         = df.get("identifier").apply(lambda x: map_to_oms_sku(x, mapping))
        df["TxnDate"]         = pd.to_datetime(df.get("order_date"), errors="coerce")
        df["Quantity"]        = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
        df["Source"]          = "Meesho"
        df["TxnType"]         = "Shipment"
        df["Units_Effective"] = df["Quantity"]
        df["OrderId"]         = df.get("sub_order_num", np.nan)
        result = df[["OMS_SKU","TxnDate","TxnType","Quantity","Units_Effective","Source","OrderId"]].copy()
        result.columns = ["Sku","TxnDate","Transaction Type","Quantity","Units_Effective","Source","OrderId"]
        return result.dropna(subset=["TxnDate"])
    except Exception as e:
        st.error(f"Error loading Meesho: {e}")
        return pd.DataFrame()

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
def calculate_po_base(
    sales_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    period_days: int,
    lead_time: int,
    target_days: int,
    demand_basis: str = "Sold",
    min_denominator: int = 7,
    use_seasonality: bool = False,
    seasonal_weight: float = 0.5
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
    sold.columns = ["OMS_SKU","Sold_Units"]
    returns = recent[recent["Transaction Type"]=="Refund"].groupby("Sku")["Quantity"].sum().reset_index()
    returns.columns = ["OMS_SKU","Return_Units"]
    net = recent.groupby("Sku")["Units_Effective"].sum().reset_index()
    net.columns = ["OMS_SKU","Net_Units"]

    summary = sold.merge(returns, on="OMS_SKU", how="outer").merge(net, on="OMS_SKU", how="outer").fillna(0)
    po_df = pd.merge(inv_df, summary, on="OMS_SKU", how="left").fillna({"Sold_Units":0,"Return_Units":0,"Net_Units":0})

    denom = max(period_days, min_denominator)
    demand_units = po_df["Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["Sold_Units"]
    po_df["Recent_ADS"] = (demand_units / denom).fillna(0)

    if use_seasonality and target_days > 0:
        future_start = max_date + timedelta(days=lead_time)
        future_end   = future_start + timedelta(days=target_days)
        ly_start     = future_start - timedelta(days=365)
        ly_end       = future_end - timedelta(days=365)

        ly_sales = df[(df["TxnDate"] >= ly_start) & (df["TxnDate"] < ly_end)].copy()

        if not ly_sales.empty:
            ly_sold = ly_sales[ly_sales["Transaction Type"]=="Shipment"].groupby("Sku")["Quantity"].sum().reset_index()
            ly_sold.columns = ["OMS_SKU", "LY_Sold_Units"]
            ly_net = ly_sales.groupby("Sku")["Units_Effective"].sum().reset_index()
            ly_net.columns = ["OMS_SKU", "LY_Net_Units"]
            ly_summary = ly_sold.merge(ly_net, on="OMS_SKU", how="outer").fillna(0)
            po_df = pd.merge(po_df, ly_summary, on="OMS_SKU", how="left").fillna(0)

            ly_days_count = (ly_end - ly_start).days
            ly_demand = po_df["LY_Net_Units"].clip(lower=0) if demand_basis == "Net" else po_df["LY_Sold_Units"]
            po_df["LY_ADS"] = (ly_demand / ly_days_count).fillna(0)
            po_df["ADS"] = np.where(
                po_df["LY_ADS"] > 0,
                (po_df["Recent_ADS"] * (1 - seasonal_weight)) + (po_df["LY_ADS"] * seasonal_weight),
                po_df["Recent_ADS"]
            )
        else:
            po_df["ADS"]    = po_df["Recent_ADS"]
            po_df["LY_ADS"] = 0
    else:
        po_df["ADS"]    = po_df["Recent_ADS"]
        po_df["LY_ADS"] = 0

    po_df["Stockout_Flag"] = ""
    po_df.loc[(po_df["ADS"] > 0) & (po_df["Total_Inventory"] <= 0), "Stockout_Flag"] = "⚠️ OOS"
    return po_df

# ══════════════════════════════════════════════════════════════
# 12) SIDEBAR — FILE UPLOADS
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📂 Data Upload")
map_file = st.sidebar.file_uploader("1️⃣ SKU Mapping (Required)", type=["xlsx"])

st.sidebar.markdown("### ⚙️ Amazon Settings")
st.session_state.amazon_date_basis    = st.sidebar.selectbox("Date Basis", ["Shipment Date","Invoice Date","Order Date"], index=0)
st.session_state.include_replacements = st.sidebar.checkbox("Include FreeReplacement", value=False)
st.sidebar.divider()

st.sidebar.markdown("### 2️⃣ MTR Reports (Amazon Tax)")
mtr_main_zip = st.sidebar.file_uploader("MTR — Main ZIP (all months)", type=["zip"], key="mtr_main_zip")
st.sidebar.divider()

st.sidebar.markdown("### 3️⃣ Sales Data (Units)")
f_b2c      = st.sidebar.file_uploader("Amazon B2C (ZIP)", type=["zip"], key="b2c")
f_b2b      = st.sidebar.file_uploader("Amazon B2B (ZIP)", type=["zip"], key="b2b")
f_transfer = st.sidebar.file_uploader("Stock Transfer (ZIP)", type=["zip"], key="transfer")
f_fk       = st.sidebar.file_uploader("Flipkart (Excel)", type=["xlsx"], key="fk")
f_meesho   = st.sidebar.file_uploader("Meesho (ZIP)", type=["zip"], key="meesho")
st.sidebar.divider()

st.sidebar.markdown("### 4️⃣ Inventory Data")
i_oms    = st.sidebar.file_uploader("OMS (CSV)",      type=["csv"], key="oms")
i_fk     = st.sidebar.file_uploader("Flipkart (CSV)", type=["csv"], key="fk_inv")
i_myntra = st.sidebar.file_uploader("Myntra (CSV)",   type=["csv"], key="myntra")
i_amz    = st.sidebar.file_uploader("Amazon (CSV)",   type=["csv"], key="amz")
st.sidebar.divider()

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
                    if not mtr_combined.empty:
                        year_range = f"{mtr_combined['Date'].dt.year.min()}–{mtr_combined['Date'].dt.year.max()}"
                        st.sidebar.success(f"✅ MTR loaded: {csv_count} files | {len(mtr_combined):,} rows | Years: {year_range}")
                    if mtr_skipped:
                        st.sidebar.warning(f"⚠️ {len(mtr_skipped)} MTR files had issues. Check logs.")
                        with st.sidebar.expander("MTR Error Logs"):
                            for s in mtr_skipped:
                                st.write(s)

                sales_parts = []
                if f_b2c:    sales_parts.append(load_amazon_sales(f_b2c,   st.session_state.sku_mapping, "Amazon B2C", config))
                if f_b2b:    sales_parts.append(load_amazon_sales(f_b2b,   st.session_state.sku_mapping, "Amazon B2B", config))
                if f_fk:     sales_parts.append(load_flipkart_sales(f_fk,  st.session_state.sku_mapping))
                if f_meesho: sales_parts.append(load_meesho_sales(f_meesho,st.session_state.sku_mapping))
                if sales_parts:
                    st.session_state.sales_df = pd.concat([d for d in sales_parts if not d.empty], ignore_index=True)

                st.session_state.inventory_df_variant = load_inventory_consolidated(i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=False)
                st.session_state.inventory_df_parent  = load_inventory_consolidated(i_oms, i_fk, i_myntra, i_amz, st.session_state.sku_mapping, group_by_parent=True)

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

if not st.session_state.sku_mapping:
    st.info("👋 **Welcome!** Upload SKU Mapping and click **Load All Data** to begin.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# 14) MAIN TABS
# ══════════════════════════════════════════════════════════════
tab_dash, tab_mtr, tab_inv, tab_po, tab_logistics, tab_forecast, tab_drill = st.tabs([
    "📊 Dashboard", "📑 MTR Analytics", "📦 Inventory", "🎯 PO Engine", "🚚 Logistics", "📈 AI Forecast", "🔍 Deep Dive",
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
        # ── FIX 4: Show year coverage banner so user can immediately verify correct years ──
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

        # ── FIX 5: Cast Month column to str before isin to prevent type-mismatch silent fail ──
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

            # B2B vs B2C comparison table
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
# TAB 3 — INVENTORY
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
# TAB 4 — PO ENGINE
# ══════════════════════════════════════════════════════════════
with tab_po:
    st.subheader("🎯 Purchase Order Recommendations")
    if st.session_state.sales_df.empty or (st.session_state.inventory_df_variant.empty and st.session_state.inventory_df_parent.empty):
        st.warning("⚠️ Please load Sales data and Inventory data first, then click Load All Data.")
    else:
        view_mode = st.radio("Group By", ["By Variant (Size/Color)", "By Parent SKU (Style Only)"], key="po_view_mode")
        st.divider()

        st.markdown("### ⚙️ PO Parameters")
        c1, c2, c3, c4, c5 = st.columns(5)
        velocity    = c1.selectbox("Recent Velocity Period", ["Last 7 Days","Last 30 Days","Last 60 Days","Last 90 Days"], key="po_velocity")
        base_days   = 7 if "7" in velocity else 30 if "30" in velocity else 60 if "60" in velocity else 90
        grace_days  = c2.number_input("Grace Days", 0, 14, 7)
        lead_time   = c3.number_input("Lead Time (Days)", 1, 180, 15)
        target_days = c4.number_input("Target Stock (Days)", 0, 180, 60)
        safety_pct  = c5.slider("Safety Stock %", 0, 100, 20)

        st.markdown("### 📅 Seasonal Forecasting (YoY)")
        sc1, sc2 = st.columns(2)
        use_seasonality = sc1.checkbox("Blend with Last Year's Seasonality?", value=False)
        seasonal_weight = sc2.slider("Historical Weight %", 0, 100, 50) if use_seasonality else 0

        demand_basis = st.selectbox("Demand Basis", ["Sold","Net"], index=0)
        min_den      = st.number_input("Min ADS Denominator", 1, 60, 7)
        total_period = int(base_days + grace_days)

        if "Parent" in view_mode:
            inv_for_po   = st.session_state.inventory_df_parent.copy()
            sales_for_po = st.session_state.sales_df.copy()
            sales_for_po["Sku"] = sales_for_po["Sku"].apply(get_parent_sku)
        else:
            inv_for_po   = st.session_state.inventory_df_variant.copy()
            sales_for_po = st.session_state.sales_df.copy()

        po_df = calculate_po_base(
            sales_df=sales_for_po, inv_df=inv_for_po, period_days=total_period,
            lead_time=lead_time, target_days=target_days, demand_basis=demand_basis,
            min_denominator=int(min_den), use_seasonality=use_seasonality,
            seasonal_weight=seasonal_weight / 100.0
        )

        if po_df.empty:
            st.warning("No PO calculations available. Check that sales and inventory data overlap.")
        else:
            po_df["Days_Left"]        = np.where(po_df["ADS"] > 0, po_df["Total_Inventory"] / po_df["ADS"], 999)
            po_df["Lead_Time_Demand"] = po_df["ADS"] * lead_time
            po_df["Target_Stock"]     = po_df["ADS"] * target_days
            po_df["Base_Requirement"] = po_df["Lead_Time_Demand"] + po_df["Target_Stock"]
            po_df["Safety_Stock"]     = po_df["Base_Requirement"] * (safety_pct / 100)
            po_df["Total_Required"]   = po_df["Base_Requirement"] + po_df["Safety_Stock"]
            po_df["PO_Recommended"]   = (
                np.ceil((po_df["Total_Required"] - po_df["Total_Inventory"]).clip(lower=0) / 5) * 5
            ).astype(int)

            def get_priority(row):
                if row["Days_Left"] < lead_time     and row["PO_Recommended"] > 0: return "🔴 URGENT"
                if row["Days_Left"] < lead_time + 7 and row["PO_Recommended"] > 0: return "🟡 HIGH"
                if row["PO_Recommended"] > 0:                                       return "🟢 MEDIUM"
                return "⚪ OK"

            po_df["Priority"] = po_df.apply(get_priority, axis=1)
            po_needed = po_df[po_df["PO_Recommended"] > 0].sort_values(["Priority","Days_Left"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔴 Urgent", len(po_needed[po_needed["Priority"]=="🔴 URGENT"]))
            m2.metric("🟡 High",   len(po_needed[po_needed["Priority"]=="🟡 HIGH"]))
            m3.metric("🟢 Medium", len(po_needed[po_needed["Priority"]=="🟢 MEDIUM"]))
            m4.metric("📦 Total Units", f"{po_needed['PO_Recommended'].sum():,}")

            st.divider()

            display_cols = ["Priority","OMS_SKU","Total_Inventory","Recent_ADS"]
            if use_seasonality:
                display_cols.append("LY_ADS")
            display_cols.extend(["ADS","Days_Left","PO_Recommended","Stockout_Flag"])
            display_cols = [c for c in display_cols if c in po_needed.columns]

            def highlight_priority(row):
                result = []
                for col in row.index:
                    if col == "Priority":
                        if   "🔴" in str(row[col]): result.append("background-color:#fee2e2;font-weight:bold")
                        elif "🟡" in str(row[col]): result.append("background-color:#fef3c7")
                        else:                        result.append("background-color:#d1fae5")
                    elif col == "PO_Recommended":
                        result.append("background-color:#dbeafe;font-weight:bold")
                    elif col == "Days_Left" and float(row[col]) < float(lead_time):
                        result.append("background-color:#fee2e2;font-weight:bold")
                    else:
                        result.append("")
                return result

            fmt_dict = {c: "{:.3f}" if "ADS" in c else "{:.1f}" if c == "Days_Left" else "{:.0f}"
                        for c in display_cols if c not in ["Priority","OMS_SKU","Stockout_Flag"]}

            st.dataframe(
                po_needed[display_cols].head(200).style.apply(highlight_priority, axis=1).format(fmt_dict),
                use_container_width=True, height=520
            )

            suffix = "parent" if "Parent" in view_mode else "variant"
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "📥 Download PO (CSV)",
                    po_needed[display_cols].to_csv(index=False).encode("utf-8"),
                    f"po_{suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", use_container_width=True
                )
            with c_dl2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    po_needed[display_cols].to_excel(w, sheet_name="PO_Recommendations", index=False)
                st.download_button(
                    "📥 Download PO (Excel)", buf.getvalue(),
                    f"po_{suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# ══════════════════════════════════════════════════════════════
# TAB 5 — LOGISTICS
# ══════════════════════════════════════════════════════════════
with tab_logistics:
    st.subheader("🚚 Logistics Information")
    transfer_df = st.session_state.transfer_df
    if transfer_df.empty:
        st.info("📦 Upload Amazon Stock Transfer file to view logistics data.")
    else:
        st.dataframe(transfer_df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 6 — AI FORECAST
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
                    fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast", line=dict(dash="dash")))
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"🤖 Predicted demand (next {days} days): **{int(fut['yhat'].sum())} units**")
                except Exception as e:
                    st.error(f"Forecast error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 7 — DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab_drill:
    st.subheader("🔍 Deep Dive & Panel Analysis")
    st.info("Utilize this tab for individual SKU breakdown.")

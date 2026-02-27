"""
Flipkart loader — extracted 1-for-1 from app.py.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .helpers import map_to_oms_sku, clean_sku


def _fk_month_from_filename(fname: str):
    base = Path(fname).stem.upper()
    _MON = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        "MARCH": 3, "APRIL": 4, "JUNE": 6, "JULY": 7, "SEPT": 9, "AUGUST": 8,
    }
    parts = re.split(r"[-_\s]", base)
    for p in parts:
        mon_num = _MON.get(p[:5]) or _MON.get(p[:4]) or _MON.get(p[:3])
        if mon_num:
            for q in parts:
                if re.fullmatch(r"20\d{2}", q):
                    return f"{q}-{mon_num:02d}"
    return None


def _parse_flipkart_xlsx(
    file_bytes: bytes, fname: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Sales Report")
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        date_col = "Buyer Invoice Date" if "Buyer Invoice Date" in xl.columns else "Order Date"
        xl["Date"] = pd.to_datetime(xl[date_col], errors="coerce")

        file_month = _fk_month_from_filename(fname)
        if file_month:
            xl["Month"] = file_month
        else:
            xl["Month"] = xl["Date"].dt.to_period("M").astype(str)

        def _fk_txn(event):
            e = str(event).strip()
            if e == "Sale":                 return "Shipment"
            if e == "Return":               return "Refund"
            if e == "Cancellation":         return "Cancel"
            if e == "Return Cancellation":  return "ReturnCancel"
            return "Shipment"

        xl["TxnType"]        = xl["Event Sub Type"].apply(_fk_txn)
        xl["Quantity"]       = pd.to_numeric(xl.get("Item Quantity", 1), errors="coerce").fillna(0).astype("float32")
        xl["Invoice_Amount"] = pd.to_numeric(xl.get("Buyer Invoice Amount", 0), errors="coerce").fillna(0).astype("float32")
        xl["OMS_SKU"]        = xl["SKU"].apply(clean_sku).apply(lambda x: map_to_oms_sku(x, mapping))

        state_col = next((c for c in xl.columns if "Delivery State" in c), None)
        xl["State"] = xl[state_col].fillna("").astype(str).str.upper().str.strip() if state_col else ""

        xl["OrderId"]        = xl.get("Order ID",         xl.get("Order Id",         "")).astype(str)
        xl["BuyerInvoiceId"] = xl.get("Buyer Invoice ID", xl.get("Buyer Invoice Id", "")).astype(str)

        out = xl[["Date", "Month", "TxnType", "Quantity", "Invoice_Amount",
                  "OMS_SKU", "State", "OrderId", "BuyerInvoiceId"]].copy()
        return out.dropna(subset=["Date"])

    except Exception:
        return pd.DataFrame()


def load_flipkart_from_zip(
    zip_bytes: bytes, mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Parse Flipkart master ZIP containing monthly Sales Report XLSXs.
    Returns (combined_df, xlsx_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open Flipkart ZIP: {e}"]

    xlsx_items = [n for n in root_zf.namelist() if n.lower().endswith(".xlsx")]

    for item_name in xlsx_items:
        base = Path(item_name).name
        try:
            file_bytes = root_zf.read(item_name)
            df = _parse_flipkart_xlsx(file_bytes, base, mapping)
            if df.empty:
                skipped.append(f"{base}: no data / unrecognised format")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{base}: {e}")

    if not dfs:
        return pd.DataFrame(), len(xlsx_items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    return combined, len(xlsx_items), skipped


def flipkart_to_sales_rows(fk_df: pd.DataFrame) -> pd.DataFrame:
    if fk_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              fk_df["OMS_SKU"],
        "TxnDate":          fk_df["Date"],
        "Transaction Type": fk_df["TxnType"],
        "Quantity":         fk_df["Quantity"],
        "Units_Effective":  np.where(fk_df["TxnType"] == "Refund",       -fk_df["Quantity"],
                            np.where(fk_df["TxnType"] == "Cancel",         0,
                            np.where(fk_df["TxnType"] == "ReturnCancel",   fk_df["Quantity"],
                                     fk_df["Quantity"]))),
        "Source":           "Flipkart",
        "OrderId":          fk_df["OrderId"],
    })
    return out

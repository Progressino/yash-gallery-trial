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
    # ── Try legacy "Sales Report" sheet first ──────────────────────────────
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Sales Report")
        xl.columns = xl.columns.str.strip()
        if not xl.empty and "Event Sub Type" in xl.columns:
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
        pass

    # ── Try new "Order Export" format (first sheet, columns: SKU ID / Gross Units / Order Date) ──
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, dtype=str)
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        # Detect new format by required columns
        if not ("SKU ID" in xl.columns and "Gross Units" in xl.columns and "Order Date" in xl.columns):
            return pd.DataFrame()

        xl["Date"] = pd.to_datetime(xl["Order Date"], errors="coerce")
        xl = xl.dropna(subset=["Date"])
        if xl.empty:
            return pd.DataFrame()

        file_month = _fk_month_from_filename(fname)

        def _get_month(d):
            return file_month if file_month else d.to_period("M").strftime("%Y-%m")

        for col in ["Gross Units", "Final Sale Units", "Return Units", "Cancellation Units",
                    "Final Sale Amount", "Return Amount"]:
            xl[col] = pd.to_numeric(xl.get(col, 0), errors="coerce").fillna(0)

        xl["OMS_SKU"] = xl["SKU ID"].apply(
            lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
        )

        # Synthetic OrderId: ProductId + SKU + date string (no real order ID available)
        product_id_col = "Product Id" if "Product Id" in xl.columns else ""
        if product_id_col:
            xl["_OrderId"] = (
                xl[product_id_col].astype(str) + "_" +
                xl["SKU ID"].astype(str) + "_" +
                xl["Date"].dt.strftime("%Y%m%d")
            )
        else:
            xl["_OrderId"] = xl["SKU ID"].astype(str) + "_" + xl["Date"].dt.strftime("%Y%m%d")

        rows: List[pd.DataFrame] = []

        # Shipment rows — use Gross Units (all orders placed, including cancelled)
        ship = xl[xl["Gross Units"] > 0].copy()
        if not ship.empty:
            rows.append(pd.DataFrame({
                "Date":           ship["Date"],
                "Month":          ship["Date"].apply(_get_month),
                "TxnType":        "Shipment",
                "Quantity":       ship["Gross Units"].astype("float32"),
                "Invoice_Amount": ship["Final Sale Amount"].astype("float32"),
                "OMS_SKU":        ship["OMS_SKU"],
                "State":          "",
                "OrderId":        ship["_OrderId"],
                "BuyerInvoiceId": "",
            }))

        # Refund rows
        ret = xl[xl["Return Units"] > 0].copy()
        if not ret.empty:
            rows.append(pd.DataFrame({
                "Date":           ret["Date"],
                "Month":          ret["Date"].apply(_get_month),
                "TxnType":        "Refund",
                "Quantity":       ret["Return Units"].astype("float32"),
                "Invoice_Amount": ret["Return Amount"].astype("float32"),
                "OMS_SKU":        ret["OMS_SKU"],
                "State":          "",
                "OrderId":        ret["_OrderId"],
                "BuyerInvoiceId": "",
            }))

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

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


def _parse_flipkart_orders_sheet(
    file_bytes: bytes, fname: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Parse the 'Orders' sheet from Flipkart Seller Hub payment/settlement reports
    (UUID-named XLSX files).  Header structure has 3 lead rows:
      row 0 = group labels, row 1 = column names, row 2 = sub-headers / blanks.
    Data starts at row 3.
    """
    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Orders", header=None, dtype=str)
        if len(raw) < 4:
            return pd.DataFrame()

        # Row 1 (0-indexed) has the actual column names
        col_names = raw.iloc[1].fillna("").astype(str).tolist()
        df = raw.iloc[3:].reset_index(drop=True)
        df.columns = col_names

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Date: use Order Date (not Due Date which is payment settlement date)
        date_col = next((c for c in df.columns if "order date" in c.lower()), None)
        if not date_col:
            return pd.DataFrame()
        df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_Date"])
        if df.empty:
            return pd.DataFrame()

        # SKU
        sku_col = next((c for c in df.columns if "seller sku" in c.lower()), None)
        if not sku_col:
            return pd.DataFrame()
        df["_OMS_SKU"] = df[sku_col].apply(lambda x: map_to_oms_sku(clean_sku(str(x)), mapping))

        # Quantity
        qty_col = next((c for c in df.columns if c.strip().lower() == "quantity"), None)
        df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1).astype("float32") if qty_col else 1.0

        # Revenue
        rev_col = next((c for c in df.columns if "sale amount" in c.lower()), None)
        df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0).astype("float32") if rev_col else 0.0

        # Transaction type: Return Type not-null → Refund, else Shipment
        ret_col = next((c for c in df.columns if "return type" in c.lower()), None)
        if ret_col:
            df["_TxnType"] = df[ret_col].apply(
                lambda x: "Refund" if pd.notna(x) and str(x).strip() not in ("", "nan") else "Shipment"
            )
        else:
            df["_TxnType"] = "Shipment"

        # Order ID
        order_col = next((c for c in df.columns if c.strip().lower() == "order id"), None)
        df["_OrderId"] = df[order_col].fillna("").astype(str) if order_col else ""

        # State (not available in this format)
        file_month = _fk_month_from_filename(fname)

        out = pd.DataFrame({
            "Date":           df["_Date"],
            "Month":          file_month if file_month else df["_Date"].dt.to_period("M").astype(str),
            "TxnType":        df["_TxnType"],
            "Quantity":       df["_Qty"],
            "Invoice_Amount": df["_Rev"],
            "OMS_SKU":        df["_OMS_SKU"],
            "State":          "",
            "OrderId":        df["_OrderId"],
            "BuyerInvoiceId": "",
        })
        return out.dropna(subset=["Date"])

    except Exception:
        return pd.DataFrame()


def _parse_flipkart_earn_more(
    file_bytes: bytes, fname: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Parse Flipkart 'earn_more_report' sheet (daily seller performance report).
    Columns: Product Id, SKU ID, Category, Brand, Vertical, Order Date,
             Fulfillment Type, Location Id, Gross Units, GMV,
             Cancellation Units, Cancellation Amount,
             Return Units, Return Amount, Final Sale Units, Final Sale Amount.
    Expands aggregated rows into Shipment/Refund/Cancel rows.
    """
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name="earn_more_report", dtype=str)
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        # Normalize date
        xl["Date"] = pd.to_datetime(xl["Order Date"], errors="coerce")
        xl = xl.dropna(subset=["Date"])
        if xl.empty:
            return pd.DataFrame()

        file_month = _fk_month_from_filename(fname)

        def _get_month(d):
            return file_month if file_month else d.to_period("M").strftime("%Y-%m")

        # Normalize units columns
        for col in ["Gross Units", "Final Sale Units", "Return Units", "Cancellation Units",
                    "Final Sale Amount", "Return Amount"]:
            xl[col] = pd.to_numeric(xl.get(col, 0), errors="coerce").fillna(0)

        # SKU mapping
        xl["OMS_SKU"] = xl["SKU ID"].apply(
            lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
        )

        rows: List[pd.DataFrame] = []

        # Shipment rows — use Gross Units (all orders placed, including cancelled)
        ship = xl[xl["Gross Units"] > 0].copy()
        if not ship.empty:
            rows.append(pd.DataFrame({
                "Date":           ship["Date"],
                "Month":          ship["Date"].apply(_get_month),
                "TxnType":        "Shipment",
                "Quantity":       ship["Gross Units"].astype("float32"),
                "Invoice_Amount": ship["Final Sale Amount"].astype("float32"),
                "OMS_SKU":        ship["OMS_SKU"],
                "State":          "",
                "OrderId":        "",
                "BuyerInvoiceId": "",
            }))

        # Refund rows
        ret = xl[xl["Return Units"] > 0].copy()
        if not ret.empty:
            rows.append(pd.DataFrame({
                "Date":           ret["Date"],
                "Month":          ret["Date"].apply(_get_month),
                "TxnType":        "Refund",
                "Quantity":       ret["Return Units"].astype("float32"),
                "Invoice_Amount": ret["Return Amount"].astype("float32"),
                "OMS_SKU":        ret["OMS_SKU"],
                "State":          "",
                "OrderId":        "",
                "BuyerInvoiceId": "",
            }))

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    except Exception:
        return pd.DataFrame()


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

"""
Meesho loader — extracted 1-for-1 from app.py.
"""
import io
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _parse_meesho_inner_zip(inner_zf) -> pd.DataFrame:
    files = {f.lower(): f for f in inner_zf.namelist()}
    rows  = []

    def _best_date_col(df, prefer_return: bool = False) -> str:
        cols_lower = {c.lower(): c for c in df.columns}
        if prefer_return:
            for candidate in ["return_date", "return_created_date", "return_pickup_date",
                               "pickup_date", "reverse_pickup_date"]:
                if candidate in cols_lower:
                    return cols_lower[candidate]
        for candidate in ["order_date", "created_date", "order_created_date"]:
            if candidate in cols_lower:
                return cols_lower[candidate]
        return None

    if "tcs_sales.xlsx" in files:
        with inner_zf.open(files["tcs_sales.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=False)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("total_invoice_value", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state_new", "")
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Shipment"
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_Month"]])

    if "tcs_sales_return.xlsx" in files:
        with inner_zf.open(files["tcs_sales_return.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=True)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("total_invoice_value", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state_new", "")
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Refund"
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_Month"]])

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
            df["_Month"]   = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_Month"]])

    if "reverse.xlsx" in files and not any(
        (r["_TxnType"] == "Refund").any() for r in rows if not r.empty
    ):
        with inner_zf.open(files["reverse.xlsx"]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=True)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("meesho_price", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state", df.get("state", ""))
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            df["_TxnType"] = "Refund"
            df["_Month"]   = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_Month"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.columns = ["Date", "TxnType", "Quantity", "Invoice_Amount", "State", "OrderId", "_Month"]
    out["Date"]           = pd.to_datetime(out["Date"], errors="coerce")
    out["Quantity"]       = out["Quantity"].astype("float32")
    out["Invoice_Amount"] = out["Invoice_Amount"].astype("float32")
    out["State"]          = out["State"].astype(str).str.upper().str.strip()
    out["Month"]          = out["_Month"].where(
        out["_Month"].notna(),
        out["Date"].dt.to_period("M").astype(str)
    )
    out = out.drop(columns=["_Month"])
    return out.dropna(subset=["Date"])


def parse_meesho_csv(csv_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    """
    Parse a Meesho daily orders CSV report (non-ZIP format).
    Handles columns: Reason for Credit Entry, Sub Order No, Order Date,
    Customer State, SKU, Quantity, Supplier Discounted Price, etc.
    Returns (df, status_message) with columns matching meesho_df schema.
    """
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False,
                             on_bad_lines="skip", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return pd.DataFrame(), f"Parse error: {e}"
    else:
        return pd.DataFrame(), "Encoding error"

    if df.empty:
        return pd.DataFrame(), "Empty file"

    df.columns = df.columns.str.strip().str.lower()

    # Date column: "order date"
    date_col = next((c for c in df.columns if "order date" in c or c == "date"), None)
    if not date_col:
        return pd.DataFrame(), "No date column found"
    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame(), "All dates invalid"

    # Status: "reason for credit entry" (SHIPPED/CANCELLED/RETURNED) or "order status"
    status_col = next((c for c in df.columns if "reason" in c or "order status" in c
                       or c == "status"), None)
    def _txn(s):
        s = str(s).strip().upper()
        if "RETURN" in s or "RTO" in s: return "Refund"
        if "CANCEL" in s: return "Cancel"
        return "Shipment"
    df["_TxnType"] = df[status_col].apply(_txn) if status_col else "Shipment"

    # Quantity
    qty_col = next((c for c in df.columns if c == "quantity"), None)
    df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1) if qty_col else 1.0

    # Revenue: prefer discounted price
    rev_col = next((c for c in df.columns if "discounted price" in c
                    or "selling price" in c or "listed price" in c), None)
    df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0) if rev_col else 0.0

    # State
    state_col = next((c for c in df.columns if "customer state" in c or c == "state"), None)

    # Order ID
    order_col = next((c for c in df.columns if "sub order" in c or "order no" in c
                      or "packet id" in c or "packet" in c), None)

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"].astype("float32"),
        "Invoice_Amount": df["_Rev"].astype("float32"),
        "State":          df[state_col].fillna("").str.upper().str.strip() if state_col else "",
        "OrderId":        df[order_col].fillna("").astype(str) if order_col else "",
    })
    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    return out.dropna(subset=["Date"]), "OK"


def load_meesho_from_zip(zip_bytes: bytes) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Parse Meesho master ZIP (ZIP of ZIPs).
    Returns (combined_df, inner_zip_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open Meesho ZIP: {e}"]

    items = [n for n in root_zf.namelist() if n.lower().endswith(".zip")]

    for item_name in items:
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

    if not dfs:
        return pd.DataFrame(), len(items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    return combined, len(items), skipped


def meesho_to_sales_rows(meesho_df: pd.DataFrame) -> pd.DataFrame:
    if meesho_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              "MEESHO_TOTAL",
        "TxnDate":          meesho_df["Date"],
        "Transaction Type": meesho_df["TxnType"],
        "Quantity":         meesho_df["Quantity"],
        "Units_Effective":  np.where(meesho_df["TxnType"] == "Refund", -meesho_df["Quantity"],
                            np.where(meesho_df["TxnType"] == "Cancel",  0, meesho_df["Quantity"])),
        "Source":           "Meesho",
        "OrderId":          meesho_df["OrderId"],
    })
    return out

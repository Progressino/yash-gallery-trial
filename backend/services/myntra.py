"""
Myntra PPMP loader — extracted 1-for-1 from app.py.
"""
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .helpers import map_to_oms_sku


def _parse_myntra_csv(
    csv_bytes: bytes, filename: str, mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, low_memory=False, on_bad_lines="skip")
    except Exception as e:
        return pd.DataFrame(), f"CSV parse error: {e}"

    if df.empty:
        return pd.DataFrame(), "Empty file"

    df.columns = df.columns.str.strip().str.lower()

    # PPMP: "order_created_date"; Seller Orders Report: "created on" or "order date"
    date_col = next((c for c in df.columns if
                     "order_created_date" in c or "order_date" in c or
                     c in ("created date", "created_date", "order date", "purchase date",
                           "created on")), None)
    # Generic fallback: any column whose name ends with "date" or "on" containing "creat"
    if not date_col:
        date_col = next((c for c in df.columns if c.endswith("date") or c.endswith("_date")), None)
    if not date_col:
        return pd.DataFrame(), "No date column found"

    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    null_mask = df["_Date"].isna()
    if null_mask.any():
        df.loc[null_mask, "_Date"] = pd.to_datetime(
            df.loc[null_mask, date_col].astype(str), format="%Y%m%d", errors="coerce"
        )
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame(), "All dates invalid"

    # PPMP: "sku_id"; Seller Orders Report: "sku id" or "sku"
    sku_col = next((c for c in df.columns if c in ["sku_id", "skuid", "sku", "sku id", "seller sku"]), None)
    if not sku_col:
        return pd.DataFrame(), "No SKU column"
    df["_OMS_SKU"] = df[sku_col].apply(lambda x: map_to_oms_sku(str(x).strip(), mapping))

    qty_col = next((c for c in df.columns if c == "quantity"), None)
    df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1) if qty_col else 1.0

    rev_col = next(
        (c for c in df.columns if c in ["invoiceamount", "invoice_amount", "net_amount", "shipment_value"]),
        None,
    )
    df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0) if rev_col else 0.0

    # Status column detection (priority order)
    # Handles both PPMP (underscore_separated) and Seller Orders Report (space separated)
    _status_candidates = (
        [c for c in df.columns if "order_status" in c or "order status" in c]
        + [c for c in df.columns if c in (
            "status", "packet_status", "shipment_status",
            "sub_order_status", "current_status", "item_status",
            "article_status", "delivery_status",
            "order status", "item status", "shipment status",
        )]
        + [c for c in df.columns if "forward" in c or "reverse" in c]
        + [c for c in df.columns if "status" in c]
        + [c for c in df.columns if c in (
            "transaction_type", "txn_type", "return_type", "order_type", "type",
        )]
    )
    status_col = next(iter(dict.fromkeys(_status_candidates)), None)

    def _myntra_txn(s):
        s = str(s).strip().upper()
        if s in ("FORWARD", "FWD"):
            return "Shipment"
        if s in ("REVERSE", "REV"):
            return "Refund"
        if ("RETURN" in s or "REVERSE" in s or s.startswith("RTO")
                or s.startswith("RTD") or s in ("R", "RS", "RD", "RTOD")):
            return "Refund"
        if "CANCEL" in s or s in ("F", "IC", "FAILED"):
            return "Cancel"
        if s in ("C", "SH", "PK", "D", "S", "SHIPPED", "CONFIRMED", "DELIVERED",
                 "PACKED", "PACKING_IN_PROGRESS", "READY_FOR_DISPATCH",
                 "MANIFESTED", "OUT_FOR_DELIVERY", "WP"):
            return "Shipment"
        return "Shipment"

    df["_TxnType"] = df[status_col].apply(_myntra_txn) if status_col else "Shipment"

    state_col  = next((c for c in df.columns if c in [
        "state", "customer_delivery_state_code", "buyer state", "ship state",
    ]), None)
    pm_col     = next((c for c in df.columns if "payment_method" in c or "payment method" in c), None)
    wh_col     = next((c for c in df.columns if "warehouse_id" in c or "warehouse id" in c), None)
    # PPMP: "order_id" / "packet_id"; Seller Orders Report: "order id" / "seller order id"
    order_col  = next((c for c in df.columns if c in [
        "order_id", "packet_id", "order id", "sub order id", "suborder id",
        "seller order id", "order id fk", "store order id",
    ]), None)
    _raw_status = df[status_col].fillna("").astype(str).str.strip() if status_col else ""

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "OMS_SKU":        df["_OMS_SKU"],
        "TxnType":        df["_TxnType"],
        "RawStatus":      _raw_status,
        "Quantity":       df["_Qty"].astype("float32"),
        "Invoice_Amount": df["_Rev"].astype("float32"),
        "State":          df[state_col].fillna("").str.upper().str.strip() if state_col else "",
        "Payment_Method": df[pm_col].fillna("") if pm_col else "",
        "Warehouse_Id":   df[wh_col].fillna("") if wh_col else "",
        "OrderId":        df[order_col].fillna("") if order_col else "",
    })
    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    return out, f"OK | status_col={status_col!r}"


def load_myntra_from_zip(
    zip_bytes: bytes, mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Parse Myntra PPMP master ZIP containing monthly CSVs.
    Returns (combined_df, csv_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open ZIP: {e}"]

    csv_items = [n for n in root_zf.namelist() if n.lower().endswith(".csv")]

    for item_name in csv_items:
        base = Path(item_name).name
        try:
            data = root_zf.read(item_name)
            df, msg = _parse_myntra_csv(data, base, mapping)
            if df.empty:
                skipped.append(f"{base}: {msg}")
            else:
                dfs.append(df)
                if not msg.startswith("OK"):
                    skipped.append(f"{base}: Partial ({msg})")
        except Exception as e:
            skipped.append(f"{base}: {e}")

    if not dfs:
        return pd.DataFrame(), len(csv_items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["OrderId", "OMS_SKU", "TxnType", "Date"], keep="first"
    )
    return combined, len(csv_items), skipped


def myntra_to_sales_rows(myntra_df: pd.DataFrame) -> pd.DataFrame:
    if myntra_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Sku":              myntra_df["OMS_SKU"],
        "TxnDate":          myntra_df["Date"],
        "Transaction Type": myntra_df["TxnType"],
        "Quantity":         myntra_df["Quantity"],
        "Units_Effective":  np.where(myntra_df["TxnType"] == "Refund", -myntra_df["Quantity"],
                            np.where(myntra_df["TxnType"] == "Cancel",  0, myntra_df["Quantity"])),
        "Source":           "Myntra",
        "OrderId":          myntra_df["OrderId"],
    })
    return out

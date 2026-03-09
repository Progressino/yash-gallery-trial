"""
Snapdeal loader — parses monthly Excel/ZIP reports from Snapdeal Seller Panel.

Typical Snapdeal report columns (fuzzy-matched):
  Order ID / Order Code / Suborder ID
  Order Date / Created Date / Order Created Date
  SKU / Product SKU / Seller SKU Code / SKU Code
  Quantity / Qty
  Status / Order Status / Sub Status / Fulfillment Status
    → Shipped / Delivered / RTO / Returned / Cancelled
  Sale Price / Selling Price / Deal Price / MRP / Total Amount
  State / Delivery State / Customer State

Returns a DataFrame with:
  Date, TxnType (Shipment/Refund/Cancel), Quantity, Invoice_Amount,
  State, OrderId, OMS_SKU, Month
"""
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers import map_to_oms_sku, clean_sku


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_col_fuzzy(cols: List[str], keywords: List[str]) -> Optional[str]:
    for col in cols:
        col_l = col.lower()
        if any(kw in col_l for kw in keywords):
            return col
    return None


def _snapdeal_txn(status: str) -> str:
    """Map Snapdeal order status → Shipment / Refund / Cancel."""
    s = str(status).strip().lower()
    if any(x in s for x in ["return", "rto", "refund", "returned"]):
        return "Refund"
    if any(x in s for x in ["cancel", "cancelled"]):
        return "Cancel"
    # Delivered / Shipped / Dispatched / Picked / Processing
    return "Shipment"


def _parse_snapdeal_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Given a raw DataFrame (from Excel/CSV), extract and normalise columns.
    Returns cleaned DataFrame with: Date, TxnType, Quantity, Invoice_Amount,
    State, OrderId, OMS_SKU, Month.
    """
    if df.empty:
        return pd.DataFrame()

    # Normalise column names for fuzzy search
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    # ── Date ──────────────────────────────────────────────────
    date_col = _find_col(cols, [
        "Order Date", "Order Created Date", "Created Date",
        "Order Placed Date", "Placed Date",
    ]) or _find_col_fuzzy(cols, ["order date", "created date", "placed date"])

    if date_col is None:
        return pd.DataFrame()

    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame()

    # ── Order ID ──────────────────────────────────────────────
    order_col = _find_col(cols, [
        "Order ID", "Order Code", "Suborder ID", "Sub Order ID",
        "Order Number", "Snapdeal Order ID",
    ]) or _find_col_fuzzy(cols, ["order id", "order code", "suborder"])
    df["_OrderId"] = df[order_col].fillna("").astype(str) if order_col else ""

    # ── SKU ───────────────────────────────────────────────────
    sku_col = _find_col(cols, [
        "SKU", "Product SKU", "Seller SKU Code", "SKU Code",
        "Seller SKU", "Item SKU",
    ]) or _find_col_fuzzy(cols, ["sku", "seller sku"])
    if sku_col:
        df["_OMS_SKU"] = df[sku_col].apply(
            lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
        )
    else:
        df["_OMS_SKU"] = "UNKNOWN"

    # ── Quantity ──────────────────────────────────────────────
    qty_col = _find_col(cols, ["Quantity", "Qty", "Item Qty"]) or \
              _find_col_fuzzy(cols, ["quantity", "qty"])
    df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1).astype("float32") \
                 if qty_col else 1.0

    # ── Revenue ───────────────────────────────────────────────
    rev_col = _find_col(cols, [
        "Sale Price", "Selling Price", "Deal Price",
        "Total Amount", "Invoice Amount", "Net Amount", "Buyer Price",
    ]) or _find_col_fuzzy(cols, ["sale price", "selling price", "total amount", "net amount"])
    df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0).astype("float32") \
                 if rev_col else 0.0

    # ── Transaction Type ─────────────────────────────────────
    status_col = _find_col(cols, [
        "Status", "Order Status", "Sub Status", "Fulfillment Status",
        "Shipment Status", "Item Status",
    ]) or _find_col_fuzzy(cols, ["status"])
    if status_col:
        df["_TxnType"] = df[status_col].apply(_snapdeal_txn)
    else:
        df["_TxnType"] = "Shipment"

    # ── State ─────────────────────────────────────────────────
    state_col = _find_col(cols, [
        "State", "Delivery State", "Customer State",
        "Shipping State", "Ship State",
    ]) or _find_col_fuzzy(cols, ["state"])
    df["_State"] = df[state_col].fillna("").astype(str).str.upper().str.strip() \
                   if state_col else ""

    # ── Build output ──────────────────────────────────────────
    out = pd.DataFrame({
        "Date":           df["_Date"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"],
        "Invoice_Amount": df["_Rev"],
        "State":          df["_State"],
        "OrderId":        df["_OrderId"],
        "OMS_SKU":        df["_OMS_SKU"],
    })
    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    return out.dropna(subset=["Date"])


def _parse_snapdeal_file(
    file_bytes: bytes,
    fname: str,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Parse a single Snapdeal Excel or CSV report."""
    fn_lower = fname.lower()
    try:
        if fn_lower.endswith(".csv"):
            for enc in ("utf-8", "ISO-8859-1"):
                try:
                    raw = pd.read_csv(
                        io.BytesIO(file_bytes), dtype=str,
                        encoding=enc, on_bad_lines="skip",
                    )
                    break
                except UnicodeDecodeError:
                    continue
        else:
            # Try all sheets, use the largest one with data
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            dfs_by_sheet = []
            for sheet in xl.sheet_names:
                try:
                    _df = xl.parse(sheet, dtype=str)
                    if not _df.empty:
                        dfs_by_sheet.append(_df)
                except Exception:
                    continue
            if not dfs_by_sheet:
                return pd.DataFrame()
            # Pick sheet with most rows
            raw = max(dfs_by_sheet, key=len)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    return _parse_snapdeal_df(raw, mapping)


def load_snapdeal_from_zip(
    zip_bytes: bytes,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Parse Snapdeal master ZIP (may contain .xlsx/.csv files, optionally nested ZIPs).
    Returns (combined_df, file_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open Snapdeal ZIP: {e}"]

    all_names = root_zf.namelist()
    # Collect all Excel/CSV files (including inside nested ZIPs)
    file_items: List[Tuple[str, bytes]] = []

    for name in all_names:
        base = Path(name).name
        if "__MACOSX" in name or base.startswith("."):
            continue
        lower = base.lower()
        if lower.endswith(".zip"):
            # Nested ZIP (monthly ZIP of files)
            try:
                inner_data = root_zf.read(name)
                with zipfile.ZipFile(io.BytesIO(inner_data)) as inner_zf:
                    for inner_name in inner_zf.namelist():
                        inner_base = Path(inner_name).name
                        if inner_base.lower().endswith((".xlsx", ".xls", ".csv")):
                            file_items.append((inner_base, inner_zf.read(inner_name)))
            except Exception as e:
                skipped.append(f"{base}: {e}")
        elif lower.endswith((".xlsx", ".xls", ".csv")):
            try:
                file_items.append((base, root_zf.read(name)))
            except Exception as e:
                skipped.append(f"{base}: {e}")

    for fname, file_bytes in file_items:
        try:
            df = _parse_snapdeal_file(file_bytes, fname, mapping)
            if df.empty:
                skipped.append(f"{fname}: No recognised data / empty")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{fname}: {e}")

    if not dfs:
        return pd.DataFrame(), len(file_items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    return combined, len(file_items), skipped


def snapdeal_to_sales_rows(snapdeal_df: pd.DataFrame) -> pd.DataFrame:
    """Convert snapdeal_df to unified sales_df format."""
    if snapdeal_df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Sku":              snapdeal_df["OMS_SKU"],
        "TxnDate":          snapdeal_df["Date"],
        "Transaction Type": snapdeal_df["TxnType"],
        "Quantity":         snapdeal_df["Quantity"],
        "Units_Effective":  np.where(
            snapdeal_df["TxnType"] == "Refund",  -snapdeal_df["Quantity"],
            np.where(snapdeal_df["TxnType"] == "Cancel", 0, snapdeal_df["Quantity"])
        ),
        "Source":           "Snapdeal",
        "OrderId":          snapdeal_df["OrderId"],
    })

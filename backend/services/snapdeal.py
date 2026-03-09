"""
Snapdeal loader — parses monthly Excel/ZIP reports from Snapdeal Seller Panel.

Snapdeal reports often have 1-5 title/metadata rows before the actual column
headers, so we auto-detect the real header row before parsing.

Known Snapdeal column names (from actual seller panel exports):
  Order Code / Sub Order Code / Order ID
  Order Date / Order Placed Date
  Product Name
  Seller SKU Code / Snap SKU / SKU
  Quantity
  Order Status / Status / Sub Status
  Deal Price / Selling Price / Sale Price / Total Amount
  State/UT / State / Delivery State
  City / Buyer Name / Pincode

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


# ── Keywords that appear in the REAL header row of Snapdeal reports ──────────
_HEADER_KEYWORDS = [
    "order code", "order date", "order id", "sub order",
    "seller sku", "snap sku", "order status", "deal price",
    "selling price", "quantity", "state/ut", "buyer name",
    "product name", "order placed", "suborder",
]


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


def _detect_header_row(raw: pd.DataFrame) -> int:
    """
    Scan the first 15 rows to find the row that most looks like a header
    (contains the most recognised Snapdeal column keyword matches).
    Returns the 0-based row index, or 0 if nothing found.
    """
    best_row, best_score = 0, 0
    for i in range(min(15, len(raw))):
        row_vals = [str(v).lower().strip() for v in raw.iloc[i].values]
        score = sum(
            1 for cell in row_vals
            if any(kw in cell for kw in _HEADER_KEYWORDS)
        )
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _snapdeal_txn(status: str) -> str:
    """Map Snapdeal order status → Shipment / Refund / Cancel."""
    s = str(status).strip().lower()
    if any(x in s for x in ["return", "rto", "refund", "returned", "reverse"]):
        return "Refund"
    if any(x in s for x in ["cancel", "cancelled", "cancellation"]):
        return "Cancel"
    # Delivered / Shipped / Dispatched / Picked / Processing / Approved
    return "Shipment"


def _parse_snapdeal_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Given a raw DataFrame (already with correct headers), extract and normalise.
    Returns cleaned DataFrame with: Date, TxnType, Quantity, Invoice_Amount,
    State, OrderId, OMS_SKU, Month.
    """
    if df.empty:
        return pd.DataFrame()

    # Normalise column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop rows where all values are NaN
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame()

    cols = list(df.columns)

    # ── Date ──────────────────────────────────────────────────
    date_col = _find_col(cols, [
        "Order Date", "Order Placed Date", "Order Created Date",
        "Created Date", "Placed Date", "Date",
        "Order_Date", "OrderDate",
    ]) or _find_col_fuzzy(cols, ["order date", "placed date", "created date"])

    if date_col is None:
        return pd.DataFrame()

    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame()

    # ── Order ID — prefer Sub Order Code (most specific) ─────
    order_col = _find_col(cols, [
        "Sub Order Code", "Suborder Code", "Sub Order ID",
        "Order Code", "Order ID", "Snapdeal Order ID", "Order Number",
    ]) or _find_col_fuzzy(cols, ["sub order", "order code", "order id"])
    df["_OrderId"] = df[order_col].fillna("").astype(str) if order_col else ""

    # ── SKU ───────────────────────────────────────────────────
    sku_col = _find_col(cols, [
        "Seller SKU Code", "Seller SKU", "Snap SKU", "SKU Code",
        "SKU", "Product SKU", "Item SKU", "SellerSKUCode",
    ]) or _find_col_fuzzy(cols, ["seller sku", "snap sku", "sku code", "sku"])
    if sku_col:
        df["_OMS_SKU"] = df[sku_col].apply(
            lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
        )
    else:
        df["_OMS_SKU"] = "UNKNOWN"

    # ── Quantity ──────────────────────────────────────────────
    qty_col = _find_col(cols, [
        "Quantity", "Qty", "Item Qty", "No. of Units", "Units",
    ]) or _find_col_fuzzy(cols, ["quantity", "qty", "units"])
    df["_Qty"] = pd.to_numeric(
        df[qty_col], errors="coerce"
    ).fillna(1).astype("float32") if qty_col else 1.0

    # ── Revenue ───────────────────────────────────────────────
    rev_col = _find_col(cols, [
        "Deal Price", "Selling Price", "Sale Price",
        "Total Amount", "Invoice Amount", "Net Amount",
        "Buyer Price", "MRP", "Product Price",
    ]) or _find_col_fuzzy(cols, [
        "deal price", "selling price", "sale price",
        "total amount", "net amount", "invoice",
    ])
    df["_Rev"] = pd.to_numeric(
        df[rev_col], errors="coerce"
    ).fillna(0).astype("float32") if rev_col else 0.0

    # ── Transaction Type ─────────────────────────────────────
    status_col = _find_col(cols, [
        "Order Status", "Status", "Sub Status",
        "Fulfillment Status", "Shipment Status", "Item Status",
        "Order Sub Status",
    ]) or _find_col_fuzzy(cols, ["status"])
    if status_col:
        df["_TxnType"] = df[status_col].apply(_snapdeal_txn)
    else:
        df["_TxnType"] = "Shipment"

    # ── State ─────────────────────────────────────────────────
    state_col = _find_col(cols, [
        "State/UT", "State", "Delivery State", "Customer State",
        "Shipping State", "Ship State", "Buyer State",
    ]) or _find_col_fuzzy(cols, ["state"])
    df["_State"] = (
        df[state_col].fillna("").astype(str).str.upper().str.strip()
        if state_col else ""
    )

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
    """
    Parse a single Snapdeal Excel or CSV report.
    Auto-detects the real header row (Snapdeal reports have title rows on top).
    """
    fn_lower = fname.lower()

    try:
        if fn_lower.endswith(".csv"):
            # CSV: try auto-detect header row too
            raw_no_header: Optional[pd.DataFrame] = None
            for enc in ("utf-8", "ISO-8859-1"):
                try:
                    raw_no_header = pd.read_csv(
                        io.BytesIO(file_bytes), dtype=str, header=None,
                        encoding=enc, on_bad_lines="skip",
                    )
                    break
                except UnicodeDecodeError:
                    continue
            if raw_no_header is None or raw_no_header.empty:
                return pd.DataFrame()
            hdr_row = _detect_header_row(raw_no_header)
            cols = raw_no_header.iloc[hdr_row].astype(str).tolist()
            raw = raw_no_header.iloc[hdr_row + 1:].reset_index(drop=True)
            raw.columns = cols
        else:
            # Excel: iterate sheets, auto-detect header row on each
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            best_df: Optional[pd.DataFrame] = None
            best_rows = 0

            for sheet in xl.sheet_names:
                try:
                    raw_no_header = xl.parse(sheet, header=None, dtype=str)
                    if raw_no_header.empty:
                        continue
                    hdr_row = _detect_header_row(raw_no_header)
                    cols = raw_no_header.iloc[hdr_row].astype(str).tolist()
                    candidate = raw_no_header.iloc[hdr_row + 1:].reset_index(drop=True)
                    candidate.columns = cols
                    candidate = candidate.dropna(how="all")
                    if len(candidate) > best_rows:
                        best_rows = len(candidate)
                        best_df = candidate
                except Exception:
                    continue

            if best_df is None or best_df.empty:
                return pd.DataFrame()
            raw = best_df

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
    Parse Snapdeal master ZIP (may contain .xlsx/.csv files or nested ZIPs).
    Returns (combined_df, file_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open Snapdeal ZIP: {e}"]

    all_names = root_zf.namelist()
    file_items: List[Tuple[str, bytes]] = []

    for name in all_names:
        base = Path(name).name
        if "__MACOSX" in name or base.startswith("."):
            continue
        lower = base.lower()
        if lower.endswith(".zip"):
            # Nested ZIP
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

"""
Snapdeal loader — parses monthly Excel/ZIP reports from Snapdeal Seller Panel.

Uses a two-stage approach:
 1. Scan every possible header row (0-15) for each sheet
 2. For each candidate layout, try full column detection
 3. Fall back to value-based date detection if name-based fails

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


# ── Keywords that indicate we're looking at a real header row ─────────────────
_HEADER_KEYWORDS = [
    "order", "date", "sku", "quantity", "qty", "status", "price",
    "state", "city", "product", "seller", "snap", "buyer", "amount",
    "suborder", "sub order", "deal", "shipment", "dispatch",
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


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 20) -> int:
    """
    Return the 0-based row index most likely to be the real header.
    Picks the row with the most keyword hits across its cell values.
    """
    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw))):
        row_vals = [str(v).lower().strip() for v in raw.iloc[i].values
                    if pd.notna(v) and str(v).strip()]
        score = sum(
            1 for cell in row_vals
            if any(kw in cell for kw in _HEADER_KEYWORDS)
        )
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _find_date_col_by_value(df: pd.DataFrame) -> Optional[str]:
    """
    Scan all columns for one whose non-null values parse as dates ≥50% of the time.
    Returns the first such column name, or None.
    """
    for col in df.columns:
        try:
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            hit_rate = parsed.notna().mean()
            if hit_rate >= 0.5:
                # Also verify they look like real dates (year 2000-2030)
                years = parsed.dropna().dt.year
                if years.between(2000, 2030).all():
                    return col
        except Exception:
            continue
    return None


def _snapdeal_txn(status: str) -> str:
    """Map Snapdeal order status → Shipment / Refund / Cancel."""
    s = str(status).strip().lower()
    if any(x in s for x in ["return", "rto", "refund", "returned", "reverse"]):
        return "Refund"
    if any(x in s for x in ["cancel", "cancelled", "cancellation"]):
        return "Cancel"
    return "Shipment"


def _parse_snapdeal_df(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Given a DataFrame (already with correct column headers), parse and normalise.
    Returns (result_df, debug_msg, field_map).
    result_df is empty on failure; debug_msg explains why.
    field_map shows which raw column was mapped to each field.
    """
    if df.empty:
        return pd.DataFrame(), "empty", {}

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame(), "all-NaN rows", {}

    cols = list(df.columns)

    field_map: dict = {"raw_cols": cols}

    # ── Date — name-based first, then value-based fallback ───────
    date_col = _find_col(cols, [
        "Order Date", "Order Placed Date", "Order Created Date",
        "Created Date", "Placed Date", "Date", "Order_Date",
        "Dispatch Date", "Ship Date", "Settlement Date", "Payment Date",
    ]) or _find_col_fuzzy(cols, ["order date", "placed date", "created date", "dispatch date", "settlement date", "payment date"])

    if date_col is None:
        date_col = _find_date_col_by_value(df)

    field_map["date_col"] = date_col

    if date_col is None:
        return pd.DataFrame(), f"no date col (cols: {cols[:15]})", field_map

    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame(), f"date col '{date_col}' yielded 0 valid dates", field_map

    # Sanity: require reasonable date range
    years = df["_Date"].dt.year
    if not years.between(2010, 2035).any():
        return pd.DataFrame(), f"date col '{date_col}' has no dates in 2010-2035", field_map

    # ── Order ID ──────────────────────────────────────────────
    order_col = _find_col(cols, [
        "Sub Order Code", "Suborder Code", "Sub Order ID", "Sub Order No",
        "Order Code", "Order ID", "Snapdeal Order ID", "Order Number",
        "SubOrder Code", "Sub-Order Code",
    ]) or _find_col_fuzzy(cols, ["sub order", "suborder", "order code", "order id", "order no"])
    field_map["order_col"] = order_col
    df["_OrderId"] = df[order_col].fillna("").astype(str) if order_col else ""

    # ── SKU ───────────────────────────────────────────────────
    sku_col = _find_col(cols, [
        "Seller SKU Code", "Seller SKU", "Snap SKU", "SKU Code",
        "SKU", "Product SKU", "Item SKU", "SellerSKUCode",
        "Seller_SKU", "Article Code", "EAN Code", "EAN", "Bar Code",
        "Barcode", "Product Code", "Vendor SKU", "Listing SKU",
        "Seller Product Code", "Style Code", "Style ID",
    ]) or _find_col_fuzzy(cols, ["seller sku", "snap sku", "sku code", "sku", "article code", "ean", "bar code", "barcode", "product code", "vendor sku", "style code"])
    field_map["sku_col"] = sku_col
    if sku_col:
        df["_OMS_SKU"] = df[sku_col].apply(
            lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
        )
    else:
        df["_OMS_SKU"] = "UNKNOWN"

    # ── Quantity ──────────────────────────────────────────────
    qty_col = _find_col(cols, [
        "Quantity", "Qty", "Item Qty", "No. of Units", "Units", "Pieces",
    ]) or _find_col_fuzzy(cols, ["quantity", "qty", "units", "pieces"])
    field_map["qty_col"] = qty_col
    df["_Qty"] = (
        pd.to_numeric(df[qty_col], errors="coerce").fillna(1).astype("float32")
        if qty_col else 1.0
    )

    # ── Revenue ───────────────────────────────────────────────
    rev_col = _find_col(cols, [
        "Deal Price", "Selling Price", "Sale Price",
        "Total Amount", "Invoice Amount", "Net Amount",
        "Buyer Price", "MRP", "Product Price", "Gross Amount",
        "Order Amount", "Item Price",
    ]) or _find_col_fuzzy(cols, [
        "deal price", "selling price", "sale price", "total amount",
        "net amount", "invoice", "order amount", "item price", "settlement amount",
    ])
    field_map["rev_col"] = rev_col
    df["_Rev"] = (
        pd.to_numeric(df[rev_col], errors="coerce").fillna(0).astype("float32")
        if rev_col else 0.0
    )

    # ── Transaction Type ─────────────────────────────────────
    status_col = _find_col(cols, [
        "Order Status", "Status", "Sub Status", "Fulfillment Status",
        "Shipment Status", "Item Status", "Order Sub Status", "Current Status",
        "Current Order Status", "Dispatch Status", "Return Status",
        "Transaction Type", "Transaction_Type", "Order Type",
    ]) or _find_col_fuzzy(cols, ["status", "txn type", "transaction type", "order type", "sale type"])
    field_map["status_col"] = status_col
    df["_TxnType"] = (
        df[status_col].apply(_snapdeal_txn)
        if status_col else "Shipment"
    )

    # ── State ─────────────────────────────────────────────────
    state_col = _find_col(cols, [
        "State/UT", "State", "Delivery State", "Customer State",
        "Shipping State", "Ship State", "Buyer State", "Delivery State/UT",
    ]) or _find_col_fuzzy(cols, ["state", "buyer state", "delivery state", "customer state"])
    field_map["state_col"] = state_col
    df["_State"] = (
        df[state_col].fillna("").astype(str).str.upper().str.strip()
        if state_col else ""
    )

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
    result = out.dropna(subset=["Date"])
    return result, f"ok ({len(result)} rows)", field_map


def _try_parse_with_header(
    raw: pd.DataFrame,
    hdr_row: int,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, dict]:
    """Apply header row `hdr_row` and attempt parse. Returns (df, field_map)."""
    if hdr_row >= len(raw):
        return pd.DataFrame(), {}
    cols = raw.iloc[hdr_row].astype(str).tolist()
    data = raw.iloc[hdr_row + 1:].reset_index(drop=True)
    data.columns = cols
    data = data.dropna(how="all")
    result, _, field_map = _parse_snapdeal_df(data, mapping)
    return result, field_map


def _parse_snapdeal_file(
    file_bytes: bytes,
    fname: str,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Parse a single Snapdeal Excel or CSV file.
    Returns (df, debug_info, field_map).
    Tries every header row 0-15 until one yields valid data.
    """
    fn_lower = fname.lower()

    raw_candidates: List[pd.DataFrame] = []   # list of raw (no-header) DataFrames to try

    try:
        if fn_lower.endswith(".csv"):
            for enc in ("utf-8", "ISO-8859-1"):
                try:
                    raw = pd.read_csv(
                        io.BytesIO(file_bytes), dtype=str, header=None,
                        encoding=enc, on_bad_lines="skip",
                    )
                    raw_candidates.append(raw)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xl.sheet_names:
                try:
                    raw = xl.parse(sheet, header=None, dtype=str)
                    if not raw.empty:
                        raw_candidates.append(raw)
                except Exception:
                    continue
    except Exception as e:
        return pd.DataFrame(), f"read error: {e}", {}

    if not raw_candidates:
        return pd.DataFrame(), "no sheets / empty file", {}

    # For each sheet (raw), try every header row from 0 to 15
    best_df: pd.DataFrame = pd.DataFrame()
    best_debug = ""
    best_field_map: dict = {}

    for raw in raw_candidates:
        # First: try the heuristic best row
        hdr_row = _detect_header_row(raw)
        candidate, field_map = _try_parse_with_header(raw, hdr_row, mapping)
        if not candidate.empty:
            if len(candidate) > len(best_df):
                best_df = candidate
                best_debug = f"header row {hdr_row}"
                best_field_map = field_map
            continue  # already found something on this sheet

        # Brute force: try every row 0-15
        for r in range(min(16, len(raw))):
            if r == hdr_row:
                continue
            candidate, field_map = _try_parse_with_header(raw, r, mapping)
            if not candidate.empty and len(candidate) > len(best_df):
                best_df = candidate
                best_debug = f"header row {r} (brute force)"
                best_field_map = field_map
                break

    if best_df.empty:
        # Return column names from first 5 rows for diagnostics
        debug_cols = []
        raw_cols_found = []
        for raw in raw_candidates[:1]:
            for r in range(min(5, len(raw))):
                row_vals = [str(v).strip() for v in raw.iloc[r].values
                            if pd.notna(v) and str(v).strip()]
                if row_vals:
                    debug_cols.append(f"row{r}: {row_vals[:8]}")
                    if r == 0:
                        raw_cols_found = row_vals
        return pd.DataFrame(), " | ".join(debug_cols) or "no data", {"raw_cols": raw_cols_found}

    return best_df, best_debug, best_field_map


def load_snapdeal_from_zip(
    zip_bytes: bytes,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, int, List[str], dict]:
    """
    Parse Snapdeal master ZIP (may contain .xlsx/.csv or nested ZIPs).
    Returns (combined_df, file_count, skipped_list, parse_info).
    parse_info maps fname → field_map for diagnostics.
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []
    parse_info: dict = {}  # fname → field_map

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open ZIP: {e}"], {}

    file_items: List[Tuple[str, bytes]] = []

    for name in root_zf.namelist():
        base = Path(name).name
        if "__MACOSX" in name or base.startswith("."):
            continue
        lower = base.lower()
        if lower.endswith(".zip"):
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

    for fname, fbytes in file_items:
        try:
            df, debug, field_map = _parse_snapdeal_file(fbytes, fname, mapping)
            parse_info[fname] = field_map
            if df.empty:
                skipped.append(f"{fname}: {debug}")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{fname}: {e}")

    if not dfs:
        return pd.DataFrame(), len(file_items), skipped, parse_info

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    return combined, len(file_items), skipped, parse_info


def snapdeal_to_sales_rows(snapdeal_df: pd.DataFrame) -> pd.DataFrame:
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

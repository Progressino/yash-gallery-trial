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

# ── Sheet name → TxnType override (used when status col is absent) ────────────
_SHEET_TXNTYPE: Dict[str, str] = {
    "return":        "Refund",
    "returns":       "Refund",
    "rto":           "Refund",
    "reverse":       "Refund",
    "returned":      "Refund",
    "cancel":        "Cancel",
    "cancelled":     "Cancel",
    "cancellation":  "Cancel",
}
# Sheets to skip entirely (summary/report sheets with no row-level order data)
_SKIP_SHEET_KEYWORDS = ["summary", "report", "dashboard", "pivot", "master"]


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
    # OMS files have "Product Sku Code" (correct YK SKU) AND "Listing Sku Code"
    # (may have "PL" prefix or other variants). Prefer Product Sku Code.
    # Note: "Product Code" excluded — Snapdeal's internal numeric deal IDs.
    sku_col = _find_col(cols, [
        "Product Sku Code", "Product SKU Code",
        "Seller SKU Code", "Seller SKU", "Snap SKU", "SKU Code",
        "SKU ID", "SKU", "Product SKU", "Item SKU", "SellerSKUCode",
        "Seller_SKU", "Seller Item Code", "Seller Product SKU",
        "Vendor SKU", "Seller Product Code",
        "Style Code", "Style ID", "Item Code",
    ]) or _find_col_fuzzy(cols, [
        "product sku", "seller sku", "snap sku", "sku code", "sku id",
        "vendor sku", "seller item", "seller product",
        "style code", "item code",
    ])
    # Fallback listing SKU col for description/variant matching
    listing_sku_col = _find_col(cols, ["Listing Sku Code", "Listing SKU Code", "Listing SKU"]) \
        or _find_col_fuzzy(cols, ["listing sku"])
    # Last resort: "Product Code" — only use when values look like seller SKUs (contain letters).
    if sku_col is None:
        pc_col = _find_col(cols, ["Product Code", "Product ID", "Deal Code", "Catalog ID"])
        if pc_col:
            sample = df[pc_col].dropna().astype(str).str.strip().head(30)
            numeric_ratio = sample.str.match(r'^\d+$').mean() if len(sample) > 0 else 1.0
            if numeric_ratio < 0.5:
                sku_col = pc_col

    field_map["sku_col"] = sku_col
    field_map["listing_sku_col"] = listing_sku_col
    if sku_col:
        def _resolve_sku(row_sku, row_listing_sku=None) -> str:
            """Resolve SKU with Listing SKU fallback and PL-prefix stripping."""
            import re as _re
            c = clean_sku(str(row_sku))
            resolved = mapping.get(c, c)
            # If we have a listing SKU column and it differs, try it as fallback
            if listing_sku_col and row_listing_sku is not None:
                listing = clean_sku(str(row_listing_sku))
                if listing and listing != c:
                    # Try direct listing SKU match
                    if listing in mapping:
                        return mapping[listing]
                    # Strip common Snapdeal listing prefixes (e.g. 1253PLYK... → 1253YK...)
                    stripped = _re.sub(r'^(\d+)PL(YK)', r'\1\2', listing, flags=_re.I)
                    if stripped != listing:
                        if stripped in mapping:
                            return mapping[stripped]
                        if stripped:
                            resolved = stripped  # use stripped listing SKU as fallback
            return resolved

        if listing_sku_col:
            df["_OMS_SKU"] = df.apply(
                lambda r: _resolve_sku(r[sku_col], r[listing_sku_col]), axis=1
            )
        else:
            df["_OMS_SKU"] = df[sku_col].apply(
                lambda x: map_to_oms_sku(clean_sku(str(x)), mapping)
            )
        # Drop rows where SKU resolved to empty / literal "nan" / "unknown" / purely numeric Snapdeal IDs
        df = df[~df["_OMS_SKU"].str.upper().isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])]
        df = df[~df["_OMS_SKU"].str.match(r'^\d+$')]
        if df.empty:
            return pd.DataFrame(), f"sku col '{sku_col}' yielded no valid SKUs", field_map
    else:
        return pd.DataFrame(), f"no sku col found (cols: {cols[:15]})", field_map

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

    # ── Company / Channel ─────────────────────────────────────
    # OMS reports have "Channel Name" = "Yash gallery private limited - Snapdeal",
    # "Aashirwad Garments - Snapdeal", etc.  Strip the " - Snapdeal" suffix.
    channel_col = _find_col(cols, ["Channel Name", "Channel", "Seller Name", "Company"]) \
        or _find_col_fuzzy(cols, ["channel name", "channel", "seller name"])
    field_map["channel_col"] = channel_col
    if channel_col:
        df["_Company"] = (
            df[channel_col].fillna("").astype(str).str.strip()
            .str.replace(r"\s*-\s*snapdeal\s*$", "", case=False, regex=True)
            .str.strip()
        )
    else:
        df["_Company"] = ""

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"],
        "Invoice_Amount": df["_Rev"],
        "State":          df["_State"],
        "OrderId":        df["_OrderId"],
        "OMS_SKU":        df["_OMS_SKU"],
        "Company":        df["_Company"],
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


def _txntype_from_sheet(sheet_name: str) -> Optional[str]:
    """Return a forced TxnType based on the sheet name, or None if no hint."""
    s = sheet_name.lower().strip()
    for kw, txn in _SHEET_TXNTYPE.items():
        if kw in s:
            return txn
    return None


def _should_skip_sheet(sheet_name: str) -> bool:
    s = sheet_name.lower().strip()
    return any(kw in s for kw in _SKIP_SHEET_KEYWORDS)


def _parse_snapdeal_file(
    file_bytes: bytes,
    fname: str,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Parse a single Snapdeal Excel or CSV file.
    Returns (df, debug_info, field_map).

    KEY BEHAVIOUR: ALL sheets are parsed and combined.
    Sheet name is used to override TxnType when no status column exists
    (e.g. a 'Returns' sheet → all rows become Refund).
    """
    fn_lower = fname.lower()

    # (sheet_name, raw_df) pairs to try
    raw_candidates: List[Tuple[str, pd.DataFrame]] = []

    try:
        if fn_lower.endswith(".csv"):
            for enc in ("utf-8", "ISO-8859-1"):
                try:
                    raw = pd.read_csv(
                        io.BytesIO(file_bytes), dtype=str, header=None,
                        encoding=enc, on_bad_lines="skip",
                    )
                    raw_candidates.append(("Sheet1", raw))
                    break
                except UnicodeDecodeError:
                    continue
        else:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            # Snapdeal monthly *payment settlement* XLSX (Total_Suboders, Returns, …)
            # has invoice lines but no seller SKU — cannot attribute to products in SKU Deepdive.
            _tsheet = next(
                (s for s in xl.sheet_names
                 if "total" in s.lower().replace(" ", "") and "subod" in s.lower().replace(" ", "")),
                None,
            )
            if _tsheet is not None:
                probe_hdr = xl.parse(_tsheet, header=0, nrows=1, dtype=str)
                col_txt = " ".join(str(c).lower() for c in probe_hdr.columns)
                if "sku" not in col_txt and "style" not in col_txt and "product code" not in col_txt:
                    return pd.DataFrame(), (
                        "Snapdeal payment/settlement XLSX (no Seller SKU column). "
                        "Use the Snapdeal **seller order / OMS** export for per-SKU sales."
                    ), {}

            for sheet in xl.sheet_names:
                if _should_skip_sheet(sheet):
                    continue
                try:
                    raw = xl.parse(sheet, header=None, dtype=str)
                    if not raw.empty:
                        raw_candidates.append((sheet, raw))
                except Exception:
                    continue
    except Exception as e:
        return pd.DataFrame(), f"read error: {e}", {}

    if not raw_candidates:
        return pd.DataFrame(), "no sheets / empty file", {}

    all_sheet_dfs: List[pd.DataFrame] = []
    best_field_map: dict = {}
    sheet_results: List[str] = []

    for sheet_name, raw in raw_candidates:
        txn_hint = _txntype_from_sheet(sheet_name)

        # Try heuristic header row first, then brute-force
        hdr_row = _detect_header_row(raw)
        candidate, field_map = _try_parse_with_header(raw, hdr_row, mapping)

        if candidate.empty:
            for r in range(min(16, len(raw))):
                if r == hdr_row:
                    continue
                candidate, field_map = _try_parse_with_header(raw, r, mapping)
                if not candidate.empty:
                    break

        if candidate.empty:
            sheet_results.append(f"'{sheet_name}': no data")
            continue

        # Apply sheet-name TxnType override when no status column was detected
        if txn_hint is not None and field_map.get("status_col") is None:
            candidate = candidate.copy()
            candidate["TxnType"] = txn_hint

        all_sheet_dfs.append(candidate)
        if not best_field_map:
            best_field_map = field_map
        sheet_results.append(f"'{sheet_name}'({txn_hint or 'mixed'}): {len(candidate)} rows")

    if not all_sheet_dfs:
        # Diagnostics: show first 5 rows from the first sheet
        debug_cols = []
        raw_cols_found: List[str] = []
        for _, raw in raw_candidates[:1]:
            for r in range(min(5, len(raw))):
                row_vals = [str(v).strip() for v in raw.iloc[r].values
                            if pd.notna(v) and str(v).strip()]
                if row_vals:
                    debug_cols.append(f"row{r}: {row_vals[:8]}")
                    if r == 0:
                        raw_cols_found = row_vals
        return pd.DataFrame(), " | ".join(debug_cols) or "no data", {"raw_cols": raw_cols_found}

    combined = pd.concat(all_sheet_dfs, ignore_index=True)
    debug_msg = f"ok ({len(combined)} rows from {len(all_sheet_dfs)} sheets: {', '.join(sheet_results)})"
    return combined, debug_msg, best_field_map


def load_snapdeal_from_zip(
    zip_bytes: bytes,
    mapping: Dict[str, str],
    filename: str = "upload",
) -> Tuple[pd.DataFrame, int, List[str], dict]:
    """
    Parse Snapdeal files: ZIP (with .xlsx/.csv or nested ZIPs), or a plain CSV/Excel.
    Returns (combined_df, file_count, skipped_list, parse_info).
    parse_info maps fname → field_map for diagnostics.
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []
    parse_info: dict = {}  # fname → field_map

    # ── If not a ZIP, try parsing as a direct CSV/Excel file ──────────────────
    fn_lower = filename.lower()
    if not fn_lower.endswith(".zip"):
        df, debug, field_map = _parse_snapdeal_file(zip_bytes, filename, mapping)
        parse_info[filename] = field_map
        if df.empty:
            return pd.DataFrame(), 1, [f"{filename}: {debug}"], parse_info
        return df, 1, [], parse_info

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        # Last resort: maybe it's a CSV with a .zip extension or misdetected
        df, debug, field_map = _parse_snapdeal_file(zip_bytes, filename.replace(".zip", ".csv"), mapping)
        if not df.empty:
            parse_info[filename] = field_map
            return df, 1, [], parse_info
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
    out = pd.DataFrame({
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
        "LineKey":          "",
    })
    if "Company" in snapdeal_df.columns:
        co = snapdeal_df["Company"].fillna("").astype(str).str.strip()
        out["Company"] = co.values
        out["DSR_Segment"] = co.values
    else:
        out["DSR_Segment"] = ""
    return out

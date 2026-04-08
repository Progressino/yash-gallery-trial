"""
Meesho loader — extracted 1-for-1 from app.py.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _clean_meesho_cell(value) -> str:
    """Strip Excel float noise (1158.0 → 1158) and null tokens; keep listing case for mapping."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and np.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    t = str(value).strip().replace(",", "")
    if not t or t.lower() in ("nan", "none", "<na>", "nat"):
        return ""
    try:
        f = float(t)
        if np.isfinite(f) and f == int(f) and abs(f) < 1e15:
            return str(int(f))
    except ValueError:
        pass
    return t.strip()


def _clean_meesho_str_series(series: pd.Series) -> pd.Series:
    return series.map(_clean_meesho_cell)


def _norm_meesho_size(s: str) -> str:
    """Normalize size tokens (Meesho Order CSV / API): XXXL → 3XL, etc."""
    s = str(s).strip().upper()
    if not s or s in ("NAN", "NONE", "NULL"):
        return ""
    m = re.match(r"^(X{3,})L$", s)  # XXXL, XXXXL, …
    if m:
        return f"{len(m.group(1))}XL"
    return s


# Trailing size when SKU and size are pasted in one cell: "1158YKGREEN XL" → "1158YKGREEN-XL"
# Includes kids' bands e.g. "1158YKGREEN 7-8" → "1158YKGREEN-7-8"
_TRAILING_SIZE_RE = re.compile(
    r"\s+(XS|2XL|3XL|4XL|5XL|6XL|[2-6]XL|XXS|XXL|XXXL|XXXXL|XXXXXL|S|M|L|XL|"
    r"X{3,}L|\d{1,2}-\d{1,2})\s*$",
    re.IGNORECASE,
)


def _maybe_split_space_sku_size(series: pd.Series) -> pd.Series:
    def one(x: str) -> str:
        x = str(x).strip()
        if not x:
            return ""
        m = _TRAILING_SIZE_RE.search(x)
        if m:
            left = x[: m.start()].strip()
            right = _norm_meesho_size(m.group(1))
            if left and right:
                return f"{left}-{right}"
        return x

    return series.apply(one)


def _meesho_size_column(df: pd.DataFrame) -> Optional[str]:
    """Original column name in df for Meesho size (df may be lowercased headers)."""
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for candidate in (
        "size",
        "size_name",
        "variant_size",
        "product_size",
        "product size",
        "variant size",
        "item_size",
        "item size",
        "sku size",
        "size (india)",
        "select size",
        "standard size",
        "standard_size",
        "indian size",
        "indian_size",
        "buyer size",
        "buyer_size",
    ):
        if candidate in cols_lower:
            return cols_lower[candidate]
    for k, orig in cols_lower.items():
        if "resize" in k:
            continue
        if k == "size" or k.endswith(" size"):
            return orig
    return None


def _combine_meesho_sku_size(
    base: pd.Series, size_ser: Optional[pd.Series]
) -> pd.Series:
    """
    Build variant SKU: base + "-" + size (e.g. 1158YKGREEN + XL → 1158YKGREEN-XL).
    If a separate size column is missing or empty for a row, try splitting "SKU SIZE" in base.
    """
    b = _clean_meesho_str_series(base.fillna(""))
    if size_ser is None:
        return _maybe_split_space_sku_size(b)
    z = _clean_meesho_str_series(size_ser.fillna("")).str.upper()
    z = z.replace({"NAN": "", "NONE": "", "NULL": ""})
    z = z.apply(_norm_meesho_size)
    out = b.copy().astype(str)
    both = (b != "") & (z != "")
    out.loc[both] = (b[both] + "-" + z[both]).astype(str).values
    need_split = (b != "") & (z == "")
    if need_split.any():
        out.loc[need_split] = _maybe_split_space_sku_size(b.loc[need_split]).values
    out.loc[b == ""] = ""
    return out


_TIER1_MEESHO_SKU_HEADERS = (
    "sku",
    "sku id",
    "sku_id",
    "sku code",
    "sku_code",
    "seller_sku",
    "seller sku",
    "product_sku",
    "item_sku",
    "listing_sku",
    "listing sku",
    "meesho_sku",
    "meesho sku",
    "supplier_sku",
    "supplier sku",
    "merchant_sku",
    "merchant sku",
    "partner_sku",
    "catalog_sku",
    "brand_sku",
    "inventory_sku",
    "inventory sku",
    "packet_sku",
    "warehouse_sku",
)


def _meesho_sku_base_series(df: pd.DataFrame) -> pd.Series:
    """
    Detect the column that holds the listing / replace-SKU token across TCS, return, and
    order CSV layouts. Falls back to any column whose name contains 'sku' (excluding order ids).
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=str)
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    def _nonempty(ser: pd.Series) -> int:
        s = _clean_meesho_str_series(ser)
        return int((s.str.len() > 0).sum())

    for cand in _TIER1_MEESHO_SKU_HEADERS:
        if cand not in cols_lower:
            continue
        ser = _clean_meesho_str_series(df[cols_lower[cand]])
        if _nonempty(ser) > 0:
            return ser

    tier2 = (
        "sub_catalog_name",
        "catalog_name",
        "product_name",
        "item_name",
        "article_name",
        "sub_catalog_id",
        "catalog_id",
        "product_id",
        "article_id",
        "product_code",
        "item_code",
        "variant_name",
        "variant_id",
        "style_code",
        "style_id",
    )
    best_ser: Optional[pd.Series] = None
    best_n = 0
    for cand in tier2:
        if cand not in cols_lower:
            continue
        ser = _clean_meesho_str_series(df[cols_lower[cand]])
        nn = _nonempty(ser)
        if nn > best_n:
            best_n = nn
            best_ser = ser
    if best_ser is not None and best_n > 0:
        return best_ser

    for k, orig in sorted(cols_lower.items()):
        if "sku" not in k:
            continue
        if any(x in k for x in ("order", "sub_order", "commission", "packet_id")):
            continue
        ser = _clean_meesho_str_series(df[orig])
        if _nonempty(ser) > 0:
            return ser

    return pd.Series([""] * n, index=df.index, dtype=str)


def refresh_meesho_dataframe_oms_inplace(df: pd.DataFrame, mapping: Optional[dict]) -> None:
    """Normalize SKU cells; set OMS_SKU via Replace-SKU / Meesho sheet mapping (mutates df)."""
    if df.empty or "SKU" not in df.columns:
        return
    from .helpers import map_to_oms_sku

    df["SKU"] = _clean_meesho_str_series(df["SKU"])
    if mapping:
        df["OMS_SKU"] = df["SKU"].map(lambda s: map_to_oms_sku(s, mapping) if s else "")
    else:
        df["OMS_SKU"] = df["SKU"]


def _parse_meesho_inner_zip(inner_zf) -> pd.DataFrame:
    # Basename index: Supplier archives often use paths like Report_Dec/tcs_sales.xlsx
    files: dict[str, str] = {}
    for f in inner_zf.namelist():
        if f.endswith("/"):
            continue
        files[Path(f).name.lower()] = f
    rows: List[pd.DataFrame] = []

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
            _sz = _meesho_size_column(df)
            df["_SKU"]     = _combine_meesho_sku_size(_meesho_sku_base_series(df), df[_sz] if _sz else None)
            df["_TxnType"] = "Shipment"
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_SKU", "_Month"]])

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
            _sz = _meesho_size_column(df)
            df["_SKU"]     = _combine_meesho_sku_size(_meesho_sku_base_series(df), df[_sz] if _sz else None)
            df["_TxnType"] = "Refund"
            if "financial_year" in df.columns and "month_number" in df.columns:
                df["_Month"] = df.apply(
                    lambda r: f"{int(r['financial_year'])}-{int(r['month_number']):02d}"
                    if pd.notna(r.get("financial_year")) and pd.notna(r.get("month_number"))
                    else None, axis=1
                )
            else:
                df["_Month"] = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_SKU", "_Month"]])

    # ForwardReports = shipments (used when no TCS data present)
    _fwd_key = next((k for k in ["forwardreports.xlsx"] if k in files), None)
    if _fwd_key and not rows:
        with inner_zf.open(files[_fwd_key]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=False)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("meesho_price", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state", df.get("state", ""))
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            _sz = _meesho_size_column(df)
            df["_SKU"]     = _combine_meesho_sku_size(_meesho_sku_base_series(df), df[_sz] if _sz else None)
            def _meesho_txn(s):
                s = str(s).lower()
                if "return" in s or "rto" in s: return "Refund"
                if "cancel" in s:               return "Cancel"
                return "Shipment"
            df["_TxnType"] = df.get("order_status", "").apply(_meesho_txn)
            df["_Month"]   = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_SKU", "_Month"]])

    # Reverse / AdjustmentFileReverse = returns (mutually exclusive filenames)
    _rev_key = next((k for k in ["reverse.xlsx", "adjustmentfilereverse.xlsx"] if k in files), None)
    if _rev_key and not any(
        (r["_TxnType"] == "Refund").any() for r in rows if not r.empty
    ):
        with inner_zf.open(files[_rev_key]) as fh:
            df = pd.read_excel(fh)
        if not df.empty:
            date_col = _best_date_col(df, prefer_return=True)
            df["_Date"]    = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            df["_Qty"]     = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
            df["_Rev"]     = pd.to_numeric(df.get("meesho_price", 0), errors="coerce").fillna(0)
            df["_State"]   = df.get("end_customer_state", df.get("state", ""))
            df["_OrderId"] = df.get("sub_order_num", "").astype(str)
            _sz = _meesho_size_column(df)
            df["_SKU"]     = _combine_meesho_sku_size(_meesho_sku_base_series(df), df[_sz] if _sz else None)
            df["_TxnType"] = "Refund"
            df["_Month"]   = None
            rows.append(df[["_Date", "_TxnType", "_Qty", "_Rev", "_State", "_OrderId", "_SKU", "_Month"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.columns = ["Date", "TxnType", "Quantity", "Invoice_Amount", "State", "OrderId", "SKU", "_Month"]
    out["Date"]           = pd.to_datetime(out["Date"], errors="coerce")
    out["Quantity"]       = out["Quantity"].astype("float32")
    out["Invoice_Amount"] = out["Invoice_Amount"].astype("float32")
    out["State"]          = out["State"].astype(str).str.upper().str.strip()
    out["SKU"]            = _clean_meesho_str_series(out["SKU"])
    out["OMS_SKU"]        = out["SKU"]   # alias expected by platform_metrics / PO engine
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

    # Status: "reason for credit entry" (DELIVERED/RETURNED/RTO/CANCELLED) or "order status"
    status_col = next((c for c in df.columns if "reason" in c or "order status" in c
                       or c == "status"), None)
    def _txn(s):
        s = str(s).lower().strip()
        if "return" in s or "rto" in s:                          return "Refund"
        if "cancel" in s:                                         return "Cancel"
        if s in ("ready_to_ship", "hold", "pending", "new"):     return "Cancel"  # pre-shipment, exclude
        return "Shipment"  # DELIVERED, SHIPPED, DOOR_STEP_EXCHANGED, etc.
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

    # Order ID: prefer "sub order no" / "packet id"
    order_col = next((c for c in df.columns if "sub order" in c or "order no" in c
                      or "packet id" in c or "packet" in c), None)

    # Size column: combine with SKU → 1158YKGREEN + XL → 1158YKGREEN-XL (also split "1158YKGREEN XL" in one cell).
    base_sku = _meesho_sku_base_series(df)
    sz_col = _meesho_size_column(df)
    sku_series = _combine_meesho_sku_size(base_sku, df[sz_col] if sz_col else None)

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"].astype("float32"),
        "Invoice_Amount": df["_Rev"].astype("float32"),
        "State":          df[state_col].fillna("").str.upper().str.strip() if state_col else "",
        "OrderId":        df[order_col].fillna("").astype(str) if order_col else "",
        "SKU":            sku_series,
    })
    out["SKU"] = _clean_meesho_str_series(out["SKU"])
    out["OMS_SKU"] = out["SKU"]   # alias expected by platform_metrics / PO engine
    out["Month"]   = out["Date"].dt.to_period("M").astype(str)
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

    # Single-file download: TCS xlsx directly inside outer zip (no nested .zip)
    if not dfs:
        try:
            flat = _parse_meesho_inner_zip(root_zf)
            if not flat.empty:
                dfs.append(flat)
        except Exception as e:
            skipped.append(f"Flat archive: {e}")

    if not dfs:
        return pd.DataFrame(), len(items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(keep="first")
    zip_count = len(items) if items else (1 if not combined.empty else 0)
    return combined, zip_count, skipped


def meesho_to_sales_rows(meesho_df: pd.DataFrame, sku_mapping: dict | None = None) -> pd.DataFrame:
    """
    Convert meesho_df to the unified sales_df schema.
    Uses SKU column if present; applies sku_mapping to resolve OMS SKU (Replace SKU / Meesho master).
    Falls back to 'MEESHO_TOTAL' only when no SKU data is available.
    """
    if meesho_df.empty:
        return pd.DataFrame()

    from .helpers import map_to_oms_sku

    # Resolve OMS SKU: use the SKU column from the parsed data if available
    if "SKU" in meesho_df.columns:
        raw_sku = _clean_meesho_str_series(meesho_df["SKU"])
        has_sku = raw_sku.str.len() > 0
        has_sku &= ~raw_sku.str.lower().isin(["nan", "none"])
        if sku_mapping:
            mapped = raw_sku.map(lambda s: map_to_oms_sku(s, sku_mapping) if s else "")
        else:
            mapped = raw_sku
        mapped = mapped.astype(str).str.strip()
        empty_mapped = mapped.str.len() == 0
        sku_series = mapped.where(~(has_sku & empty_mapped), raw_sku)
        sku_series = sku_series.where(has_sku, "MEESHO_TOTAL")
    else:
        sku_series = "MEESHO_TOTAL"

    out = pd.DataFrame({
        "Sku":              sku_series,
        "TxnDate":          meesho_df["Date"],
        "Transaction Type": meesho_df["TxnType"],
        "Quantity":         meesho_df["Quantity"],
        "Units_Effective":  np.where(meesho_df["TxnType"] == "Refund", -meesho_df["Quantity"],
                            np.where(meesho_df["TxnType"] == "Cancel",  0, meesho_df["Quantity"])),
        "Source":           "Meesho",
        "OrderId":          meesho_df["OrderId"],
    })
    return out

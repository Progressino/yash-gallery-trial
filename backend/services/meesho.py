"""
Meesho loader — extracted 1-for-1 from app.py.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers import clean_line_id_series, is_likely_non_sku_notes_value, looks_like_seller_listing_sku


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


def _tier1_series_overwhelmingly_aggregate(ser: pd.Series) -> bool:
    """Skip tier-1 columns that are mostly MEESHO_TOTAL / *_TOTAL bucket rows."""
    s = _clean_meesho_str_series(ser.fillna(""))
    nonempty = s.str.len() > 0
    if not nonempty.any():
        return True
    u = s.str.upper()
    bad = u.eq("MEESHO_TOTAL") | ((u.str.endswith("_TOTAL")) & (u.str.len() <= 28))
    return bool(bad.sum() / int(nonempty.sum()) >= 0.85)


def _listing_sku_fraction(ser: pd.Series) -> Tuple[float, int]:
    s = _clean_meesho_str_series(ser.fillna(""))
    nonempty = s.str.len() > 0
    n = int(nonempty.sum())
    if n == 0:
        return 0.0, 0
    good = int(s.loc[nonempty].map(looks_like_seller_listing_sku).sum())
    return good / n, n


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
        if _nonempty(ser) > 0 and not _tier1_series_overwhelmingly_aggregate(ser):
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
    best_key: Tuple[float, int] = (0.0, 0)
    for cand in tier2:
        if cand not in cols_lower:
            continue
        ser = _clean_meesho_str_series(df[cols_lower[cand]])
        frac, nn = _listing_sku_fraction(ser)
        if nn == 0 or frac < 0.22:
            continue
        key = (frac, nn)
        if key > best_key:
            best_key = key
            best_ser = ser
    if best_ser is not None and best_key[1] > 0:
        return best_ser

    for k, orig in sorted(cols_lower.items()):
        if "sku" not in k:
            continue
        if any(
            x in k
            for x in (
                "order",
                "sub_order",
                "commission",
                "packet_id",
                "reason",
                "credit",
                "return",
                "adjust",
                "resize",
            )
        ):
            continue
        ser = _clean_meesho_str_series(df[orig])
        frac, nn = _listing_sku_fraction(ser)
        if nn > 0 and frac >= 0.2:
            return ser

    return pd.Series([""] * n, index=df.index, dtype=str)


def _meesho_line_dedup_series(df: pd.DataFrame) -> pd.Series:
    """Prefer Sub Order ID over packet id — DSR and supplier exports are sub-order grain."""
    if df.empty:
        return pd.Series(dtype=str)
    keys = pd.Series("", index=df.index, dtype=str)
    for col in (
        "sub order no",
        "sub order",
        "suborder id",
        "suborder no",
        "sub_order_no",
        "sub_order_id",
        "packet id",
        "packet_id",
        "catalog id",
    ):
        if col not in df.columns:
            continue
        cand = clean_line_id_series(df[col])
        need = keys.eq("") & cand.ne("")
        keys = keys.mask(need, cand)
    return keys


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
    out["RawStatus"] = out["TxnType"].astype(str)
    lk = clean_line_id_series(out["OrderId"])
    miss = lk.eq("")
    if miss.any():
        dt = out["Date"].dt.strftime("%Y%m%d")
        sku = out["SKU"].astype(str).str.strip()
        txn = out["TxnType"].astype(str)
        q = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0).astype(str)
        rev = pd.to_numeric(out["Invoice_Amount"], errors="coerce").fillna(0).astype(str)
        lk = lk.mask(miss, "MEETCS|" + dt + "|" + sku + "|" + txn + "|" + q + "|" + rev)
    out["LineKey"] = lk
    oid0 = clean_line_id_series(out["OrderId"])
    out["OrderId"] = oid0.where(oid0.ne(""), lk)
    out["MeeshoSubOrder"] = clean_line_id_series(out["OrderId"])
    return out.dropna(subset=["Date"])


def _norm_export_header(c) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def looks_like_meesho_order_export(df: pd.DataFrame) -> bool:
    """Unified sales export: TxnDate + Transaction Type + Sku + Quantity (app / ERP export)."""
    if df.empty or len(df.columns) < 4:
        return False
    keys = {_norm_export_header(c) for c in df.columns}
    return {"txndate", "transactiontype", "sku", "quantity"}.issubset(keys)


_ALT_ORDER_EXPORT_SKU_NORMS = frozenset(
    {
        "sku1",
        "sku2",
        "sku3",
        "listingsku",
        "sellersku",
        "merchantsku",
        "suppliersku",
        "productsku",
        "catalogsku",
        "replacementsku",
        "variantsku",
        "skuid",
        "meeshosku",
        "marketplacesku",
        "channelsku",
    }
)


def _order_export_alt_sku_columns(col_map: dict[str, object], primary: object) -> List[object]:
    """Extra listing / seller SKU columns in unified sales Excel (beyond pandas' Sku.1)."""
    raw: List[object] = []
    for norm, orig in sorted(col_map.items(), key=lambda x: x[0]):
        if orig == primary:
            continue
        if norm in _ALT_ORDER_EXPORT_SKU_NORMS:
            raw.append(orig)
        elif norm.startswith("sku") and norm != "sku" and len(norm) > 3 and norm[3:].isdigit():
            raw.append(orig)
    seen: set[int] = set()
    out: List[object] = []
    for o in raw:
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        out.append(o)
    return out


def parse_meesho_order_export_xlsx(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    """
    Re-import an ERP unified sales Excel export (Meesho rows with TxnDate, Sku, …).
    When Sku is MEESHO_TOTAL, uses listing column Sku.1 / sku1 (pandas duplicate rename).
    """
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        return pd.DataFrame(), f"Excel read error: {e}"
    if df.empty:
        return pd.DataFrame(), "Empty workbook"
    if not looks_like_meesho_order_export(df):
        return pd.DataFrame(), "Not a unified sales export (need TxnDate, Transaction Type, Sku, Quantity)"

    col_map: dict[str, object] = {}
    for c in df.columns:
        col_map.setdefault(_norm_export_header(c), c)

    date_c = col_map.get("txndate")
    txn_c = col_map.get("transactiontype")
    sku_c = col_map.get("sku")
    qty_c = col_map.get("quantity")
    order_c = col_map.get("orderid")
    state_c = col_map.get("state") or col_map.get("customerstate")
    rev_c = col_map.get("invoiceamount") or col_map.get("invoice_amount")
    source_c = col_map.get("source")
    sku_alt_c = col_map.get("sku1") or col_map.get("sku2")

    if not all([date_c, txn_c, sku_c, qty_c]):
        return pd.DataFrame(), "Missing required columns in export"

    if source_c is not None:
        m = df[source_c].astype(str).str.strip().str.lower() == "meesho"
        df = df.loc[m].copy()
        if df.empty:
            return pd.DataFrame(), "No rows with Source=Meesho"

    sku_series = df[sku_c].astype(str).str.strip()

    def _row_needs_listing_coalesce(s: pd.Series) -> pd.Series:
        return (
            s.str.upper().eq("MEESHO_TOTAL")
            | s.isin(["", "NAN", "NONE"])
            | s.map(lambda x: is_likely_non_sku_notes_value(x))
        )

    bad = _row_needs_listing_coalesce(sku_series)
    alt_cols: List[object] = []
    if sku_alt_c is not None:
        alt_cols.append(sku_alt_c)
    for c in _order_export_alt_sku_columns(col_map, sku_c):
        if c not in alt_cols:
            alt_cols.append(c)
    for col in alt_cols:
        alt = df[col].astype(str).str.strip()
        meesho_agg = sku_series.str.upper().eq("MEESHO_TOTAL")
        ok_alt = alt.map(looks_like_seller_listing_sku) | (
            meesho_agg
            & (alt.str.len() >= 4)
            & (alt.str.len() <= 48)
            & ~alt.map(is_likely_non_sku_notes_value)
            & alt.str.contains(r"\d", regex=True, na=False)
        )
        sku_series = sku_series.where(~(bad & ok_alt), alt)
        bad = _row_needs_listing_coalesce(sku_series)
        if not bad.any():
            break

    def _txn(s) -> str:
        u = str(s).strip().upper()
        if "REFUND" in u or "RETURN" in u or "RTO" in u:
            return "Refund"
        if "CANCEL" in u:
            return "Cancel"
        return "Shipment"

    rev = (
        pd.to_numeric(df[rev_c], errors="coerce").fillna(0).astype("float32")
        if rev_c is not None
        else pd.Series(0.0, index=df.index, dtype="float32")
    )
    oid_raw = df[order_c].fillna("").astype(str) if order_c is not None else pd.Series("", index=df.index)
    oid_raw = clean_line_id_series(oid_raw)
    txn_ser = df[txn_c].map(_txn)
    dt_key = pd.to_datetime(df[date_c], errors="coerce").dt.strftime("%Y%m%d")
    sku_key = _clean_meesho_str_series(sku_series).astype(str)
    q_key = pd.to_numeric(df[qty_c], errors="coerce").fillna(1).astype(str)
    lk = oid_raw.where(
        oid_raw.ne(""),
        "MEEEXP|" + dt_key.fillna("") + "|" + sku_key + "|" + txn_ser.astype(str) + "|" + q_key,
    )
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_c], errors="coerce"),
            "TxnType": txn_ser,
            "Quantity": pd.to_numeric(df[qty_c], errors="coerce").fillna(1).astype("float32").abs(),
            "Invoice_Amount": rev,
            "State": (
                df[state_c].fillna("").astype(str).str.upper().str.strip()
                if state_c is not None
                else ""
            ),
            "OrderId": oid_raw.where(oid_raw.ne(""), lk),
            "LineKey": lk,
            "RawStatus": df[txn_c].fillna("").astype(str).str.strip(),
            "SKU": _clean_meesho_str_series(sku_series),
            "MeeshoSubOrder": oid_raw.mask(oid_raw.eq(""), clean_line_id_series(lk)),
        }
    )
    out["OMS_SKU"] = out["SKU"]
    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    out = out.dropna(subset=["Date"])
    return out, "OK"


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
        if "return" in s or "rto" in s:
            return "Refund"
        if "cancel" in s:
            return "Cancel"
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

    # Order / LineKey: Sub Order ID must win over packet id (see _meesho_line_dedup_series).
    order_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "sub order" in cl or "suborder" in cl:
            order_col = c
            break
    if order_col is None:
        for c in df.columns:
            cl = str(c).lower()
            if "order no" in cl or cl in ("order id", "order_id"):
                order_col = c
                break
    if order_col is None:
        for c in df.columns:
            if "packet" in str(c).lower():
                order_col = c
                break

    # Size column: combine with SKU → 1158YKGREEN + XL → 1158YKGREEN-XL (also split "1158YKGREEN XL" in one cell).
    base_sku = _meesho_sku_base_series(df)
    sz_col = _meesho_size_column(df)
    sku_series = _combine_meesho_sku_size(base_sku, df[sz_col] if sz_col else None)

    line_keys = _meesho_line_dedup_series(df)
    oid_fb = (
        clean_line_id_series(df[order_col])
        if order_col
        else pd.Series("", index=df.index, dtype=str)
    )
    oid_out = line_keys.where(line_keys.ne(""), oid_fb)
    _raw_st = df[status_col].fillna("").astype(str).str.strip() if status_col else ""
    _sub_ord = (
        clean_line_id_series(df[order_col])
        if order_col
        else pd.Series("", index=df.index, dtype=str)
    )
    _sub_ord = _sub_ord.where(_sub_ord.ne(""), clean_line_id_series(oid_out))

    out = pd.DataFrame({
        "Date":           df["_Date"],
        "TxnType":        df["_TxnType"],
        "Quantity":       df["_Qty"].astype("float32"),
        "Invoice_Amount": df["_Rev"].astype("float32"),
        "State":          df[state_col].fillna("").str.upper().str.strip() if state_col else "",
        "OrderId":        oid_out,
        "LineKey":        line_keys,
        "RawStatus":      _raw_st,
        "SKU":            sku_series,
        "MeeshoSubOrder": _sub_ord,
    })
    out["SKU"] = _clean_meesho_str_series(out["SKU"])
    out["OMS_SKU"] = out["SKU"]   # alias expected by platform_metrics / PO engine
    out["Month"]   = out["Date"].dt.to_period("M").astype(str)
    oi = clean_line_id_series(out["OrderId"])
    el = clean_line_id_series(out["LineKey"])
    miss = oi.eq("") & el.eq("")
    if miss.any():
        dt = out["Date"].dt.strftime("%Y%m%d")
        # Do not embed RawStatus: SHIPPED vs DELIVERED for the same sub-order would get
        # different synthetic keys and double-count in Tier-3 dedup (same as physical line).
        syn = (
            "MEECSV|"
            + dt
            + "|"
            + out["SKU"].astype(str).str.strip()
            + "|"
            + out["TxnType"].astype(str)
            + "|"
            + pd.to_numeric(out["Quantity"], errors="coerce").fillna(0).astype(str)
        )
        out.loc[miss, "LineKey"] = syn.loc[miss]
        out.loc[miss, "OrderId"] = syn.loc[miss]
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
    from .daily_store import _dedup_platform_df

    combined = _dedup_platform_df(combined, "meesho")
    zip_count = len(items) if items else (1 if not combined.empty else 0)
    return combined, zip_count, skipped


def meesho_export_sku_recovery_maps(
    meesho_df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    """
    Build lookups from raw Meesho rows: (OrderId, YYYY-MM-DD) -> listing SKU, and OrderId -> SKU
    when there is exactly one plausible listing SKU for that key (multi-item orders are skipped).
    """
    if meesho_df.empty or "OrderId" not in meesho_df.columns or "SKU" not in meesho_df.columns:
        return {}, {}
    m = meesho_df.copy()
    m["_oid"] = m["OrderId"].astype(str).str.strip()
    m["_sku"] = _clean_meesho_str_series(m["SKU"])
    dt = pd.to_datetime(m["Date"], errors="coerce")
    m["_day"] = dt.dt.strftime("%Y-%m-%d")
    m.loc[dt.isna(), "_day"] = ""

    good = m["_oid"].ne("") & ~m["_oid"].str.lower().isin(["nan", "none"])
    good &= m["_sku"].str.len() > 0
    good &= ~m["_sku"].str.upper().eq("MEESHO_TOTAL")
    good &= ~m["_sku"].map(is_likely_non_sku_notes_value)
    good &= m["_sku"].map(looks_like_seller_listing_sku)
    m = m.loc[good]
    if m.empty:
        return {}, {}

    day_map: Dict[Tuple[str, str], str] = {}
    for (oid, day), grp in m.groupby(["_oid", "_day"]):
        day_s = str(day).strip() if day is not None and str(day) not in ("NaT", "nan") else ""
        if not day_s:
            continue
        skus = list(dict.fromkeys(grp["_sku"].tolist()))
        if len(skus) == 1:
            day_map[(str(oid), day_s)] = skus[0]

    oid_map: Dict[str, str] = {}
    for oid, grp in m.groupby("_oid"):
        skus = list(dict.fromkeys(grp["_sku"].tolist()))
        if len(skus) == 1:
            oid_map[str(oid).strip()] = skus[0]

    return day_map, oid_map


def _export_row_needs_meesho_sku_recovery(sku: object) -> bool:
    if sku is None or (isinstance(sku, float) and pd.isna(sku)):
        return True
    s = str(sku).strip()
    if not s or s.lower() in ("nan", "none"):
        return True
    sl = s.lower()
    if (
        sl == "meesho_total"
        or "_total" in sl
        or (sl.startswith("total") and len(sl) <= 24)
        or (sl.endswith("total") and len(sl) <= 24)
    ):
        return True
    if is_likely_non_sku_notes_value(s):
        return True
    return False


def apply_meesho_listing_sku_recovery_for_export(
    export_df: pd.DataFrame, meesho_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace MEESHO_TOTAL / note-like Sku values on Meesho export rows when meesho_df has a single
    unambiguous listing SKU for the same OrderId (and calendar day when dates align).
    """
    if export_df.empty or meesho_df.empty:
        return export_df
    need_cols = {"Source", "Sku", "OrderId", "TxnDate"}
    if not need_cols.issubset(export_df.columns):
        return export_df
    day_map, oid_map = meesho_export_sku_recovery_maps(meesho_df)
    if not day_map and not oid_map:
        return export_df
    out = export_df.copy()
    is_m = out["Source"].astype(str).str.strip().str.lower() == "meesho"
    needs = is_m & out["Sku"].map(_export_row_needs_meesho_sku_recovery)
    if not needs.any():
        return out
    sub = out.loc[needs]
    oidv = sub["OrderId"].astype(str).str.strip()
    dtv = pd.to_datetime(sub["TxnDate"], errors="coerce")
    bad_day = dtv.isna()
    dk = np.where(bad_day, "", dtv.dt.strftime("%Y-%m-%d"))
    mk = oidv + "|" + pd.Series(dk, index=sub.index).astype(str)
    flat_day = {f"{a}|{b}": v for (a, b), v in day_map.items()}
    rday = mk.map(flat_day).fillna("")
    roid = oidv.map(oid_map).fillna("")
    rec = rday.where(rday.astype(str).str.len() > 0, roid)
    rec = rec.astype(str).str.strip()
    good = rec.map(looks_like_seller_listing_sku) & rec.str.len().gt(0)
    if good.any():
        out.loc[good.index[good], "Sku"] = rec.loc[good]
    return out


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

    oid = meesho_df["OrderId"].astype(str).str.strip()
    if "LineKey" in meesho_df.columns:
        lk = meesho_df["LineKey"].astype(str).str.strip()
        use = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
        oid = oid.where(~use, lk)
    lk_sales = (
        clean_line_id_series(meesho_df["LineKey"])
        if "LineKey" in meesho_df.columns
        else pd.Series("", index=meesho_df.index, dtype=str)
    )
    out = pd.DataFrame({
        "Sku":              sku_series,
        "TxnDate":          meesho_df["Date"],
        "Transaction Type": meesho_df["TxnType"],
        "Quantity":         meesho_df["Quantity"],
        "Units_Effective":  np.where(meesho_df["TxnType"] == "Refund", -meesho_df["Quantity"],
                            np.where(meesho_df["TxnType"] == "Cancel", -meesho_df["Quantity"],
                                     meesho_df["Quantity"])),
        "Source":           "Meesho",
        "OrderId":          oid,
        "LineKey":          lk_sales,
    })
    return out

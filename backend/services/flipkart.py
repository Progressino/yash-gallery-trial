"""
Flipkart loader — extracted 1-for-1 from app.py.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .helpers import clean_line_id_series, clean_sku, map_to_oms_sku

_FK_SKU_PLACEHOLDER_RE = re.compile(r"^[\s\-\u2014\u2013\.]*$", re.U)


def _fk_is_sku_placeholder(val) -> bool:
    """Flipkart reports often use '-' / blank in SKU when the real id lives in SKU ID / FSN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    s = str(val).strip().upper()
    if not s or s in ("NAN", "NONE", "N/A", "NA", "#N/A", "NULL", "UNDEFINED", "SKU"):
        return True
    return bool(_FK_SKU_PLACEHOLDER_RE.fullmatch(s))


def _fk_col_ci(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _fk_series_optional_col(df: pd.DataFrame, *header_names: str) -> pd.Series:
    """Column as string series, or empty strings when the report omits that field."""
    c = _fk_col_ci(df, *header_names)
    if c:
        return df[c].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=str)


def _fk_skuish_column_noise(cl: str) -> bool:
    """True if header looks like totals/qty/amount — not a listing identifier column."""
    return any(
        x in cl
        for x in (
            "total",
            "quantity",
            "qty",
            "count",
            "rate",
            "amount",
            "value",
            "gmv",
            "price",
            "commission",
            "fee",
            "tax",
        )
    )


def _fk_fill_placeholders_from_skuish_columns(
    xl: pd.DataFrame, out: pd.Series, used: Set[str]
) -> pd.Series:
    """
    Flipkart sheets often put “-” in the primary SKU cell while the real seller code
    lives under a variant header (Merchant SKU, typo “Skuu”, etc.). Scan remaining
    columns whose names contain “sku” (case-insensitive).
    """
    bad = out.map(_fk_is_sku_placeholder)
    if not bad.any():
        return out
    for col in xl.columns:
        if col in used:
            continue
        cl = str(col).strip().lower()
        if "sku" not in cl and "fsn" not in cl:
            continue
        if _fk_skuish_column_noise(cl):
            continue
        alt = xl[col].astype(str).str.strip()
        out = out.where(~bad, alt)
        bad = out.map(_fk_is_sku_placeholder)
        if not bad.any():
            break
    return out


def _fk_coalesced_listing_sku_series(xl: pd.DataFrame) -> pd.Series:
    """First non-placeholder token per row across common Flipkart Sales Report columns."""
    groups = (
        ("SKU", "Sku"),
        ("Seller SKU", "Seller Sku", "seller sku"),
        ("Merchant SKU", "Merchant Sku", "Partner SKU", "Vendor SKU", "Your SKU"),
        ("SKU ID", "SKU Id", "Sku Id", "sku id"),
        ("FSN", "fsn"),
        ("Listing ID", "Listing Id", "listing id"),
    )
    cols: List[str] = []
    for names in groups:
        c = None
        for nm in names:
            c = _fk_col_ci(xl, nm)
            if c:
                break
        if c and c not in cols:
            cols.append(c)
    n = len(xl)
    if not cols:
        out = pd.Series([""] * n, index=xl.index, dtype=str)
        return _fk_fill_placeholders_from_skuish_columns(xl, out, set())
    out = xl[cols[0]].astype(str).str.strip()
    for c in cols[1:]:
        alt = xl[c].astype(str).str.strip()
        bad = out.map(_fk_is_sku_placeholder)
        out = out.where(~bad, alt)
    return _fk_fill_placeholders_from_skuish_columns(xl, out, set(cols))


def _fk_map_listing_to_oms(raw, mapping: Dict[str, str]) -> str:
    if _fk_is_sku_placeholder(raw):
        return ""
    return map_to_oms_sku(clean_sku(str(raw)), mapping)


def _fk_parse_excel_order_dates(series: pd.Series) -> pd.Series:
    """
    Flipkart exports store dates as:
    - Excel day serials (float/int, e.g. 46126) in XLSX/CSV exports
    - YYYYMMDD integers (e.g. 20260410) in XLSB / MACO aggregate exports
    - Datetime strings (e.g. "2026-04-10") in some API exports
    Plain ``pd.to_datetime`` treats floats as nanoseconds (wrong) and
    leaves serial strings as NaT — both drop real rows.
    """
    s = series.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    if pd.api.types.is_numeric_dtype(s):
        # YYYYMMDD integers (20000101–20991231) vs Excel day serials (~20000–65000)
        yyyymmdd_mask = (s >= 20_000_101) & (s <= 20_991_231)
        if yyyymmdd_mask.all():
            return pd.to_datetime(
                s.astype("int64").astype(str), format="%Y%m%d", errors="coerce"
            )
        elif yyyymmdd_mask.any():
            out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            out.loc[yyyymmdd_mask] = pd.to_datetime(
                s.loc[yyyymmdd_mask].astype("int64").astype(str), format="%Y%m%d", errors="coerce"
            )
            out.loc[~yyyymmdd_mask] = pd.to_datetime(
                s.loc[~yyyymmdd_mask], unit="D", origin="1899-12-30", errors="coerce"
            )
            return out
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

    # object / string / mixed
    num = pd.to_numeric(s, errors="coerce")
    yyyymmdd_mask = num.notna() & (num >= 20_000_101) & (num <= 20_991_231)
    serial_mask   = num.notna() & (num >= 20_000) & (num <= 65_000) & ~yyyymmdd_mask
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if yyyymmdd_mask.any():
        out.loc[yyyymmdd_mask] = pd.to_datetime(
            num.loc[yyyymmdd_mask].astype("int64").astype(str), format="%Y%m%d", errors="coerce"
        )
    if serial_mask.any():
        out.loc[serial_mask] = pd.to_datetime(
            num.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    rest = ~yyyymmdd_mask & ~serial_mask & s.notna()
    if rest.any():
        out.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce")
    return out


def _fk_sales_report_date_column(xl: pd.DataFrame) -> Optional[str]:
    """
    Choose which Sales Report column drives ``Date`` for filtering and dashboards.

    Flipkart Seller Hub “units for day X” usually follows **dispatch / order** timing.
    **Buyer Invoice Date** often lags by a day or more, so preferring it alone caused
    rows to fall outside the user’s chosen calendar day (under-count vs gross in FK UI).
    """
    low = {str(c).strip().lower(): c for c in xl.columns}
    for cand in (
        "dispatch date",
        "dispatch date (seller)",
        "shipment date",
        "shipped on",
        "shipped date",
        "packed on",
        "packed date",
        "handover date",
        "order date",
        "order created date",
        "sale date",
        "buyer invoice date",
    ):
        if cand in low:
            return low[cand]
    return None


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


def _fk_finalize_out_with_line_keys(out: pd.DataFrame, raw_status: Optional[pd.Series] = None) -> pd.DataFrame:
    """Add RawStatus + LineKey so Tier-3 merges dedupe even when OMS mapping changes."""
    if out.empty:
        return out
    df = out.copy()
    if raw_status is not None:
        df["RawStatus"] = raw_status.fillna("").astype(str).str.strip()
    else:
        df["RawStatus"] = df["TxnType"].astype(str)
    oid = clean_line_id_series(df["OrderId"])
    inv = (
        clean_line_id_series(df["BuyerInvoiceId"])
        if "BuyerInvoiceId" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    oitem = (
        clean_line_id_series(df["OrderItemId"])
        if "OrderItemId" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    # One marketplace order id can span multiple invoice / item lines — without this,
    # unified sales dedupe collapses rows and understates units vs Flipkart totals.
    base_lk = oid
    base_lk = base_lk + np.where(inv.ne(""), "|" + inv, "")
    base_lk = base_lk + np.where(oitem.ne(""), "|" + oitem, "")
    # Rare exports omit invoice / item ids — disambiguate duplicate keys within the file.
    _dup_idx = base_lk.groupby(base_lk).cumcount().astype(str)
    base_lk = base_lk + np.where(_dup_idx != "0", "|L" + _dup_idx, "")
    if "LineKey" not in df.columns:
        df["LineKey"] = base_lk
    else:
        lk = clean_line_id_series(df["LineKey"])
        df["LineKey"] = lk.where(lk.ne(""), base_lk)
    df["OrderId"] = oid.where(oid.ne(""), df["LineKey"])
    return df


def _parse_flipkart_xlsx(
    file_bytes: bytes, fname: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    # ── Try legacy "Sales Report" sheet first ──────────────────────────────
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Sales Report")
        xl.columns = xl.columns.str.strip()
        if not xl.empty and "Event Sub Type" in xl.columns:
            date_col = _fk_sales_report_date_column(xl)
            if date_col:
                xl["Date"] = _fk_parse_excel_order_dates(xl[date_col])

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
                _eff_sku = _fk_coalesced_listing_sku_series(xl)
                xl["OMS_SKU"]        = _eff_sku.map(lambda x: _fk_map_listing_to_oms(x, mapping))

                state_col = next((c for c in xl.columns if "Delivery State" in c), None)
                xl["State"] = (
                    xl[state_col].fillna("").astype(str).str.upper().str.strip()
                    if state_col
                    else pd.Series("", index=xl.index, dtype=str)
                )

                xl["OrderId"] = _fk_series_optional_col(xl, "Order ID", "Order Id")
                xl["BuyerInvoiceId"] = _fk_series_optional_col(xl, "Buyer Invoice ID", "Buyer Invoice Id")
                xl["OrderItemId"] = _fk_series_optional_col(
                    xl,
                    "Order Item ID",
                    "Order Item Id",
                    "Sales Order Item ID",
                    "Sales Order Item Id",
                    "Order Item ID (Sale)",
                )
                xl["Brand"] = _fk_series_optional_col(xl, "Brand")

                out = xl[["Date", "Month", "TxnType", "Quantity", "Invoice_Amount",
                          "OMS_SKU", "State", "OrderId", "BuyerInvoiceId", "OrderItemId", "Brand"]].copy()
                out = _fk_finalize_out_with_line_keys(
                    out, raw_status=xl["Event Sub Type"].fillna("").astype(str).str.strip()
                )
                return out.dropna(subset=["Date"])
    except Exception:
        pass

    # ── Try new "Order Export" format (first sheet, columns: SKU ID / Gross Units / Order Date) ──
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        # Detect new format by required columns
        if not ("SKU ID" in xl.columns and "Gross Units" in xl.columns and "Order Date" in xl.columns):
            return pd.DataFrame()

        xl["Date"] = _fk_parse_excel_order_dates(xl["Order Date"])
        xl = xl.dropna(subset=["Date"])
        if xl.empty:
            return pd.DataFrame()

        file_month = _fk_month_from_filename(fname)

        def _get_month(d):
            return file_month if file_month else d.to_period("M").strftime("%Y-%m")

        for col in ["Gross Units", "Final Sale Units", "Return Units", "Cancellation Units",
                    "Final Sale Amount", "Return Amount"]:
            xl[col] = pd.to_numeric(xl.get(col, 0), errors="coerce").fillna(0)

        sku_id_col = "SKU ID"
        _sid = xl[sku_id_col].astype(str).str.strip()
        _used: Set[str] = {sku_id_col}
        for nm in (
            "Seller SKU",
            "Seller Sku",
            "SKU",
            "Merchant SKU",
            "Merchant Sku",
            "Partner SKU",
            "FSN",
            "fsn",
        ):
            c = _fk_col_ci(xl, nm)
            if c:
                _used.add(c)
                alt = xl[c].astype(str).str.strip()
                bad = _sid.map(_fk_is_sku_placeholder)
                _sid = _sid.where(~bad, alt)
        _sid = _fk_fill_placeholders_from_skuish_columns(xl, _sid, _used)
        xl["OMS_SKU"] = _sid.map(lambda x: _fk_map_listing_to_oms(x, mapping))
        xl["Brand"] = _fk_series_optional_col(xl, "Brand")

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

        # Final Sale Units = Gross − Cancellation − Return (already net of both).
        # Using Final Sale for Shipment qty and ALSO creating separate Refund rows
        # would double-deduct returns. We use Final Sale directly — no Refund rows.
        xl["_ship_qty"] = pd.to_numeric(xl.get("Final Sale Units", xl.get("Gross Units", 0)), errors="coerce").fillna(0)
        ship = xl[xl["_ship_qty"] > 0].copy()
        if ship.empty:
            return pd.DataFrame()

        out = pd.DataFrame({
            "Date":           ship["Date"],
            "Month":          ship["Date"].apply(_get_month),
            "TxnType":        "Shipment",
            "Quantity":       ship["_ship_qty"].astype("float32"),
            "Invoice_Amount": ship["Final Sale Amount"].astype("float32"),
            "OMS_SKU":        ship["OMS_SKU"],
            "State":          "",
            "OrderId":        ship["_OrderId"],
            "BuyerInvoiceId": "",
            "Brand":          ship["Brand"],
        })
        return _fk_finalize_out_with_line_keys(out, raw_status=None)

    except Exception:
        return pd.DataFrame()


def _parse_flipkart_xlsb(
    file_bytes: bytes, fname: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Parse Flipkart MACO / aggregate .xlsb (Binary Excel) report.
    Columns match the Order Export format: Product Id, SKU ID, Brand, Order Date
    (YYYYMMDD int), Gross Units, Cancellation Units, Return Units, Final Sale Units,
    Final Sale Amount, etc.
    Final Sale Units = Gross − Cancellation − Return — no separate Refund rows generated.
    Requires the pyxlsb package (add to requirements.txt).
    """
    try:
        import pyxlsb
    except ImportError:
        return pd.DataFrame()

    try:
        rows_data: List[List] = []
        with pyxlsb.open_workbook(io.BytesIO(file_bytes)) as wb:
            if not wb.sheets:
                return pd.DataFrame()
            names = [str(s) for s in wb.sheets]
            lower = {n.lower().strip(): n for n in names}
            sheet_name = lower.get("flipkart sale") or names[0]
            with wb.get_sheet(sheet_name) as ws:
                for row in ws.rows():
                    rows_data.append([c.v for c in row])

        if len(rows_data) < 2:
            return pd.DataFrame()

        headers = [
            str(h).strip() if h is not None else f"_col{i}"
            for i, h in enumerate(rows_data[0])
        ]
        xl = pd.DataFrame(rows_data[1:], columns=headers)
        xl = xl.dropna(how="all")
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        if "SKU ID" not in xl.columns or "Order Date" not in xl.columns:
            return pd.DataFrame()

        xl["Date"] = _fk_parse_excel_order_dates(xl["Order Date"])
        xl = xl.dropna(subset=["Date"])
        if xl.empty:
            return pd.DataFrame()

        file_month = _fk_month_from_filename(fname)

        def _get_month(d):
            return file_month if file_month else d.to_period("M").strftime("%Y-%m")

        for col in ["Gross Units", "Final Sale Units", "Return Units", "Cancellation Units",
                    "Final Sale Amount", "Return Amount"]:
            xl[col] = pd.to_numeric(xl.get(col, 0), errors="coerce").fillna(0)

        sku_id_col = "SKU ID"
        _sid = xl[sku_id_col].astype(str).str.strip()
        _used: Set[str] = {sku_id_col}
        for nm in ("Seller SKU", "Seller Sku", "SKU", "Merchant SKU",
                   "Merchant Sku", "Partner SKU", "FSN", "fsn"):
            c = _fk_col_ci(xl, nm)
            if c:
                _used.add(c)
                alt = xl[c].astype(str).str.strip()
                bad = _sid.map(_fk_is_sku_placeholder)
                _sid = _sid.where(~bad, alt)
        _sid = _fk_fill_placeholders_from_skuish_columns(xl, _sid, _used)
        xl["OMS_SKU"] = _sid.map(lambda x: _fk_map_listing_to_oms(x, mapping))
        xl["Brand"] = _fk_series_optional_col(xl, "Brand")

        product_id_col = "Product Id" if "Product Id" in xl.columns else ""
        if product_id_col:
            xl["_OrderId"] = (
                xl[product_id_col].astype(str) + "_" +
                xl["SKU ID"].astype(str) + "_" +
                xl["Date"].dt.strftime("%Y%m%d")
            )
        else:
            xl["_OrderId"] = xl["SKU ID"].astype(str) + "_" + xl["Date"].dt.strftime("%Y%m%d")

        # Final Sale Units = Gross − Cancellation − Return (already net).
        # Do NOT create separate Refund rows.
        xl["_ship_qty"] = pd.to_numeric(
            xl.get("Final Sale Units", xl.get("Gross Units", 0)), errors="coerce"
        ).fillna(0)
        ship = xl[xl["_ship_qty"] > 0].copy()
        if ship.empty:
            return pd.DataFrame()

        out = pd.DataFrame({
            "Date":           ship["Date"],
            "Month":          ship["Date"].apply(_get_month),
            "TxnType":        "Shipment",
            "Quantity":       ship["_ship_qty"].astype("float32"),
            "Invoice_Amount": ship["Final Sale Amount"].astype("float32"),
            "OMS_SKU":        ship["OMS_SKU"],
            "State":          "",
            "OrderId":        ship["_OrderId"],
            "BuyerInvoiceId": "",
            "Brand":          ship["Brand"],
        })
        return _fk_finalize_out_with_line_keys(out, raw_status=None)

    except Exception:
        return pd.DataFrame()


def load_flipkart_from_zip(
    zip_bytes: bytes, mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Parse Flipkart master ZIP containing monthly Sales Report XLSXs (and optional XLSB).
    Returns (combined_df, file_count, skipped_list).
    """
    dfs: List[pd.DataFrame] = []
    skipped: List[str] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open Flipkart ZIP: {e}"]

    xlsx_items = [n for n in root_zf.namelist() if n.lower().endswith(".xlsx")]
    xls_legacy = [
        n for n in root_zf.namelist()
        if n.lower().endswith(".xls") and not n.lower().endswith(".xlsx")
    ]
    xlsb_items = [n for n in root_zf.namelist() if n.lower().endswith(".xlsb")]
    all_items = xlsx_items + xls_legacy + xlsb_items

    for item_name in xlsx_items + xls_legacy:
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

    for item_name in xlsb_items:
        base = Path(item_name).name
        try:
            file_bytes = root_zf.read(item_name)
            df = _parse_flipkart_xlsb(file_bytes, base, mapping)
            if df.empty:
                skipped.append(f"{base}: no data / unrecognised XLSB format")
            else:
                dfs.append(df)
        except Exception as e:
            skipped.append(f"{base}: {e}")

    if not dfs:
        return pd.DataFrame(), len(all_items), skipped

    combined = pd.concat(dfs, ignore_index=True)
    from .daily_store import _dedup_platform_df

    combined = _dedup_platform_df(combined, "flipkart")
    return combined, len(all_items), skipped


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
        df["_Date"] = _fk_parse_excel_order_dates(df[date_col])
        df = df.dropna(subset=["_Date"])
        if df.empty:
            return pd.DataFrame()

        # SKU — Seller SKU first; fall back to SKU ID / FSN when cell is '-' or blank
        sku_col = next((c for c in df.columns if "seller sku" in c.lower()), None)
        if not sku_col:
            return pd.DataFrame()
        eff = df[sku_col].astype(str).str.strip()
        _used_o: Set[str] = {sku_col}
        for nm in ("SKU ID", "SKU Id", "Sku Id", "Merchant SKU", "Partner SKU", "FSN", "fsn"):
            alt_c = _fk_col_ci(df, nm)
            if alt_c:
                _used_o.add(alt_c)
                alt = df[alt_c].astype(str).str.strip()
                bad = eff.map(_fk_is_sku_placeholder)
                eff = eff.where(~bad, alt)
        eff = _fk_fill_placeholders_from_skuish_columns(df, eff, _used_o)
        df["_OMS_SKU"] = eff.map(lambda x: _fk_map_listing_to_oms(x, mapping))

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

        brand_col = next((c for c in df.columns if c.strip().lower() == "brand"), None)
        df["_Brand"] = df[brand_col].fillna("").astype(str).str.strip() if brand_col else ""

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
            "Brand":          df["_Brand"],
        })
        out = _fk_finalize_out_with_line_keys(out, raw_status=None)
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
    Expands aggregated rows into Shipment/Refund rows (ship qty = Final Sale Units when present).
    """
    try:
        xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name="earn_more_report")
        if xl.empty:
            return pd.DataFrame()

        xl.columns = xl.columns.str.strip()

        # Normalize date (Excel serials / strings must not use naive to_datetime — see _fk_parse_excel_order_dates)
        xl["Date"] = _fk_parse_excel_order_dates(xl["Order Date"])
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

        # SKU mapping (coalesce when SKU ID is a placeholder)
        _sid = xl["SKU ID"].astype(str).str.strip()
        _used_e: Set[str] = {"SKU ID"}
        for nm in (
            "Seller SKU",
            "Seller Sku",
            "SKU",
            "Merchant SKU",
            "Partner SKU",
            "FSN",
            "fsn",
        ):
            c = _fk_col_ci(xl, nm)
            if c:
                _used_e.add(c)
                alt = xl[c].astype(str).str.strip()
                bad = _sid.map(_fk_is_sku_placeholder)
                _sid = _sid.where(~bad, alt)
        _sid = _fk_fill_placeholders_from_skuish_columns(xl, _sid, _used_e)
        xl["OMS_SKU"] = _sid.map(lambda x: _fk_map_listing_to_oms(x, mapping))
        xl["Brand"] = _fk_series_optional_col(xl, "Brand")

        pid_col = "Product Id" if "Product Id" in xl.columns else None

        def _pid_series(sdf: pd.DataFrame) -> pd.Series:
            if pid_col and pid_col in sdf.columns:
                return sdf[pid_col].fillna("").astype(str).str.strip()
            return pd.Series("", index=sdf.index, dtype=str)

        # Final Sale Units = Gross − Cancellation − Return (already net of both).
        # Using Final Sale for Shipment qty and ALSO creating separate Refund rows
        # would double-deduct returns. We use Final Sale directly — no Refund rows.
        xl["_ship_qty"] = pd.to_numeric(
            xl.get("Final Sale Units", xl.get("Gross Units", 0)), errors="coerce"
        ).fillna(0)
        ship = xl[xl["_ship_qty"] > 0].copy()
        if ship.empty:
            return pd.DataFrame()

        pid = _pid_series(ship)
        sk = ship["SKU ID"].astype(str).str.strip()
        dts = ship["Date"].dt.strftime("%Y%m%d")
        sq = pd.to_numeric(ship["_ship_qty"], errors="coerce").fillna(0).astype(np.int64).astype(str)
        ship_lk = "FKEM|" + pid + "|" + sk + "|" + dts + "|SHIP|" + sq
        out = pd.DataFrame({
            "Date":           ship["Date"],
            "Month":          ship["Date"].apply(_get_month),
            "TxnType":        "Shipment",
            "Quantity":       ship["_ship_qty"].astype("float32"),
            "Invoice_Amount": ship["Final Sale Amount"].astype("float32"),
            "OMS_SKU":        ship["OMS_SKU"],
            "State":          "",
            "OrderId":        ship_lk,
            "LineKey":        ship_lk,
            "BuyerInvoiceId": "",
            "Brand":          ship["Brand"],
        })
        out["RawStatus"] = "Shipment"
        return out

    except Exception:
        return pd.DataFrame()


def flipkart_to_sales_rows(fk_df: pd.DataFrame) -> pd.DataFrame:
    if fk_df.empty:
        return pd.DataFrame()
    if "Brand" in fk_df.columns:
        _brand = fk_df["Brand"].fillna("").astype(str).str.strip()
    else:
        _brand = pd.Series("", index=fk_df.index, dtype=str)
    oid = fk_df["OrderId"].astype(str).str.strip()
    if "LineKey" in fk_df.columns:
        lk = fk_df["LineKey"].astype(str).str.strip()
        use = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
        oid = oid.where(~use, lk)
    lk_sales = (
        clean_line_id_series(fk_df["LineKey"])
        if "LineKey" in fk_df.columns
        else pd.Series("", index=fk_df.index, dtype=str)
    )
    # Units_Effective must mirror MACO / Order Export "Final Sale Units":
    # Sales Report emits paired Sale + Cancellation rows — cancellations must subtract
    # (previously Cancel used 0, so net units overstated vs seller hub aggregates).
    out = pd.DataFrame({
        "Sku":              fk_df["OMS_SKU"],
        "TxnDate":          fk_df["Date"],
        "Transaction Type": fk_df["TxnType"],
        "Quantity":         fk_df["Quantity"],
        "Units_Effective":  np.where(fk_df["TxnType"] == "Refund",       -fk_df["Quantity"],
                            np.where(fk_df["TxnType"] == "Cancel",       -fk_df["Quantity"],
                            np.where(fk_df["TxnType"] == "ReturnCancel",   fk_df["Quantity"],
                                     fk_df["Quantity"]))),
        "Source":           "Flipkart",
        "OrderId":          oid,
        "DSR_Segment":      _brand,
        "LineKey":          lk_sales,
    })
    return out

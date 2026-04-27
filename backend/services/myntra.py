"""
Myntra PPMP loader — extracted 1-for-1 from app.py.
"""
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .helpers import (
    canonical_pl_sku_key,
    clean_line_id_series,
    clean_sku,
    is_non_rto_forward_milestone_status,
    map_to_oms_sku,
    normalized_sku_forms_for_lookup,
)


_DMY_SLASH_DASH_RE = re.compile(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s|$)")
_ISO_RANGE_IN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_-](\d{4}-\d{2}-\d{2})")
_TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*$")


def _parse_myntra_datetime_series(series: pd.Series) -> pd.Series:
    """
    Parse Myntra report date/time cells with a day-first fallback for D/M/Y strings.
    Prevents month swaps like ``05-03-2026`` being read as May 3rd.
    """
    raw = series.astype(str)
    out = pd.to_datetime(raw, errors="coerce")
    # Pandas interprets time-only tokens as "today HH:MM", which is wrong for reports.
    time_only = raw.str.match(_TIME_ONLY_RE, na=False)
    if time_only.any():
        out = out.mask(time_only, pd.NaT)
    dmy_mask = raw.str.match(_DMY_SLASH_DASH_RE, na=False)
    if dmy_mask.any():
        dmy = pd.to_datetime(raw.where(dmy_mask), errors="coerce", dayfirst=True)
        use = dmy.notna()
        out = out.mask(use, dmy)
    return out


def _myntra_filename_date_fallback(filename: str) -> pd.Timestamp | None:
    """
    Seller exports can contain time-only values (e.g. ``20:02.0``) in date columns.
    Fall back to filename range ``YYYY-MM-DD_YYYY-MM-DD`` and use the range end date.
    """
    s = str(filename or "")
    m = _ISO_RANGE_IN_NAME_RE.search(s)
    if not m:
        return None
    try:
        d1 = pd.Timestamp(m.group(2))
        return d1.normalize()
    except Exception:
        return None


def _tokens_for_myntra_lookup(raw) -> List[str]:
    """Expand a cell value into lookup tokens (88022920.0 vs 88022920, scientific notation, | splits)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s0 = (
        str(raw)
        .strip()
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\xa0", " ")
    )
    if not s0 or s0.lower() in ("nan", "none", ""):
        return []
    # PPMP / Excel: "9609|1589" or pasted glitch — try whole cell and each segment.
    pieces: List[str] = [s0]
    if "|" in s0:
        for p in s0.split("|"):
            pp = p.strip()
            if pp and pp not in pieces:
                pieces.append(pp)

    out: List[str] = []
    seen: set[str] = set()

    def add(t: str):
        tt = t.strip()
        if not tt or tt.upper() in ("NAN", "NONE"):
            return
        if tt not in seen:
            seen.add(tt)
            out.append(tt)

    for s in pieces:
        add(s)
        add(s.upper())
        cs = clean_sku(s)
        if cs:
            add(cs)
        try:
            f = float(str(s).replace(",", "").strip())
            if not np.isfinite(f):
                continue
            if f == int(f) and abs(f) < 1e16:
                add(str(int(f)))
        except ValueError:
            pass
    # YARYKASS100672680 → 100672680 when PPMP sends only the numeric tail
    for t in list(out):
        for m in re.finditer(r"\d{6,}", t):
            add(m.group(0))
    for t in list(out):
        for nf in normalized_sku_forms_for_lookup(t):
            add(nf)
    return out


def resolve_myntra_row_keys_to_oms(mapping: Dict[str, str], raw_values: List) -> str:
    """
    Map a Myntra line to OMS using the SKU master. Mapping-sheet YRN column and PPMP
    "Myntra SKU code" share the same keys. Tries each raw field in order
    (YRN / Myntra SKU code → other SKU columns → style / article / EAN…) with numeric normalization.
    """
    ordered_tokens: List[str] = []
    for raw in raw_values:
        for t in _tokens_for_myntra_lookup(raw):
            if t not in ordered_tokens:
                ordered_tokens.append(t)
    if not ordered_tokens:
        return ""

    for k in ordered_tokens:
        if k in mapping:
            return mapping[k]
        alt = canonical_pl_sku_key(k)
        if alt != k and alt in mapping:
            return mapping[alt]
    for k in ordered_tokens:
        o = map_to_oms_sku(k, mapping)
        if o != k:
            return o
    if ordered_tokens:
        return map_to_oms_sku(ordered_tokens[0], mapping)
    return ""


def resolve_myntra_row_to_oms(
    mapping: Dict[str, str],
    yrn_raw,
    sku_raw,
    style_raw=None,
) -> str:
    """Backward-compatible ordering: YRN, seller SKU, Style ID."""
    return resolve_myntra_row_keys_to_oms(mapping, [yrn_raw, sku_raw, style_raw])


def _ordered_myntra_identifier_columns(cols: List[str]) -> List[str]:
    """PPMP / Seller Orders / Returns: collect column names in mapping priority order (headers lowercased)."""
    seen: set[str] = set()
    out: List[str] = []

    def extend(bucket: List[str]) -> None:
        for c in bucket:
            if c not in seen:
                seen.add(c)
                out.append(c)

    extend([c for c in cols if "yrn" in c])
    # PPMP "Myntra SKU code" equals the mapping sheet YRN column — same priority as YRN.
    extend([
        c for c in cols
        if c not in seen
        and (
            c in ("myntra sku code", "myntra_sku_code", "myntra sku", "merchant sku code")
            or ("myntra" in c and "sku" in c and "oms" not in c)
        )
    ])

    _sku_exact = {
        "sku_id", "skuid", "sku", "sku id", "seller sku", "seller sku code",
        "myntra sku code", "seller_sku_code", "product sku", "packet sku",
        "item sku", "variant sku", "merchant sku", "partner sku",
        "article id", "article_id",
    }
    extend([c for c in cols if c in _sku_exact])

    extend([
        c for c in cols
        if c not in seen
        and "sku" in c
        and "oms" not in c
        and "style" not in c
        and "yrn" not in c
        and "total" not in c
    ])

    extend([
        c for c in cols
        if c not in seen
        and "reverse" not in c
        and (
            ("style" in c and "id" in c)
            or c in ("style_id", "style code", "stylecode", "master style id", "global style id",
                     "style id", "oms style id", "child style id")
        )
    ])

    extend([
        c for c in cols
        if c not in seen
        and (
            ("article" in c and ("id" in c or "code" in c or "sku" in c))
            or "packet_article" in c
            or "seller_article" in c
            or "article_type" in c
            or c in ("article code", "article_code", "base article id")
        )
    ])

    extend([c for c in cols if c not in seen and ("ean" in c or "upc" in c or "gtin" in c)])

    # PPMP / report variants: style as "code", vendor/brand/merchant SKUs, catalog ids (low priority).
    _bad_token = (
        "oms", "warehouse", "pincode", "amount", "price", "invoice", "discount",
        "commission", "fee", "tax", "gst", "quantity", "total",
    )

    def _ok_extra(c: str) -> bool:
        if c in seen or any(b in c for b in _bad_token):
            return False
        if c in ("packet_id", "order_id", "sub_order_id", "suborder id", "order id"):
            return False
        return True

    extend([
        c for c in cols
        if _ok_extra(c)
        and (
            c
            in (
                "product_id",
                "item_id",
                "listing_id",
                "catalog_id",
                "model_no",
                "model_id",
                "fsn",
            )
            or (
                c.endswith("_sku")
                and not c.startswith("total")
            )
            or ("style" in c and "code" in c)
            or (
                ("vendor" in c or "merchant" in c or "brand" in c)
                and ("sku" in c or "code" in c or "article" in c)
            )
        )
    ])

    return out


def _myntra_line_dedup_series(df: pd.DataFrame) -> pd.Series:
    """
    Stable per-line id for Tier-3 dedup and unified sales (order line > packet > seller ids).
    Must stay consistent across re-uploads so merge_platform_data / SQLite + warm-cache merges
    do not double-count the same row.
    """
    if df.empty:
        return pd.Series(dtype=str)
    keys = pd.Series("", index=df.index, dtype=str)
    for col in (
        "order line id",
        "packet id",
        "seller order id",
        "order id fk",
    ):
        if col not in df.columns:
            continue
        cand = clean_line_id_series(df[col])
        need = keys.eq("") & cand.ne("")
        keys = keys.mask(need, cand)
    if keys.eq("").any() and "store order id" in df.columns:
        cand = clean_line_id_series(df["store order id"])
        need = keys.eq("") & cand.ne("")
        keys = keys.mask(need, cand)
    return keys


def _coalesce_myntra_dispatch_over_order_date(df: pd.DataFrame, order_dates: pd.Series) -> pd.Series:
    """
    Prefer actual dispatch / shipment / packed timestamps when present so daily totals
    align with seller DSR spreadsheets that bucket by fulfilment date, not order_created.
    """
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    # Headers are lowercased in _parse_myntra_csv; seller reports often use spaces ("shipped on").
    candidates = (
        "dispatch_date",
        "dispatched_date",
        "actual_dispatch_date",
        "dispatch date",
        "dispatch on",
        "shipment_date",
        "shipment date",
        "shipped_date",
        "shipped on",
        "ship_date",
        "ship date",
        "shipping_date",
        "shipping date",
        "packed_date",
        "packed on",
        "handover_date",
        "handover date",
        "courier_handover_date",
        "manifest_date",
        "logistics_dispatch_date",
        "out_for_delivery_date",
        "out for delivery date",
        "inscanned on",
        "inscanned_date",
        "delivered on",
        "delivered_date",
        "delivery_date",
    )
    for col in candidates:
        if col not in df.columns:
            continue
        d2 = _parse_myntra_datetime_series(df[col])
        if d2.isna().all():
            continue
        need = out.isna() & d2.notna()
        out = out.mask(need, d2)

    # Safety cap: if dispatch date is more than 60 days after the order date, it is
    # almost certainly a data artifact (e.g. Jan-Mar archive rows where the *actual*
    # dispatch date falls in April).  Fall back to order_date for those rows so they
    # stay bucketed in the correct reporting month instead of polluting a future date.
    _MAX_DISPATCH_LAG = pd.Timedelta(days=60)
    _order = pd.to_datetime(order_dates, errors="coerce")
    _lag_exceeded = out.notna() & _order.notna() & ((out - _order) > _MAX_DISPATCH_LAG)
    out = out.mask(_lag_exceeded, _order)

    return out.fillna(order_dates)


def _myntra_use_dispatch_date() -> bool:
    """
    When enabled, prefer dispatch/shipment dates over order-created date.
    Default OFF to keep monthly totals aligned with seller reports that pivot on
    created/order date.
    """
    v = (os.environ.get("MYNTRA_USE_DISPATCH_DATE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _myntra_reverse_status_is_refund(val: str) -> bool:
    """
    True only for reverse statuses that clearly indicate a return/reverse leg.
    Avoid treating arbitrary non-empty placeholders as refunds.
    """
    s = str(val).strip().upper()
    if not s or s in {"NAN", "NONE", "NULL", "-", "N/A", "NA", "NO"}:
        return False
    if "CANCEL" in s:
        return False
    if is_non_rto_forward_milestone_status(s):
        return False
    return (
        "RETURN" in s
        or "REVERSE" in s
        or "REFUND" in s
        or s.startswith("RTO")
        or s.startswith("RTD")
        or s.startswith("RVP")
        or s in {"R", "RS", "RD", "RTOD", "RVP", "RTN", "RSHIP", "REVERSE", "REV"}
    )


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

    df["_Date"] = _parse_myntra_datetime_series(df[date_col])
    null_mask = df["_Date"].isna()
    if null_mask.any():
        df.loc[null_mask, "_Date"] = pd.to_datetime(
            df.loc[null_mask, date_col].astype(str), format="%Y%m%d", errors="coerce"
        )
    # Some seller CSVs have time-only values in created-on/shipped-on; recover month/day
    # from filename date range so rows are not dropped as invalid.
    if df["_Date"].isna().any():
        _hint = _myntra_filename_date_fallback(filename)
        if _hint is not None:
            df["_Date"] = df["_Date"].fillna(_hint)

    # Default to order-created date for stable month totals. Dispatch-date bucketing is
    # available via MYNTRA_USE_DISPATCH_DATE=1 when explicitly needed.
    if _myntra_use_dispatch_date():
        df["_Date"] = _coalesce_myntra_dispatch_over_order_date(df, df["_Date"])
    df = df.dropna(subset=["_Date"])
    if df.empty:
        return pd.DataFrame(), "All dates invalid"

    id_cols = _ordered_myntra_identifier_columns(list(df.columns))
    if not id_cols:
        return pd.DataFrame(), "No SKU / YRN / Style / Article identifier column"

    def _row_oms(row: pd.Series) -> str:
        return resolve_myntra_row_keys_to_oms(mapping, [row[c] for c in id_cols])

    df["_OMS_SKU"] = df.apply(_row_oms, axis=1)

    qty_col = next((c for c in df.columns if c == "quantity"), None)
    df["_Qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1) if qty_col else 1.0

    rev_col = next(
        (c for c in df.columns if c in [
            "invoiceamount", "invoice_amount", "net_amount", "shipment_value",
            "final amount", "seller price",
        ]),
        None,
    )
    df["_Rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0) if rev_col else 0.0

    # Status column detection (priority order)
    # Handles both PPMP (underscore_separated) and Seller Orders Report (space separated).
    # Important: never use reverse-status columns as the primary classifier.
    _preferred = (
        "forward_order_status",
        "order_status",
        "order status",
        "status",
        "shipment_status",
        "shipment status",
        "sub_order_status",
        "current_status",
        "item_status",
        "article_status",
        "delivery_status",
        "transaction_type",
        "txn_type",
    )
    status_col = next((c for c in _preferred if c in df.columns), None)
    if status_col is None:
        _fallback = [
            c for c in df.columns
            if ("status" in c and "reverse" not in c)
        ]
        if _fallback:
            status_col = _fallback[0]

    # ── Dedicated reverse_order_status column (Myntra PPMP key return signal) ──
    # PPMP files have BOTH forward_order_status AND reverse_order_status.
    # A row with a non-empty reverse_order_status means it is a return —
    # regardless of what forward_order_status says (often "DELIVERED").
    reverse_col = next(
        (c for c in df.columns if c == "reverse_order_status"
         or ("reverse" in c and "order" in c and "status" in c)),
        None,
    )

    def _myntra_txn(s):
        s = str(s).strip().upper()
        if s in ("FORWARD", "FWD"):
            return "Shipment"
        if s in ("REVERSE", "REV", "RVP"):                    # RVP = Reverse Pickup
            return "Refund"
        if not is_non_rto_forward_milestone_status(s) and (
                "RETURN" in s or "REVERSE" in s or s.startswith("RTO")
                or s.startswith("RTD") or s.startswith("RVP")
                or s in ("R", "RS", "RD", "RTOD", "RVP", "RTN", "RSHIP")):
            return "Refund"
        # "F" and "IC" frequently appear in seller exports while quantities still need to be
        # counted in gross month/day checks done by ops; treat both as Shipment.
        # Keep explicit textual cancels and FAILED as cancel.
        if s in ("F", "IC"):
            return "Shipment"
        if "CANCEL" in s or s in ("FAILED",):
            return "Cancel"
        if s in ("C", "SH", "PK", "D", "S", "SHIPPED", "CONFIRMED", "DELIVERED",
                 "PACKED", "PACKING_IN_PROGRESS", "READY_FOR_DISPATCH",
                 "MANIFESTED", "OUT_FOR_DELIVERY", "WP"):
            return "Shipment"
        return "Shipment"

    # Step 1: classify from the detected forward status column
    df["_TxnType"] = df[status_col].apply(_myntra_txn) if status_col else "Shipment"

    # Step 1b: RT files are customer-return files — every row is a Refund.
    # The filename starts with "RT " (e.g. "RT April-2024.csv").
    # These files have order_status=C (delivered) but fr_is_refunded=1, so the
    # normal status-based classification wrongly marks them as Shipments.
    if filename.upper().startswith("RT "):
        df["_TxnType"] = "Refund"

    # Step 2: if reverse_order_status exists, override rows that explicitly signal
    # reverse/return states. Do not treat arbitrary non-empty values as returns.
    if reverse_col is not None:
        _rev_vals = (
            df[reverse_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        _is_return = _rev_vals.map(_myntra_reverse_status_is_refund)
        df.loc[_is_return, "_TxnType"] = "Refund"

    state_col  = next((c for c in df.columns if c in [
        "state", "customer_delivery_state_code", "buyer state", "ship state",
    ]), None)
    pm_col     = next((c for c in df.columns if "payment_method" in c or "payment method" in c), None)
    wh_col     = next((c for c in df.columns if "warehouse_id" in c or "warehouse id" in c), None)
    # Line-level id for dedup: must not use ``store order id`` first (parent order shared by many
    # lines — was chosen because it appears before ``order line id`` in CSV column order).
    _order_id_priority = (
        "order line id",
        "seller order id",
        "order_id",
        "packet_id",
        "packet id",
        "sub order id",
        "suborder id",
        "order id",
        "order id fk",
        "order release id",
        "store order id",
    )
    order_col = next((c for c in _order_id_priority if c in df.columns), None)
    _raw_status = df[status_col].fillna("").astype(str).str.strip() if status_col else ""

    line_keys = _myntra_line_dedup_series(df)
    oid_fb = (
        clean_line_id_series(df[order_col])
        if order_col
        else pd.Series("", index=df.index, dtype=str)
    )
    oid_out = line_keys.where(line_keys.ne(""), oid_fb)

    parent_order = (
        clean_line_id_series(df["store order id"])
        if "store order id" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )

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
        "OrderId":        oid_out,
        "LineKey":        line_keys,
        "ParentOrderId":  parent_order,
    })
    out["Month"]       = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    return out, f"OK | status_col={status_col!r}"


def load_myntra_from_zip(
    zip_bytes: bytes,
    mapping: Dict[str, str],
    source_filename: str | None = None,
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
    from .daily_store import _dedup_platform_df
    from .helpers import apply_dsr_segment_from_upload_filename

    combined = _dedup_platform_df(combined, "myntra")
    combined = apply_dsr_segment_from_upload_filename(combined, source_filename, "Myntra")
    return combined, len(csv_items), skipped


def myntra_to_sales_rows(myntra_df: pd.DataFrame) -> pd.DataFrame:
    if myntra_df.empty:
        return pd.DataFrame()
    oid = myntra_df["OrderId"].astype(str).str.strip()
    if "LineKey" in myntra_df.columns:
        lk = myntra_df["LineKey"].astype(str).str.strip()
        use = lk.ne("") & ~lk.str.lower().isin(["nan", "none"])
        oid = oid.where(~use, lk)
    lk_out = (
        clean_line_id_series(myntra_df["LineKey"])
        if "LineKey" in myntra_df.columns
        else pd.Series("", index=myntra_df.index, dtype=str)
    )
    out = pd.DataFrame({
        "Sku":              myntra_df["OMS_SKU"],
        "TxnDate":          myntra_df["Date"],
        "Transaction Type": myntra_df["TxnType"],
        "Quantity":         myntra_df["Quantity"],
        "Units_Effective":  np.where(myntra_df["TxnType"] == "Refund", -myntra_df["Quantity"],
                            np.where(myntra_df["TxnType"] == "Cancel", -myntra_df["Quantity"],
                                     myntra_df["Quantity"])),
        "Source":           "Myntra",
        "OrderId":          oid,
        "LineKey":          lk_out,
    })
    if "DSR_Segment" in myntra_df.columns:
        out["DSR_Segment"] = myntra_df["DSR_Segment"].astype(str).fillna("").values
    return out

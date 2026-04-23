"""
MTR (Amazon Tax Report) loader — extracted 1-for-1 from app.py.
Key functions: parse_mtr_csv(), load_mtr_from_zip()
"""
import gc
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ── Date parsing ─────────────────────────────────────────────

def _parse_date_flexible(series: pd.Series) -> pd.Series:
    priority_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
        "%d-%b-%Y", "%d/%b/%Y", "%m/%d/%Y",
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",       # ISO 8601 without tz
        "%d %b %Y", "%d %B %Y",    # "12 Dec 2025", "12 December 2025"
    ]
    non_null = series.dropna()
    threshold = max(int(len(non_null) * 0.70), 1) if len(non_null) > 0 else 1
    for fmt in priority_formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() >= threshold:
                return parsed
        except Exception:
            continue
    # Fallback: ISO 8601 with timezone offset (e.g. "2026-03-23T22:37:50+05:30").
    # Strip the offset and keep the LOCAL datetime — do NOT convert to UTC, because
    # converting IST→UTC shifts early-morning dates (before 05:30 IST) to the previous
    # day, causing March 23 orders to appear as March 22.
    import re as _re
    try:
        stripped = series.astype(str).str.replace(r'\s*[+-]\d{2}:?\d{2}$', '', regex=True)
        parsed = pd.to_datetime(stripped, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
        if parsed.notna().sum() >= threshold:
            return parsed
    except Exception:
        pass
    # Last resort UTC conversion (may shift dates for non-IST timezones)
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        if parsed.notna().sum() >= threshold:
            return parsed.dt.tz_convert(None)
    except Exception:
        pass
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _downcast_mtr(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Report_Type", "Transaction_Type", "Ship_To_State",
              "Warehouse_Id", "Fulfillment", "Payment_Method",
              "IRN_Status", "Month", "Month_Label"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in ["Quantity", "Invoice_Amount", "Total_Tax", "CGST", "SGST", "IGST"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    return df


# ── Single-file CSV parser ───────────────────────────────────

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_mtr_csv(csv_bytes: bytes, source_file: str) -> Tuple[pd.DataFrame, str]:
    """Parse a single MTR CSV. Returns (df, status_message)."""
    try:
        raw = pd.read_csv(
            io.BytesIO(csv_bytes), dtype=str, low_memory=False,
            encoding="utf-8", on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        try:
            raw = pd.read_csv(
                io.BytesIO(csv_bytes), dtype=str, low_memory=False,
                encoding="ISO-8859-1", on_bad_lines="skip",
            )
        except Exception:
            return pd.DataFrame(), "Encoding Error"
    except Exception as e:
        return pd.DataFrame(), f"Parse Error: {e}"

    if raw.empty:
        return pd.DataFrame(), "Empty file"

    # Normalize: strip, lowercase, replace dashes with spaces (handles Amazon Order Report format)
    raw.columns = raw.columns.astype(str).str.strip().str.lower().str.replace("-", " ", regex=False)
    is_b2b = "buyer name" in raw.columns or "customer bill to gstid" in raw.columns
    report_type = "B2B" if is_b2b else "B2C"

    want_b2c = {
        "shipment date", "invoice date", "transaction type", "sku",
        "quantity", "invoice amount", "total tax amount",
        "cgst tax", "sgst tax", "igst tax", "ship to state", "warehouse id",
        "fulfillment channel", "payment method code", "order id", "invoice number",
        # Amazon Order Report (after dash → space normalization):
        "amazon order id", "merchant order id", "purchase date", "last updated date",
        "order status", "order total", "item price", "ship date", "ship state",
        "buyer name", "product name",
        # Amazon FBA Shipment Report (numeric-named files):
        "customer shipment date", "merchant sku", "product amount", "shipping amount",
        "shipment to state", "shipment to city", "fnsku", "asin", "fc",
    }
    want = (want_b2c | {"irn filing status", "customer bill to gstid"}) if is_b2b else want_b2c
    raw = raw[[c for c in raw.columns if c in want]]

    date_col = next(
        (d for d in [
            "shipment date", "invoice date", "transaction date", "order date",
            "purchase date", "ship date", "last updated date",
            "customer shipment date",   # Amazon FBA Shipment Report
        ]
         if d in raw.columns), None,
    )
    raw["_Date"] = _parse_date_flexible(raw[date_col]) if date_col else pd.NaT

    # Fallback: other date columns
    for alt in ["invoice date", "shipment date", "transaction date", "order date"]:
        if alt in raw.columns and alt != date_col:
            null_mask = raw["_Date"].isna()
            if null_mask.any():
                raw.loc[null_mask, "_Date"] = _parse_date_flexible(raw.loc[null_mask, alt])

    # Fallback: filename month/year
    fn_lower = source_file.lower()
    fn_match = re.search(
        r"-(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)-(\d{4})", fn_lower,
    )
    fallback_ts = pd.NaT
    if fn_match:
        try:
            fallback_ts = pd.Timestamp(
                year=int(fn_match.group(2)),
                month=_MONTH_MAP[fn_match.group(1)],
                day=1,
            )
        except Exception:
            pass

    filled_dates = 0
    if fallback_ts is not pd.NaT:
        still_null = raw["_Date"].isna()
        filled_dates = int(still_null.sum())
        if filled_dates:
            raw.loc[still_null, "_Date"] = fallback_ts

    initial_len = len(raw)
    raw = raw.dropna(subset=["_Date"])
    dropped_dates = initial_len - len(raw)
    if raw.empty:
        return pd.DataFrame(), f"All {initial_len} rows had invalid/missing dates."

    current_year = datetime.now().year
    valid_mask = raw["_Date"].dt.year.between(2018, current_year + 1)
    ghost_rows = (~valid_mask).sum()
    raw = raw[valid_mask]
    if raw.empty:
        return pd.DataFrame(), "All rows had out-of-range years."

    def g(name):
        return (raw[name].fillna("").astype(str).str.strip()
                if name in raw.columns
                else pd.Series("", index=raw.index, dtype=str))

    def gn(name):
        return (pd.to_numeric(raw[name], errors="coerce").fillna(0).astype("float32")
                if name in raw.columns
                else pd.Series(0.0, index=raw.index, dtype="float32"))

    txn = g("transaction type").str.lower()
    txn_std = pd.Series("Shipment", index=raw.index, dtype=str)
    txn_std[txn.str.contains(
        r"return|refund|reimbursement|chargeback|reversal|credit\s*note|reimbursed",
        na=False,
    )] = "Refund"
    txn_std[txn.str.contains("cancel", na=False)] = "Cancel"

    # Order / FBA reports often leave "transaction type" empty for every row; use order status.
    os_lower = g("order status").str.lower()
    still_ship = txn_std == "Shipment"
    txn_std[still_ship & os_lower.str.contains(r"return|refund|reimbursed", na=False)] = "Refund"
    txn_std[still_ship & os_lower.str.contains("cancel", na=False)] = "Cancel"

    # Resolve column aliases (MTR / Order Report / FBA Shipment Report)
    _sku_col      = next((c for c in ["sku", "merchant sku"] if c in raw.columns), None)
    _order_id_col = next((c for c in ["order id", "amazon order id", "merchant order id"] if c in raw.columns), None)
    _inv_amt_col  = next((c for c in ["invoice amount", "product amount", "order total", "item price"] if c in raw.columns), None)
    _state_col    = next((c for c in ["ship to state", "shipment to state", "ship state"] if c in raw.columns), None)

    out = pd.DataFrame({
        "Date":             raw["_Date"],
        "Report_Type":      report_type,
        "Transaction_Type": txn_std,
        "SKU":              g(_sku_col) if _sku_col else pd.Series("", index=raw.index, dtype=str),
        "Quantity":         gn("quantity"),
        "Invoice_Amount":   gn(_inv_amt_col) if _inv_amt_col else pd.Series(0.0, index=raw.index, dtype="float32"),
        "Total_Tax":        gn("total tax amount"),
        "CGST":             gn("cgst tax"),
        "SGST":             gn("sgst tax"),
        "IGST":             gn("igst tax"),
        "Ship_To_State":    g(_state_col).str.upper() if _state_col else pd.Series("", index=raw.index, dtype=str),
        "Warehouse_Id":     g("warehouse id"),
        "Fulfillment":      g("fulfillment channel"),
        "Payment_Method":   g("payment method code"),
        "Order_Id":         g(_order_id_col) if _order_id_col else pd.Series("", index=raw.index, dtype=str),
        "Invoice_Number":   g("invoice number"),
        "Buyer_Name":       g("buyer name"),
        "IRN_Status":       g("irn filing status"),
    })
    del raw

    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    out = _downcast_mtr(out)

    msgs: list[str] = []
    if filled_dates:
        label = fallback_ts.strftime("%b %Y") if fallback_ts is not pd.NaT else "?"
        msgs.append(f"Filled {filled_dates} rows using filename date ({label}).")
    if dropped_dates:
        msgs.append(f"Dropped {dropped_dates} rows — no date after fallback.")
    if ghost_rows:
        msgs.append(f"Dropped {ghost_rows} rows with out-of-range years.")

    from .helpers import apply_dsr_segment_from_upload_filename

    out = apply_dsr_segment_from_upload_filename(out, source_file, "Amazon")

    return out, ("OK" if not msgs else " | ".join(msgs))


# ── ZIP loader ───────────────────────────────────────────────

def _collect_csv_entries(main_zip_file, depth: int = 0):
    entries, skipped = [], []
    for item_name in main_zip_file.namelist():
        base = Path(item_name).name
        if not base:
            continue
        if base.lower().endswith(".zip") and depth < 3:
            try:
                data = main_zip_file.read(item_name)
                sub_zf = zipfile.ZipFile(io.BytesIO(data))
                sub_entries, sub_skipped = _collect_csv_entries(sub_zf, depth + 1)
                entries.extend(sub_entries)
                skipped.extend(sub_skipped)
                del data
            except Exception as e:
                skipped.append(f"{base}: Zip extraction error {e}")
        elif base.lower().endswith(".csv"):
            entries.append((main_zip_file, item_name, base))
    return entries, skipped


def _amazon_fba_aggregate_order_lines(df: pd.DataFrame, *, has_order_id: bool) -> pd.DataFrame:
    """
    FBA / order-keyed rows without a tax invoice: Amazon often emits the same
    Customer Shipment event as multiple CSV lines (exact clones or split qty rows
    with the same timestamp). Drop exact duplicates, then sum quantity and amounts
    per (Order, SKU, txn, shipment second) so each fulfilment line is counted once
    with the correct unit total.
    """
    if df.empty:
        return df
    work = df.copy()
    work = work.drop(columns=["SKU", "_day", "_qty"], errors="ignore")
    work["_ts"] = pd.to_datetime(work["Date"], errors="coerce").dt.floor("s")

    dup_subset = ["_sk", "Transaction_Type", "_ts", "Quantity"]
    if "Invoice_Amount" in work.columns:
        dup_subset.append("Invoice_Amount")
    if has_order_id:
        dup_subset = ["Order_Id"] + dup_subset
    work = work.drop_duplicates(subset=dup_subset, keep="first")

    sum_cols = [
        c for c in (
            "Quantity", "Invoice_Amount", "Total_Tax", "CGST", "SGST", "IGST",
        ) if c in work.columns
    ]
    first_cols = [
        c for c in (
            "Report_Type", "Ship_To_State", "Warehouse_Id", "Fulfillment",
            "Payment_Method", "Invoice_Number", "Buyer_Name", "IRN_Status",
        ) if c in work.columns and c not in sum_cols
    ]
    agg: dict = {"Date": "max"}
    for c in sum_cols:
        agg[c] = "sum"
    for c in first_cols:
        agg[c] = "first"

    # One logical line per Amazon order + SKU + txn: sum qty across shipment instants
    # (seconds apart) after exact duplicate removal — matches seller "unique line" exports.
    gcols = (["Order_Id", "_sk", "Transaction_Type"] if has_order_id
             else ["_sk", "Transaction_Type"])
    work = work.reset_index(drop=True)
    # pandas 3.x: ``as_index=False`` + dict agg can corrupt grouper alignment; use reset_index().
    gb = work.groupby(gcols, sort=False, observed=True)
    out = gb.agg(agg).reset_index()
    if "DSR_Segment" in work.columns:
        dsr_one = work.groupby(gcols, sort=False, observed=True)["DSR_Segment"].first().reset_index()
        out = out.merge(dsr_one, on=gcols, how="left")
    out = out.rename(columns={"_sk": "SKU"})
    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    out["Month_Label"] = out["Date"].dt.strftime("%b %Y")
    if not has_order_id:
        out["Order_Id"] = ""
    return out


def dedup_amazon_mtr_dataframe(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate lines when multiple Tier-1 ZIPs / reports overlap (same
    shipment appears in more than one file). Uses stable business keys instead
    of raw timestamps so minor date differences do not create extra rows.

    - Invoice rows: Invoice_Number + canonical SKU + transaction type + qty
      (date omitted — avoids duplicate from report-vs-shipment date noise).
    - Invoice rows suppress duplicate FBA-style lines for the same Amazon order.
    - Order-keyed rows **without** invoice: exact duplicate rows removed, then
      quantities and tax amounts **summed** per Order_Id + SKU + txn +
      shipment timestamp (second) so split/clone FBA Customer Shipment lines
      are one logical line.
    - Rows with neither id: same aggregation on SKU + txn + shipment second.
    """
    if combined.empty:
        return combined
    req = {"Invoice_Number", "Order_Id", "SKU", "Transaction_Type", "Quantity", "Date"}
    if not req.issubset(combined.columns):
        return combined

    from .sales import canonical_sales_sku  # lazy: avoids import cycles at module load

    d = combined.copy()
    inv_raw = d["Invoice_Number"].astype(str).str.strip()
    has_inv = inv_raw.ne("") & ~inv_raw.str.lower().isin(["nan", "none", "-", "n/a", "na"])

    d["_sk"] = d["SKU"].map(canonical_sales_sku)
    d["_day"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
    d["_qty"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0).round().astype("int64")

    inv_rows = d[has_inv].drop_duplicates(
        subset=["Invoice_Number", "_sk", "Transaction_Type", "_qty"], keep="last"
    )
    cov = set(inv_rows["Order_Id"].astype(str).str.strip().unique()) - {"", "nan", "none"}

    rest = d[~has_inv].copy()
    if cov:
        oid = rest["Order_Id"].astype(str).str.strip()
        txn_rest = rest["Transaction_Type"].astype(str).str.strip()
        # Invoice-keyed shipments suppress duplicate FBA shipment lines for the same order.
        # Refunds/credits usually share Order_Id but have no Invoice_Number — do not drop them.
        suppress_oid = oid.isin(cov) & (txn_rest == "Shipment")
        rest = rest[~suppress_oid]

    oid_ok = rest["Order_Id"].astype(str).str.strip()
    oid_ok = oid_ok.ne("") & ~oid_ok.str.lower().isin(["nan", "none"])
    by_oid = _amazon_fba_aggregate_order_lines(rest.loc[oid_ok], has_order_id=True)
    no_oid = _amazon_fba_aggregate_order_lines(rest.loc[~oid_ok], has_order_id=False)

    out = pd.concat([inv_rows, by_oid, no_oid], ignore_index=True)
    out = out.drop(columns=["_sk", "_day", "_qty"], errors="ignore")
    gc.collect()
    return _downcast_mtr(out)


def mtr_deduplicate(combined: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible name — delegates to dedup_amazon_mtr_dataframe."""
    return dedup_amazon_mtr_dataframe(combined)


def mtr_concat_and_dedup(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate parsed MTR DataFrames and apply standard dedup + downcast."""
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    gc.collect()
    combined = mtr_deduplicate(combined)
    return _downcast_mtr(combined)


def load_mtr_from_extracted_files(files: List[Tuple[str, bytes]]) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Build one MTR frame from extracted inner files (e.g. RAR contents): each
    entry may be a nested MTR ZIP or a single MTR CSV. Cross-file dedup applied once.
    """
    parts: List[pd.DataFrame] = []
    skipped: List[str] = []
    parsed_files = 0
    for name, blob in files:
        lower = name.lower()
        if lower.endswith(".zip"):
            part, n_csv, sk = load_mtr_from_zip(blob)
            skipped.extend(sk)
            if not part.empty:
                parts.append(part)
                parsed_files += 1
        elif lower.endswith(".csv"):
            df, msg = parse_mtr_csv(blob, Path(name).name)
            if df.empty:
                skipped.append(f"{name}: {msg}")
            else:
                parts.append(df)
                parsed_files += 1
                if msg != "OK":
                    skipped.append(f"{name}: Partial ({msg})")
    if not parts:
        return pd.DataFrame(), 0, skipped
    combined = mtr_concat_and_dedup(parts)
    return combined, parsed_files, skipped


def load_mtr_from_zip(zip_bytes: bytes) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Load all MTR CSVs from a master ZIP.
    Returns (combined_df, csv_count, skipped_list).
    """
    skipped: List[str] = []
    dfs: List[pd.DataFrame] = []

    try:
        root_zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        return pd.DataFrame(), 0, [f"Cannot open ZIP: {e}"]

    entries, skipped = _collect_csv_entries(root_zf)
    if not entries:
        return pd.DataFrame(), 0, skipped

    for zf, item_name, base in entries:
        try:
            data = zf.read(item_name)
            df, msg = parse_mtr_csv(data, base)
            del data
            gc.collect()
            if df.empty:
                skipped.append(f"{base}: {msg}")
            else:
                dfs.append(df)
                if msg != "OK":
                    skipped.append(f"{base}: Partial ({msg})")
        except Exception as e:
            skipped.append(f"{base}: Critical Error — {e}")

    if not dfs:
        return pd.DataFrame(), 0, skipped

    csv_count = len(dfs)
    combined = mtr_concat_and_dedup(dfs)
    del dfs
    gc.collect()

    return combined, csv_count, skipped

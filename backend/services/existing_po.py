"""
Existing PO sheet loader.
Parses an Excel/CSV uploaded by the user that contains open/pending PO quantities.

Returns a DataFrame with:
  - OMS_SKU
  - PO_Pipeline_Total   (total in-pipeline, used for net PO deduction)
  - PO_Qty_Ordered      (original qty ordered with manufacturer, if col found)
  - Pending_Cutting     (units awaiting cutting, if col found)
  - Balance_to_Dispatch (units cut but not dispatched, if col found)
"""
import io
from typing import Optional

import pandas as pd


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Case-insensitive exact column finder."""
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_col_fuzzy(cols: list[str], keywords: list[str]) -> Optional[str]:
    """Find a column whose name *contains* any of the given keywords (case-insensitive)."""
    for col in cols:
        col_l = col.lower()
        if any(kw in col_l for kw in keywords):
            return col
    return None


def parse_existing_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an existing PO tracking sheet.
    Accepts .xlsx or .csv.

    Extracts (all optional except SKU):
      - PO_Qty_Ordered      : qty originally ordered with manufacturer
                              (NEW ORDER / PO Qty / Ordered Qty)
      - Pending_Cutting     : units awaiting cutting
                              (Pending Cutting / Cut Pending)
      - Balance_to_Dispatch : units cut but not yet dispatched
                              (Balance to dispatch / Dispatch Balance)
      - Total_Balance       : total pipeline (Pending_Cutting + Balance_to_Dispatch + other)
                              → becomes PO_Pipeline_Total for engine deduction

    If none of the specific columns are found, falls back to any
    generic balance/qty column as PO_Pipeline_Total.
    """
    fn_lower = filename.lower()

    # ── Read file ────────────────────────────────────────────────
    try:
        if fn_lower.endswith(".csv"):
            try:
                raw = pd.read_csv(
                    io.BytesIO(file_bytes), dtype=str, encoding="utf-8", on_bad_lines="skip"
                )
            except UnicodeDecodeError:
                raw = pd.read_csv(
                    io.BytesIO(file_bytes), dtype=str, encoding="ISO-8859-1", on_bad_lines="skip"
                )
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")

    if raw.empty:
        raise ValueError("File is empty.")

    raw.columns = raw.columns.astype(str).str.strip()
    cols = list(raw.columns)

    # ── Find SKU column ─────────────────────────────────────────
    sku_col = _find_col(
        cols,
        ["OMS SKU", "OMS_SKU", "SKU", "Style Code", "Style", "Item Code",
         "Product Code", "ASIN", "Product ID", "Seller SKU"],
    )
    if sku_col is None:
        raise ValueError(
            f"Cannot find a SKU column. Available columns: {cols[:20]}"
        )

    # ── Find specific breakdown columns (all optional) ──────────

    # Originally ordered with manufacturer
    # Column may include a date suffix, e.g. "new order 23-02-2026" — fuzzy match handles this
    ordered_col = _find_col(cols, [
        "NEW ORDER", "New Order", "PO Qty", "PO Quantity",
        "Ordered Qty", "Ordered Quantity", "Order Qty",
    ])
    if ordered_col is None:
        ordered_col = _find_col_fuzzy(cols, ["new order", "po qty", "ordered qty", "order qty"])

    # Units awaiting cutting
    cutting_col = _find_col(cols, [
        "Pending Cutting", "Pending Cut", "Cutting Pending",
        "Pend Cutting", "Cut Pending", "Pending Cuts",
    ])
    if cutting_col is None:
        cutting_col = _find_col_fuzzy(cols, ["pending cut", "cut pending", "pending cutting"])

    # Units cut but not yet dispatched to warehouse
    dispatch_col = _find_col(cols, [
        "Balance to dispatch", "Dispatch Balance", "Bal to Dispatch",
        "Balance Dispatch", "Pending dispatch", "Pending Dispatch",
    ])
    if dispatch_col is None:
        dispatch_col = _find_col_fuzzy(cols, ["dispatch"])

    # Total pipeline balance (= Pending Cutting + Balance to Dispatch + any other stage)
    total_col = _find_col(cols, [
        "TOTAL BALANCE From Latest Status", "Total Balance From Latest Status",
        "Total Balance", "Total Bal", "Total Pending", "Net Balance", "TOTAL BALANCE",
    ])
    if total_col is None:
        total_col = _find_col_fuzzy(cols, ["total balance", "total bal"])

    # ── Generic fallback if NO specific columns found ────────────
    fallback_col: Optional[str] = None
    if not any([ordered_col, cutting_col, dispatch_col, total_col]):
        fallback_col = _find_col(
            cols,
            ["Balance Qty", "Balance Quantity", "Open Qty", "Open Quantity",
             "Pending Qty", "Pending Quantity", "Units", "Balance",
             "Qty", "Balance to dispatch", "TOTAL BALANCE From Latest Status",
             "Total Balance", "Dispatch Balance", "Pending dispatch",
             "New Order", "NEW ORDER"],
        )
        if fallback_col is None:
            fallback_col = _find_col_fuzzy(
                cols, ["balance", "dispatch", "pending", "open qty", "po qty", "new order"]
            )
        if fallback_col is None:
            raise ValueError(
                f"Cannot find a quantity/balance column. Available columns: {cols[:20]}"
            )

    # ── Build working DataFrame ──────────────────────────────────
    specific_cols = [
        c for c in [ordered_col, cutting_col, dispatch_col, total_col, fallback_col] if c
    ]
    all_needed = list(dict.fromkeys([sku_col] + specific_cols))  # deduplicate, preserve order
    df = raw[all_needed].copy()

    df[sku_col] = df[sku_col].astype(str).str.strip()
    df = df[df[sku_col].str.len() > 0]
    df = df[~df[sku_col].str.lower().isin(["sku", "oms_sku", "nan", "none", ""])]

    if df.empty:
        raise ValueError("No valid SKU rows found after parsing.")

    for c in specific_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Aggregate — multiple PO lines for the same SKU
    agg_dict = {c: "sum" for c in specific_cols}
    result = df.groupby(sku_col, as_index=False).agg(agg_dict)
    result = result.rename(columns={sku_col: "OMS_SKU"})

    # ── Map raw column names → semantic names ────────────────────
    rename_map: dict[str, str] = {}
    if ordered_col:   rename_map[ordered_col]  = "PO_Qty_Ordered"
    if cutting_col:   rename_map[cutting_col]   = "Pending_Cutting"
    if dispatch_col:  rename_map[dispatch_col]  = "Balance_to_Dispatch"
    if total_col:     rename_map[total_col]     = "Total_Balance"
    if fallback_col and fallback_col not in rename_map:
        rename_map[fallback_col] = "PO_Qty_Ordered"
    result = result.rename(columns=rename_map)

    # ── Derive PO_Pipeline_Total (used by engine to net off Gross PO) ──
    # Priority: Total_Balance > Pending_Cutting + Balance_to_Dispatch > PO_Qty_Ordered
    if "Total_Balance" in result.columns:
        result["PO_Pipeline_Total"] = result["Total_Balance"].clip(lower=0).astype(int)
        result = result.drop(columns=["Total_Balance"])          # absorbed into PO_Pipeline_Total
    elif "Pending_Cutting" in result.columns and "Balance_to_Dispatch" in result.columns:
        result["PO_Pipeline_Total"] = (
            result["Pending_Cutting"] + result["Balance_to_Dispatch"]
        ).clip(lower=0).astype(int)
    elif "PO_Qty_Ordered" in result.columns:
        result["PO_Pipeline_Total"] = result["PO_Qty_Ordered"].clip(lower=0).astype(int)
    else:
        result["PO_Pipeline_Total"] = 0

    # Ensure all pipeline cols are non-negative ints
    for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total"]:
        if c in result.columns:
            result[c] = result[c].clip(lower=0).astype(int)

    return result

"""
Existing PO sheet loader.
Parses an Excel/CSV uploaded by the user that contains open/pending PO quantities.
Returns a DataFrame with columns: OMS_SKU, PO_Pipeline_Total
"""
import io
from typing import Optional

import pandas as pd


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Case-insensitive column finder."""
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_existing_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an existing PO sheet.
    Accepts .xlsx or .csv.
    Looks for:
      - A SKU column (sku, oms sku, product code, item code, asin, style code, style)
      - A quantity/balance column (balance qty, open qty, po qty, quantity, balance, pending qty, units)
    Returns DataFrame with [OMS_SKU, PO_Pipeline_Total].
    """
    fn_lower = filename.lower()

    # ── Read file ────────────────────────────────────────────────
    try:
        if fn_lower.endswith(".csv"):
            try:
                raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                raw = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="ISO-8859-1", on_bad_lines="skip")
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")

    if raw.empty:
        raise ValueError("File is empty.")

    raw.columns = raw.columns.astype(str).str.strip()

    # ── Find SKU column ─────────────────────────────────────────
    sku_col = _find_col(
        list(raw.columns),
        ["OMS SKU", "OMS_SKU", "SKU", "Style Code", "Style", "Item Code",
         "Product Code", "ASIN", "Product ID", "Seller SKU"],
    )
    if sku_col is None:
        raise ValueError(
            f"Cannot find a SKU column. Available columns: {list(raw.columns)[:20]}"
        )

    # ── Find quantity column ─────────────────────────────────────
    qty_col = _find_col(
        list(raw.columns),
        ["Balance Qty", "Balance Quantity", "Open Qty", "Open Quantity",
         "PO Qty", "PO Quantity", "Pending Qty", "Pending Quantity",
         "Ordered Qty", "Ordered Quantity", "Quantity", "Units", "Balance",
         "Qty"],
    )
    if qty_col is None:
        raise ValueError(
            f"Cannot find a quantity/balance column. Available columns: {list(raw.columns)[:20]}"
        )

    df = raw[[sku_col, qty_col]].copy()
    df.columns = ["OMS_SKU", "Qty"]
    df["OMS_SKU"] = df["OMS_SKU"].astype(str).str.strip()
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

    # Drop blanks / header rows
    df = df[df["OMS_SKU"].str.len() > 0]
    df = df[~df["OMS_SKU"].str.lower().isin(["sku", "oms_sku", "nan", "none", ""])]

    if df.empty:
        raise ValueError("No valid SKU rows found after parsing.")

    # Aggregate — multiple PO lines for the same SKU
    result = df.groupby("OMS_SKU", as_index=False)["Qty"].sum()
    result = result.rename(columns={"Qty": "PO_Pipeline_Total"})
    result["PO_Pipeline_Total"] = result["PO_Pipeline_Total"].clip(lower=0).astype(int)

    return result

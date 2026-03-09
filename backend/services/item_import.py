"""
Excel / CSV bulk import parser for the Item Master module.

Expected columns (fuzzy-matched, case-insensitive):
  item_code, item_name, item_type, hsn_code, season,
  merchant_code, selling_price, purchase_price, sizes, launch_date

Returns a list of dicts ready to pass to item_db.bulk_create_items().
"""
import io
from typing import Optional

import pandas as pd


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_col_fuzzy(cols: list[str], keywords: list[str]) -> Optional[str]:
    for col in cols:
        col_l = col.lower()
        if any(kw in col_l for kw in keywords):
            return col
    return None


def parse_item_import(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse an Excel or CSV file and return a list of item dicts.
    Raises ValueError on unreadable files or missing required columns.
    """
    fn_lower = filename.lower()
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

    # ── Required: SKU / item code ─────────────────────────────────────────────
    code_col = _find_col(cols, ["item_code", "item code", "sku", "style code", "style"]) or \
               _find_col_fuzzy(cols, ["item_code", "item code", "style code"])
    if code_col is None:
        raise ValueError(
            f"Cannot find item_code column. Available: {cols[:15]}"
        )

    # ── Required: item name ───────────────────────────────────────────────────
    name_col = _find_col(cols, ["item_name", "item name", "style name", "name", "description"]) or \
               _find_col_fuzzy(cols, ["item_name", "item name", "style name"])
    if name_col is None:
        raise ValueError(
            f"Cannot find item_name column. Available: {cols[:15]}"
        )

    # ── Optional columns ──────────────────────────────────────────────────────
    type_col    = _find_col(cols, ["item_type", "item type", "type", "category"]) or \
                  _find_col_fuzzy(cols, ["item_type", "item type", "category"])
    hsn_col     = _find_col(cols, ["hsn_code", "hsn code", "hsn"]) or \
                  _find_col_fuzzy(cols, ["hsn"])
    season_col  = _find_col(cols, ["season"]) or _find_col_fuzzy(cols, ["season"])
    merch_col   = _find_col(cols, ["merchant_code", "merchant code", "merchant"]) or \
                  _find_col_fuzzy(cols, ["merchant"])
    sell_col    = _find_col(cols, ["selling_price", "selling price", "mrp", "sale price"]) or \
                  _find_col_fuzzy(cols, ["selling", "mrp"])
    purch_col   = _find_col(cols, ["purchase_price", "purchase price", "cost price", "cost"]) or \
                  _find_col_fuzzy(cols, ["purchase", "cost"])
    sizes_col   = _find_col(cols, ["sizes", "size range", "size list"]) or \
                  _find_col_fuzzy(cols, ["sizes", "size range"])
    launch_col  = _find_col(cols, ["launch_date", "launch date", "launch"]) or \
                  _find_col_fuzzy(cols, ["launch"])

    def _val(row, col: Optional[str], default: str = "") -> str:
        if col is None:
            return default
        v = row.get(col, default)
        return "" if pd.isna(v) else str(v).strip()

    results: list[dict] = []
    for _, row in raw.iterrows():
        code = _val(row, code_col)
        name = _val(row, name_col)
        if not code or code.lower() in ("nan", "none", ""):
            continue

        sizes_raw = _val(row, sizes_col)
        sizes = [s.strip() for s in sizes_raw.split(",") if s.strip()] if sizes_raw else []

        try:
            sp = float(_val(row, sell_col) or 0)
        except (ValueError, TypeError):
            sp = 0.0
        try:
            pp = float(_val(row, purch_col) or 0)
        except (ValueError, TypeError):
            pp = 0.0

        results.append({
            "item_code":      code,
            "item_name":      name or code,
            "item_type":      _val(row, type_col, "FG"),
            "hsn_code":       _val(row, hsn_col),
            "season":         _val(row, season_col),
            "merchant_code":  _val(row, merch_col),
            "selling_price":  sp,
            "purchase_price": pp,
            "sizes":          sizes,
            "launch_date":    _val(row, launch_col),
        })

    if not results:
        raise ValueError("No valid rows found in file.")

    return results

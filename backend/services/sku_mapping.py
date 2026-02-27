"""
SKU Mapping loader — extracted 1-for-1 from app.py load_sku_mapping().
"""
import io
from typing import Dict

import pandas as pd


def parse_sku_mapping(file_bytes: bytes) -> Dict[str, str]:
    """
    Parse a multi-sheet Excel SKU mapping file.
    Returns {seller_sku_upper → oms_sku} dict.
    """
    mapping: Dict[str, str] = {}
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        if df.empty or len(df.columns) < 2:
            continue

        seller_col, oms_col = None, None
        for col in df.columns:
            col_lower = str(col).lower()
            if (
                any(k in col_lower for k in ["seller", "myntra", "meesho", "snapdeal", "sku id"])
                and "sku" in col_lower
            ):
                seller_col = col
            if "oms" in col_lower and "sku" in col_lower:
                oms_col = col

        if seller_col is None and len(df.columns) > 1:
            seller_col = df.columns[1]
        if oms_col is None:
            oms_col = df.columns[-1]

        if seller_col and oms_col:
            for _, row in df.iterrows():
                s = _clean(row.get(seller_col, ""))
                o = _clean(row.get(oms_col, ""))
                if s and o and s != "nan" and o != "nan":
                    mapping[s] = o

    return mapping


def _clean(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip().upper()

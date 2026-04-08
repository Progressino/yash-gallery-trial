"""
Shared helper functions used across all services.
Extracted 1-for-1 from app.py.
"""
import re
import io
import zipfile
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── SKU helpers ────────────────────────────────────────────────

def clean_sku(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip().upper()


# Match Amazon PL listing spellings (1023PLYK* → 1023YK*) for map lookups.
_PL_YK = re.compile(r"^(\d+)PL(YK)", re.I)


def canonical_pl_sku_key(sku: str) -> str:
    c = clean_sku(sku)
    if not c:
        return c
    return _PL_YK.sub(r"\1\2", c)


def map_to_oms_sku(seller_sku, mapping: Dict[str, str]) -> str:
    if pd.isna(seller_sku):
        return seller_sku
    c = clean_sku(seller_sku)
    if not c:
        return c
    if c in mapping:
        return mapping[c]
    alt = canonical_pl_sku_key(c)
    if alt != c and alt in mapping:
        return mapping[alt]
    # Excel / CSV: style ids as 1.02E+08 — normalize to integer string for map keys
    try:
        f = float(str(seller_sku).strip().replace(",", ""))
        if np.isfinite(f) and f == int(f) and abs(f) < 1e16:
            ik = str(int(f))
            if ik != c and ik in mapping:
                return mapping[ik]
            pl = canonical_pl_sku_key(ik)
            if pl in mapping:
                return mapping[pl]
    except (ValueError, OverflowError):
        pass
    return c


def mapping_lookup_sets(mapping: Dict[str, str]) -> Tuple[Set[str], Set[str]]:
    """
    Normalized map keys (plus PL-alias of each key) and OMS values.
    Used to tell whether a sales/export token is covered by the master sheet.
    """
    key_set: Set[str] = set()
    for k in mapping.keys():
        kk = clean_sku(k)
        if kk:
            key_set.add(kk)
            key_set.add(canonical_pl_sku_key(kk))
    val_set = {clean_sku(v) for v in mapping.values() if clean_sku(v)}
    return key_set, val_set


def sku_recognized_in_master(
    token: str,
    mapping: Dict[str, str],
    *,
    key_set: Optional[Set[str]] = None,
    val_set: Optional[Set[str]] = None,
) -> bool:
    """True if token appears as a seller/marketplace key or as an OMS value in the master."""
    if not mapping:
        return True
    c = clean_sku(token)
    if not c:
        return True
    if key_set is None or val_set is None:
        key_set, val_set = mapping_lookup_sets(mapping)
    pl = canonical_pl_sku_key(c)
    return c in key_set or pl in key_set or c in val_set


def get_parent_sku(oms_sku) -> str:
    if pd.isna(oms_sku):
        return oms_sku
    s = str(oms_sku).strip()
    marketplace_suffixes = [
        "_Myntra", "_Flipkart", "_Amazon", "_Meesho",
        "_MYNTRA", "_FLIPKART", "_AMAZON", "_MEESHO",
    ]
    for suf in marketplace_suffixes:
        if s.endswith(suf):
            s = s.replace(suf, "")
            break
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2:
            last = parts[-1].upper()
            size_patterns = {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL"}
            common_colors = {
                "RED", "BLUE", "GREEN", "BLACK", "WHITE", "YELLOW", "PINK", "PURPLE",
                "ORANGE", "BROWN", "GREY", "GRAY", "NAVY", "MAROON", "BEIGE", "CREAM",
                "GOLD", "SILVER", "TAN", "KHAKI", "OLIVE", "TEAL", "CORAL", "PEACH",
            }
            is_size  = (last in size_patterns or last.endswith("XL") or last.isdigit()
                        or (len(last) <= 4 and any(c in last for c in ["S", "M", "L", "X"])))
            is_color = (last in common_colors) or any(c in last for c in common_colors)
            if is_size or is_color:
                s = "-".join(parts[:-1])
    return s


# ── DataFrame helpers ──────────────────────────────────────────

def _downcast_sales(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Transaction Type", "Source"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in ["Quantity", "Units_Effective"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    return df


def read_csv_safe(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()


def read_zip_csv(zip_bytes: bytes) -> pd.DataFrame:
    """Read the first CSV inside a ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return pd.DataFrame()
            data = zf.read(csv_names[0])
            return pd.read_csv(io.BytesIO(data), dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _coerce_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert category columns to string so parquet can handle them."""
    out = df.copy()
    for col in out.select_dtypes(include="category").columns:
        out[col] = out[col].astype(str)
    return out

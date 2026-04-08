"""
Shared helper functions used across all services.
Extracted 1-for-1 from app.py.
"""
import re
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── SKU helpers ────────────────────────────────────────────────

def clean_sku(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip().upper()


# Match Amazon PL listing spellings (1023PLYK* → 1023YK*) for map lookups.
_PL_YK = re.compile(r"^(\d+)PL(YK)", re.I)
_HYPHEN_SPACES = re.compile(r"\s*-\s*")
_DOT_BEFORE_HYPHEN = re.compile(r"\.(?=-)")


def canonical_pl_sku_key(sku: str) -> str:
    c = clean_sku(sku)
    if not c:
        return c
    return _PL_YK.sub(r"\1\2", c)


def normalized_sku_forms_for_lookup(value) -> List[str]:
    """
    Spacing around hyphens and stray dots before size suffix (e.g. GREEN.-3XL → GREEN-3XL),
    plus PL↔YK variants — common between marketplace exports and the master sheet.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    c = clean_sku(value)
    if not c:
        return []
    bases = [c]
    hy = _HYPHEN_SPACES.sub("-", c)
    if hy != c:
        bases.append(hy)
    dot = _DOT_BEFORE_HYPHEN.sub("", c)
    if dot != c:
        bases.append(dot)
    dot_hy = _DOT_BEFORE_HYPHEN.sub("", hy)
    if dot_hy not in bases:
        bases.append(dot_hy)
    seen: Set[str] = set()
    out: List[str] = []
    for b in bases:
        if not b:
            continue
        for x in (b, canonical_pl_sku_key(b)):
            if x and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def integer_token_variants(s: str) -> Set[str]:
    """
    Align YRN / style IDs across Excel, pandas, and CSV: 100672680.0 vs 100672680 vs 1.0067268E+8.
    Non-numeric SKUs return only the cleaned string.
    """
    out: Set[str] = set()
    if not s:
        return out
    t = str(s).strip().replace(",", "")
    if t:
        out.add(t)
        ut = t.upper()
        if ut != t:
            out.add(ut)
    try:
        f = float(t)
        if np.isfinite(f) and f == int(f) and abs(f) < 1e16:
            ik = str(int(f))
            out.add(ik)
    except (ValueError, OverflowError):
        pass
    return out


def map_to_oms_sku(seller_sku, mapping: Dict[str, str]) -> str:
    if pd.isna(seller_sku):
        return seller_sku
    c = clean_sku(seller_sku)
    if not c:
        return c
    for form in normalized_sku_forms_for_lookup(seller_sku):
        for tok in integer_token_variants(form):
            if tok in mapping:
                return mapping[tok]
            pl = canonical_pl_sku_key(tok)
            if pl in mapping:
                return mapping[pl]
    # Excel / CSV: style ids as 1.02E+08 — normalize to integer string for map keys
    try:
        f = float(str(seller_sku).strip().replace(",", ""))
        if np.isfinite(f) and f == int(f) and abs(f) < 1e16:
            ik = str(int(f))
            for cand in (ik, canonical_pl_sku_key(ik)):
                if cand in mapping:
                    return mapping[cand]
    except (ValueError, OverflowError):
        pass
    return c


def _tokens_one_master_cell(value) -> Set[str]:
    """All lookup tokens derived from one key or OMS cell (spacing, PL, ints)."""
    out: Set[str] = set()
    for form in normalized_sku_forms_for_lookup(value):
        for tok in integer_token_variants(form):
            out.add(tok)
            out.add(canonical_pl_sku_key(tok))
    return out


def mapping_lookup_sets(mapping: Dict[str, str]) -> Tuple[Set[str], Set[str]]:
    """
    Normalized map keys and OMS values (incl. PL↔YK for both — sales often canonicalise OMS to YK).
    Used to tell whether a sales/export token is covered by the master sheet.
    Integer YRNs include 100672680.0-style aliases so they match Excel/pandas floats.
    """
    key_set: Set[str] = set()
    for k in mapping.keys():
        key_set.update(_tokens_one_master_cell(k))
    val_set: Set[str] = set()
    for v in mapping.values():
        val_set.update(_tokens_one_master_cell(v))
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
    cand: Set[str] = set()
    for form in normalized_sku_forms_for_lookup(token):
        for tok in integer_token_variants(form):
            cand.add(tok)
            cand.add(canonical_pl_sku_key(tok))
    if cand & key_set:
        return True
    if cand & val_set:
        return True
    return False


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

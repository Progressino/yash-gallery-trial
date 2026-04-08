"""
SKU Mapping loader — multi-sheet Excel (Amazon, Flipkart, Myntra, Meesho, Snapdeal).
Also loads bundled master from backend/data/yash_sku_mapping_master.json (fast) or .xlsx.
After editing the xlsx, run: python scripts/regenerate_bundled_sku_map.py
"""
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_BUNDLED_SKU_MAP_CACHE: Optional[Dict[str, str]] = None


def bundled_sku_mapping_json_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "yash_sku_mapping_master.json"


def bundled_sku_mapping_xlsx_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "yash_sku_mapping_master.xlsx"


def load_bundled_sku_mapping() -> Dict[str, str]:
    """Load repo-shipped master map (JSON preferred for cold start). Cached in-process."""
    global _BUNDLED_SKU_MAP_CACHE
    if _BUNDLED_SKU_MAP_CACHE is not None:
        return _BUNDLED_SKU_MAP_CACHE
    if os.environ.get("SKIP_BUNDLED_SKU_MAPPING", "").strip() in ("1", "true", "yes"):
        _BUNDLED_SKU_MAP_CACHE = {}
        return _BUNDLED_SKU_MAP_CACHE
    pj = bundled_sku_mapping_json_path()
    px = bundled_sku_mapping_xlsx_path()
    try:
        if pj.is_file():
            _BUNDLED_SKU_MAP_CACHE = json.loads(pj.read_text(encoding="utf-8"))
        elif px.is_file():
            _BUNDLED_SKU_MAP_CACHE = parse_sku_mapping(px.read_bytes())
        else:
            _BUNDLED_SKU_MAP_CACHE = {}
    except Exception:
        _BUNDLED_SKU_MAP_CACHE = {}
    return _BUNDLED_SKU_MAP_CACHE


def clear_bundled_sku_mapping_cache() -> None:
    """Tests / hot-reload."""
    global _BUNDLED_SKU_MAP_CACHE
    _BUNDLED_SKU_MAP_CACHE = None


def ensure_default_sku_mapping_from_bundle(sess) -> None:
    """
    If the session has no SKU map, load the bundled Yash master sheet.
    Skipped when pause_auto_data_restore (explicit wipe) is on.
    """
    if getattr(sess, "pause_auto_data_restore", False):
        return
    if sess.sku_mapping:
        return
    b = load_bundled_sku_mapping()
    if b:
        sess.sku_mapping = b


def _clean(sku) -> str:
    if pd.isna(sku):
        return ""
    return str(sku).strip().replace('"""', "").replace("SKU:", "").strip().upper()


def _excel_lookup_keys_from_cell(raw) -> List[str]:
    """Keys stored in sku_mapping for a spreadsheet cell (88022920 vs 88022920.0)."""
    out: List[str] = []
    if raw is None:
        return out
    if isinstance(raw, float) and math.isnan(raw):
        return out
    s = _clean(raw)
    if s and s.upper() not in ("NAN", "NONE") and s not in ("STYLE ID", "YRN NUMBER", "YRN", "DATE"):
        out.append(s)
    try:
        f = float(str(raw).replace(",", "").strip())
        if math.isfinite(f) and f == int(f) and abs(f) < 1e16:
            ik = str(int(f))
            if ik not in out:
                out.append(ik)
    except ValueError:
        pass
    return out


def _is_oms_column(name: str) -> bool:
    s = str(name).lower().strip()
    if s in ("omssku", "oms sku", "oms sku code", "oms_sku"):
        return True
    if "oms" in s and ("sku" in s or "code" in s):
        return True
    return False


def _is_replace_sku_column(name: str) -> bool:
    """Workbook column for 'Replace SKU' masters — order/listing SKU to translate → OMS."""
    s = str(name).lower().strip()
    if _is_oms_column(name):
        return False
    return "replace" in s and "sku" in s


def _sheet_named_replace_sku(sheet_name: str) -> bool:
    sl = str(sheet_name).lower().replace("_", " ")
    return "replace" in sl and "sku" in sl


def _is_seller_column(name: str) -> bool:
    s = str(name).lower().strip()
    if s in ("date", "dt", "day"):
        return False
    if _is_oms_column(name):
        return False
    if "style" in s and "id" in s:
        return False
    if "yrn" in s:
        return False
    if "brand" in s and "sku" not in s:
        return False
    # Marketplaces + Meesho family companies (Yash Gallery, Akiko, Ashirwad/Ashirward, Pushpa, mall, etc.)
    keys = (
        "seller", "myntra", "meesho", "messho", "mesho", "snapdeal",
        "flipkart", "fsn", "merchant", "listing", "article",
        "pushpa", "garments", "mall", "akiko", "ashir", "yash",
        "supplier", "catalog",
    )
    if any(k in s for k in keys) and "sku" in s:
        return True
    if "sku id" in s or s.endswith("sku code"):
        return True
    if _is_replace_sku_column(name):
        return True
    return False


def _sheet_needs_meesho_style_fallback(sheet_name: str) -> bool:
    sl = str(sheet_name).lower()
    return any(
        x in sl
        for x in ("meesho", "messho", "mesho", "ashir", "akiko", "pushpa", "garments", "mall")
    ) and "flipkart" not in sl and "amazon" not in sl


def _sheet_needs_loose_column_fallback(sheet_name: str) -> bool:
    """Meesho-family tabs or 'Replace SKU' tabs with minimal headers."""
    return _sheet_needs_meesho_style_fallback(sheet_name) or _sheet_named_replace_sku(sheet_name)


def _pick_seller_oms_columns(
    df: pd.DataFrame, sheet_name: str = ""
) -> tuple[Optional[object], Optional[object]]:
    cols = list(df.columns)
    oms_candidates = [c for c in cols if _is_oms_column(str(c))]
    seller_candidates = [c for c in cols if _is_seller_column(str(c))]

    oms_col = oms_candidates[-1] if oms_candidates else None
    seller_col: Optional[object] = None
    # Replace SKU column = primary listing/order token (Meesho combined SKU, etc.).
    replace_col = next((c for c in cols if _is_replace_sku_column(str(c))), None)
    if replace_col is not None:
        seller_col = replace_col
    # Meesho family sheets: "Meesho SKU" when no Replace SKU column.
    elif _sheet_needs_meesho_style_fallback(sheet_name):
        seller_col = next(
            (
                c
                for c in cols
                if "meesho" in str(c).lower()
                and "sku" in str(c).lower()
                and not _is_oms_column(str(c))
            ),
            None,
        )
    if seller_col is None:
        seller_col = seller_candidates[0] if seller_candidates else None

    data_cols = [c for c in cols if str(c).lower().strip() not in ("date", "dt", "day", "brand")
                        and not (str(c).lower() == "date")]

    if seller_col is None and len(data_cols) >= 2:
        seller_col = data_cols[0]
    if oms_col is None and len(data_cols) >= 2:
        oms_col = data_cols[-1]
    elif oms_col is None and len(cols) > 1:
        oms_col = cols[-1]
    if seller_col is None and len(cols) > 1:
        seller_col = cols[1]

    if seller_col is not None and oms_col is not None and seller_col == oms_col and len(data_cols) >= 2:
        seller_col = data_cols[0]
        oms_col = data_cols[-1]

    return seller_col, oms_col


def parse_sku_mapping(file_bytes: bytes) -> Dict[str, str]:
    """
    Parse a multi-sheet Excel SKU mapping file.
    Returns {seller_sku_upper → oms_sku} with extra keys for STYLE ID and YRN (Myntra).

    Supports **Replace SKU** columns/sheets: listing/order SKU (after Meesho SKU+size combine)
    maps to OMS. **Meesho SKU** / **Myntra SKU code** columns on the same row are also
    registered as keys. **YRN** column keys match PPMP Myntra SKU code → OMS.
    """
    mapping: Dict[str, str] = {}
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    for _sheet_name in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=_sheet_name)
        if df.empty or len(df.columns) < 2:
            continue

        style_col = next(
            (c for c in df.columns if "style" in str(c).lower() and "id" in str(c).lower()),
            None,
        )
        yrn_col = next((c for c in df.columns if "yrn" in str(c).lower()), None)

        seller_col, oms_col = _pick_seller_oms_columns(df, _sheet_name)

        if (seller_col is None or oms_col is None) and _sheet_needs_loose_column_fallback(_sheet_name):
            data_cols = [
                c for c in df.columns
                if str(c).lower().strip() not in ("date", "brand", "dt")
                and "date" not in str(c).lower()
            ]
            if len(data_cols) >= 2:
                seller_col, oms_col = data_cols[0], data_cols[-1]

        if seller_col is None or oms_col is None:
            continue

        meesho_sheet = (
            _sheet_needs_meesho_style_fallback(_sheet_name)
            or _sheet_named_replace_sku(_sheet_name)
        )

        skip_extra = {c for c in (seller_col, oms_col, style_col, yrn_col) if c is not None}
        extra_key_cols: List[object] = []
        _seen_x = set()
        for c in df.columns:
            if c in skip_extra or c in _seen_x:
                continue
            cl = str(c).lower()
            if _is_oms_column(str(c)):
                continue
            if ("replace" in cl and "sku" in cl) or (
                "myntra" in cl and "sku" in cl and "oms" not in cl
            ) or ("meesho" in cl and "sku" in cl):
                extra_key_cols.append(c)
                _seen_x.add(c)

        def _put_row_keys(raw_cell, o_val: str) -> None:
            if not o_val:
                return
            for k in _excel_lookup_keys_from_cell(raw_cell):
                if not k:
                    continue
                mapping[k] = o_val
                if meesho_sheet and " " in k:
                    mapping[re.sub(r"\s+", "-", k.strip())] = o_val

        for _, row in df.iterrows():
            s = _clean(row.get(seller_col, ""))
            o = _clean(row.get(oms_col, ""))
            if o in ("", "NAN", "OMS SKU", "SELLER-SKU"):
                continue
            if s and s not in ("NAN", "OMS SKU", "SELLER-SKU", "SELLER SKU", "DATE"):
                _put_row_keys(row.get(seller_col, ""), o)
            for ec in extra_key_cols:
                _put_row_keys(row.get(ec, ""), o)
            if style_col:
                for sid in _excel_lookup_keys_from_cell(row.get(style_col, "")):
                    if sid and o:
                        mapping.setdefault(sid, o)
            # YRN ↔ Myntra SKU code from PPMP; overwrite so YRN wins on conflicts.
            if yrn_col:
                for yid in _excel_lookup_keys_from_cell(row.get(yrn_col, "")):
                    if yid and o:
                        mapping[yid] = o

    return mapping

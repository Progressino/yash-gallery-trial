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


def sku_mapping_disk_path() -> Path:
    """Server-wide SKU map on the warm-cache volume (survives container restarts)."""
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    return Path(base) / "sku_mapping.json"


def load_sku_mapping_from_disk() -> Dict[str, str]:
    p = sku_mapping_disk_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merge_sku_mapping_upload(
    existing: Optional[Dict[str, str]],
    parsed: Dict[str, str],
) -> Dict[str, str]:
    """Overlay uploaded rows onto the current master map (uploaded keys win)."""
    merged = dict(existing or {})
    merged.update(parsed or {})
    return merged


def merge_sku_mapping_layers(
    *layers: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Merge mapping dicts left→right (later layers win on key conflicts)."""
    out: Dict[str, str] = {}
    for layer in layers:
        if isinstance(layer, dict) and layer:
            out.update(layer)
    return out


def resolve_sku_mapping_base(sess) -> Dict[str, str]:
    """Best-effort master map before merging a supplemental upload.

    Always starts from the bundled master so warehouse typos / marketplace
    aliases that ship in-repo are never dropped when an older disk map exists.
    """
    bundled = load_bundled_sku_mapping()
    disk = load_sku_mapping_from_disk()
    cur = getattr(sess, "sku_mapping", None) or {}
    return merge_sku_mapping_layers(bundled, disk, cur)


def ensure_sku_mapping_merged_globally(sess=None) -> Dict[str, str]:
    """Merge bundled + disk (+ session) and persist so every session sees full map."""
    bundled = load_bundled_sku_mapping()
    disk = load_sku_mapping_from_disk()
    cur = getattr(sess, "sku_mapping", None) if sess is not None else None
    merged = merge_sku_mapping_layers(bundled, disk, cur if isinstance(cur, dict) else None)
    if merged:
        persist_sku_mapping_globally(merged)
        if sess is not None:
            sess.sku_mapping = dict(merged)
    return merged


def persist_sku_mapping_globally(mapping: Dict[str, str]) -> None:
    """Write SKU map to disk warm-cache and in-memory warm cache (shared across sessions)."""
    if not mapping:
        return
    snap = dict(mapping)
    try:
        p = sku_mapping_disk_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    try:
        import backend.main as _main

        if not isinstance(_main._warm_cache, dict):
            _main._warm_cache = {}
        _main._warm_cache["sku_mapping"] = snap
    except Exception:
        pass


def restore_sku_mapping_to_session(sess) -> bool:
    """
    Fill / refresh session SKU map from bundled master + disk/warm overlays.
    Returns True if a map was applied.
    """
    paused = bool(getattr(sess, "pause_auto_data_restore", False))

    warm: Dict[str, str] = {}
    try:
        import backend.main as _main

        wc = _main._warm_cache.get("sku_mapping") if isinstance(_main._warm_cache, dict) else None
        if isinstance(wc, dict) and wc:
            warm = dict(wc)
    except Exception:
        pass

    disk = load_sku_mapping_from_disk()
    gh: Dict[str, str] = {}
    if not paused and not disk and not warm:
        try:
            from .github_cache import load_sku_mapping_from_drive

            loaded = load_sku_mapping_from_drive()
            if loaded:
                gh = dict(loaded)
        except Exception:
            pass

    bundled = load_bundled_sku_mapping()
    existing = getattr(sess, "sku_mapping", None) or {}
    # Bundled is the base; disk/warm/github/session overlays win so operator uploads stick.
    merged = merge_sku_mapping_layers(bundled, gh, disk, warm, existing)
    if not merged:
        return False
    if existing and merged == existing:
        return False
    sess.sku_mapping = merged
    # Keep disk warm-cache complete so PO/hydrate always see typo aliases.
    if len(merged) > len(disk or {}):
        persist_sku_mapping_globally(merged)
    return True


def ensure_default_sku_mapping_from_bundle(sess) -> None:
    """Backward-compatible entry: restore from any server source, then bundled master."""
    restore_sku_mapping_to_session(sess)


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


def _embedded_numeric_keys_from_cell(raw) -> List[str]:
    """
    Pull 6+ digit runs from cleaned text and from Excel-style normalized keys (int form of
    scientific / float). Covers YARYKASS100506552 → 100506552 and 1.00506552E+8 strings.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    texts: List[str] = []
    t = _clean(raw)
    if t and t not in ("STYLE ID", "YRN NUMBER", "YRN", "DATE"):
        texts.append(t)
    for k in _excel_lookup_keys_from_cell(raw):
        if k and k not in texts:
            texts.append(k)
    seen: set[str] = set()
    out: List[str] = []
    for b in texts:
        for m in re.finditer(r"\d{6,}", b):
            g = m.group(0)
            if g not in seen:
                seen.add(g)
                out.append(g)
    return out


def _sheet_is_myntra_tab(sheet_name: str) -> bool:
    return "myntra" in str(sheet_name).lower()


def _is_oms_column(name: str) -> bool:
    s = str(name).lower().strip()
    if s in (
        "omssku",
        "oms sku",
        "oms sku code",
        "oms_sku",
        "oms",
        "oms code",
        "internal oms",
        "master oms",
    ):
        return True
    if "oms" in s and ("sku" in s or "code" in s):
        return True
    return False


def _is_right_sku_target_column(name: str) -> bool:
    """Target/canonical SKU column on supplemental replace sheets (typo 'Righ Sku' included)."""
    s = str(name).lower().strip()
    if _is_oms_column(name):
        return False
    return ("right" in s or "righ" in s) and "sku" in s


def _ordered_oms_columns(df: pd.DataFrame, primary: Optional[object]) -> List[object]:
    """Try preferred target column first, then every other OMS-like column left-to-right."""
    all_oms = [c for c in df.columns if _is_oms_column(str(c))]
    if primary is not None:
        rest = [c for c in all_oms if c != primary]
        if primary in all_oms:
            return [primary] + rest
        # Supplemental sheets use "Right/Righ Sku" as the canonical target, not OMS-named cols.
        return [primary] + rest
    return all_oms


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
    # Supplemental replace workbook: old OMS_SKU → Right/Righ Sku (canonical).
    right_col = next((c for c in cols if _is_right_sku_target_column(str(c))), None)
    if right_col is not None and oms_candidates:
        return oms_candidates[0], right_col
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

    OMS columns are forward-filled (Excel merged cells). Per row, every OMS-like
    column is scanned so YRN still maps when the primary OMS cell is empty but
    another OMS column on the same row is filled.
    """
    mapping: Dict[str, str] = {}
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    for _sheet_name in xls.sheet_names:
        # Reuse open workbook — re-reading file_bytes per sheet is very slow on large maps.
        df = pd.read_excel(xls, sheet_name=_sheet_name)
        if df.empty or len(df.columns) < 2:
            continue

        # Combo / DPT BOM sheets (one listing → many OMS components) must not be
        # collapsed into last-wins 1:1 mapping — handled by combo_sku_map.
        try:
            from .combo_sku_map import sheet_looks_like_combo_bom

            _norms = [str(c).strip().lower() for c in df.columns]
            _combo_named = any(
                n in ("dpt sku", "combo sku", "dpt_sku", "combo_sku")
                or n.startswith("dpt ")
                or (n.startswith("combo") and "sku" in n and "stock" not in n)
                for n in _norms
            )
            if _combo_named or sheet_looks_like_combo_bom(df):
                continue
        except Exception:
            pass

        # Excel merged cells often leave OMS blank on continuation rows — forward-fill OMS columns only.
        _oms_like = [c for c in df.columns if _is_oms_column(str(c))]
        if _oms_like:
            df = df.copy()
            df[_oms_like] = df[_oms_like].ffill()

        style_col = next(
            (c for c in df.columns if "style" in str(c).lower() and "id" in str(c).lower()),
            None,
        )
        yrn_col = next((c for c in df.columns if "yrn" in str(c).lower()), None)

        # Numeric tails from seller / extra columns (PPMP YRN-style ids) — Myntra tab or any sheet with YRN column
        embed_myntra_numerics = _sheet_is_myntra_tab(_sheet_name) or (yrn_col is not None)

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

        oms_col_order = _ordered_oms_columns(df, oms_col)

        def _row_oms(row: pd.Series) -> str:
            for oc in oms_col_order:
                if oc is None:
                    continue
                v = _clean(row.get(oc, ""))
                if v and v not in ("NAN", "OMS SKU", "SELLER-SKU"):
                    return v
            return ""

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

        for row in df.to_dict(orient="records"):
            s = _clean(row.get(seller_col, ""))
            o = _row_oms(row)
            if o in ("", "NAN", "OMS SKU", "SELLER-SKU"):
                continue
            if s and s not in ("NAN", "OMS SKU", "SELLER-SKU", "SELLER SKU", "DATE"):
                _put_row_keys(row.get(seller_col, ""), o)
            for ec in extra_key_cols:
                _put_row_keys(row.get(ec, ""), o)
            if style_col:
                raw_st = row.get(style_col, "")
                for sid in _excel_lookup_keys_from_cell(raw_st):
                    if sid and o:
                        mapping.setdefault(sid, o)
                # Myntra STYLE ID / catalog id often matches PPMP numeric-only columns.
                if embed_myntra_numerics:
                    for nid in _embedded_numeric_keys_from_cell(raw_st):
                        if nid and o:
                            mapping[nid] = o
            # YRN ↔ Myntra SKU code from PPMP; overwrite so YRN wins on conflicts.
            if yrn_col:
                raw_y = row.get(yrn_col, "")
                for yid in _excel_lookup_keys_from_cell(raw_y):
                    if yid and o:
                        mapping[yid] = o
                for yid in _embedded_numeric_keys_from_cell(raw_y):
                    if yid and o:
                        mapping[yid] = o
            # MYNTRA SKU CODE / primary seller col may be plain numeric or PREFIX+digits.
            if embed_myntra_numerics:
                for nid in _embedded_numeric_keys_from_cell(row.get(seller_col, "")):
                    if nid and o:
                        mapping[nid] = o
                for ec in extra_key_cols:
                    for nid in _embedded_numeric_keys_from_cell(row.get(ec, "")):
                        if nid and o:
                            mapping[nid] = o

    return mapping

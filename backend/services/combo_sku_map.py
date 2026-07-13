"""
Combo / DPT SKU BOM map — one listing SKU → one or more OMS component SKUs.

Example (Combo Sku Map.xlsx):
  1003DPT21MULTI-3XL → 1003YKMUSTARD-3XL + DPT21MULTI
  1003YK5027DPT21-L  → 1003YKMUSTARD-L + 5027YKMULTI-L + DPT21MULTI

Used so PO / quarterly demand attributes each combo sale to the inventory
components that actually stock out (kurta sizes + dupatta styles).
"""
from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .helpers import clean_sku, collapse_duplicate_trailing_size_suffix

# combo_key → [(component_oms_sku, qty_per_combo), ...]
ComboBom = Dict[str, List[Tuple[str, float]]]

_BUNDLED_COMBO_CACHE: Optional[ComboBom] = None

_COMBO_KEY_HINTS = (
    "dpt sku",
    "dpt_sku",
    "combo sku",
    "combo_sku",
    "combo listing",
    "combo listing sku",
)
_COMPONENT_HINTS = (
    "component",
    "oms sku",
    "oms_sku",
    "right sku",
    "sku",
)


def bundled_combo_sku_map_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "combo_sku_map.json"


def combo_sku_map_disk_path() -> Path:
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    return Path(base) / "combo_sku_map.json"


def clear_bundled_combo_sku_map_cache() -> None:
    global _BUNDLED_COMBO_CACHE
    _BUNDLED_COMBO_CACHE = None


def _norm_key(raw) -> str:
    return collapse_duplicate_trailing_size_suffix(clean_sku(raw))


def _col_norm(name) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def sheet_looks_like_combo_bom(df: pd.DataFrame) -> bool:
    """True when a sheet is a combo BOM (DPT/Combo → component Sku), not 1:1 master."""
    if df is None or df.empty or len(df.columns) < 2:
        return False
    norms = [_col_norm(c) for c in df.columns]
    has_combo_key = any(
        n in _COMBO_KEY_HINTS or n.startswith("dpt") or "combo" in n
        for n in norms
    )
    if not has_combo_key:
        # Heuristic: same left key maps to 2+ distinct right values → BOM sheet.
        left, right = df.columns[0], df.columns[1]
        g = (
            df[[left, right]]
            .dropna()
            .assign(
                _l=lambda d: d[left].map(_norm_key),
                _r=lambda d: d[right].map(_norm_key),
            )
        )
        g = g[(g["_l"] != "") & (g["_r"] != "")]
        if g.empty:
            return False
        return bool(g.groupby("_l")["_r"].nunique().gt(1).any())
    return True


def _pick_combo_columns(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Return (combo_col, component_col, qty_col)."""
    combo_col = None
    component_col = None
    qty_col = None
    for c in df.columns:
        n = _col_norm(c)
        if combo_col is None and (
            n in ("dpt sku", "dpt_sku", "combo sku", "combo_sku", "combo listing sku")
            or n.startswith("dpt ")
            or (n.startswith("combo") and "sku" in n and "qty" not in n and "stock" not in n)
        ):
            combo_col = c
        if n in ("qty", "quantity", "qty per", "component qty", "combo qty") and "stock" not in n:
            qty_col = c
    for c in df.columns:
        if c == combo_col:
            continue
        n = _col_norm(c)
        if n in ("sku", "component sku", "component", "oms sku", "oms_sku", "right sku"):
            component_col = c
            break
    if combo_col is None and len(df.columns) >= 2:
        combo_col = df.columns[0]
    if component_col is None and len(df.columns) >= 2:
        # Prefer a column literally named Sku over others.
        for c in df.columns:
            if c == combo_col:
                continue
            if _col_norm(c) == "sku":
                component_col = c
                break
        if component_col is None:
            for c in df.columns:
                if c != combo_col and c != qty_col:
                    component_col = c
                    break
    return combo_col, component_col, qty_col


def parse_combo_sku_map(file_bytes: bytes) -> ComboBom:
    """
    Parse Combo Sku Map Excel → {COMBO_SKU: [(COMPONENT_OMS, qty), ...]}.

    Accepts columns like ``DPT Sku`` + ``Sku`` (qty defaults to 1).
    Duplicate (combo, component) rows sum qty.
    """
    out: Dict[str, Dict[str, float]] = {}
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    for sheet in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
        if not sheet_looks_like_combo_bom(df):
            continue
        combo_col, component_col, qty_col = _pick_combo_columns(df)
        if combo_col is None or component_col is None:
            continue
        for _, row in df.iterrows():
            ck = _norm_key(row.get(combo_col, ""))
            comp = _norm_key(row.get(component_col, ""))
            if not ck or not comp or ck in ("NAN", "DPT SKU", "COMBO SKU"):
                continue
            if comp in ("NAN", "SKU", "OMS SKU"):
                continue
            qty = 1.0
            if qty_col is not None:
                try:
                    qty = float(pd.to_numeric(row.get(qty_col), errors="coerce") or 1.0)
                except Exception:
                    qty = 1.0
            if qty <= 0:
                continue
            bucket = out.setdefault(ck, {})
            bucket[comp] = bucket.get(comp, 0.0) + qty
    return {k: sorted(((c, q) for c, q in comps.items()), key=lambda t: t[0]) for k, comps in out.items()}


def merge_combo_sku_map(existing: Optional[ComboBom], parsed: ComboBom) -> ComboBom:
    """Overlay uploaded combo keys (uploaded keys fully replace prior component lists)."""
    merged: ComboBom = {k: list(v) for k, v in (existing or {}).items()}
    for k, comps in (parsed or {}).items():
        merged[k] = list(comps)
    return merged


def combo_bom_to_jsonable(bom: ComboBom) -> Dict[str, List[List[object]]]:
    return {k: [[c, q] for c, q in comps] for k, comps in (bom or {}).items()}


def combo_bom_from_jsonable(data) -> ComboBom:
    if not isinstance(data, dict):
        return {}
    out: ComboBom = {}
    for k, comps in data.items():
        ck = _norm_key(k)
        if not ck or not isinstance(comps, list):
            continue
        parsed: List[Tuple[str, float]] = []
        for item in comps:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                c = _norm_key(item[0])
                q = float(item[1]) if len(item) > 1 else 1.0
            elif isinstance(item, dict):
                c = _norm_key(item.get("oms_sku") or item.get("sku") or "")
                q = float(item.get("qty") or 1.0)
            elif isinstance(item, str):
                c, q = _norm_key(item), 1.0
            else:
                continue
            if c and q > 0:
                parsed.append((c, q))
        if parsed:
            # collapse dupes
            acc: Dict[str, float] = {}
            for c, q in parsed:
                acc[c] = acc.get(c, 0.0) + q
            out[ck] = sorted(acc.items(), key=lambda t: t[0])
    return out


def load_bundled_combo_sku_map() -> ComboBom:
    global _BUNDLED_COMBO_CACHE
    if _BUNDLED_COMBO_CACHE is not None:
        return _BUNDLED_COMBO_CACHE
    p = bundled_combo_sku_map_path()
    if not p.is_file():
        _BUNDLED_COMBO_CACHE = {}
        return _BUNDLED_COMBO_CACHE
    try:
        _BUNDLED_COMBO_CACHE = combo_bom_from_jsonable(
            json.loads(p.read_text(encoding="utf-8"))
        )
    except Exception:
        _BUNDLED_COMBO_CACHE = {}
    return _BUNDLED_COMBO_CACHE


def load_combo_sku_map_from_disk() -> ComboBom:
    p = combo_sku_map_disk_path()
    if not p.is_file():
        return {}
    try:
        return combo_bom_from_jsonable(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return {}


def persist_combo_sku_map_globally(bom: ComboBom) -> None:
    if not bom:
        return
    snap = combo_bom_to_jsonable(bom)
    try:
        p = combo_sku_map_disk_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    try:
        import backend.main as _main

        if not isinstance(_main._warm_cache, dict):
            _main._warm_cache = {}
        _main._warm_cache["combo_sku_map"] = bom
    except Exception:
        pass


def restore_combo_sku_map_to_session(sess) -> bool:
    """Fill empty session combo map from warm → disk → bundled."""
    if getattr(sess, "combo_sku_map", None):
        return False
    try:
        import backend.main as _main

        warm = (_main._warm_cache or {}).get("combo_sku_map")
        if isinstance(warm, dict) and warm:
            sess.combo_sku_map = warm if _is_bom(warm) else combo_bom_from_jsonable(warm)
            if sess.combo_sku_map:
                return True
    except Exception:
        pass
    disk = load_combo_sku_map_from_disk()
    if disk:
        sess.combo_sku_map = disk
        return True
    bundled = load_bundled_combo_sku_map()
    if bundled:
        sess.combo_sku_map = bundled
        return True
    return False


def _is_bom(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    sample = next(iter(obj.values()))
    return isinstance(sample, list) and (
        not sample or isinstance(sample[0], tuple) or (
            isinstance(sample[0], (list, tuple)) and len(sample[0]) >= 1
        )
    )


def resolve_active_combo_sku_map(
    explicit: Optional[ComboBom] = None,
    sess=None,
) -> ComboBom:
    if explicit:
        return explicit if _is_bom(explicit) else combo_bom_from_jsonable(explicit)
    if sess is not None:
        cur = getattr(sess, "combo_sku_map", None)
        if cur:
            return cur if _is_bom(cur) else combo_bom_from_jsonable(cur)
    try:
        import backend.main as _main

        warm = (_main._warm_cache or {}).get("combo_sku_map")
        if isinstance(warm, dict) and warm:
            return warm if _is_bom(warm) else combo_bom_from_jsonable(warm)
    except Exception:
        pass
    disk = load_combo_sku_map_from_disk()
    if disk:
        return disk
    return load_bundled_combo_sku_map()


def lookup_combo_components(token: str, combo_map: ComboBom) -> Optional[List[Tuple[str, float]]]:
    if not combo_map or not token:
        return None
    k = _norm_key(token)
    if not k:
        return None
    comps = combo_map.get(k)
    if comps:
        return list(comps)
    return None


def resolve_demand_components(
    raw,
    sku_mapping: Optional[Dict[str, str]],
    combo_map: Optional[ComboBom],
    *,
    strip_pl: bool = False,
    attribute_combo_to_listing_only: bool = False,
) -> List[Tuple[str, float]]:
    """
    Map one listing/seller SKU to demand targets for PO nets.

    Prefers combo BOM on the raw listing key (before 1:1 mapping can drop
    sibling components). Falls back to canonical OMS key.

    When ``attribute_combo_to_listing_only`` is True (quarterly File-matching),
    combo listing keys stay on the listing identity and are NOT collapsed via
    the 1:1 master map onto a component OMS SKU.
    """
    from .po_engine import _PL_RE, _strip_pl, canonical_oms_key

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    stripped = str(raw).strip()
    if not stripped or stripped.lower() in ("nan", "none", "nat"):
        return []

    bom = combo_map or {}
    candidates = [_norm_key(stripped)]
    pl_stripped = _PL_RE.sub(r"\1\2", stripped.upper())
    if pl_stripped and pl_stripped not in candidates:
        candidates.append(_norm_key(pl_stripped))

    for cand in candidates:
        comps = lookup_combo_components(cand, bom)
        if comps:
            if attribute_combo_to_listing_only:
                listing = cand or _norm_key(stripped) or stripped
                return [(listing, 1.0)] if listing else []
            return [(c, float(q)) for c, q in comps if c]

    if strip_pl:
        if sku_mapping:
            mapped = _strip_pl(stripped, sku_mapping)
        else:
            mapped = pl_stripped
        mapped = _norm_key(mapped)
    else:
        mapped = canonical_oms_key(stripped, sku_mapping)

    if mapped:
        # Combo stubs in the master map (listing → first component) must not
        # collapse listing sales onto components for File-matching history.
        if attribute_combo_to_listing_only:
            mapped_comps = lookup_combo_components(mapped, bom)
            raw_is_listing = any(lookup_combo_components(c, bom) for c in candidates)
            if raw_is_listing:
                listing = candidates[0] or stripped
                return [(listing, 1.0)]
            # If 1:1 map pointed at a combo listing key, keep that listing.
            if mapped_comps:
                return [(mapped, 1.0)]
        comps = lookup_combo_components(mapped, bom)
        if comps:
            if attribute_combo_to_listing_only:
                return [(mapped, 1.0)]
            return [(c, float(q)) for c, q in comps if c]
        return [(mapped, 1.0)]
    return []


def build_sku_explode_map(
    unique_skus: Iterable,
    sku_mapping: Optional[Dict[str, str]],
    combo_map: Optional[ComboBom],
    *,
    strip_pl: bool = False,
    retain_combo_listings: bool = False,
    attribute_combo_to_listing_only: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    bom = combo_map or {}
    for s in unique_skus:
        key = s if s is None or isinstance(s, str) else str(s)
        comps = resolve_demand_components(
            key,
            sku_mapping,
            bom,
            strip_pl=strip_pl,
            attribute_combo_to_listing_only=attribute_combo_to_listing_only,
        )
        if not comps:
            out[key] = []
            continue
        if (
            retain_combo_listings
            and not attribute_combo_to_listing_only
            and lookup_combo_components(key, bom)
        ):
            # Keep the listing row (sales visibility) and still fan demand to components.
            listing = _norm_key(key) or str(key).strip()
            retained: List[Tuple[str, float]] = []
            if listing:
                retained.append((listing, 1.0))
            for c, q in comps:
                if c and c != listing:
                    retained.append((c, float(q)))
            out[key] = retained if retained else comps
        else:
            out[key] = comps if comps else []
    return out


def explode_sku_qty_dataframe(
    df: pd.DataFrame,
    *,
    sku_col: str,
    qty_col: str,
    sku_mapping: Optional[Dict[str, str]] = None,
    combo_map: Optional[ComboBom] = None,
    strip_pl: bool = False,
    retain_combo_listings: bool = False,
    attribute_combo_to_listing_only: bool = False,
) -> pd.DataFrame:
    """
    Fan listing SKUs out to combo components (qty × component multiplier).

    When ``retain_combo_listings`` is True, combo keys also keep a listing row at
    the original qty (for PO/quarterly visibility) while components still receive
    exploded demand for stocking.

    When ``attribute_combo_to_listing_only`` is True, combo listing keys keep their
    own identity (no component fan, no 1:1 collapse onto a component OMS) — used
    for quarterly history that must match File/Deepdive SKU sales.

    When combo_map is empty, still applies 1:1 canonical / strip_pl resolution
    so callers can use this as a single canonicalize+explode step.
    """
    if df is None or df.empty or sku_col not in df.columns:
        return df
    work = df
    raw = work[sku_col]
    uniq = pd.unique(raw.to_numpy())
    explode_map = build_sku_explode_map(
        uniq,
        sku_mapping,
        combo_map or {},
        strip_pl=strip_pl,
        retain_combo_listings=retain_combo_listings,
        attribute_combo_to_listing_only=attribute_combo_to_listing_only,
    )

    # Drop rows that resolve to nothing.
    lengths = raw.map(lambda s: len(explode_map.get(s, []))).astype(int)
    if (lengths == 0).all():
        return work.iloc[0:0].copy()

    multi = bool((lengths > 1).any())
    scaled = False
    if not multi:
        # Single target each — remap in place; scale qty if multiplier ≠ 1.
        out = work.loc[lengths > 0].copy()
        src = out[sku_col]
        out[sku_col] = src.map(lambda s: explode_map[s][0][0])
        mults = src.map(lambda s: float(explode_map[s][0][1]))
        if qty_col in out.columns and not (mults == 1.0).all():
            out[qty_col] = pd.to_numeric(out[qty_col], errors="coerce").fillna(0) * mults
            scaled = True
        if "Units_Effective" in out.columns and not (mults == 1.0).all():
            out["Units_Effective"] = (
                pd.to_numeric(out["Units_Effective"], errors="coerce").fillna(0) * mults
            )
            scaled = True
        _ = scaled
        if retain_combo_listings:
            out["_Combo_Fan"] = False
        return out

    keep = work.loc[lengths > 0]
    keep_len = lengths.loc[lengths > 0].to_numpy()
    idx = np.repeat(keep.index.to_numpy(), keep_len)
    out = keep.loc[idx].reset_index(drop=True)
    new_skus: List[str] = []
    new_mults: List[float] = []
    is_fan: List[bool] = []
    for s, n in zip(keep[sku_col].tolist(), keep_len.tolist()):
        comps = explode_map[s]
        listing = _norm_key(s) or str(s).strip()
        for i in range(n):
            c, q = comps[i]
            new_skus.append(c)
            new_mults.append(float(q))
            # Retained listing row (first identity) is not fan demand; component
            # copies are. Used so quarterly can ignore combo-inflated components.
            is_fan.append(bool(retain_combo_listings and c != listing))
    out[sku_col] = new_skus
    mult_arr = np.asarray(new_mults, dtype=float)
    if qty_col in out.columns:
        out[qty_col] = pd.to_numeric(out[qty_col], errors="coerce").fillna(0) * mult_arr
    if "Units_Effective" in out.columns:
        out["Units_Effective"] = (
            pd.to_numeric(out["Units_Effective"], errors="coerce").fillna(0) * mult_arr
        )
    if retain_combo_listings:
        out["_Combo_Fan"] = is_fan
    return out


def combo_keys_as_identity_sku_mapping(bom: ComboBom) -> Dict[str, str]:
    """
    Optional 1:1 stubs so combo listing keys are recognized in the master map.

    Maps each combo key to its first sized component when available, else first
    component. Real demand still comes from explode — this only aids recognition.
    """
    out: Dict[str, str] = {}
    size_re = re.compile(
        r"-(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL|7XL|8XL)$", re.I
    )
    for k, comps in (bom or {}).items():
        if not comps:
            continue
        sized = next((c for c, _ in comps if size_re.search(c)), None)
        out[k] = sized or comps[0][0]
    return out

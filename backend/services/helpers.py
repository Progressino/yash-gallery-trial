"""
Shared helper functions used across all services.
Extracted 1-for-1 from app.py.
"""
import re
import io
import unicodedata
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


def normalize_id_token_for_mapping(value) -> str:
    """
    Normalise marketplace / Excel tokens before map lookup (YRN, Flipkart SKU ID).
    Handles '47061570.0' and scientific notation strings without changing alphabetic SKUs.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    t = str(value).strip()
    if not t or t.lower() in ("nan", "none"):
        return t
    try:
        if re.fullmatch(r"\d+\.0+", t):
            return str(int(float(t)))
    except ValueError:
        pass
    try:
        if re.match(r"^\d+(\.\d+)?[eE][+\-]?\d+$", t):
            f = float(t)
            if np.isfinite(f) and f == int(f) and abs(f) < 1e16:
                return str(int(f))
    except ValueError:
        pass
    return t


def clean_line_id_series(series: pd.Series) -> pd.Series:
    """
    Normalize marketplace id columns for row-level dedup (strip noise, Excel float tails).
    Used for LineKey / OrderId alignment across re-uploads and cache merges.
    """
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"nan": "", "none": "", "<na>": "", "NaT": ""})
    num = pd.to_numeric(s, errors="coerce")
    is_whole = num.notna() & np.isfinite(num) & (num == np.floor(num))
    s = s.mask(is_whole, num[is_whole].astype(np.int64).astype(str))
    return s


# "NON RTO DELIVERED" / "NON-RETURN" are forward milestones; naive ``"rto" in s`` / ``"return" in s`` false-positive.
_NON_RTO_FORWARD = re.compile(r"(?i)NON[-\s]?RTO|NON[-\s]?RETURN")


def is_non_rto_forward_milestone_status(status) -> bool:
    """
    True when a marketplace status negates RTO/return (successful forward delivery).
    Without this guard, substring checks mark the row as Refund; tier-3 dedup then
    drops the Shipment twin and gross delivered units disappear from dashboards.
    """
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return False
    return bool(_NON_RTO_FORWARD.search(str(status)))


# Phrases from return / credit / adjustment columns that are often mis-read as SKUs.
_NON_SKU_NOTE_PHRASE = re.compile(
    r"SIZE\s*CHANGE|BAAKI|SE\s+ADJUST|\bADJUST\b|\bEXCHANGE\b|REASON\s+FOR|"
    r"\bRTO\b|CREDIT\s+ENTRY|REPLACEMENT\s|DAMAG|COMPLAINT|RETURN\s+REASON",
    re.I,
)


def looks_like_seller_listing_sku(value) -> bool:
    """Heuristic: Meesho / supplier listing codes (digits+YK, compact alnum), not prose notes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    t = str(value).strip()
    if not t or t.lower() in ("nan", "none", "<na>", "nat"):
        return False
    if _NON_SKU_NOTE_PHRASE.search(t):
        return False
    if len(t) > 72:
        return False
    if re.search(r"\d{3,}\s*YK", t, re.I):
        return True
    cu = clean_sku(t)
    if re.match(r"^\d{4,}[A-Z]", cu):
        return True
    if re.match(r"^[A-Z0-9]{4,}(?:-[A-Z0-9]+)+$", cu):
        return True
    if 4 <= len(cu) <= 36 and re.search(r"\d", cu) and "  " not in t and t.count(" ") <= 3:
        return True
    return False


def is_likely_non_sku_notes_value(value) -> bool:
    """True for return/adjustment prose — should not appear as Sku / OMS in exports."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    t = unicodedata.normalize("NFKC", str(value).strip())
    if not t:
        return False
    if _NON_SKU_NOTE_PHRASE.search(t):
        return True
    if len(t) > 55 and t.count(" ") >= 3:
        return True
    return False


# Match Amazon PL listing spellings (1023PLYK* → 1023YK*) for map lookups.
_PL_YK = re.compile(r"^(\d+)PL(YK)", re.I)
_HYPHEN_SPACES = re.compile(r"\s*-\s*")
_DOT_BEFORE_HYPHEN = re.compile(r"\.(?=-)")


def canonical_pl_sku_key(sku: str) -> str:
    c = clean_sku(sku)
    if not c:
        return c
    return _PL_YK.sub(r"\1\2", c)


def _listing_hyphen_and_pla_variants(cleaned_upper: str) -> List[str]:
    """
    Extra shapes seen across marketplaces for the same listing:
    - Hyphen between numeric style id and YK block: 165-YK251MUSTRAD ↔ 165YK251MUSTRAD
    - Stray hyphen after Y: 165Y-K251… → 165YK251…
    - Amazon-style PLA… where YK was split: 165PLAK251… → 165PLYK251… → (PL strip) 165YK251…
    """
    out: List[str] = []
    if not cleaned_upper:
        return out
    cu = cleaned_upper
    m = re.match(r"^(\d+)-(YK[A-Z0-9\-]+)$", cu, re.I)
    if m:
        g2 = m.group(2).upper()
        glued = f"{m.group(1)}{g2}"
        if glued != cu:
            out.append(glued)
    m = re.match(r"^(\d+)(YK[A-Z0-9\-]+)$", cu, re.I)
    if m:
        g2 = m.group(2).upper()
        split = f"{m.group(1)}-{g2}"
        if split != cu:
            out.append(split)
    if re.match(r"^\d+Y-K[0-9A-Z]", cu, re.I):
        alt = re.sub(r"^(\d+)Y-K", r"\1YK", cu, count=1, flags=re.I).upper()
        if alt != cu:
            out.append(alt)
    m = re.match(r"^(\d+)PLA(YK[A-Z0-9\-]+)$", cu, re.I)
    if m:
        out.append(f"{m.group(1)}PL{m.group(2).upper()}")
    #165PLAK251… → 165PLYK251… (listing typo / panel quirk)
    m = re.match(r"^(\d+)PLAK(\d+[A-Z0-9\-]*)$", cu, re.I)
    if m:
        out.append(f"{m.group(1)}PLYK{m.group(2).upper()}")
    return out


def _glued_myntra_size_hyphen_variants(s: str) -> List[str]:
    """
    PPMP sometimes omits hyphens around size: 1061YKBLUE4XLBLUE → 1061YKBLUE-4XL-BLUE.
    1378YKMULTI3XLMULTI → 1378YKMULTI-3XL-MULTI and shortened 1378YKMULTI-3XL when color repeats.
    """
    c = clean_sku(s)
    if not c or "-" in c:
        return []
    m = re.match(r"^(\d+YK[A-Z]+)((?:XXXL|XXL|[2345]XL))([A-Z]+)$", c)
    if not m:
        return []
    full = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    out = [full]
    base, sz, tail = m.group(1), m.group(2), m.group(3)
    if base.endswith(tail):
        out.append(f"{base}-{sz}")
    return out


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
    if "MUSTRAD" in c:
        alt = c.replace("MUSTRAD", "MUSTARD")
        if alt != c:
            bases.append(alt)
    for gl in _glued_myntra_size_hyphen_variants(c):
        if gl not in bases:
            bases.append(gl)
    hy = _HYPHEN_SPACES.sub("-", c)
    if hy != c:
        bases.append(hy)
    dot = _DOT_BEFORE_HYPHEN.sub("", c)
    if dot != c:
        bases.append(dot)
    dot_hy = _DOT_BEFORE_HYPHEN.sub("", hy)
    if dot_hy not in bases:
        bases.append(dot_hy)
    i = 0
    while i < len(bases):
        for v in _listing_hyphen_and_pla_variants(bases[i]):
            if v not in bases:
                bases.append(v)
        i += 1
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
    raw = str(seller_sku).strip()
    root = normalize_id_token_for_mapping(raw) or raw
    c = clean_sku(root)
    if not c:
        return c
    for form in normalized_sku_forms_for_lookup(root):
        for tok in integer_token_variants(form):
            if tok in mapping:
                return mapping[tok]
            pl = canonical_pl_sku_key(tok)
            if pl in mapping:
                return mapping[pl]
    # Excel / CSV: style ids as 1.02E+08 — normalize to integer string for map keys
    try:
        f = float(str(root).replace(",", ""))
        if np.isfinite(f) and f == int(f) and abs(f) < 1e16:
            ik = str(int(f))
            for cand in (ik, canonical_pl_sku_key(ik)):
                if cand in mapping:
                    return mapping[cand]
    except (ValueError, OverflowError):
        pass
    ne = _cached_numeric_embed_index(mapping)
    if c.isdigit() and len(c) >= 6 and c in ne:
        return ne[c]
    # Myntra / catalog vendor SKUs: DSBYDRSS131185528, YARYKASS100672680 — match6+ digit style/YRN tail
    for m in re.finditer(r"\d{6,}", c):
        g = m.group(0)
        if g in ne:
            return ne[g]
    return c


def _tokens_one_master_cell(value) -> Set[str]:
    """All lookup tokens derived from one key or OMS cell (spacing, PL, ints)."""
    out: Set[str] = set()
    for form in normalized_sku_forms_for_lookup(value):
        for tok in integer_token_variants(form):
            out.add(tok)
            out.add(canonical_pl_sku_key(tok))
    return out


_NUMERIC_EMBED_CACHE: Dict[int, Dict[str, str]] = {}


def _build_numeric_embed_index(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Bare catalog ids (e.g. 100506552) often match a 6+ digit run inside YARY…100506552 or OMS text.
    Last mapping row wins on rare collisions.
    """
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        oms = clean_sku(v)
        if not oms:
            continue
        for tok in _tokens_one_master_cell(str(k)):
            for m in re.finditer(r"\d{6,}", tok):
                out[m.group(0)] = oms
        vv = clean_sku(str(v))
        if vv:
            for m in re.finditer(r"\d{6,}", vv):
                out[m.group(0)] = vv
    return out


def _cached_numeric_embed_index(mapping: Dict[str, str]) -> Dict[str, str]:
    i = id(mapping)
    if i not in _NUMERIC_EMBED_CACHE:
        _NUMERIC_EMBED_CACHE[i] = _build_numeric_embed_index(mapping)
        if len(_NUMERIC_EMBED_CACHE) > 64:
            _NUMERIC_EMBED_CACHE.clear()
    return _NUMERIC_EMBED_CACHE[i]


def mapping_lookup_sets(mapping: Dict[str, str]) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """
    Normalized map keys and OMS values (incl. PL↔YK for both — sales often canonicalise OMS to YK).
    Third return: numeric id → OMS for digits embedded in keys/values (Myntra YRN tails, style ids).
    Used to tell whether a sales/export token is covered by the master sheet.
    Integer YRNs include 100672680.0-style aliases so they match Excel/pandas floats.
    """
    key_set: Set[str] = set()
    for k in mapping.keys():
        key_set.update(_tokens_one_master_cell(k))
    val_set: Set[str] = set()
    for v in mapping.values():
        val_set.update(_tokens_one_master_cell(v))
    num_embed = _cached_numeric_embed_index(mapping)
    return key_set, val_set, num_embed


def sku_recognized_in_master(
    token: str,
    mapping: Dict[str, str],
    *,
    key_set: Optional[Set[str]] = None,
    val_set: Optional[Set[str]] = None,
    numeric_embed: Optional[Dict[str, str]] = None,
) -> bool:
    """True if token appears as a seller/marketplace key or as an OMS value in the master."""
    if not mapping:
        return True
    c = clean_sku(token)
    if not c:
        return True
    if key_set is None or val_set is None:
        key_set, val_set, numeric_embed = mapping_lookup_sets(mapping)
    elif numeric_embed is None:
        numeric_embed = _cached_numeric_embed_index(mapping)
    cand: Set[str] = set()
    for form in normalized_sku_forms_for_lookup(token):
        for tok in integer_token_variants(form):
            cand.add(tok)
            cand.add(canonical_pl_sku_key(tok))
    if cand & key_set:
        return True
    if cand & val_set:
        return True
    for tok in cand:
        if tok.isdigit() and len(tok) >= 6 and tok in numeric_embed:
            return True
    for m in re.finditer(r"\d{6,}", c):
        if m.group(0) in numeric_embed:
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
            # Two-part SKUs like ``AK-139`` or ``STYLE-1657``: trailing token is a numeric
            # style id, not a size — do not strip (``139``.isdigit() used to collapse to ``AK``).
            if len(parts) == 2 and last.isdigit() and len(last) >= 3:
                return s
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
        # Inventory uploads are often XLSX despite CSV-like naming.
        # Fallback to Excel so callers (inventory parsers) don't silently drop files.
        try:
            xls = pd.read_excel(io.BytesIO(file_bytes))
            if isinstance(xls, pd.DataFrame):
                return xls
            return pd.DataFrame()
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


def infer_dsr_label_from_upload_filename(filename: Optional[str], platform_word: str) -> str:
    """
    Pull seller/account token from upload names such as
    ``..._YG Myntra 21-22-4-26.csv``, ``..._Akiko Meesho ....csv``,
    ``..._Ikrass Flipkart ....xlsx`` (pattern: ``<label> <Platform>``).
    """
    if not filename or not platform_word:
        return ""
    name = Path(str(filename)).name.strip()
    if not name:
        return ""
    for _ in range(3):
        m = re.match(r"(.+)(\.(?:[a-z0-9]{1,12}))$", name, re.I)
        if not m:
            break
        name = m.group(1)
    pat = re.compile(rf"\s+{re.escape(platform_word)}\b", re.I)
    mm = pat.search(name)
    if not mm:
        return ""
    before = name[: mm.start()].strip()
    if not before:
        return ""
    token = before.split("_")[-1].strip()
    return (token if token else before).strip()


def apply_dsr_segment_from_upload_filename(
    df: pd.DataFrame,
    filename: Optional[str],
    platform_word: str,
) -> pd.DataFrame:
    """Set ``DSR_Segment`` on every row from the upload filename when a label is detected."""
    if df.empty:
        return df
    label = infer_dsr_label_from_upload_filename(filename, platform_word)
    if not label:
        return df
    out = df.copy()
    out["DSR_Segment"] = label
    return out


def apply_dsr_segment_to_df_inplace(
    df: pd.DataFrame,
    filename: Optional[str],
    platform_word: str,
) -> None:
    """
    Like ``apply_dsr_segment_from_upload_filename`` but mutates ``df`` so callers
    that pass the same object to SQLite + session merge always see ``DSR_Segment``.
    """
    if df.empty or not filename or not platform_word:
        return
    label = infer_dsr_label_from_upload_filename(filename, platform_word)
    if not label:
        return
    df["DSR_Segment"] = label

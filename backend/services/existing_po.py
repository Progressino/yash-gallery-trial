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
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)

# Heuristic used only for auto-detection (headerless sheets, ambiguous SKU columns).
# Must be permissive: "OMS_SKU" can be size SKUs (1917YKBLUE-3XL), parent-like tokens,
# or vendor/article ids (V1-A). The main goal is to reject non-SKU labels like "New SKU".
_OMS_SKU_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,}$", re.I)
_PO_SIZE_TOKENS = frozenset(
    {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"}
)
_SUMMARY_SKU_TOKENS = frozenset(
    {
        "TOTAL",
        "GRANDTOTAL",
        "SUBTOTAL",
        "SUMMARY",
        "SUM",
        "GRAND",
    }
)

# Accept common unicode dash variants from Excel/WhatsApp copies.
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_DASH_SPLIT_RE = re.compile(r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]+")


def _is_summary_sku(value: object) -> bool:
    """True for sheet footer/header aggregate rows (Total, Grand Total, …)."""
    s = str(value or "").strip().upper()
    if not s:
        return False
    compact = re.sub(r"[\s\-_]+", "", s)
    if compact in _SUMMARY_SKU_TOKENS:
        return True
    if compact.startswith("GRANDTOTAL") or compact.startswith("SUBTOTAL"):
        return True
    if re.fullmatch(r"TOTAL\S*", compact):
        return True
    return False


def _normalize_sku_text(value: object) -> str:
    """Normalize SKU tokens for stable grouping/joins (unicode dashes, stray whitespace)."""
    from .helpers import collapse_duplicate_trailing_size_suffix

    s = str(value or "").strip().upper()
    if not s:
        return ""
    s = _DASH_RE.sub("-", s)
    # Collapse " - " / "--" to "-" and trim around separators.
    parts = [p.strip() for p in s.split("-")]
    parts = [p for p in parts if p]
    return collapse_duplicate_trailing_size_suffix("-".join(parts))


_PL_INFIX_RE = re.compile(r"^(.*?)-PL-(.*?)$", re.I)


def existing_po_merge_key(raw: object) -> str:
    """
    Canonicalize Existing PO / pipeline merge keys without sku_mapping.

    sku_mapping maps individual sizes (1917YKBLUE-4XL) to bundled listings
    (1917YKBLUE-4XL-5XL). The uploaded PO sheet already uses the correct per-size
    keys — applying sku_mapping would collapse pipeline onto bundled rows.
    """
    try:
        from ..services.po_engine import normalize_id_token_for_mapping, clean_sku
        from .helpers import collapse_duplicate_trailing_size_suffix

        t = normalize_id_token_for_mapping(str(raw or "").strip())
        t = clean_sku(t or raw)
        if not t:
            t = str(raw or "").strip().upper()
        stripped = _PL_INFIX_RE.sub(r"\1-\2", str(t).strip().upper())
        return collapse_duplicate_trailing_size_suffix(stripped)
    except Exception:
        return _normalize_sku_text(raw)


_SKU_EXACT = [
    "OMS SKU", "OMS_SKU", "OMS SKU Code", "OMS",
    "SKU", "Seller SKU", "Merchant SKU", "Listing SKU", "Parent SKU",
    "Style Code", "Style", "Item Code", "Product Code", "Product ID",
    "ASIN", "Vendor Article Number", "Vendor Article", "Article Code",
    "Article", "Vendor Style", "MRP Article",
]


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


def _resolve_sku_column(cols: list[str]) -> Optional[str]:
    """Exact names first, then a safe fuzzy match (avoid order/sub-order id columns)."""
    hit = _find_col(cols, _SKU_EXACT)
    if hit:
        return hit
    skip_sub = ("order", "sub order", "packet", "line id", "release")
    for col in cols:
        cl = str(col).lower().strip()
        if not cl or cl.startswith("unnamed"):
            continue
        if any(x in cl for x in skip_sub):
            continue
        if "sku" in cl or cl.endswith("sku"):
            return col
        if cl in ("style code", "item code", "article code", "vendor article"):
            return col
    return None


def _dedupe_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.loc[:, ~raw.columns.duplicated(keep="first")].copy()
    out.columns = out.columns.astype(str).str.strip()
    return out


def _read_po_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(
            io.BytesIO(file_bytes), dtype=str, encoding="utf-8", on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            io.BytesIO(file_bytes), dtype=str, encoding="ISO-8859-1", on_bad_lines="skip"
        )
    return _dedupe_columns(raw)


def _read_po_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Try every sheet and header row 0..5 — real templates often have a title row
    or put the table on sheet 2, which breaks a naive read_excel(sheet=0).
    """
    bio = io.BytesIO(file_bytes)
    try:
        xl = pd.ExcelFile(bio)
    except Exception as e:
        raise ValueError(f"Cannot open Excel file: {e}") from e

    candidates: list[tuple[int, pd.DataFrame, str, int]] = []
    for sheet in xl.sheet_names:
        for header_row in range(0, 6):
            try:
                raw = pd.read_excel(xl, sheet_name=sheet, header=header_row, dtype=str)
            except Exception:
                continue
            if raw.empty or len(raw.columns) < 2:
                continue
            raw = _dedupe_columns(raw)
            cols = list(raw.columns)
            sku_guess = _resolve_sku_column(cols)
            if not sku_guess:
                continue
            sku_series = raw[sku_guess].astype(str).str.strip()
            n_ok = int(sku_series.map(_looks_like_oms_sku).sum())
            if n_ok == 0:
                # Column name looked like SKU-ish (e.g. "New SKU") but values do not.
                # Skip this candidate; otherwise headerless sheets get mis-detected.
                continue
            candidates.append((int(n_ok), raw, sheet, header_row))

    if not candidates:
        raise ValueError(
            "Could not find a SKU column on any sheet (tried multiple header rows). "
            "Add a column such as: OMS SKU, SKU, Style Code, Item Code, Seller SKU, "
            "Vendor Article — or move the table to start on row 1 with headers in one row."
        )

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _read_raw_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith(".csv"):
            raw = _read_po_csv(file_bytes)
        else:
            raw = _read_po_excel(file_bytes)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}") from e

    if raw.empty:
        raise ValueError("File is empty.")

    return raw


def _looks_like_oms_sku(value: object) -> bool:
    s = str(value or "").strip().upper()
    if len(s) < 3 or s in {"SKU", "OMS_SKU", "NAN", "NONE"}:
        return False
    if _OMS_SKU_RE.match(s):
        parts = s.split("-")
        # Size SKUs (and bundled size ranges).
        if len(parts) >= 2 and parts[-1] in _PO_SIZE_TOKENS:
            return True
        if len(parts) >= 3 and parts[-1] in _PO_SIZE_TOKENS and parts[-2] in _PO_SIZE_TOKENS:
            return True
        # Vendor/article ids usually include digits.
        if any(ch.isdigit() for ch in s):
            return True
        # Generic "ABC-RED-L" style tokens.
        if "-" in s and len(s) >= 6:
            return True
    return False


def _column_numeric_ratio(series: pd.Series) -> float:
    nums = pd.to_numeric(series, errors="coerce")
    if len(nums) == 0:
        return 0.0
    return float(nums.notna().mean())


def _column_looks_numeric_qty(series: pd.Series) -> bool:
    return _column_numeric_ratio(series) >= 0.6


def _read_headerless_po(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Sheets whose first row is data (no header) — common in operator exports."""
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith(".csv"):
            raw = pd.read_csv(
                io.BytesIO(file_bytes), header=None, dtype=str, encoding="utf-8", on_bad_lines="skip"
            )
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=str)
    except Exception:
        return None
    if raw.empty or raw.shape[1] < 4:
        return None
    raw = raw.dropna(how="all").reset_index(drop=True)
    sku_series = raw.iloc[:, 0].astype(str).str.strip()
    hits = sku_series.map(_looks_like_oms_sku).sum()
    if hits < max(2, int(len(raw) * 0.5)):
        return None
    out = raw.copy()
    out.columns = [f"_c{i}" for i in range(out.shape[1])]
    return out


def _infer_quantity_columns(raw: pd.DataFrame, sku_col: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    When named headers are missing, pick numeric columns by data shape.
    Typical Yash layout: SKU | status | style | name | pending | total | total | 0
    """
    numeric: list[tuple[str, float, float]] = []
    for col in raw.columns:
        if col == sku_col:
            continue
        series = raw[col]
        if not _column_looks_numeric_qty(series):
            continue
        nums = pd.to_numeric(series, errors="coerce").fillna(0)
        total = float(nums.sum())
        if total <= 0:
            continue
        numeric.append((col, total, float(nums.max())))

    if not numeric:
        return None, None, None, None

    numeric.sort(key=lambda x: (-x[1], -x[2]))
    total_col = numeric[0][0]
    total_series = pd.to_numeric(raw[total_col], errors="coerce").fillna(0)
    cutting_col = None
    if len(numeric) > 1:
        # Smallest non-zero numeric column that is not a duplicate of the total column.
        rest = []
        for col, sm, mx in numeric[1:]:
            if col == total_col:
                continue
            other = pd.to_numeric(raw[col], errors="coerce").fillna(0)
            if other.equals(total_series):
                continue
            rest.append((col, sm, mx))
        if rest:
            cutting_col = min(rest, key=lambda x: x[1])[0]
    return None, cutting_col, None, total_col


def _split_bundled_po_sku(sku: str) -> list[str]:
    """1917YKBLUE-XXL-3XL → [1917YKBLUE-XXL, 1917YKBLUE-3XL]."""
    s = _normalize_sku_text(sku)
    # Split on any dash variant / repeated separators.
    parts = [p.strip() for p in _DASH_SPLIT_RE.split(s) if p.strip()]
    if (
        len(parts) >= 3
        and parts[-1] in _PO_SIZE_TOKENS
        and parts[-2] in _PO_SIZE_TOKENS
        and parts[-1] != parts[-2]
    ):
        base = "-".join(parts[:-2])
        return [f"{base}-{parts[-2]}", f"{base}-{parts[-1]}"]
    return [s]


def is_bundled_size_range_sku(sku: object) -> bool:
    """True when SKU is a combined size band like 1917YKBLUE-XXL-3XL (not L-L typos)."""
    parts = _DASH_SPLIT_RE.split(_normalize_sku_text(sku))
    return (
        len(parts) >= 3
        and parts[-1] in _PO_SIZE_TOKENS
        and parts[-2] in _PO_SIZE_TOKENS
        and parts[-1] != parts[-2]
    )


def bundle_band_label(sku: object) -> str:
    """L-XL from 1917YKBLUE-L-XL; empty when not a bundled size-range SKU."""
    parts = _DASH_SPLIT_RE.split(_normalize_sku_text(sku))
    if (
        len(parts) >= 3
        and parts[-1] in _PO_SIZE_TOKENS
        and parts[-2] in _PO_SIZE_TOKENS
        and parts[-1] != parts[-2]
    ):
        return f"{parts[-2]}-{parts[-1]}"
    return ""


def build_bundle_listing_map(*sku_sources) -> dict[str, str]:
    """Map per-size and bundled OMS_SKU → bundle band label (e.g. L-XL)."""
    out: dict[str, str] = {}
    for source in sku_sources:
        if source is None:
            continue
        try:
            skus = source
            if hasattr(source, "astype"):
                skus = source.astype(str)
            for raw in skus:
                sku = _normalize_sku_text(raw)
                if not is_bundled_size_range_sku(sku):
                    continue
                band = bundle_band_label(sku)
                if not band:
                    continue
                out[sku] = band
                for child in _split_bundled_po_sku(sku):
                    out[_normalize_sku_text(child)] = band
        except Exception:
            continue
    return out


def _split_integer_qty(total: int, n: int) -> list[int]:
    """Split a whole quantity across *n* sizes, preserving the sum."""
    total = int(total or 0)
    n = int(n or 1)
    if n <= 1:
        return [total]
    base, rem = divmod(max(0, total), n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def bundled_listing_skus(skus) -> set[str]:
    """Normalized bundled size-range SKUs from any iterable."""
    out: set[str] = set()
    if skus is None:
        return out
    if hasattr(skus, "tolist"):
        items = skus.tolist()
    elif isinstance(skus, (str, bytes)):
        items = [skus]
    else:
        items = list(skus)
    for raw in items:
        sku = _normalize_sku_text(raw)
        if sku and is_bundled_size_range_sku(sku):
            out.add(sku)
    return out


def per_size_covered_by_bundled_listing(child_sku: object, bundled_skus: set[str]) -> bool:
    """True when *child_sku* is a size inside a bundled listing present in *bundled_skus*."""
    if not bundled_skus:
        return False
    ck = _normalize_sku_text(child_sku)
    if not ck:
        return False
    for band in bundled_skus:
        children = {_normalize_sku_text(c) for c in _split_bundled_po_sku(band)}
        if ck in children:
            return True
    return False


def expand_bundled_po_skus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fan out combined size-range PO lines (S-M, XXL-3XL) to individual OMS SKUs
    so PO Engine rows like 1917YKBLUE-3XL receive pipeline quantities.
    """
    if df is None or df.empty:
        return df
    breakdown = [
        c
        for c in ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total")
        if c in df.columns
    ]
    rows: list[dict] = []
    for _, r in df.iterrows():
        sku = _normalize_sku_text(r.get("OMS_SKU"))
        if not sku:
            continue
        expanded = _split_bundled_po_sku(sku)
        n = len(expanded)

        for esku in expanded:
            row = {"OMS_SKU": esku}
            for c in breakdown:
                val = pd.to_numeric(r.get(c), errors="coerce")
                val = 0 if pd.isna(val) else int(val)
                row[c] = val // n if n > 1 else val
            rows.append(row)
    if not rows:
        return df
    out = pd.DataFrame(rows)
    if breakdown:
        out = out.groupby("OMS_SKU", as_index=False)[breakdown].sum()
    return out


def fan_out_bundled_listing_metrics(
    metric_df: pd.DataFrame,
    metric_cols: list[str],
    *,
    retain_bundled_listing_skus: set[str] | None = None,
) -> pd.DataFrame:
    """
    Split bundled-listing metrics (e.g. 1917YKBLUE-L-XL sales) across per-size children.
    Skips children when inventory still lists the bundled listing (pipeline/sales stay on band).
    """
    if metric_df is None or metric_df.empty or "OMS_SKU" not in metric_df.columns:
        return pd.DataFrame(columns=["OMS_SKU"] + list(metric_cols))
    m = metric_df.copy()
    m["OMS_SKU"] = m["OMS_SKU"].astype(str).map(_normalize_sku_text)
    for c in metric_cols:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)
    bundled = m[m["OMS_SKU"].map(is_bundled_size_range_sku)]
    if bundled.empty:
        return pd.DataFrame(columns=["OMS_SKU"] + list(metric_cols))
    rows: list[dict] = []
    for _, br in bundled.iterrows():
        bk = str(br["OMS_SKU"])
        children = _split_bundled_po_sku(bk)
        n = len(children)
        if n <= 1:
            continue
        shares_by_col: dict[str, list[int | float]] = {}
        for c in metric_cols:
            if c not in br.index:
                shares_by_col[c] = [0] * n
                continue
            val = float(br[c] or 0)
            if val <= 0:
                shares_by_col[c] = [0] * n
            elif c.endswith("d") or c.endswith("_Units") or "Units" in c or c in {
                "Sold_Units",
                "Return_Units",
                "Net_Units",
                "Ship_Units_150d",
                "ADS_Sold_Units",
                "ADS_Net_Units",
                "Units_90d",
                "Units_30d",
                "Freq_30d",
            }:
                shares_by_col[c] = _split_integer_qty(int(val), n)
            else:
                shares_by_col[c] = [val / n] * n
        retain = retain_bundled_listing_skus or set()
        for i, kid in enumerate(children):
            kid_norm = _normalize_sku_text(kid)
            if per_size_covered_by_bundled_listing(kid_norm, retain):
                continue
            row: dict = {"OMS_SKU": kid_norm}
            for c in metric_cols:
                sh = shares_by_col.get(c, [0] * n)
                row[c] = sh[i] if i < len(sh) else 0
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["OMS_SKU"] + list(metric_cols))
    out = pd.DataFrame(rows)
    return out.groupby("OMS_SKU", as_index=False)[metric_cols].sum()


def session_has_fresh_existing_po(sess) -> bool:
    """True when this session holds an uploaded Existing PO newer than generic cache."""
    ep = getattr(sess, "existing_po_df", None)
    if ep is None or getattr(ep, "empty", True):
        return False
    gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    if gen > 0:
        return True
    # Sessions restored before generation tracking still have uploaded_at metadata.
    return bool(str(getattr(sess, "existing_po_uploaded_at", "") or "").strip())


def existing_po_needs_recalc(sess) -> bool:
    """PO table is stale until Calculate PO runs after the latest Existing PO upload."""
    if not session_has_fresh_existing_po(sess):
        return False
    cur = int(getattr(sess, "existing_po_generation", 0) or 0)
    done = int(getattr(sess, "po_calculate_existing_po_generation", -1) or -1)
    return done != cur


def _existing_po_disk_dir() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def existing_po_meta_bundle(sess) -> dict:
    """Serializable metadata for warm-cache / disk restore."""
    ep = getattr(sess, "existing_po_df", None)
    rows = int(len(ep)) if ep is not None and not getattr(ep, "empty", True) else 0
    sku_n = 0
    if ep is not None and not getattr(ep, "empty", True) and "OMS_SKU" in ep.columns:
        sku_n = int(ep["OMS_SKU"].astype(str).nunique())
    manual_skus = getattr(sess, "existing_po_manual_raise_skus", None) or []
    return {
        "existing_po_generation": int(getattr(sess, "existing_po_generation", 0) or 0),
        "existing_po_uploaded_at": str(getattr(sess, "existing_po_uploaded_at", "") or ""),
        "existing_po_filename": str(getattr(sess, "existing_po_filename", "") or ""),
        "existing_po_rows": rows,
        "existing_po_skus": sku_n,
        "existing_po_manual_raise_date": str(getattr(sess, "existing_po_manual_raise_date", "") or ""),
        "existing_po_manual_raise_skus_count": int(len(manual_skus)),
        "existing_po_manual_upload": bool(getattr(sess, "existing_po_manual_upload", False)),
    }


def persist_manual_raise_skus(sess) -> None:
    """Sidecar list of SKUs on the latest manual Existing PO raise sheet."""
    skus = getattr(sess, "existing_po_manual_raise_skus", None)
    if not skus:
        return
    try:
        path = _existing_po_disk_dir() / "existing_po_manual_raise_skus.json"
        path.write_text(json.dumps(list(skus), ensure_ascii=False), encoding="utf-8")
    except Exception:
        _log.exception("persist_manual_raise_skus failed")


def load_manual_raise_skus_into_session(sess) -> None:
    path = _existing_po_disk_dir() / "existing_po_manual_raise_skus.json"
    if not path.is_file():
        return
    try:
        skus = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(skus, list) and skus:
            sess.existing_po_manual_raise_skus = [str(s).strip().upper() for s in skus if str(s).strip()]
    except Exception:
        _log.exception("load_manual_raise_skus_into_session failed")


def manual_existing_po_raise_skus(sess) -> set[str]:
    skus = getattr(sess, "existing_po_manual_raise_skus", None)
    if skus:
        return {str(s).strip().upper() for s in skus if str(s).strip()}
    load_manual_raise_skus_into_session(sess)
    skus = getattr(sess, "existing_po_manual_raise_skus", None) or []
    return {str(s).strip().upper() for s in skus if str(s).strip()}


def apply_existing_po_session_meta(sess, meta: dict) -> None:
    """Copy Existing PO upload metadata onto a session (not the dataframe)."""
    if not isinstance(meta, dict):
        return
    gen = int(meta.get("existing_po_generation") or 0)
    if gen > int(getattr(sess, "existing_po_generation", 0) or 0):
        sess.existing_po_generation = gen
    for key in ("existing_po_uploaded_at", "existing_po_filename"):
        if meta.get(key):
            setattr(sess, key, str(meta[key]))
    if meta.get("existing_po_manual_raise_date"):
        sess.existing_po_manual_raise_date = str(meta["existing_po_manual_raise_date"])[:10]
    sess.existing_po_manual_upload = bool(meta.get("existing_po_manual_upload"))
    load_manual_raise_skus_into_session(sess)


def persist_existing_po_to_disk(sess) -> bool:
    """Write latest Existing PO upload to warm-cache disk (survives restarts / warm-cache clobber)."""
    ep = getattr(sess, "existing_po_df", None)
    if ep is None or getattr(ep, "empty", True):
        return False
    try:
        from .helpers import _coerce_df_for_parquet

        root = _existing_po_disk_dir()
        root.mkdir(parents=True, exist_ok=True)
        _coerce_df_for_parquet(ep).to_parquet(root / "existing_po_df.parquet", index=False)
        meta = existing_po_meta_bundle(sess)
        (root / "existing_po_meta.json").write_text(
            json.dumps(meta, default=str),
            encoding="utf-8",
        )
        persist_manual_raise_skus(sess)
        manifest_path = root / "_manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = set(manifest.get("keys") or [])
        keys.add("existing_po_df")
        keys.add("existing_po_meta")
        manifest["keys"] = sorted(keys)
        manifest_path.write_text(json.dumps(manifest, default=str), encoding="utf-8")
        _log.info(
            "Existing PO disk save: %s rows (gen=%s)",
            meta.get("existing_po_rows"),
            meta.get("existing_po_generation"),
        )
        return True
    except Exception:
        _log.exception("persist_existing_po_to_disk failed")
        return False


def read_existing_po_disk_meta() -> Optional[dict]:
    try:
        meta_path = _existing_po_disk_dir() / "existing_po_meta.json"
        if not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_existing_po_df_from_disk() -> Optional[pd.DataFrame]:
    try:
        path = _existing_po_disk_dir() / "existing_po_df.parquet"
        if not path.is_file():
            return None
        df = pd.read_parquet(path)
        return df if not getattr(df, "empty", True) else None
    except Exception:
        _log.exception("load existing_po_df from disk failed")
        return None


def count_per_size_pipeline_skus(ep: pd.DataFrame) -> int:
    """Individual-size SKUs with pipeline (not bundled bands like L-XL)."""
    if ep is None or ep.empty or "OMS_SKU" not in ep.columns:
        return 0
    skus = ep["OMS_SKU"].astype(str).str.strip().str.upper()
    pipe = pd.to_numeric(ep.get("PO_Pipeline_Total"), errors="coerce").fillna(0) > 0
    bundled = skus.map(is_bundled_size_range_sku)
    return int((~bundled & pipe).sum())


def existing_po_pipeline_sku_count(ep: pd.DataFrame | None) -> int:
    if ep is None or getattr(ep, "empty", True):
        return 0
    pipe = pd.to_numeric(ep.get("PO_Pipeline_Total"), errors="coerce").fillna(0)
    return int((pipe > 0).sum())


def existing_po_new_order_sku_count(ep: pd.DataFrame | None) -> int:
    if ep is None or getattr(ep, "empty", True) or "PO_Qty_Ordered" not in ep.columns:
        return 0
    ordered = pd.to_numeric(ep["PO_Qty_Ordered"], errors="coerce").fillna(0)
    return int((ordered > 0).sum())


def session_should_keep_existing_po(sess, warm_df: Optional[pd.DataFrame] = None) -> bool:
    """True when the session sheet is complete and should not be replaced by warm-cache copy."""
    if not session_has_fresh_existing_po(sess):
        return False
    ep = getattr(sess, "existing_po_df", None)
    if ep is None or getattr(ep, "empty", True):
        return False
    if existing_po_looks_aggregated_bundled_only(ep):
        return False
    try:
        from .daily_inventory_history import upload_timestamp_epoch

        sess_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
        sess_at = upload_timestamp_epoch(str(getattr(sess, "existing_po_uploaded_at", "") or ""))
        import backend.main as _main

        for meta in (
            _main._warm_cache.get(_main._EXISTING_PO_META_WARM_KEY),
            read_existing_po_disk_meta(),
        ):
            if not isinstance(meta, dict):
                continue
            mg = int(meta.get("existing_po_generation") or 0)
            ma = upload_timestamp_epoch(str(meta.get("existing_po_uploaded_at") or ""))
            if mg > sess_gen or ma > sess_at + 0.5:
                return False
    except Exception:
        pass
    if warm_df is None or getattr(warm_df, "empty", True):
        return True
    sess_rows = int(len(ep))
    warm_rows = int(len(warm_df))
    if warm_rows > sess_rows + 50:
        return False
    sess_per = count_per_size_pipeline_skus(ep)
    warm_per = count_per_size_pipeline_skus(warm_df)
    if warm_per > sess_per + 200:
        return False
    return True


def seed_existing_po_warm_cache_from_disk() -> bool:
    """Mirror on-disk Existing PO into server warm cache (survives restarts / deploys)."""
    df = _load_existing_po_df_from_disk()
    if df is None:
        return False
    meta = read_existing_po_disk_meta()
    try:
        import backend.main as _main

        wc = _main._warm_cache.get("existing_po_df")
        wc_rows = int(len(wc)) if wc is not None and not getattr(wc, "empty", True) else 0
        disk_rows = int(len(df))
        disk_per = count_per_size_pipeline_skus(df)
        wc_per = count_per_size_pipeline_skus(wc) if wc is not None else 0
        disk_gen = int((meta or {}).get("existing_po_generation") or 0)
        wc_meta = _main._warm_cache.get(_main._EXISTING_PO_META_WARM_KEY)
        wc_gen = int((wc_meta or {}).get("existing_po_generation") or 0) if isinstance(wc_meta, dict) else 0
        should_seed = (
            wc_rows == 0
            or disk_gen > wc_gen
            or disk_rows > wc_rows + 50
            or disk_per > wc_per + 200
            or existing_po_looks_aggregated_bundled_only(wc)
        )
        if not should_seed:
            return False
        if not _main._warm_cache:
            _main._warm_cache = {}
        _main._warm_cache["existing_po_df"] = df.copy()
        if meta:
            _main._warm_cache[_main._EXISTING_PO_META_WARM_KEY] = dict(meta)
        _log.info(
            "Existing PO seeded into warm cache from disk (%s rows, gen %s→%s)",
            disk_rows,
            wc_gen,
            disk_gen,
        )
        return True
    except Exception:
        _log.exception("seed_existing_po_warm_cache_from_disk failed")
        return False


def existing_po_looks_aggregated_bundled_only(ep: pd.DataFrame) -> bool:
    """
    True when the sheet has pipeline mostly on bundled SKUs (L-XL) with summed qty,
    but almost no per-size rows — typical of a stale warm-cache blob, not a full export.
    """
    if ep is None or ep.empty:
        return False
    per = count_per_size_pipeline_skus(ep)
    if per >= 500:
        return False
    skus = ep["OMS_SKU"].astype(str)
    bundled = skus.map(is_bundled_size_range_sku)
    pipe = pd.to_numeric(ep.get("PO_Pipeline_Total"), errors="coerce").fillna(0) > 0
    bundled_active = int((bundled & pipe).sum())
    if bundled_active == 0:
        return False
    if per == 0 and len(ep) <= 40:
        return True
    return len(ep) > 50 and per < max(80, bundled_active * 3)


def _apply_existing_po_df_to_session(sess, df: pd.DataFrame, meta: Optional[dict] = None) -> None:
    old_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    sess.existing_po_df = df
    if meta:
        apply_existing_po_session_meta(sess, meta)
    new_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    if new_gen > old_gen:
        sess.po_calculate_existing_po_generation = -1


def restore_existing_po_from_warm_cache(sess) -> bool:
    """Prefer shared warm-cache Existing PO when it is newer or has more per-size rows."""
    try:
        import backend.main as _main

        wc = _main._warm_cache.get("existing_po_df")
    except Exception:
        return False
    if wc is None or getattr(wc, "empty", True):
        return False
    cur = getattr(sess, "existing_po_df", None)
    wc_per = count_per_size_pipeline_skus(wc)
    cur_per = count_per_size_pipeline_skus(cur) if cur is not None else 0
    meta = None
    try:
        import backend.main as _main

        meta = _main._warm_cache.get(_main._EXISTING_PO_META_WARM_KEY)
    except Exception:
        pass
    newer = False
    if isinstance(meta, dict):
        try:
            from .daily_inventory_history import upload_timestamp_epoch

            mg = int(meta.get("existing_po_generation") or 0)
            sg = int(getattr(sess, "existing_po_generation", 0) or 0)
            ma = upload_timestamp_epoch(str(meta.get("existing_po_uploaded_at") or ""))
            sa = upload_timestamp_epoch(str(getattr(sess, "existing_po_uploaded_at", "") or ""))
            newer = mg > sg or ma > sa + 0.5
        except Exception:
            pass
    if not newer and wc_per <= cur_per + 200:
        return False
    _apply_existing_po_df_to_session(sess, wc.copy(), meta if isinstance(meta, dict) else None)
    _log.info("Existing PO restored from warm cache: %s per-size pipeline SKUs", wc_per)
    return True


def restore_existing_po_from_disk(sess) -> bool:
    """Hydrate session from disk when warm cache left a partial or stale Existing PO."""
    meta = read_existing_po_disk_meta()
    df = _load_existing_po_df_from_disk()
    if df is None:
        return False

    disk_gen = int(meta.get("existing_po_generation") or 0)
    disk_rows = int(meta.get("existing_po_rows") or len(df))
    sess_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    ep = getattr(sess, "existing_po_df", None)
    sess_rows = int(len(ep)) if ep is not None and not getattr(ep, "empty", True) else 0
    disk_per = count_per_size_pipeline_skus(df)
    sess_per = count_per_size_pipeline_skus(ep) if ep is not None else 0
    disk_fn = str(meta.get("existing_po_filename") or "").strip()
    sess_fn = str(getattr(sess, "existing_po_filename", "") or "").strip()
    try:
        from .daily_inventory_history import upload_timestamp_epoch

        disk_at = upload_timestamp_epoch(str(meta.get("existing_po_uploaded_at") or ""))
        sess_at = upload_timestamp_epoch(str(getattr(sess, "existing_po_uploaded_at", "") or ""))
    except Exception:
        disk_at = 0.0
        sess_at = 0.0

    newer_gen = disk_gen > sess_gen
    newer_upload = disk_at > sess_at + 0.5
    partial_session = sess_rows > 0 and disk_rows > sess_rows + 50
    same_gen_partial = disk_gen == sess_gen and sess_rows > 0 and disk_rows > sess_rows + 50
    empty_session = sess_rows == 0 and disk_rows > 0
    aggregated_session = sess_rows > 0 and disk_per > sess_per + 200
    filename_changed = bool(disk_fn and sess_fn and disk_fn != sess_fn)
    if not (
        newer_gen
        or newer_upload
        or partial_session
        or same_gen_partial
        or empty_session
        or aggregated_session
        or filename_changed
    ):
        return False

    _apply_existing_po_df_to_session(sess, df, meta)
    _log.info(
        "Existing PO restored from disk: %s rows, %s per-size pipeline (gen %s→%s)",
        disk_rows,
        disk_per,
        sess_gen,
        disk_gen,
    )
    return True


def ensure_existing_po_hydrated(sess) -> bool:
    """Before PO calculate, guarantee the full per-size uploaded sheet is in memory."""
    changed = False
    ep = getattr(sess, "existing_po_df", None)
    if existing_po_looks_aggregated_bundled_only(ep):
        changed = restore_existing_po_from_disk(sess) or changed
        changed = restore_existing_po_from_warm_cache(sess) or changed
    else:
        changed = restore_existing_po_from_disk(sess) or changed
        ep = getattr(sess, "existing_po_df", None)
        ep_empty = ep is None or getattr(ep, "empty", True)
        if existing_po_looks_aggregated_bundled_only(ep) or ep_empty:
            changed = restore_existing_po_from_warm_cache(sess) or changed
    if changed:
        try:
            if getattr(sess, "po_calculate_status", "idle") != "running":
                from ..services.po_raise_remove import invalidate_po_calculate_result

                invalidate_po_calculate_result(sess)
                from ..services.po_shared_cache import invalidate_all_shared_caches

                invalidate_all_shared_caches()
        except Exception:
            pass
    return changed


def _bundled_row_has_per_size_children(sku: str, sku_set: set[str]) -> bool:
    """True when this bundled band already has separate per-size rows on the sheet."""
    if not is_bundled_size_range_sku(sku):
        return False
    for child in _split_bundled_po_sku(sku):
        if child in sku_set:
            return True
    return False


def _expand_bundled_po_rows_without_children(
    ep: pd.DataFrame,
    inventory_skus: set[str] | None = None,
) -> pd.DataFrame:
    """Fan out only bundled rows whose per-size children are absent from the sheet."""
    if ep is None or ep.empty:
        return ep
    sku_set = set(ep["OMS_SKU"].astype(str))
    inv = {_normalize_sku_text(s) for s in (inventory_skus or set())}
    to_expand = ep[
        ep["OMS_SKU"].astype(str).map(
            lambda s: is_bundled_size_range_sku(s)
            and not _bundled_row_has_per_size_children(s, sku_set)
            and _normalize_sku_text(s) not in inv
        )
    ]
    if to_expand.empty:
        return ep
    keep = ep[~ep.index.isin(to_expand.index)]
    expanded = expand_bundled_po_skus(to_expand)
    out = pd.concat([keep, expanded], ignore_index=True)
    breakdown = [
        c
        for c in ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total")
        if c in out.columns
    ]
    if breakdown:
        out = out.groupby("OMS_SKU", as_index=False)[breakdown].sum()
    return out


def prepare_existing_po_for_merge(
    existing_po_df: pd.DataFrame,
    canonical_fn,
    inventory_skus: set[str] | None = None,
) -> pd.DataFrame:
    """
    Canonicalize Existing PO rows for exact OMS_SKU merge.

    Operator sheets list bundled listings (4XL-5XL) and individual sizes (4XL, 5XL)
    as separate rows with separate quantities — do not fan out when both exist.

    When the sheet only has bundled size-range rows, fan out to per-size SKUs only if
    inventory does not still list the bundled listing (e.g. 1917YKBLUE-L-XL with stock).
    """
    if existing_po_df is None or existing_po_df.empty:
        return pd.DataFrame()
    ep = existing_po_df.copy()
    if "PO_Pipeline_Total" not in ep.columns:
        return pd.DataFrame()
    ep["OMS_SKU"] = (
        ep["OMS_SKU"].astype(str).map(existing_po_merge_key).astype(str).str.strip().str.upper()
    )
    ep = ep[ep["OMS_SKU"].str.len() > 0]
    breakdown = [
        c
        for c in ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total")
        if c in ep.columns
    ]
    if breakdown:
        for c in breakdown:
            ep[c] = pd.to_numeric(ep[c], errors="coerce").fillna(0)
        ep = ep.groupby("OMS_SKU", as_index=False)[breakdown].sum()
    else:
        ep = ep.drop_duplicates(subset=["OMS_SKU"], keep="last")
    if not ep.empty and any(is_bundled_size_range_sku(s) for s in ep["OMS_SKU"].astype(str)):
        ep = _expand_bundled_po_rows_without_children(ep, inventory_skus=inventory_skus)
    return ep


def _ep_row_has_pipeline(ep_idx: pd.DataFrame, sku: str, breakdown: list[str]) -> bool:
    if sku not in ep_idx.index:
        return False
    for col in breakdown:
        if col not in ep_idx.columns:
            continue
        if float(pd.to_numeric(ep_idx.at[sku, col], errors="coerce") or 0) > 0:
            return True
    return False


def unbundle_inventory_rows_for_existing_po(
    po_df: pd.DataFrame,
    ep: pd.DataFrame,
    breakdown_cols: list[str],
    canonical_fn=None,
) -> pd.DataFrame:
    """
    Inventory often lists combined sizes (4XL-5XL) while the Existing PO sheet has
    per-size rows (4XL: 170, 5XL: 150). Add per-size PO rows for sheet pipeline only.

    Bundled listing stock does **not** fan out to individual sizes — a 4XL-5XL listing
    with 18 units is not 9+9 on 4XL and 5XL unless those SKUs exist separately in inventory.
    Keep the bundled listing row (with its inventory) when the sheet has a bundled line.
    """
    if po_df is None or po_df.empty or ep is None or ep.empty:
        return po_df
    breakdown = list(
        dict.fromkeys(
            breakdown_cols
            + [c for c in ("PO_Pipeline_Total", "PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch") if c in ep.columns]
        )
    )
    if not breakdown:
        return po_df

    for col in breakdown:
        if col not in po_df.columns:
            po_df[col] = 0

    merge_key = canonical_fn or existing_po_merge_key
    ep_idx = ep.set_index("OMS_SKU")
    split_metrics = [
        "Total_Inventory",
        "OMS_Inventory",
        "Sold_Units",
        "Return_Units",
        "Net_Units",
        "Recent_ADS",
        "ADS",
        "LY_ADS",
        "Seasonal_Month_ADS",
        "Flat30_ADS",
        "Ship_Units_150d",
        "Gross_PO_Qty",
        # Bundled rows may inherit style-level active days; per-size sheet rows must not.
        "Eff_Days",
    ]

    drop_idx: list = []
    new_rows: list[dict] = []
    po_skus = po_df["OMS_SKU"].astype(str).str.strip()

    # Pre-filter to bundled-SKU rows (vectorized) — iterrows() over the full
    # po_df (thousands of rows) is the dominant cost when almost all rows
    # are non-bundled and would just `continue`.
    _is_bundled = po_df["OMS_SKU"].astype(str).map(_normalize_sku_text).map(is_bundled_size_range_sku)
    for idx, row in po_df[_is_bundled].iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        sheet_children = [
            c for c in (_normalize_sku_text(ch) for ch in children)
            if _ep_row_has_pipeline(ep_idx, merge_key(c), breakdown)
        ]
        if not sheet_children:
            continue

        bundled_key = merge_key(sku)
        bundled_pipe = 0.0
        if "PO_Pipeline_Total" in po_df.columns:
            bundled_pipe = float(
                pd.to_numeric(po_df.at[idx, "PO_Pipeline_Total"], errors="coerce") or 0
            )
        child_ep_sum = 0.0
        for child in sheet_children:
            ck = merge_key(child)
            if ck in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
                child_ep_sum += float(
                    pd.to_numeric(ep_idx.at[ck, "PO_Pipeline_Total"], errors="coerce") or 0
                )
        per_size_rows_exist = any((po_skus == merge_key(c)).any() for c in sheet_children)
        if (
            not per_size_rows_exist
            and child_ep_sum > 0
            and bundled_pipe >= child_ep_sum
        ):
            continue

        base = row.to_dict()
        for child in sheet_children:
            child_key = merge_key(child)
            pipe_vals: dict[str, object] = {}
            for col in breakdown:
                if col in ep_idx.columns:
                    val = (
                        pd.to_numeric(ep_idx.at[child_key, col], errors="coerce")
                        if child_key in ep_idx.index
                        else 0
                    )
                    pipe_vals[col] = 0 if pd.isna(val) else val

            existing = po_skus == child_key
            if existing.any():
                for col, val in pipe_vals.items():
                    cur = pd.to_numeric(po_df.loc[existing, col], errors="coerce").fillna(0)
                    new_val = pd.to_numeric(val, errors="coerce")
                    new_val = 0 if pd.isna(new_val) else new_val
                    po_df.loc[existing, col] = np.maximum(cur, float(new_val)).astype(int)
                continue

            child_row = dict(base)
            child_row["OMS_SKU"] = child_key
            for m in split_metrics:
                if m in child_row:
                    child_row[m] = 0
            for col, val in pipe_vals.items():
                child_row[col] = val
            new_rows.append(child_row)

        bundled_key = merge_key(sku)
        if _ep_row_has_pipeline(ep_idx, bundled_key, breakdown):
            for col in breakdown:
                if col in ep_idx.columns and bundled_key in ep_idx.index:
                    val = pd.to_numeric(ep_idx.at[bundled_key, col], errors="coerce")
                    po_df.at[idx, col] = 0 if pd.isna(val) else val
        else:
            # Keep bundled listing rows when they still hold marketplace inventory.
            inv_on_row = pd.to_numeric(row.get("Total_Inventory"), errors="coerce")
            if pd.isna(inv_on_row) or float(inv_on_row) <= 0:
                drop_idx.append(idx)
            else:
                for col in breakdown:
                    if col in po_df.columns:
                        po_df.at[idx, col] = 0

    if not drop_idx and not new_rows:
        return po_df
    out = po_df.drop(index=drop_idx, errors="ignore")
    if new_rows:
        add = pd.DataFrame(new_rows)
        for c in out.columns:
            if c not in add.columns:
                add[c] = 0 if out[c].dtype != object else ""
        out = pd.concat([out, add[out.columns]], ignore_index=True)
    return out


def rollup_pipeline_onto_bundled_rows(
    po_df: pd.DataFrame,
    ep: pd.DataFrame,
    *,
    bundled_from_per_size_inv: set[str] | None = None,
) -> pd.DataFrame:
    """
    Inventory often uses bundled SKUs (4XL-5XL) while Existing PO lists individual sizes.
    Sum child pipeline onto the bundled inventory row when direct merge missed.

    When per-size children already exist as separate PO rows, keep the bundled sheet line
    qty only (e.g. 4XL-5XL: 4 units) — do not add child totals on top.
    """
    if po_df is None or po_df.empty or ep is None or ep.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    breakdown = [
        c
        for c in ("PO_Pipeline_Total", "PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch")
        if c in ep.columns
    ]
    if not breakdown:
        return po_df
    ep_idx = ep.set_index("OMS_SKU")
    out = po_df.copy()
    mapped_bundles = {str(s).strip().upper() for s in (bundled_from_per_size_inv or set())}
    po_skus = set(out["OMS_SKU"].astype(str).str.strip())

    for col in breakdown:
        if col not in out.columns:
            out[col] = 0
    # Pre-filter to bundled-SKU rows (vectorized) — see comment in
    # unbundle_inventory_rows_for_existing_po for why this matters.
    _is_bundled = out["OMS_SKU"].astype(str).map(_normalize_sku_text).map(is_bundled_size_range_sku)
    for idx, row in out[_is_bundled].iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        child_keys = [existing_po_merge_key(c) for c in children]
        children_with_pipeline = []
        for ck in child_keys:
            if ck not in po_skus:
                continue
            rows = out[out["OMS_SKU"].astype(str).str.strip() == ck]
            if rows.empty:
                continue
            pipe = float(pd.to_numeric(rows["PO_Pipeline_Total"], errors="coerce").fillna(0).max() or 0)
            if pipe > 0:
                children_with_pipeline.append(ck)
        bundled_key = existing_po_merge_key(sku)
        direct_total = 0.0
        if bundled_key in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
            direct_total = float(
                pd.to_numeric(ep_idx.at[bundled_key, "PO_Pipeline_Total"], errors="coerce") or 0
            )
        if children_with_pipeline:
            continue
        if direct_total > 0 and bundled_key not in mapped_bundles:
            continue
        for col in breakdown:
            total = 0.0
            found_child = False
            for child in children:
                ck = existing_po_merge_key(child)
                if ck in ep_idx.index:
                    found_child = True
                    total += float(pd.to_numeric(ep_idx.at[ck, col], errors="coerce") or 0)
            if found_child and total > 0:
                out.at[idx, col] = int(total) if col == "PO_Pipeline_Total" else total
    return out


_PIPELINE_COALESCE_COLS = (
    "PO_Pipeline_Total",
    "PO_Qty_Ordered",
    "Pending_Cutting",
    "Balance_to_Dispatch",
    "PO_Confirmed_Raise_Pipeline",
    "PO_Pipeline_Effective",
)


def coalesce_pipeline_columns_on_po_df(po_df: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing per-size pipeline on PO rows from the Existing PO sheet.

    Inventory may use bundled listing SKUs (via sku_mapping) while the uploaded PO
    sheet lists individual sizes — the left merge on OMS_SKU alone can miss those rows.
    """
    if po_df is None or po_df.empty or ep is None or ep.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    breakdown = [c for c in _PIPELINE_COALESCE_COLS if c in ep.columns]
    if not breakdown:
        return po_df
    ep_idx = ep.set_index("OMS_SKU")
    out = po_df.copy()
    for col in breakdown:
        if col not in out.columns:
            out[col] = 0
    keys = out["OMS_SKU"].astype(str).map(existing_po_merge_key)
    for idx, key in zip(out.index, keys):
        if not key or key not in ep_idx.index:
            continue
        for col in breakdown:
            cur = float(pd.to_numeric(out.at[idx, col], errors="coerce") or 0)
            ep_val = float(pd.to_numeric(ep_idx.at[key, col], errors="coerce") or 0)
            if ep_val > cur:
                if col in (
                    "PO_Pipeline_Total",
                    "PO_Qty_Ordered",
                    "Pending_Cutting",
                    "Balance_to_Dispatch",
                    "PO_Confirmed_Raise_Pipeline",
                    "PO_Pipeline_Effective",
                ):
                    out.at[idx, col] = int(ep_val)
                else:
                    out.at[idx, col] = ep_val
    return out


def per_size_pipeline_covered_by_bundled_po_row(child_sku: str, po_df: pd.DataFrame) -> bool:
    """True when a bundled inventory row already carries this child's pipeline total."""
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return False
    ck = existing_po_merge_key(child_sku)
    if is_bundled_size_range_sku(ck):
        return False
    for _, br in po_df.iterrows():
        bk = existing_po_merge_key(br.get("OMS_SKU"))
        if not is_bundled_size_range_sku(bk):
            continue
        children = {existing_po_merge_key(c) for c in _split_bundled_po_sku(bk)}
        if ck not in children:
            continue
        pipe = float(pd.to_numeric(br.get("PO_Pipeline_Total"), errors="coerce") or 0)
        if pipe > 0:
            return True
    return False


def bundled_pipeline_children_in_po(bundled_sku: str, ep: pd.DataFrame, po_keys: set[str]) -> bool:
    """True when every per-size child with pipeline on the sheet already has a PO row."""
    if not is_bundled_size_range_sku(bundled_sku) or ep is None or ep.empty:
        return False
    children = _split_bundled_po_sku(bundled_sku)
    if len(children) < 2:
        return False
    ep_idx = ep.set_index("OMS_SKU")
    saw_child_pipeline = False
    for child in children:
        ck = existing_po_merge_key(child)
        if ck not in ep_idx.index:
            continue
        pipe = float(pd.to_numeric(ep_idx.at[ck, "PO_Pipeline_Total"], errors="coerce") or 0)
        for bc in ("Pending_Cutting", "Balance_to_Dispatch", "PO_Qty_Ordered"):
            if bc in ep_idx.columns:
                pipe = max(pipe, float(pd.to_numeric(ep_idx.at[ck, bc], errors="coerce") or 0))
        if pipe <= 0:
            continue
        saw_child_pipeline = True
        if ck not in po_keys:
            return False
    return saw_child_pipeline


_BUNDLED_FAN_INTEGER_COLS = (
    "Sold_Units",
    "Return_Units",
    "Net_Units",
    "ADS_Sold_Units",
    "ADS_Net_Units",
    "Ship_Units_150d",
)
_BUNDLED_FAN_FLOAT_COLS = (
    "Recent_ADS",
    "LY_ADS",
    "Seasonal_Month_ADS",
    "Flat30_ADS",
    "ADS",
)


def _fan_metric_shares(row: pd.Series, n_children: int) -> dict[str, list[float | int]]:
    """Split bundled-row demand metrics across per-size children (sum preserved)."""
    shares: dict[str, list[float | int]] = {}
    for col in _BUNDLED_FAN_INTEGER_COLS:
        if col not in row.index:
            continue
        val = int(pd.to_numeric(row.get(col), errors="coerce") or 0)
        shares[col] = _split_integer_qty(val, n_children)
    for col in _BUNDLED_FAN_FLOAT_COLS:
        if col not in row.index:
            continue
        val = float(pd.to_numeric(row.get(col), errors="coerce") or 0)
        shares[col] = [val / n_children] * n_children
    eff = float(pd.to_numeric(row.get("Eff_Days"), errors="coerce") or 0)
    shares["Eff_Days"] = [eff] * n_children
    return shares


def fan_bundled_demand_to_sheet_children(
    po_df: pd.DataFrame,
    ep: pd.DataFrame | None,
    breakdown_cols: list[str] | None = None,
    *,
    canonical_fn=None,
) -> tuple[pd.DataFrame, set[int]]:
    """
    When the Existing PO sheet lists per-size pipeline (4XL, 5XL) for a bundled
    marketplace listing (4XL-5XL), shift demand/ADS from the band row onto those
    children so PO is raised on cuttable sizes — not on the listing SKU.

    Returns ``(dataframe, touched_row_indices)`` for ADS refresh.
    """
    touched: set[int] = set()
    if po_df is None or po_df.empty or ep is None or ep.empty or "OMS_SKU" not in po_df.columns:
        return po_df, touched
    breakdown = list(
        breakdown_cols
        or [c for c in _PIPELINE_COALESCE_COLS if c in ep.columns]
    )
    if not breakdown:
        return po_df, touched

    merge_key = canonical_fn or existing_po_merge_key
    ep_idx = ep.set_index("OMS_SKU")
    out = po_df.copy()
    po_skus = out["OMS_SKU"].astype(str).str.strip()
    is_bundled = out["OMS_SKU"].astype(str).map(_normalize_sku_text).map(is_bundled_size_range_sku)

    for idx, row in out[is_bundled].iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        sheet_children = [
            c for c in (_normalize_sku_text(ch) for ch in children)
            if _ep_row_has_pipeline(ep_idx, merge_key(c), breakdown)
        ]
        if len(sheet_children) < 1:
            continue
        bundled_key = merge_key(sku)
        bundled_on_sheet = _ep_row_has_pipeline(ep_idx, bundled_key, breakdown)
        child_keys = [merge_key(c) for c in sheet_children]
        if not any((po_skus == ck).any() for ck in child_keys):
            continue

        # Children that already received proportional sales from bundled-listing
        # fan-out during the sales merge must keep their metrics — do not re-fan.
        children_need_demand = False
        for child in sheet_children:
            child_key = merge_key(child)
            child_mask = po_skus == child_key
            if not child_mask.any():
                continue
            child_idx = out.index[child_mask][0]
            child_net = float(pd.to_numeric(out.at[child_idx, "Net_Units"], errors="coerce") or 0)
            child_sold = float(pd.to_numeric(out.at[child_idx, "Sold_Units"], errors="coerce") or 0)
            if child_net <= 0 and child_sold <= 0:
                children_need_demand = True
        if not children_need_demand:
            # Sales already fanned to per-size rows during merge — clear duplicate demand
            # on the bundled listing so PO is not raised twice (unless the sheet also
            # lists the band row as its own exclusive line alongside per-size rows).
            if bundled_on_sheet:
                continue
            child_has_metrics = False
            for child in sheet_children:
                child_key = merge_key(child)
                child_mask = po_skus == child_key
                if not child_mask.any():
                    continue
                child_idx = out.index[child_mask][0]
                child_net = float(pd.to_numeric(out.at[child_idx, "Net_Units"], errors="coerce") or 0)
                child_sold = float(pd.to_numeric(out.at[child_idx, "Sold_Units"], errors="coerce") or 0)
                if child_net > 0 or child_sold > 0:
                    child_has_metrics = True
                    touched.add(int(child_idx))
            if child_has_metrics:
                for col in _BUNDLED_FAN_INTEGER_COLS + _BUNDLED_FAN_FLOAT_COLS + ("Eff_Days",):
                    if col in out.columns:
                        out.at[idx, col] = 0
                touched.add(int(idx))
            continue

        sold = float(pd.to_numeric(row.get("Sold_Units"), errors="coerce") or 0)
        net = float(pd.to_numeric(row.get("Net_Units"), errors="coerce") or 0)
        if sold <= 0 and net <= 0:
            continue

        shares = _fan_metric_shares(row, len(sheet_children))
        for i, child in enumerate(sheet_children):
            child_key = merge_key(child)
            child_mask = po_skus == child_key
            if not child_mask.any():
                continue
            child_idx = out.index[child_mask][0]
            for col, parts in shares.items():
                if col not in out.columns:
                    continue
                val = parts[i] if i < len(parts) else 0
                if col in _BUNDLED_FAN_INTEGER_COLS:
                    out.at[child_idx, col] = int(val)
                elif col == "Eff_Days":
                    out.at[child_idx, col] = float(val)
                else:
                    out.at[child_idx, col] = round(float(val), 3)

        for col in _BUNDLED_FAN_INTEGER_COLS + _BUNDLED_FAN_FLOAT_COLS + ("Eff_Days",):
            if col in out.columns:
                out.at[idx, col] = 0
        touched.add(int(idx))
        for child in sheet_children:
            child_key = merge_key(child)
            child_mask = po_skus == child_key
            if child_mask.any():
                touched.add(int(out.index[child_mask][0]))
    return out, touched


def zero_bundled_po_when_sheet_children_present(
    po_df: pd.DataFrame,
    ep: pd.DataFrame | None,
    breakdown_cols: list[str] | None = None,
    *,
    canonical_fn=None,
) -> pd.DataFrame:
    """Bundled listing rows must not keep PO_Qty when per-size sheet children exist."""
    if po_df is None or po_df.empty or ep is None or ep.empty:
        return po_df
    breakdown = list(
        breakdown_cols
        or [c for c in _PIPELINE_COALESCE_COLS if c in ep.columns]
    )
    if not breakdown:
        return po_df
    merge_key = canonical_fn or existing_po_merge_key
    ep_idx = ep.set_index("OMS_SKU")
    out = po_df.copy()
    po_skus = set(out["OMS_SKU"].astype(str).str.strip())
    is_bundled = out["OMS_SKU"].astype(str).map(_normalize_sku_text).map(is_bundled_size_range_sku)
    for idx, row in out[is_bundled].iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        sheet_children = [
            merge_key(c)
            for c in (_normalize_sku_text(ch) for ch in children)
            if _ep_row_has_pipeline(ep_idx, merge_key(c), breakdown)
        ]
        if not sheet_children or not any(ck in po_skus for ck in sheet_children):
            continue
        bundled_key = merge_key(sku)
        if _ep_row_has_pipeline(ep_idx, bundled_key, breakdown):
            continue
        po_qty = int(pd.to_numeric(row.get("PO_Qty"), errors="coerce") or 0)
        gross = int(pd.to_numeric(row.get("Gross_PO_Qty"), errors="coerce") or 0)
        if po_qty <= 0 and gross <= 0:
            continue
        out.at[idx, "PO_Qty"] = 0
        out.at[idx, "Gross_PO_Qty"] = 0
    return out


def zero_bundled_pipeline_when_children_carry_qty(
    po_df: pd.DataFrame,
    ep: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Avoid double-counting open PO pipeline on both a size-band listing row (e.g. L-XL)
    and its per-size children when both appear in the PO output.

    When the sheet lists individual sizes plus a small exclusive band line (e.g. 4XL-5XL: 4
    while 4XL/5XL have their own rows), keep the band residual. When the band line is a
    summary of the same units already on per-size rows, clear the band row.
    """
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    breakdown = [
        c
        for c in _PIPELINE_COALESCE_COLS
        if c in po_df.columns and c not in ("PO_Confirmed_Raise_Pipeline", "PO_Pipeline_Effective")
    ]
    if not breakdown:
        return po_df
    out = po_df.copy()
    po_keys = set(out["OMS_SKU"].astype(str).str.strip())
    ep_idx = ep.set_index("OMS_SKU") if ep is not None and not ep.empty else None
    is_bundled = (
        out["OMS_SKU"].astype(str).map(_normalize_sku_text).map(is_bundled_size_range_sku)
    )
    for idx, row in out[is_bundled].iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        child_pipe = 0.0
        ep_child_sum = 0.0
        for child in children:
            ck = existing_po_merge_key(child)
            if ck not in po_keys:
                continue
            rows = out[out["OMS_SKU"].astype(str).str.strip() == ck]
            if rows.empty or "PO_Pipeline_Total" not in rows.columns:
                continue
            child_pipe += float(
                pd.to_numeric(rows["PO_Pipeline_Total"], errors="coerce").fillna(0).max() or 0
            )
            if ep_idx is not None and ck in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
                ep_child_sum += float(
                    pd.to_numeric(ep_idx.at[ck, "PO_Pipeline_Total"], errors="coerce") or 0
                )
        bundled_pipe = float(pd.to_numeric(row.get("PO_Pipeline_Total"), errors="coerce") or 0)
        if child_pipe <= 0 or bundled_pipe <= 0:
            continue
        bundled_key = existing_po_merge_key(sku)
        ep_bundled = 0.0
        if ep_idx is not None and bundled_key in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
            ep_bundled = float(
                pd.to_numeric(ep_idx.at[bundled_key, "PO_Pipeline_Total"], errors="coerce") or 0
            )
        if ep_bundled > 0 and ep_child_sum > 0:
            if ep_bundled >= ep_child_sum * 0.9:
                for col in breakdown:
                    out.at[idx, col] = 0
                continue
            for col in breakdown:
                if ep_idx is not None and bundled_key in ep_idx.index and col in ep_idx.columns:
                    val = pd.to_numeric(ep_idx.at[bundled_key, col], errors="coerce")
                    out.at[idx, col] = 0 if pd.isna(val) else int(val)
            continue
        if bundled_pipe <= child_pipe:
            for col in breakdown:
                out.at[idx, col] = 0
    return out


def bundled_band_redundant_with_child_pipeline(
    bundled_sku: str,
    po_df: pd.DataFrame,
    ep: pd.DataFrame | None = None,
) -> bool:
    """True when a band row duplicates per-size pipeline already on child rows."""
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return False
    sku = _normalize_sku_text(bundled_sku)
    if not is_bundled_size_range_sku(sku):
        return False
    po_keys = set(po_df["OMS_SKU"].astype(str).str.strip())
    child_pipe = 0.0
    ep_child_sum = 0.0
    ep_idx = ep.set_index("OMS_SKU") if ep is not None and not ep.empty else None
    for child in _split_bundled_po_sku(sku):
        ck = existing_po_merge_key(child)
        if ck not in po_keys:
            continue
        rows = po_df[po_df["OMS_SKU"].astype(str).str.strip() == ck]
        if rows.empty:
            continue
        child_pipe += float(
            pd.to_numeric(rows["PO_Pipeline_Total"], errors="coerce").fillna(0).max() or 0
        )
        if ep_idx is not None and ck in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
            ep_child_sum += float(
                pd.to_numeric(ep_idx.at[ck, "PO_Pipeline_Total"], errors="coerce") or 0
            )
    if child_pipe <= 0:
        return False
    bundled_key = existing_po_merge_key(sku)
    band_rows = po_df[po_df["OMS_SKU"].astype(str).str.strip() == bundled_key]
    if band_rows.empty:
        return False
    bundled_pipe = float(
        pd.to_numeric(band_rows["PO_Pipeline_Total"], errors="coerce").fillna(0).max() or 0
    )
    if bundled_pipe <= 0:
        return False
    ep_bundled = 0.0
    if ep_idx is not None and bundled_key in ep_idx.index and "PO_Pipeline_Total" in ep_idx.columns:
        ep_bundled = float(
            pd.to_numeric(ep_idx.at[bundled_key, "PO_Pipeline_Total"], errors="coerce") or 0
        )
    if ep_bundled > 0 and ep_child_sum > 0:
        return ep_bundled >= ep_child_sum * 0.9
    return bundled_pipe <= child_pipe


def dedupe_po_rows_by_sku(po_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate OMS_SKU rows — keep the richest row (pipeline / sales / stock)."""
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    if not po_df["OMS_SKU"].astype(str).duplicated().any():
        return po_df
    work = po_df.copy()

    def _num(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(0.0, index=work.index)
        return pd.to_numeric(work[col], errors="coerce").fillna(0)

    work["_dedupe_rank"] = (
        _num("PO_Pipeline_Total")
        + _num("Total_Inventory")
        + _num("Sold_Units")
        + _num("ADS") * 10.0
    )
    pipeline_cols = [c for c in _PIPELINE_COALESCE_COLS if c in work.columns]
    rows: list[pd.Series] = []
    for _sku, grp in work.groupby("OMS_SKU", sort=False):
        best_idx = grp["_dedupe_rank"].idxmax()
        row = grp.loc[best_idx].copy()
        for col in pipeline_cols:
            row[col] = _num(col).loc[grp.index].max()
        rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out.drop(columns=["_dedupe_rank"], errors="ignore")


def _build_po_dataframe(
    raw: pd.DataFrame,
    sku_col: str,
    *,
    ordered_col: Optional[str],
    cutting_col: Optional[str],
    dispatch_col: Optional[str],
    total_col: Optional[str],
    fallback_col: Optional[str],
) -> pd.DataFrame:
    specific_cols = [
        c for c in [ordered_col, cutting_col, dispatch_col, total_col, fallback_col] if c
    ]
    all_needed = list(dict.fromkeys([sku_col] + specific_cols))
    df = raw[all_needed].copy()

    df[sku_col] = df[sku_col].map(_normalize_sku_text)
    df = df[df[sku_col].str.len() > 0]
    df = df[~df[sku_col].isin(["SKU", "OMS_SKU", "NAN", "NONE", ""])]
    df = df[~df[sku_col].map(_is_summary_sku)]

    if df.empty:
        raise ValueError("No valid SKU rows found after parsing.")

    for c in specific_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    result = df.groupby(sku_col, as_index=False).agg({c: "sum" for c in specific_cols})
    result = result.rename(columns={sku_col: "OMS_SKU"})

    rename_map: dict[str, str] = {}
    if ordered_col:
        rename_map[ordered_col] = "PO_Qty_Ordered"
    if cutting_col:
        rename_map[cutting_col] = "Pending_Cutting"
    if dispatch_col:
        rename_map[dispatch_col] = "Balance_to_Dispatch"
    if total_col:
        rename_map[total_col] = "Total_Balance"
    if fallback_col and fallback_col not in rename_map:
        rename_map[fallback_col] = "PO_Qty_Ordered"
    result = result.rename(columns=rename_map)

    if "Total_Balance" in result.columns:
        result["PO_Pipeline_Total"] = result["Total_Balance"].clip(lower=0).astype(int)
        result = result.drop(columns=["Total_Balance"])
    elif "Pending_Cutting" in result.columns and "Balance_to_Dispatch" in result.columns:
        result["PO_Pipeline_Total"] = (
            result["Pending_Cutting"] + result["Balance_to_Dispatch"]
        ).clip(lower=0).astype(int)
    elif "Pending_Cutting" in result.columns and "PO_Pipeline_Total" not in result.columns:
        result["PO_Pipeline_Total"] = result["Pending_Cutting"].clip(lower=0).astype(int)
    elif "PO_Qty_Ordered" in result.columns:
        result["PO_Pipeline_Total"] = result["PO_Qty_Ordered"].clip(lower=0).astype(int)
    else:
        result["PO_Pipeline_Total"] = 0

    for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total"]:
        if c in result.columns:
            result[c] = result[c].clip(lower=0).astype(int)

    return result


def parse_existing_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an existing PO tracking sheet.
    Accepts .xlsx or .csv.

    Extracts (all optional except SKU):
      - PO_Qty_Ordered      : qty originally ordered with manufacturer
      - Pending_Cutting     : units awaiting cutting
      - Balance_to_Dispatch : units cut but not yet dispatched
      - Total_Balance       : total pipeline → PO_Pipeline_Total

    If none of the specific columns are found, falls back to any
    generic balance/qty column as PO_Pipeline_Total.
    """
    try:
        raw = _read_raw_po(file_bytes, filename)
    except ValueError:
        # Headerless sheets can cause the Excel header-row scan to fail entirely.
        headerless = _read_headerless_po(file_bytes, filename)
        if headerless is not None:
            raw = headerless
        else:
            raise
    cols = list(raw.columns)

    sku_col = _resolve_sku_column(cols)
    # Some operator exports omit the header row; pandas then treats the first data row as headers
    # (e.g. columns: ["1917YKBLUE-3XL", "New SKU", ..., "130"]). Detect and re-read as headerless.
    if sku_col is not None:
        try:
            _col_names = [str(c).strip() for c in cols]
            _has_numeric_headers = any(
                bool(re.fullmatch(r"\d+(?:\.\d+)?", n)) for n in _col_names
            )
            if _looks_like_oms_sku(sku_col) and _has_numeric_headers:
                headerless = _read_headerless_po(file_bytes, filename)
                if headerless is not None:
                    raw = headerless
                    cols = list(raw.columns)
                    sku_col = "_c0"
        except Exception:
            pass

    if sku_col is None:
        headerless = _read_headerless_po(file_bytes, filename)
        if headerless is not None:
            raw = headerless
            cols = list(raw.columns)
            sku_col = "_c0"
        else:
            raise ValueError(f"Cannot find a SKU column. Columns seen: {cols[:25]}")

    # ── Find specific breakdown columns (all optional) ──────────
    cutting_col = _find_col(
        cols,
        [
            "Pending Cutting", "Pending Cut", "Cutting Pending",
            "Pend Cutting", "Cut Pending", "Pending Cuts",
        ],
    )
    if cutting_col is None:
        cutting_col = _find_col_fuzzy(
            cols, ["pending cut", "cut pending", "pending cutting"]
        )

    dispatch_col = _find_col(
        cols,
        [
            "Balance to dispatch", "Dispatch Balance", "Bal to Dispatch",
            "Balance Dispatch", "Pending dispatch", "Pending Dispatch",
        ],
    )
    if dispatch_col is None:
        dispatch_col = _find_col_fuzzy(
            cols,
            [
                "balance to dispatch",
                "dispatch balance",
                "bal to dispatch",
                "qty to dispatch",
                "pending dispatch",
            ],
        )

    total_col = _find_col(
        cols,
        [
            "TOTAL BALANCE From Latest Status", "Total Balance From Latest Status",
            "Total Balance", "Total Bal", "Total Pending", "Net Balance", "TOTAL BALANCE",
        ],
    )
    if total_col is None:
        total_col = _find_col_fuzzy(cols, ["total balance", "total bal"])

    _claimed = {c for c in [cutting_col, dispatch_col, total_col] if c}
    ordered_col = _find_col(
        cols,
        [
            "NEW ORDER", "New Order", "PO Qty", "PO Quantity",
            "Ordered Qty", "Ordered Quantity", "Order Qty",
        ],
    )
    if ordered_col is None or ordered_col in _claimed:
        ordered_col = next(
            (
                col
                for col in cols
                if col not in _claimed
                and any(
                    kw in col.lower()
                    for kw in ["new order", "ordered qty", "order qty"]
                )
                and _column_looks_numeric_qty(raw[col])
            ),
            None,
        )
    elif not _column_looks_numeric_qty(raw[ordered_col]):
        ordered_col = None

    fallback_col: Optional[str] = None
    if not any([ordered_col, cutting_col, dispatch_col, total_col]):
        fallback_col = _find_col(
            cols,
            [
                "Balance Qty", "Balance Quantity", "Open Qty", "Open Quantity",
                "Pending Qty", "Pending Quantity", "Units", "Balance",
                "Qty", "Balance to dispatch", "TOTAL BALANCE From Latest Status",
                "Total Balance", "Dispatch Balance", "Pending dispatch",
            ],
        )
        if fallback_col is None:
            fallback_col = _find_col_fuzzy(
                cols, ["balance", "dispatch", "pending", "open qty", "po qty"]
            )
        if fallback_col is None or not _column_looks_numeric_qty(raw[fallback_col]):
            inferred = _infer_quantity_columns(raw, sku_col)
            ordered_col, cutting_col, dispatch_col, total_col = inferred
            fallback_col = None
            if not any([ordered_col, cutting_col, dispatch_col, total_col]):
                headerless = _read_headerless_po(file_bytes, filename)
                if headerless is not None:
                    raw = headerless
                    sku_col = "_c0"
                    ordered_col, cutting_col, dispatch_col, total_col = _infer_quantity_columns(
                        raw, sku_col
                    )
                if not any([ordered_col, cutting_col, dispatch_col, total_col]):
                    raise ValueError(
                        f"Cannot find a quantity/balance column. Columns seen: {cols[:25]}"
                    )

    return _build_po_dataframe(
        raw,
        sku_col,
        ordered_col=ordered_col,
        cutting_col=cutting_col,
        dispatch_col=dispatch_col,
        total_col=total_col,
        fallback_col=fallback_col,
    )


def audit_existing_po_upload(
    file_bytes: bytes,
    filename: str,
    parsed: pd.DataFrame,
) -> dict:
    """
    Compare parsed Existing PO totals with the sheet's own Total / summary row.

    Used on first upload so operators can confirm pending cutting, balance to dispatch,
    and total balance match before PO math runs.
    """
    out: dict = {
        "sku_rows": int(len(parsed)),
        "pipeline_units": int(pd.to_numeric(parsed.get("PO_Pipeline_Total"), errors="coerce").fillna(0).sum()),
        "pending_cutting_units": int(
            pd.to_numeric(parsed.get("Pending_Cutting"), errors="coerce").fillna(0).sum()
        )
        if "Pending_Cutting" in parsed.columns
        else None,
        "balance_to_dispatch_units": int(
            pd.to_numeric(parsed.get("Balance_to_Dispatch"), errors="coerce").fillna(0).sum()
        )
        if "Balance_to_Dispatch" in parsed.columns
        else None,
        "sheet_total_row": None,
        "totals_match": True,
        "warnings": [],
    }
    try:
        raw = _read_raw_po(file_bytes, filename)
        cols = list(raw.columns)
        sku_col = _resolve_sku_column(cols) or "_c0"
        if sku_col not in raw.columns and "_c0" in raw.columns:
            sku_col = "_c0"
        sku_series = raw[sku_col].astype(str).map(_normalize_sku_text)
        summary_mask = sku_series.map(_is_summary_sku)
        if not summary_mask.any():
            return out
        row = raw.loc[summary_mask].iloc[0]
        sheet: dict[str, int] = {}
        cutting_col = _find_col(cols, ["Pending Cutting", "Pending Cut", "Cutting Pending"]) or _find_col_fuzzy(
            cols, ["pending cut", "pending cutting"]
        )
        dispatch_col = _find_col(
            cols,
            ["Balance to dispatch", "Dispatch Balance", "Bal to Dispatch", "Pending dispatch"],
        ) or _find_col_fuzzy(cols, ["balance to dispatch", "pending dispatch"])
        total_col = _find_col(
            cols,
            ["TOTAL BALANCE From Latest Status", "Total Balance From Latest Status", "Total Balance"],
        ) or _find_col_fuzzy(cols, ["total balance"])
        if cutting_col and cutting_col in row.index:
            sheet["pending_cutting_units"] = int(pd.to_numeric(row[cutting_col], errors="coerce") or 0)
        if dispatch_col and dispatch_col in row.index:
            sheet["balance_to_dispatch_units"] = int(pd.to_numeric(row[dispatch_col], errors="coerce") or 0)
        if total_col and total_col in row.index:
            sheet["total_balance_units"] = int(pd.to_numeric(row[total_col], errors="coerce") or 0)
            sheet["pipeline_units"] = sheet["total_balance_units"]
        out["sheet_total_row"] = sheet
        tol = 1
        for key, parsed_key in (
            ("pipeline_units", "pipeline_units"),
            ("pending_cutting_units", "pending_cutting_units"),
            ("balance_to_dispatch_units", "balance_to_dispatch_units"),
        ):
            if key not in sheet or out.get(parsed_key) is None:
                continue
            if abs(int(sheet[key]) - int(out[parsed_key])) > tol:
                out["totals_match"] = False
                out["warnings"].append(
                    f"Parsed {parsed_key.replace('_', ' ')} ({out[parsed_key]:,}) "
                    f"≠ sheet Total row ({sheet[key]:,})."
                )
    except Exception as exc:
        out["warnings"].append(f"Could not read sheet Total row for audit: {exc}")
    return out


def existing_po_pipeline_totals(ep: pd.DataFrame | None) -> tuple[int, int]:
    """Authoritative pipeline units/SKU count from the uploaded Existing PO sheet."""
    if ep is None or getattr(ep, "empty", True):
        return 0, 0
    cols = [
        c
        for c in (
            "OMS_SKU",
            "PO_Pipeline_Total",
            "Pending_Cutting",
            "Balance_to_Dispatch",
            "PO_Qty_Ordered",
        )
        if c in ep.columns
    ]
    if "OMS_SKU" not in cols or "PO_Pipeline_Total" not in cols:
        return 0, 0
    shell = ep[cols].copy()
    for c in shell.columns:
        if c == "OMS_SKU":
            continue
        shell[c] = pd.to_numeric(shell[c], errors="coerce").fillna(0)
    shell["Total_Inventory"] = 0
    deduped = zero_bundled_pipeline_when_children_carry_qty(shell, ep)
    pipe = pd.to_numeric(deduped["PO_Pipeline_Total"], errors="coerce").fillna(0)
    return int(pipe.sum()), int((pipe > 0).sum())

"""
Parse SKU Status & Lead Time master (Excel/CSV) for PO Engine.

Expected columns (flexible names): SKU, Lead time (days). Status is optional — when omitted,
SKU_Sheet_Status is left blank and SKU_Sheet_Closed is derived only when status text indicates closure.
"""
from __future__ import annotations

import io
import re
from typing import BinaryIO, Optional

import pandas as pd

from .helpers import (
    clean_sku,
    collapse_duplicate_trailing_size_suffix,
    normalize_id_token_for_mapping,
)

# Match Amazon PL infix stripping used in po_engine / inventory.
_PL_RE = re.compile(r"^(\d+)PL(YK)", re.I)


def _strip_pl_sku(sku: str, mapping: dict) -> str:
    raw = str(sku).strip().upper()
    stripped = _PL_RE.sub(r"\1\2", raw)
    return mapping.get(stripped, mapping.get(raw, stripped))


def _norm_col(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _pick_status_column_fuzzy(df: pd.DataFrame, sku_col: Optional[str], lead_col: Optional[str]) -> Optional[str]:
    """When headers are non-standard (e.g. export truncation), find a status-like column."""
    skip = {_norm_col(c) for c in (sku_col or "", lead_col or "") if c}
    norm_map = {_norm_col(c): c for c in df.columns}
    for nk, orig in norm_map.items():
        if nk in skip:
            continue
        if "status" in nk or nk in ("state", "lifecycle", "availability"):
            return orig
    hints = (
        "closed sku",
        "closed style",
        "inactive",
        "discontinue",
        "discontinued",
        "open",
        "active",
        "fast moving",
        "slow selling",
        "moderate",
        "not moving",
    )
    n = len(df)
    if n == 0:
        return None
    # Small sheets (smoke tests, spot uploads) need a low threshold so we still detect status.
    min_hits = 1 if n < 12 else max(5, int(n * 0.02))
    best_col, best_hits = None, 0
    for col in df.columns:
        nk = _norm_col(col)
        if nk in skip:
            continue
        if lead_col and col == lead_col:
            continue
        if sku_col and col == sku_col:
            continue
        s = df[col].astype(str).str.strip().str.lower()
        hits = int(s.map(lambda t: any(h in t for h in hints) if t and t != "nan" else False).sum())
        if hits > best_hits:
            best_hits, best_col = hits, col
    return best_col if best_hits >= min_hits else None


def _pick_lead_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve lead-time column; tolerant of export quirks (LT, merged headers, etc.)."""
    lead_col = _pick_column(
        df,
        [
            "lead time (days)",
            "lead time days",
            "leadtimedays",
            "lead_time_days",
            "lead time",
            "leadtime",
            "lead_time",
            "manufacturing lead",
            "factory lead",
            "supplier lead",
            "po lead",
            "lead days",
            "days to ship",
            "production lead time",
            "productionleadtime",
            "manufacturing lead time",
            "total lead time",
            "totallt",
            "lt days",
            "lt",
            "mlt",
        ],
    )
    if lead_col:
        return lead_col
    norm_map = {_norm_col(c): c for c in df.columns}
    # Prefer a column whose header clearly mentions lead + days/time.
    for nk, orig in norm_map.items():
        if "lead" in nk and ("day" in nk or "time" in nk or nk.endswith("days")):
            return orig
    if "lt" in norm_map:
        return norm_map["lt"]
    if "mlt" in norm_map:
        return norm_map["mlt"]
    return None


def is_closed_sku_status(status) -> bool:
    """True when user marked SKU closed (e.g. 'Closed SKU', not generic 'closed order' notes)."""
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return False
    t = str(status).strip().lower()
    if not t:
        return False
    if "closed sku" in t:
        return True
    if "closed" in t and any(k in t for k in ("sku", "style", "item", "product")):
        return True
    if t in ("closed", "inactive", "discontinue", "discontinued"):
        return True
    if "eol" in t or "end of life" in t or "no reorder" in t or "do not reorder" in t:
        return True
    return False


def parse_sku_status_lead_dataframe(
    df: pd.DataFrame,
    sku_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Normalize to columns: OMS_SKU, SKU_Sheet_Status, Lead_Time_From_Sheet (nullable float), SKU_Sheet_Closed (bool).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["OMS_SKU", "SKU_Sheet_Status", "Lead_Time_From_Sheet", "SKU_Sheet_Closed"])

    sku_col = _pick_column(df, ["sku", "oms_sku", "oms sku", "seller sku", "style sku"])
    lead_col = _pick_lead_column(df)
    status_col = _pick_column(df, ["status", "sku status", "sku sheet status", "state"])
    if not status_col:
        status_col = _pick_status_column_fuzzy(df, sku_col, lead_col)

    if not sku_col:
        raise ValueError("Could not find a SKU column (expected names like SKU, OMS_SKU).")
    if not lead_col:
        raise ValueError(
            "Could not find a Lead time column (e.g. Lead Time (days), Lead_Time_Days, LT)."
        )

    _map = sku_mapping if sku_mapping is not None else {}

    def _norm_sku(raw) -> str:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ""
        tok = normalize_id_token_for_mapping(str(raw).strip())
        s = clean_sku(tok or raw)
        if not s:
            return ""
        return collapse_duplicate_trailing_size_suffix(_strip_pl_sku(s, _map))

    oms = df[sku_col].map(_norm_sku)
    mask = oms.str.len() > 0
    if not bool(mask.any()):
        return pd.DataFrame(columns=["OMS_SKU", "SKU_Sheet_Status", "Lead_Time_From_Sheet", "SKU_Sheet_Closed"])

    if status_col:
        status = df.loc[mask, status_col].fillna("").astype(str).str.strip()
    else:
        status = pd.Series([""] * int(mask.sum()), index=df.index[mask])
    out = pd.DataFrame(
        {
            "OMS_SKU": oms[mask].values,
            "SKU_Sheet_Status": status.values,
            "Lead_Time_From_Sheet": pd.to_numeric(df.loc[mask, lead_col], errors="coerce").values,
        }
    )
    out["SKU_Sheet_Closed"] = out["SKU_Sheet_Status"].map(is_closed_sku_status)
    # Last row wins duplicate SKUs
    out = out.drop_duplicates(subset=["OMS_SKU"], keep="last")
    return out.reset_index(drop=True)


def parse_sku_status_lead_upload(
    file: BinaryIO,
    filename: str,
    sku_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    name = (filename or "").lower()
    raw = file.read()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    else:
        df = pd.read_excel(io.BytesIO(raw))
    return parse_sku_status_lead_dataframe(df, sku_mapping=sku_mapping)

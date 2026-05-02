"""
Parse SKU Status & Lead Time master (Excel/CSV) for PO Engine.

Expected columns (flexible names): SKU, Status, Lead time (days).
Status is stored for display; values that look like "closed SKU" set SKU_Sheet_Closed in the PO table only (PO quantities use the same engine rules as without a sheet, except per-SKU lead overrides).
"""
from __future__ import annotations

import io
import re
from typing import BinaryIO, Optional

import pandas as pd

from .helpers import clean_sku, normalize_id_token_for_mapping

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
    return t in ("closed", "inactive", "discontinue", "discontinued")


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
    status_col = _pick_column(df, ["status", "sku status", "state"])
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
        ],
    )

    if not sku_col:
        raise ValueError("Could not find a SKU column (expected names like SKU, OMS_SKU).")
    if not status_col:
        raise ValueError("Could not find a Status column.")
    if not lead_col:
        raise ValueError("Could not find a Lead time column (e.g. Lead Time, Lead days).")

    _map = sku_mapping if sku_mapping is not None else {}

    out_rows = []
    for _, row in df.iterrows():
        raw = row.get(sku_col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        tok = normalize_id_token_for_mapping(str(raw).strip())
        s = clean_sku(tok or raw)
        if not s:
            continue
        s = _strip_pl_sku(s, _map)
        st = row.get(status_col)
        st_str = "" if pd.isna(st) else str(st).strip()
        ld = pd.to_numeric(row.get(lead_col), errors="coerce")
        out_rows.append(
            {
                "OMS_SKU": s,
                "SKU_Sheet_Status": st_str,
                "Lead_Time_From_Sheet": float(ld) if pd.notna(ld) else float("nan"),
                "SKU_Sheet_Closed": is_closed_sku_status(st_str),
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=["OMS_SKU", "SKU_Sheet_Status", "Lead_Time_From_Sheet", "SKU_Sheet_Closed"])
    # Last row wins duplicate SKUs
    out = out.drop_duplicates(subset=["OMS_SKU"], keep="last")
    return out


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

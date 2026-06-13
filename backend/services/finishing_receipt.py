"""Parse Finishing Dept material-issue export and refresh Existing PO pipeline balances."""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .existing_po import (
    dedupe_po_rows_by_sku,
    existing_po_merge_key,
    persist_existing_po_to_disk,
)
from .po_engine import canonical_oms_key

_FINISHING_HEADER_MARKERS = frozenset(
    {
        "designcd",
        "issqty",
        "recqty",
        "balqty",
        "issueno",
        "issdate",
        "jono",
        "jodate",
        "status",
    }
)

_SIZE_TOKENS = frozenset(
    {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"}
)

_FINISHING_META_COLS = (
    "Finishing_Issued",
    "Finishing_Received",
    "Finishing_Balance",
    "Finishing_Issue_No",
    "Finishing_Iss_Date",
    "Finishing_JO_No",
    "Finishing_JO_Date",
    "Finishing_Status",
)

_PIPELINE_COLS = ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total")


def _normalize_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(8, len(raw))):
        row = {_normalize_col(v) for v in raw.iloc[i].tolist()}
        if _FINISHING_HEADER_MARKERS.issubset(row):
            return i
    return None


def _parse_report_date(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        return value.date().isoformat()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        pass
    return s[:10]


def _design_size_to_oms(design_cd: object, size: object) -> str:
    design = str(design_cd or "").strip().upper()
    if not design or design in {"NAN", "NONE"}:
        return ""
    sz = str(size or "").strip().upper()
    if not sz or sz in {"NAN", "NONE"}:
        return existing_po_merge_key(design)
    if sz in _SIZE_TOKENS or re.fullmatch(r"\d*XL", sz):
        return existing_po_merge_key(f"{design}-{sz}")
    return existing_po_merge_key(design)


def _read_finishing_table(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, int]:
    lower = (filename or "").lower()
    bio = io.BytesIO(file_bytes)
    if lower.endswith(".csv"):
        raw = pd.read_csv(bio, header=None, dtype=str)
    else:
        raw = pd.read_excel(bio, sheet_name=0, header=None, dtype=str)
    if raw.empty:
        raise ValueError("Finishing file is empty.")
    header_row = _find_header_row(raw)
    if header_row is None:
        raise ValueError(
            "Could not find Finishing header row (expected DesignCd, IssQty, RecQty, BalQty, Issueno, IssDate)."
        )
    bio.seek(0)
    if lower.endswith(".csv"):
        df = pd.read_csv(bio, header=header_row, dtype=str)
    else:
        df = pd.read_excel(bio, sheet_name=0, header=header_row, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all"), header_row


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    norm = {_normalize_col(c): c for c in df.columns}
    for name in names:
        hit = norm.get(_normalize_col(name))
        if hit:
            return hit
    return None


def parse_finishing_receipt_workbook(
    file_bytes: bytes,
    filename: str,
    *,
    sku_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Parse Yash Gallery Finishing Dept export.

    Returns aggregated rows keyed by OMS_SKU with issued/received/balance quantities
    and the latest issue / JO metadata per SKU.
    """
    raw_df, _header_row = _read_finishing_table(file_bytes, filename)
    design_col = _col(raw_df, "DesignCd", "Design Cd", "Design")
    size_col = _col(raw_df, "Size")
    if not design_col:
        raise ValueError("Finishing file missing DesignCd column.")

    qty_cols = {
        "Finishing_Issued": _col(raw_df, "IssQty", "Iss Qty"),
        "Finishing_Received": _col(raw_df, "RecQty", "Rec Qty"),
        "Finishing_Balance": _col(raw_df, "BalQty", "Bal Qty"),
    }
    if not qty_cols["Finishing_Balance"]:
        raise ValueError("Finishing file missing BalQty column.")

    issue_col = _col(raw_df, "Issueno", "Issue No", "IssueNo")
    iss_date_col = _col(raw_df, "IssDate", "Iss Date")
    jo_no_col = _col(raw_df, "JONo", "JO No")
    jo_date_col = _col(raw_df, "JODate", "JO Date")
    status_col = _col(raw_df, "Status")

    work = raw_df.copy()
    work["_oms_key"] = [
        _design_size_to_oms(d, work.at[i, size_col] if size_col else "")
        for i, d in enumerate(work[design_col])
    ]
    work = work[work["_oms_key"].astype(str).str.len() > 0]
    if work.empty:
        raise ValueError("No valid DesignCd + Size rows found in Finishing file.")

    if sku_mapping:
        mapped = []
        for key in work["_oms_key"]:
            canon = canonical_oms_key(key, sku_mapping)
            mapped.append(existing_po_merge_key(canon or key))
        work["_oms_key"] = mapped

    for out_col, src in qty_cols.items():
        if src:
            work[out_col] = pd.to_numeric(work[src], errors="coerce").fillna(0).astype(int)
        else:
            work[out_col] = 0

    if issue_col:
        work["Finishing_Issue_No"] = work[issue_col].astype(str).str.strip()
    else:
        work["Finishing_Issue_No"] = ""
    if iss_date_col:
        work["Finishing_Iss_Date"] = work[iss_date_col].map(_parse_report_date)
    else:
        work["Finishing_Iss_Date"] = ""
    if jo_no_col:
        work["Finishing_JO_No"] = work[jo_no_col].astype(str).str.strip()
    else:
        work["Finishing_JO_No"] = ""
    if jo_date_col:
        work["Finishing_JO_Date"] = work[jo_date_col].map(_parse_report_date)
    else:
        work["Finishing_JO_Date"] = ""
    if status_col:
        work["Finishing_Status"] = work[status_col].astype(str).str.strip()
    else:
        work["Finishing_Status"] = ""

    agg: dict[str, Any] = {
        "Finishing_Issued": "sum",
        "Finishing_Received": "sum",
        "Finishing_Balance": "sum",
    }
    # Keep the latest issue row metadata (file is usually one day / one issue batch).
    for meta_col in ("Finishing_Issue_No", "Finishing_Iss_Date", "Finishing_JO_No", "Finishing_JO_Date", "Finishing_Status"):
        agg[meta_col] = "last"

    grouped = (
        work.groupby("_oms_key", as_index=False)
        .agg(agg)
        .rename(columns={"_oms_key": "OMS_SKU"})
    )

    report = {
        "filename": filename,
        "rows_read": int(len(work)),
        "skus": int(len(grouped)),
        "issued_units": int(grouped["Finishing_Issued"].sum()),
        "received_units": int(grouped["Finishing_Received"].sum()),
        "balance_units": int(grouped["Finishing_Balance"].sum()),
        "issue_numbers": sorted({s for s in grouped["Finishing_Issue_No"].astype(str) if s and s.lower() not in {"nan", "none"}}),
        "report_date": _parse_report_date(grouped["Finishing_Iss_Date"].dropna().iloc[-1] if not grouped.empty else ""),
        "non_clear_skus": int((grouped["Finishing_Balance"] > 0).sum()),
        "cleared_skus": int((grouped["Finishing_Balance"] <= 0).sum()),
    }
    return grouped, report


def _ensure_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "OMS_SKU" not in out.columns:
        out["OMS_SKU"] = ""
    for col in _PIPELINE_COLS:
        if col not in out.columns:
            out[col] = 0
    for col in _FINISHING_META_COLS:
        if col not in out.columns:
            out[col] = "" if col.endswith(("No", "Date", "Status")) else 0
    return out


def merge_finishing_into_existing_po(
    existing_po_df: pd.DataFrame,
    finishing_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Apply finishing received/balance onto Existing PO pipeline (Balance_to_Dispatch)."""
    ep = _ensure_pipeline_columns(existing_po_df)
    if not ep.empty and "OMS_SKU" in ep.columns:
        ep["OMS_SKU"] = ep["OMS_SKU"].map(existing_po_merge_key)
        ep = ep[ep["OMS_SKU"].astype(str).str.strip().str.len() > 0]
        ep = dedupe_po_rows_by_sku(ep.reset_index(drop=True))

    fin = finishing_df.copy()
    if fin.empty:
        return ep, {"updated_skus": 0, "added_skus": 0, "left_units": 0}

    fin["OMS_SKU"] = fin["OMS_SKU"].map(existing_po_merge_key)

    updated = 0
    added = 0
    left_units = 0

    for _, row in fin.iterrows():
        sku = str(row["OMS_SKU"]).strip()
        if not sku:
            continue
        bal = int(pd.to_numeric(row.get("Finishing_Balance"), errors="coerce") or 0)
        rec = int(pd.to_numeric(row.get("Finishing_Received"), errors="coerce") or 0)
        issued = int(pd.to_numeric(row.get("Finishing_Issued"), errors="coerce") or 0)
        left_units += max(0, bal)

        mask = ep["OMS_SKU"].astype(str).str.strip() == sku
        meta = {
            "Finishing_Issued": issued,
            "Finishing_Received": rec,
            "Finishing_Balance": bal,
            "Finishing_Issue_No": str(row.get("Finishing_Issue_No") or "").strip(),
            "Finishing_Iss_Date": str(row.get("Finishing_Iss_Date") or "").strip(),
            "Finishing_JO_No": str(row.get("Finishing_JO_No") or "").strip(),
            "Finishing_JO_Date": str(row.get("Finishing_JO_Date") or "").strip(),
            "Finishing_Status": str(row.get("Finishing_Status") or "").strip(),
        }

        if mask.any():
            idx = ep.index[mask][0]
            ep.at[idx, "Balance_to_Dispatch"] = max(0, bal)
            for k, v in meta.items():
                ep.at[idx, k] = v
            pending = int(pd.to_numeric(ep.at[idx, "Pending_Cutting"], errors="coerce") or 0)
            ep.at[idx, "PO_Pipeline_Total"] = max(0, pending + max(0, bal))
            updated += 1
        else:
            new_row = {
                "OMS_SKU": sku,
                "PO_Qty_Ordered": max(0, issued),
                "Pending_Cutting": 0,
                "Balance_to_Dispatch": max(0, bal),
                "PO_Pipeline_Total": max(0, bal),
                **meta,
            }
            ep = pd.concat([ep, pd.DataFrame([new_row])], ignore_index=True)
            added += 1

    ep["PO_Pipeline_Total"] = (
        pd.to_numeric(ep.get("Pending_Cutting"), errors="coerce").fillna(0)
        + pd.to_numeric(ep.get("Balance_to_Dispatch"), errors="coerce").fillna(0)
    ).clip(lower=0).astype(int)
    for c in ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch"):
        ep[c] = pd.to_numeric(ep[c], errors="coerce").fillna(0).astype(int)

    ep = dedupe_po_rows_by_sku(ep.reset_index(drop=True))

    return ep, {
        "updated_skus": updated,
        "added_skus": added,
        "left_units": left_units,
        "received_units": int(fin["Finishing_Received"].sum()),
        "issued_units": int(fin["Finishing_Issued"].sum()),
    }


def apply_finishing_receipt_import(
    sess,
    finishing_df: pd.DataFrame,
    report: dict,
    *,
    filename: str,
) -> dict:
    """Persist finishing receipt, refresh existing_po_df, and invalidate PO caches."""
    from ..services.po_raise_remove import invalidate_po_calculate_result
    from ..services.po_shared_cache import invalidate_all_shared_caches

    ep = getattr(sess, "existing_po_df", None)
    if ep is None or not hasattr(ep, "empty"):
        ep = pd.DataFrame()
    prev_name = str(getattr(sess, "finishing_receipt_filename", "") or "").strip()
    file_key = str(filename or "").strip() or "finishing.xls"
    replaced_previous = bool(prev_name)

    merged, apply_stats = merge_finishing_into_existing_po(ep, finishing_df)

    sess.existing_po_df = merged
    sess.finishing_receipt_report = {
        **dict(report),
        **apply_stats,
        "replaced_previous": replaced_previous,
    }
    sess.finishing_receipt_filename = file_key
    sess.finishing_receipt_uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sess.existing_po_generation = int(getattr(sess, "existing_po_generation", 0) or 0) + 1

    invalidate_po_calculate_result(sess)
    invalidate_all_shared_caches()
    persist_existing_po_to_disk(sess)

    left = int(apply_stats.get("left_units") or 0)
    action = "Updated" if replaced_previous else "Applied"
    msg = (
        f"Finishing receipt {action.lower()}: {apply_stats.get('updated_skus', 0):,} SKUs updated, "
        f"{apply_stats.get('added_skus', 0):,} added. "
        f"{left:,} units still at finishing (Balance to Dispatch). "
        "Click Calculate PO to refresh the PO table."
    )
    if replaced_previous:
        msg = f"Re-uploaded {file_key} — no duplicate rows. " + msg
    if left > 0:
        msg += f" ⚠ {int(report.get('non_clear_skus') or 0):,} SKUs have balance left."

    return {
        "ok": True,
        "message": msg,
        "rows": int(len(merged)),
        "skus": int(len(finishing_df)),
        "left_units": left,
        "received_units": int(apply_stats.get("received_units") or 0),
        "issued_units": int(apply_stats.get("issued_units") or 0),
        "parse_report": sess.finishing_receipt_report,
        "existing_po_generation": sess.existing_po_generation,
        "finishing_receipt_uploaded_at": sess.finishing_receipt_uploaded_at,
    }

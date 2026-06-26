"""Detect when a file belongs in a different upload section."""
from __future__ import annotations

import io
import re
from typing import Literal, Optional

import pandas as pd

DocCategory = Literal[
    "daily_inventory_history_matrix",
    "snapshot_inventory",
    "daily_sales",
    "returns",
    "sku_status_lead",
    "existing_po",
    "unknown",
]

UploadTarget = Literal[
    "snapshot_inventory",
    "daily_inventory_history",
    "daily_sales",
    "returns",
    "sku_status_lead",
    "existing_po",
]

_HISTORY_FILENAME_RE = re.compile(
    r"(inventory[\s_\-]*history|daily[\s_\-]*inv(?:entory)?|inv[\s_\-]*matrix|inventory-matrix)",
    re.I,
)
_SNAPSHOT_FILENAME_RE = re.compile(
    r"(^oms\b|\boms[\s_\-]|flipkart|myntra|amazon|ppmp|seller[\s_\-]*inventory|current[\s_\-]*inventory|\.rar$)",
    re.I,
)
_RETURN_FILENAME_RE = re.compile(
    r"(return|seller_returns|tcs_sales_return|last\s*\d+\s*days?\s*return)",
    re.I,
)
_SALES_FILENAME_RE = re.compile(
    r"(shipment|seller.?orders|orders_\d{4}|mtr|tax[\s_\-]*report|b2c|b2b|daily[\s_\-]*order)",
    re.I,
)
_EXISTING_PO_FILENAME_RE = re.compile(
    r"(existing[\s_\-]*po|open[\s_\-]*po|pipeline[\s_\-]*po|po[\s_\-]*pipeline)",
    re.I,
)
_SKU_STATUS_FILENAME_RE = re.compile(
    r"(sku[\s_\-]*status|status[\s_\-]*lead|lead[\s_\-]*time)",
    re.I,
)
_RAR_MAGIC = b"Rar!\x1a\x07\x00"

_TARGET_LABELS: dict[UploadTarget, str] = {
    "snapshot_inventory": "Upload → Daily uploads → Snapshot inventory",
    "daily_inventory_history": "Upload → History & setup → Daily inventory history matrix (PO)",
    "daily_sales": "Upload → Daily uploads → Daily order upload",
    "returns": "Upload → Daily uploads → Returns (for PO)",
    "sku_status_lead": "Upload → History & setup → SKU status & lead time",
    "existing_po": "Upload → History & setup → Existing PO",
}

_CATEGORY_LABELS: dict[DocCategory, str] = {
    "daily_inventory_history_matrix": "wide daily inventory history matrix",
    "snapshot_inventory": "today's snapshot inventory export",
    "daily_sales": "daily sales / shipment report",
    "returns": "returns report",
    "sku_status_lead": "SKU status & lead time sheet",
    "existing_po": "existing PO / pipeline sheet",
    "unknown": "unrecognized file",
}


def _plain(msg: str) -> str:
    return msg.replace("**", "")


def _read_preview_table(raw: bytes, filename: str) -> Optional[pd.DataFrame]:
    fn = (filename or "").lower()
    bio = io.BytesIO(raw)
    try:
        if fn.endswith(".csv"):
            return pd.read_csv(bio, header=None, nrows=14, dtype=str, on_bad_lines="skip")
        if fn.endswith((".xlsx", ".xls")) or raw[:4] == b"PK\x03\x04":
            xl = pd.ExcelFile(bio)
            return pd.read_excel(xl, sheet_name=0, header=None, nrows=14, dtype=str)
    except Exception:
        return None
    return None


def _joined_header_text(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return ""
    parts: list[str] = []
    for ridx in range(min(4, len(df))):
        for val in df.iloc[ridx].tolist():
            s = str(val or "").strip().lower()
            if s and s not in {"nan", "none"}:
                parts.append(s)
    return " | ".join(parts)


def _file_header_text(raw: bytes, filename: str) -> str:
    chunks: list[str] = []
    try:
        chunks.append(raw[:8000].decode("utf-8", errors="ignore").lower())
    except Exception:
        pass
    chunks.append(_joined_header_text(_read_preview_table(raw, filename)))
    return " | ".join(c for c in chunks if c)


def _wide_matrix_date_column_count(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    try:
        from .daily_inventory_history import (
            _build_column_date_map,
            _detect_parent_column,
            _detect_sku_column,
        )

        sku_idx = _detect_sku_column(df)
        if sku_idx is None:
            return 0
        parent_idx = _detect_parent_column(df, sku_idx)
        date_map, _ = _build_column_date_map(df, sku_idx, parent_idx)
        return len(date_map)
    except Exception:
        return 0


def _score_markers(text: str, markers: tuple[str, ...], weight: int = 2) -> int:
    return sum(weight for m in markers if m in text)


def _score_returns(text: str, filename: str) -> int:
    fn = (filename or "").lower()
    score = 0
    if _RETURN_FILENAME_RE.search(fn):
        score += 4
    try:
        from .po_return_import import _is_last_days_return_filename

        if _is_last_days_return_filename(filename):
            score += 5
    except Exception:
        pass
    if "return data" in fn:
        score += 3
    score += _score_markers(
        text,
        (
            "return_created_date",
            "return-date",
            "units refunded",
            "return_id",
            "return approval",
            "returned_qty",
            "return_qty",
            "return quantity",
            "seller_returns",
            "tcs_sales_return",
            "return_created",
        ),
    )
    if "event type" in text and "return" in fn:
        score += 2
    return score


def _score_sales(text: str, filename: str) -> int:
    fn = (filename or "").lower()
    score = 0
    if _SALES_FILENAME_RE.search(fn):
        score += 3
    if re.search(r"(?:^|/)orders_\d{4}-\d{2}-\d{2}", fn):
        score += 3
    score += _score_markers(
        text,
        (
            "amazon-order-id",
            "purchase-date",
            "merchant-order-id",
            "customer shipment date",
            "order_created_date",
            "product_mrp",
            "reason for credit entry",
            "sub order no",
            "sub order id",
            "shipment date",
            "invoice date",
            "transaction type",
            "buyer invoice date",
        ),
    )
    if "transaction type" in text and "return" not in fn:
        score += 1
    return score


def _score_snapshot_inventory(text: str, filename: str, date_cols: int) -> int:
    fn = (filename or "").lower()
    score = 0
    if _SNAPSHOT_FILENAME_RE.search(fn):
        score += 2
    score += _score_markers(
        text,
        (
            "buffer stock",
            "total inv",
            "inventory count",
            "ending warehouse balance",
            "available to promise",
            "live on website",
            "seller sku code",
        ),
    )
    if date_cols < 2 and ("item skucode" in text or "buffer stock" in text):
        score += 3
    if fn.endswith(".rar") and "return" not in fn and "mtr" not in fn:
        score += 1
    return score


def _score_existing_po(text: str, filename: str) -> int:
    score = 0
    if _EXISTING_PO_FILENAME_RE.search(filename or ""):
        score += 4
    score += _score_markers(
        text,
        (
            "po_pipeline",
            "pipeline total",
            "pending cutting",
            "balance to dispatch",
            "balance to despatch",
            "po qty ordered",
            "qty ordered",
            "open po",
        ),
    )
    return score


def _score_sku_status(text: str, filename: str) -> int:
    score = 0
    if _SKU_STATUS_FILENAME_RE.search(filename or ""):
        score += 4
    score += _score_markers(
        text,
        (
            "lead time",
            "leadtime",
            "lead_time",
            "manufacturing lead",
            "factory lead",
            "sku status",
            "sku_status",
            "production lead",
        ),
    )
    return score


def classify_upload_document(raw: bytes, filename: str = "") -> dict:
    """Classify an upload for routing / wrong-target warnings."""
    fn = (filename or "").strip()
    fn_l = fn.lower()
    preview = _read_preview_table(raw, fn)
    text = _file_header_text(raw, fn)
    date_cols = _wide_matrix_date_column_count(preview)

    if raw[:6] == _RAR_MAGIC or fn_l.endswith(".rar"):
        if _RETURN_FILENAME_RE.search(fn) or "return data" in fn_l:
            return {"category": "returns", "confidence": "high", "date_columns": 0}
        if _SALES_FILENAME_RE.search(fn) or "mtr" in fn_l:
            return {"category": "daily_sales", "confidence": "medium", "date_columns": 0}
        if "inventory" in fn_l or _score_snapshot_inventory(text, fn, date_cols) >= 3:
            return {"category": "snapshot_inventory", "confidence": "medium", "date_columns": 0}
        return {"category": "unknown", "confidence": "low", "date_columns": 0}

    history_fn = bool(_HISTORY_FILENAME_RE.search(fn))
    if date_cols >= 3 or (date_cols >= 2 and history_fn):
        return {
            "category": "daily_inventory_history_matrix",
            "confidence": "high" if date_cols >= 3 else "medium",
            "date_columns": date_cols,
        }

    ret = _score_returns(text, fn)
    sales = _score_sales(text, fn)
    inv = _score_snapshot_inventory(text, fn, date_cols)
    po = _score_existing_po(text, fn)
    status = _score_sku_status(text, fn)

    scores: list[tuple[int, DocCategory]] = [
        (ret, "returns"),
        (sales, "daily_sales"),
        (inv, "snapshot_inventory"),
        (po, "existing_po"),
        (status, "sku_status_lead"),
    ]
    scores.sort(key=lambda x: -x[0])
    best_score, best_cat = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0

    if best_score >= 4 and best_score >= second_score + 2:
        conf = "high" if best_score >= 6 else "medium"
        return {"category": best_cat, "confidence": conf, "date_columns": date_cols}

    if history_fn and date_cols >= 1:
        return {
            "category": "daily_inventory_history_matrix",
            "confidence": "medium",
            "date_columns": date_cols,
        }

    if inv >= 3 and date_cols < 2:
        return {"category": "snapshot_inventory", "confidence": "medium", "date_columns": date_cols}

    return {"category": "unknown", "confidence": "low", "date_columns": date_cols}


def sniff_upload_document(raw: bytes, filename: str = "") -> dict:
    """Backward-compatible alias used by inventory history checks."""
    cls = classify_upload_document(raw, filename)
    kind = cls["category"]
    if kind == "daily_inventory_history_matrix":
        filename_hint = "history"
    elif kind == "snapshot_inventory":
        filename_hint = "snapshot"
    else:
        filename_hint = None
    legacy_kind = kind if kind in {
        "daily_inventory_history_matrix",
        "snapshot_inventory",
        "unknown",
    } else "unknown"
    if kind == "snapshot_inventory":
        legacy_kind = "snapshot_inventory_oms"
    return {
        "kind": legacy_kind,
        "confidence": cls["confidence"],
        "date_columns": cls.get("date_columns", 0),
        "filename_hint": filename_hint,
        "category": kind,
    }


_CATEGORY_TO_TARGET: dict[DocCategory, UploadTarget] = {
    "daily_inventory_history_matrix": "daily_inventory_history",
    "snapshot_inventory": "snapshot_inventory",
    "daily_sales": "daily_sales",
    "returns": "returns",
    "sku_status_lead": "sku_status_lead",
    "existing_po": "existing_po",
}


def _wrong_target_message(
    filename: str,
    detected: DocCategory,
    correct_target: UploadTarget,
) -> str:
    return (
        f"“{filename}” looks like a {_CATEGORY_LABELS.get(detected, detected)}, not the file type "
        f"for this upload card. Use **{_TARGET_LABELS[correct_target]}** instead."
    )


def check_upload_target(target: UploadTarget, raw: bytes, filename: str) -> Optional[str]:
    """Return a user-visible error when *raw* is clearly for a different upload section."""
    if not raw:
        return "Empty file."
    cls = classify_upload_document(raw, filename)
    cat = cls["category"]
    conf = cls["confidence"]
    if cat == "unknown" or conf == "low":
        return None

    blocked: dict[UploadTarget, set[DocCategory]] = {
        "snapshot_inventory": {
            "daily_inventory_history_matrix",
            "returns",
            "daily_sales",
            "sku_status_lead",
            "existing_po",
        },
        "daily_inventory_history": {
            "snapshot_inventory",
            "returns",
            "daily_sales",
            "sku_status_lead",
            "existing_po",
        },
        "daily_sales": {
            "daily_inventory_history_matrix",
            "returns",
            "snapshot_inventory",
            "sku_status_lead",
            "existing_po",
        },
        "returns": {
            "daily_inventory_history_matrix",
            "snapshot_inventory",
            "daily_sales",
            "sku_status_lead",
            "existing_po",
        },
        "sku_status_lead": {
            "daily_inventory_history_matrix",
            "snapshot_inventory",
            "returns",
            "daily_sales",
            "existing_po",
        },
        "existing_po": {
            "daily_inventory_history_matrix",
            "snapshot_inventory",
            "returns",
            "daily_sales",
            "sku_status_lead",
        },
    }

    # Flipkart Sales Report is valid on both sales and returns — do not block ambiguous case.
    if cat == "daily_sales" and target == "returns" and conf != "high":
        return None
    if cat == "returns" and target == "daily_sales" and conf != "high":
        return None

    if cat in blocked.get(target, set()):
        correct = _CATEGORY_TO_TARGET.get(cat)
        if correct:
            return _plain(_wrong_target_message(filename, cat, correct))
    return None


def check_files_for_upload_target(
    target: UploadTarget,
    file_parts: list[tuple[str, bytes]],
) -> Optional[str]:
    for fname, raw in file_parts:
        msg = check_upload_target(target, raw, fname)
        if msg:
            return msg
    return None


# Legacy helpers (inventory history ↔ snapshot)
def misplaced_daily_inventory_history_message(filename: str, sniff: dict) -> str:
    cols = int(sniff.get("date_columns") or 0)
    extra = f" ({cols} daily date columns detected)" if cols >= 2 else ""
    return _plain(
        f"“{filename}” looks like the wide Daily Inventory History matrix{extra}, "
        "not today's snapshot inventory. "
        f"Use **{_TARGET_LABELS['daily_inventory_history']}**. "
        "Snapshot inventory is only for today's OMS / marketplace files (RAR, Flipkart, Myntra CSVs)."
    )


def misplaced_snapshot_inventory_message(filename: str, sniff: dict) -> str:
    return _plain(_wrong_target_message(filename, "snapshot_inventory", "snapshot_inventory"))


def check_files_for_snapshot_upload(file_parts: list[tuple[str, bytes]]) -> Optional[str]:
    return check_files_for_upload_target("snapshot_inventory", file_parts)


def check_file_for_daily_inventory_history(raw: bytes, filename: str) -> Optional[str]:
    msg = check_upload_target("daily_inventory_history", raw, filename)
    if msg:
        return msg
    cls = classify_upload_document(raw, filename)
    if cls["category"] == "daily_inventory_history_matrix":
        return None
    if cls.get("date_columns", 0) >= 3:
        return None
    return _plain(
        f"Could not recognize “{filename}” as a wide daily inventory history matrix. "
        "Expected rows = SKUs and columns = daily snapshot dates (e.g. 28-5-26, 29-5-26). "
        f"For today's OMS / marketplace stock, use **{_TARGET_LABELS['snapshot_inventory']}** instead."
    )

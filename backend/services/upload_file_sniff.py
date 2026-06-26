"""Detect when a file belongs in a different upload section (snapshot vs history matrix)."""
from __future__ import annotations

import io
import re
from typing import Literal, Optional

import pandas as pd

UploadDocKind = Literal[
    "daily_inventory_history_matrix",
    "snapshot_inventory_oms",
    "snapshot_inventory_marketplace",
    "rar_archive",
    "unknown",
]

_HISTORY_FILENAME_RE = re.compile(
    r"(inventory[\s_\-]*history|daily[\s_\-]*inv(?:entory)?|inv[\s_\-]*matrix|inventory-matrix)",
    re.I,
)
_SNAPSHOT_FILENAME_RE = re.compile(
    r"(^oms\b|\boms[\s_\-]|flipkart|myntra|amazon|ppmp|seller[\s_\-]*inventory|current[\s_\-]*inventory|\.rar$)",
    re.I,
)
_RAR_MAGIC = b"Rar!\x1a\x07\x00"


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


def sniff_upload_document(raw: bytes, filename: str = "") -> dict:
    """Classify an upload for routing / wrong-target warnings."""
    fn = (filename or "").strip()
    fn_l = fn.lower()
    if raw[:6] == _RAR_MAGIC or fn_l.endswith(".rar"):
        return {
            "kind": "rar_archive",
            "confidence": "high",
            "date_columns": 0,
            "filename_hint": "snapshot",
        }

    text = _joined_header_text(_read_preview_table(raw, fn))
    date_cols = _wide_matrix_date_column_count(_read_preview_table(raw, fn))

    marketplace = False
    if text:
        if (
            "seller sku code" in text
            or ("style id" in text and "inventory count" in text)
            or "live on website" in text
            or ("msku" in text and "ending warehouse balance" in text)
        ):
            marketplace = True

    history_fn = bool(_HISTORY_FILENAME_RE.search(fn))
    snapshot_fn = bool(_SNAPSHOT_FILENAME_RE.search(fn))

    if date_cols >= 3 or (date_cols >= 2 and history_fn):
        return {
            "kind": "daily_inventory_history_matrix",
            "confidence": "high" if date_cols >= 3 else "medium",
            "date_columns": date_cols,
            "filename_hint": "history",
        }

    if marketplace and date_cols < 2:
        return {
            "kind": "snapshot_inventory_marketplace",
            "confidence": "high",
            "date_columns": date_cols,
            "filename_hint": "snapshot",
        }

    if text and ("buffer stock" in text or "total inv" in text) and date_cols < 2:
        return {
            "kind": "snapshot_inventory_oms",
            "confidence": "high" if snapshot_fn or date_cols == 0 else "medium",
            "date_columns": date_cols,
            "filename_hint": "snapshot",
        }

    if history_fn and date_cols >= 1:
        return {
            "kind": "daily_inventory_history_matrix",
            "confidence": "medium",
            "date_columns": date_cols,
            "filename_hint": "history",
        }

    if snapshot_fn:
        return {
            "kind": "snapshot_inventory_oms",
            "confidence": "low",
            "date_columns": date_cols,
            "filename_hint": "snapshot",
        }

    return {
        "kind": "unknown",
        "confidence": "low",
        "date_columns": date_cols,
        "filename_hint": None,
    }


def misplaced_daily_inventory_history_message(filename: str, sniff: dict) -> str:
    cols = int(sniff.get("date_columns") or 0)
    extra = f" ({cols} daily date columns detected)" if cols >= 2 else ""
    return (
        f"“{filename}” looks like the wide **Daily Inventory History** matrix{extra}, "
        "not today's snapshot inventory. "
        "Upload it under **Upload → History & setup → Daily inventory history matrix (PO)**. "
        "Snapshot inventory is only for today's OMS / marketplace files (RAR, Flipkart, Myntra CSVs)."
    )


def misplaced_snapshot_inventory_message(filename: str, sniff: dict) -> str:
    kind = sniff.get("kind")
    if kind == "rar_archive":
        detail = "RAR archives belong in Snapshot inventory (Daily uploads tab)."
    elif kind == "snapshot_inventory_marketplace":
        detail = "This looks like a marketplace inventory export (Flipkart / Myntra / Amazon)."
    else:
        detail = "This looks like a single-day OMS snapshot, not a multi-day history matrix."
    return (
        f"“{filename}” is not a wide daily history matrix. {detail} "
        "Upload it under **Upload → Daily uploads → Snapshot inventory** (or the matching marketplace card)."
    )


def check_files_for_snapshot_upload(file_parts: list[tuple[str, bytes]]) -> Optional[str]:
    """Return an error message when history-matrix files were dropped on snapshot inventory."""
    for fname, raw in file_parts:
        if not raw:
            continue
        sniff = sniff_upload_document(raw, fname)
        if sniff["kind"] == "daily_inventory_history_matrix":
            return misplaced_daily_inventory_history_message(fname, sniff)
    return None


def check_file_for_daily_inventory_history(raw: bytes, filename: str) -> Optional[str]:
    """Return an error message when a snapshot file was dropped on history matrix upload."""
    if not raw:
        return "Empty file."
    sniff = sniff_upload_document(raw, filename)
    kind = sniff["kind"]
    if kind == "daily_inventory_history_matrix":
        return None
    if kind in ("rar_archive", "snapshot_inventory_marketplace", "snapshot_inventory_oms"):
        return misplaced_snapshot_inventory_message(filename, sniff)
    if sniff.get("date_columns", 0) >= 3:
        return None
    return (
        f"Could not recognize “{filename}” as a wide daily inventory history matrix. "
        "Expected rows = SKUs and columns = daily snapshot dates (e.g. 28-5-26, 29-5-26). "
        "For today's OMS / marketplace stock, use **Daily uploads → Snapshot inventory** instead."
    )

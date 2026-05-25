"""Serialize PO calculate result pages for ``GET /po/calculate/result``."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .po_result_spill import has_spill, read_page, spill_row_count

_DEFAULT_PAGE = int(os.environ.get("PO_CALC_RESULT_PAGE_SIZE", "400"))
_MAX_PAGE = int(os.environ.get("PO_CALC_RESULT_PAGE_MAX", "800"))


def default_page_size() -> int:
    return max(50, min(_MAX_PAGE, _DEFAULT_PAGE))


def _cell_py(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (pd.Timestamp,)):
        return str(v.date()) if hasattr(v, "date") else str(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return "" if np.isnan(f) or np.isinf(f) else f
    return v


def _matrix_from_df(chunk: pd.DataFrame, columns: list[str]) -> list[list[Any]]:
    work = chunk.reindex(columns=columns, fill_value="")
    return [
        [_cell_py(v) for v in row]
        for row in work.itertuples(index=False, name=None)
    ]


def build_result_page(
    *,
    session_id: str | None,
    po_df: pd.DataFrame | None,
    meta: dict,
    offset: int,
    limit: int,
    compact: bool = True,
) -> dict[str, Any]:
    """One page of PO rows — prefers parquet spill when available."""
    lim = max(1, min(int(limit or default_page_size()), _MAX_PAGE))
    off = max(0, int(offset))
    columns: list[str] = []

    if session_id and has_spill(session_id):
        total = spill_row_count(session_id)
        if meta.get("columns"):
            columns = list(meta["columns"])
        spilled = read_page(session_id, off, lim, columns=columns or None)
        if spilled is not None:
            if not columns and spilled.get("columns"):
                columns = list(spilled["columns"])
            payload: dict[str, Any] = {
                "ok": True,
                "columns": columns,
                "offset": spilled["offset"],
                "limit": spilled["limit"],
                "total": spilled["total"],
                "has_more": spilled["has_more"],
                "sales_through": meta.get("sales_through"),
                "planning_date": meta.get("planning_date"),
                "raise_ledger_rows": meta.get("raise_ledger_rows"),
                "ledger_auto_import": meta.get("ledger_auto_import"),
            }
            if compact:
                payload["rows_matrix"] = spilled.get("rows_matrix") or []
            else:
                cols = payload["columns"] or []
                payload["rows"] = [
                    dict(zip(cols, row))
                    for row in (spilled.get("rows_matrix") or [])
                ]
            return payload

    if po_df is None or not hasattr(po_df, "empty") or po_df.empty:
        rows = meta.get("rows") or []
        return {
            **meta,
            "ok": True,
            "rows": rows,
            "offset": 0,
            "limit": len(rows),
            "total": len(rows),
            "has_more": False,
        }

    columns = list(po_df.columns)
    total = int(len(po_df))
    end = min(off + lim, total)
    chunk = po_df.iloc[off:end]
    payload = {
        "ok": True,
        "columns": columns,
        "offset": off,
        "limit": lim,
        "total": total,
        "has_more": end < total,
        "sales_through": meta.get("sales_through"),
        "planning_date": meta.get("planning_date"),
        "raise_ledger_rows": meta.get("raise_ledger_rows"),
        "ledger_auto_import": meta.get("ledger_auto_import"),
    }
    if compact:
        payload["rows_matrix"] = _matrix_from_df(chunk, columns)
    else:
        payload["rows"] = chunk.fillna("").astype(object).to_dict(orient="records")
    return payload

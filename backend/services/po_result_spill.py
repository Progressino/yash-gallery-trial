"""Disk spill for large PO calculate results — fast paginated reads without huge JSON builds."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _result_dir() -> Path:
    """Writable directory for spilled PO parquet (env override or dev fallback)."""
    raw = (os.environ.get("PO_RESULT_SPILL_DIR") or "").strip()
    if raw:
        p = Path(raw)
    else:
        p = Path("/data/po_results")
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        dev = Path("./po_results_dev")
        dev.mkdir(parents=True, exist_ok=True)
        return dev


def _path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (session_id or ""))[:64]
    return _result_dir() / f"{safe}.parquet"


def clear_spill(session_id: str | None) -> None:
    if not session_id:
        return
    try:
        p = _path(session_id)
        if p.is_file():
            p.unlink()
    except Exception:
        _log.exception("clear_po_result_spill failed")


def spill_df(session_id: str, df: pd.DataFrame) -> None:
    """Write PO result to parquet for paginated API reads."""
    if not session_id or df is None or not hasattr(df, "empty") or df.empty:
        return
    try:
        tmp = _path(session_id).with_suffix(".parquet.tmp")
        final = _path(session_id)
        df.to_parquet(tmp, index=False)
        tmp.replace(final)
        _log.info("PO result spilled (%s rows) session=%s", len(df), session_id[:8])
    except Exception:
        _log.exception("spill_po_result_df failed session=%s", session_id[:8] if session_id else "?")


def has_spill(session_id: str | None) -> bool:
    return bool(session_id and _path(session_id).is_file())


def spill_row_count(session_id: str) -> int:
    if not session_id or not has_spill(session_id):
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(_path(session_id)).metadata.num_rows)
    except Exception:
        return 0


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


def read_page(
    session_id: str,
    offset: int,
    limit: int,
    *,
    columns: list[str] | None = None,
) -> dict[str, Any] | None:
    """Read one page from spilled parquet. Returns None if no spill file."""
    path = _path(session_id)
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        total = int(table.num_rows)
        off = max(0, int(offset))
        lim = max(1, min(int(limit), 800))
        end = min(off + lim, total)
        cols = list(columns or table.column_names)
        if off >= total:
            return {
                "columns": cols,
                "rows_matrix": [],
                "offset": off,
                "limit": lim,
                "total": total,
                "has_more": False,
            }
        sub = table.slice(off, end - off)
        df = sub.to_pandas()
        df = df.reindex(columns=cols, fill_value="")
        matrix = [
            [_cell_py(v) for v in row]
            for row in df.itertuples(index=False, name=None)
        ]
        return {
            "columns": cols,
            "rows_matrix": matrix,
            "offset": off,
            "limit": lim,
            "total": total,
            "has_more": end < total,
        }
    except Exception:
        _log.exception("read_po_result_page failed session=%s", session_id[:8])
        return None


def cleanup_old_spills(max_age_hours: int = 48) -> int:
    """Best-effort removal of stale spill files."""
    root = _result_dir()
    if not root.is_dir():
        return 0
    import time

    cutoff = time.time() - max(0, int(max_age_hours)) * 3600
    removed = 0
    for p in root.glob("*.parquet"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            continue
    return removed

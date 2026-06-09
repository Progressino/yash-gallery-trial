"""In-memory PO calculate job status (survives slow session middleware on poll requests)."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def set_po_job(session_id: str, **fields: Any) -> None:
    if not session_id:
        return
    with _lock:
        row = dict(_jobs.get(session_id) or {})
        row.update(fields)
        row["updated_at"] = time.time()
        _jobs[session_id] = row


def get_po_job(session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    with _lock:
        return dict(_jobs.get(session_id) or {})


def clear_po_job(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        _jobs.pop(session_id, None)


def po_job_is_stale(session_id: str, *, max_idle_sec: float = 1200.0) -> bool:
    """True when a running job has not updated progress recently (likely crashed or OOM)."""
    if not session_id:
        return False
    with _lock:
        row = _jobs.get(session_id) or {}
    if str(row.get("status") or "") != "running":
        return False
    updated = float(row.get("updated_at") or 0)
    if updated <= 0:
        return False
    return (time.time() - updated) >= max_idle_sec

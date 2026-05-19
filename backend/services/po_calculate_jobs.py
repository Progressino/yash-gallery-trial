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

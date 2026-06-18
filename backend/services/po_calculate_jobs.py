"""In-memory PO calculate jobs keyed by job_id (survives slow session middleware on polls)."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_latest_by_session: dict[str, str] = {}


def create_po_job(session_id: str, **fields: Any) -> str:
    """Create a new PO job and return its id."""
    if not session_id:
        return ""
    job_id = str(fields.pop("job_id", "") or uuid.uuid4().hex[:12])
    with _lock:
        row = {"job_id": job_id, "session_id": session_id, **fields}
        row["updated_at"] = time.time()
        _jobs[job_id] = row
        _latest_by_session[session_id] = job_id
    return job_id


def set_po_job(job_id: str, **fields: Any) -> None:
    if not job_id:
        return
    with _lock:
        row = dict(_jobs.get(job_id) or {})
        if not row:
            return
        row.update(fields)
        row["job_id"] = job_id
        row["updated_at"] = time.time()
        _jobs[job_id] = row
        sid = str(row.get("session_id") or "")
        if sid:
            _latest_by_session[sid] = job_id


def get_po_job_by_id(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {}
    with _lock:
        return dict(_jobs.get(job_id) or {})


def get_latest_job_id(session_id: str) -> str:
    if not session_id:
        return ""
    with _lock:
        return str(_latest_by_session.get(session_id) or "")


def get_po_job(session_id: str) -> dict[str, Any]:
    """Latest job for a browser session (legacy poll path)."""
    job_id = get_latest_job_id(session_id)
    if not job_id:
        return {}
    return get_po_job_by_id(job_id)


def clear_po_job(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        job_id = _latest_by_session.pop(session_id, None)
        if job_id:
            _jobs.pop(job_id, None)


def po_job_is_stale(job_id: str, *, max_idle_sec: float = 1200.0) -> bool:
    """True when a running job has not updated progress recently (likely crashed or OOM)."""
    if not job_id:
        return False
    with _lock:
        row = _jobs.get(job_id) or {}
    if str(row.get("status") or "") != "running":
        return False
    updated = float(row.get("updated_at") or 0)
    if updated <= 0:
        return False
    progress = int(row.get("progress") or 0)
    idle_limit = max_idle_sec
    if progress <= 10:
        idle_limit = min(idle_limit, 180.0)
    elif progress <= 30:
        idle_limit = min(idle_limit, 600.0)
    return (time.time() - updated) >= idle_limit


def resolve_job_id(job_or_session_id: str) -> Optional[str]:
    """Accept job_id or session_id and return job_id when resolvable."""
    if not job_or_session_id:
        return None
    row = get_po_job_by_id(job_or_session_id)
    if row:
        return job_or_session_id
    latest = get_latest_job_id(job_or_session_id)
    return latest or None

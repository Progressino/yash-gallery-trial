"""PO quarterly jobs — one shared server build, real progress."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def set_quarterly_job(session_id: str, **fields: Any) -> None:
    if not session_id:
        return
    with _lock:
        row = dict(_jobs.get(session_id) or {})
        row.update(fields)
        row["updated_at"] = time.time()
        _jobs[session_id] = row


def get_quarterly_job(session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    with _lock:
        return dict(_jobs.get(session_id) or {})


def clear_quarterly_job(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        _jobs.pop(session_id, None)


def _sync_job_from_shared(session_id: str, key: tuple) -> None:
    from .po_quarterly_cache import get_shared_quarterly, quarterly_build_status

    shared = get_shared_quarterly(key)
    if shared and shared.get("loaded"):
        set_quarterly_job(
            session_id,
            status="ready",
            progress=100,
            message="",
            result=shared,
        )
        return
    st = quarterly_build_status()
    if st.get("building"):
        set_quarterly_job(
            session_id,
            status="running",
            progress=int(st.get("progress") or 10),
            message=str(st.get("message") or "Loading quarterly history…"),
        )


def start_quarterly_background(
    session_id: str,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> bool:
    from .po_quarterly_cache import (
        get_shared_quarterly,
        quarterly_build_status,
        start_shared_quarterly_build,
    )
    from .po_quarterly_warmup import build_quarterly_payload, quarterly_cache_key

    if not session_id:
        return False

    key = quarterly_cache_key(group_by_parent, n_quarters)
    shared = get_shared_quarterly(key)
    if shared and shared.get("loaded"):
        set_quarterly_job(session_id, status="ready", progress=100, result=shared)
        return False

    st = quarterly_build_status()
    if st.get("building"):
        _sync_job_from_shared(session_id, key)
        return False

    cur = get_quarterly_job(session_id)
    if cur.get("status") == "running":
        return False

    def _build(progress_cb):
        from ..session import store

        sess = store.get(session_id)
        if sess is None:
            raise RuntimeError("Session expired")
        return build_quarterly_payload(
            sess,
            group_by_parent=group_by_parent,
            n_quarters=n_quarters,
            progress_cb=progress_cb,
        )

    started = start_shared_quarterly_build(key, _build)
    if started:
        set_quarterly_job(
            session_id,
            status="running",
            progress=8,
            message="Building quarterly history (shared cache)…",
        )

        def _poll() -> None:
            for _ in range(600):
                time.sleep(2)
                _sync_job_from_shared(session_id, key)
                job = get_quarterly_job(session_id)
                if job.get("status") in ("ready", "error"):
                    break

        threading.Thread(
            target=_poll,
            daemon=True,
            name=f"po-qtr-poll-{session_id[:8]}",
        ).start()
    return started

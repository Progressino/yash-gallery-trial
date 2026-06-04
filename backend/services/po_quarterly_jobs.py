"""Background PO quarterly history build (avoids proxy timeouts on first load)."""
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


def start_quarterly_background(
    session_id: str,
    *,
    group_by_parent: bool = False,
    n_quarters: int = 8,
) -> bool:
    """Start daemon thread; returns False if a job is already running for this session."""
    if not session_id:
        return False
    cur = get_quarterly_job(session_id)
    if cur.get("status") == "running":
        return False

    def _run() -> None:
        from ..session import store
        from .po_quarterly_warmup import (
            build_quarterly_payload,
            quarterly_cache_key,
        )

        sess = store.get(session_id)
        if sess is None:
            set_quarterly_job(
                session_id,
                status="error",
                progress=0,
                message="Session expired — refresh and try again.",
            )
            return
        set_quarterly_job(
            session_id,
            status="running",
            progress=12,
            message="Loading sales history for quarterly columns…",
        )
        try:
            set_quarterly_job(session_id, progress=35, message="Merging platform uploads…")
            result = build_quarterly_payload(
                sess,
                group_by_parent=group_by_parent,
                n_quarters=n_quarters,
            )
            key = quarterly_cache_key(group_by_parent, n_quarters)
            sess._quarterly_cache[key] = result
            if result.get("loaded") and result.get("rows"):
                set_quarterly_job(
                    session_id,
                    status="ready",
                    progress=100,
                    message="",
                    result=result,
                )
            else:
                set_quarterly_job(
                    session_id,
                    status="error",
                    progress=0,
                    message="No quarterly data — build Sales first (upload platforms).",
                    result=result,
                )
        except Exception as exc:
            logger.exception("quarterly background job failed session=%s", session_id[:8])
            set_quarterly_job(
                session_id,
                status="error",
                progress=0,
                message=str(exc) or "Quarterly build failed",
            )

    set_quarterly_job(
        session_id,
        status="running",
        progress=5,
        message="Starting quarterly history build…",
    )
    threading.Thread(
        target=_run,
        daemon=True,
        name=f"po-qtr-{session_id[:8]}",
    ).start()
    return True

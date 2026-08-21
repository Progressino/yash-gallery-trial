"""Single-flight session hydration from shared warm cache (one worker per session)."""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable

_log = logging.getLogger(__name__)

_meta = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}
_session_lock_started: dict[str, float] = {}
# If a hydrate worker dies without releasing, clear after this many seconds.
_STALE_HYDRATE_SEC = 90.0


class HydrateSchedule(str, Enum):
    READY = "ready"
    SCHEDULED = "scheduled"
    INFLIGHT = "inflight"
    NO_SESSION = "no_session"


def _lock_for(session_id: str) -> threading.Lock:
    with _meta:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def _force_clear_stale_lock(session_id: str) -> bool:
    """Release a hydrate lock stuck longer than ``_STALE_HYDRATE_SEC``."""
    with _meta:
        started = _session_lock_started.get(session_id)
        lock = _session_locks.get(session_id)
        if not lock or started is None:
            return False
        age = time.monotonic() - started
        if age < _STALE_HYDRATE_SEC:
            return False
        try:
            # Replace the stuck lock so new hydrates can proceed.
            _session_locks[session_id] = threading.Lock()
            _session_lock_started.pop(session_id, None)
            _log.warning(
                "cleared stale session hydrate lock session=%s age=%.0fs",
                session_id[:8],
                age,
            )
            return True
        except Exception:
            _log.exception("failed clearing stale hydrate lock session=%s", session_id[:8])
            return False


def session_hydrate_inflight(session_id: str) -> bool:
    """True when another thread holds the per-session hydration lock."""
    if not session_id:
        return False
    _force_clear_stale_lock(session_id)
    lock = _lock_for(session_id)
    acquired = lock.acquire(blocking=False)
    if acquired:
        lock.release()
        return False
    return True


def session_warm_hydration_complete(sess) -> bool:
    """True when warm-cache copy into this session is not needed."""
    if sess is None:
        return False
    try:
        import backend.main as _main

        if getattr(sess, "pause_auto_data_restore", False):
            return False
        if _main.session_needs_operational_data(sess):
            return False
        if _main.session_needs_warm_cache_topup(sess):
            return False
        return True
    except Exception:
        return False


def schedule_session_hydrate(
    session_id: str,
    worker: Callable[[str], None],
    *,
    executor,
) -> HydrateSchedule:
    """
    Queue at most one hydration worker per session.

    Concurrent callers receive INFLIGHT; completed sessions receive READY.
    """
    if not session_id:
        return HydrateSchedule.NO_SESSION

    from ..session import store

    _force_clear_stale_lock(session_id)

    sess = store.get(session_id)
    if sess is not None and session_warm_hydration_complete(sess):
        return HydrateSchedule.READY

    lock = _lock_for(session_id)
    if not lock.acquire(blocking=False):
        return HydrateSchedule.INFLIGHT

    try:
        with _meta:
            _session_lock_started[session_id] = time.monotonic()
        sess = store.get(session_id)
        if sess is not None and session_warm_hydration_complete(sess):
            return HydrateSchedule.READY

        def _wrapped() -> None:
            try:
                worker(session_id)
            except Exception:
                _log.exception("session hydrate worker failed session=%s", session_id[:8])
            finally:
                with _meta:
                    _session_lock_started.pop(session_id, None)
                lock.release()

        executor.submit(_wrapped)
        return HydrateSchedule.SCHEDULED
    except Exception:
        with _meta:
            _session_lock_started.pop(session_id, None)
        lock.release()
        raise


def run_session_hydrate_blocking(
    session_id: str,
    fn: Callable[[], None],
    *,
    check_ready: Callable[[], bool] | None = None,
) -> bool:
    """
    Run ``fn`` under the session hydration lock (blocking).

    Returns True if ``fn`` ran, False if the session was already ready.
    """
    if not session_id:
        return False
    _force_clear_stale_lock(session_id)
    lock = _lock_for(session_id)
    with lock:
        with _meta:
            _session_lock_started[session_id] = time.monotonic()
        try:
            if check_ready and check_ready():
                return False
            fn()
            return True
        finally:
            with _meta:
                _session_lock_started.pop(session_id, None)

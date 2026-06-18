"""Tests for single-flight session hydration lock."""
from __future__ import annotations

import threading
import time

from backend.services import session_hydrate as sh


def test_schedule_session_hydrate_single_flight():
    calls: list[str] = []
    done = threading.Event()

    def worker(session_id: str) -> None:
        calls.append(session_id)
        time.sleep(0.05)
        done.set()

    class _Exec:
        def submit(self, fn):
            threading.Thread(target=fn, daemon=True).start()

    sid = "sess-test-1"
    s1 = sh.schedule_session_hydrate(sid, worker, executor=_Exec())
    s2 = sh.schedule_session_hydrate(sid, worker, executor=_Exec())
    assert s1 == sh.HydrateSchedule.SCHEDULED
    assert s2 == sh.HydrateSchedule.INFLIGHT
    done.wait(timeout=2.0)
    assert calls == [sid]


def test_session_hydrate_inflight_while_locked():
    sid = "sess-test-2"
    lock = sh._lock_for(sid)
    lock.acquire()
    try:
        assert sh.session_hydrate_inflight(sid) is True
    finally:
        lock.release()
    assert sh.session_hydrate_inflight(sid) is False

"""Session middleware must not block the event loop during PostgreSQL session restore."""

import time
import threading


def test_upload_does_not_block_event_loop_during_pg_restore(monkeypatch):
    """
    Simulate a slow PG restore (2s). A concurrent /api/health request must not wait on it.
    /api/health is session-lightweight; the slow restore is triggered via coverage instead.
    """
    barrier = threading.Barrier(2, timeout=5)
    restore_started = threading.Event()

    def _slow_load(session_id: str):
        restore_started.set()
        time.sleep(2)
        return None

    monkeypatch.setattr(
        "backend.db.forecast_session_pg.load_session_from_pg",
        _slow_load,
    )
    monkeypatch.setattr(
        "backend.db.forecast_session_pg.pg_session_persist_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.session._schedule_pg_session_restore",
        lambda sid: None,
    )

    import backend.session as sess_mod
    monkeypatch.setattr(sess_mod, "_PG_RESTORE_TIMEOUT_SEC", 2.0)

    from starlette.testclient import TestClient
    from backend.main import app

    client_stale = TestClient(app, raise_server_exceptions=False)
    client_stale.cookies.set("session_id", "stale-session-for-blocking-test")

    from backend.routers.auth import create_token
    token = create_token("tester", role="Admin")
    client_stale.cookies.set("auth_token", token)

    health_elapsed: list[float] = []

    def _do_stale_request():
        client_stale.get("/api/data/coverage", params={"light": True})

    t = threading.Thread(target=_do_stale_request, daemon=True)
    t.start()

    restore_started.wait(timeout=3)

    client_fresh = TestClient(app, raise_server_exceptions=False)
    t0 = time.monotonic()
    r = client_fresh.get("/api/health")
    health_elapsed.append(time.monotonic() - t0)
    t.join(timeout=5)

    assert r.status_code == 200
    assert health_elapsed[0] < 1.5, (
        f"/api/health took {health_elapsed[0]:.2f}s — event loop blocked by PG restore?"
    )

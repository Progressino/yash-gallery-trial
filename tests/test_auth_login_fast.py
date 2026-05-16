"""Login must stay fast even when PostgreSQL session restore is slow."""

import time
from io import BytesIO


def test_login_skips_slow_pg_restore(monkeypatch):
    """Simulate a 30s PG blob load — login must not wait on it."""
    calls = []

    def _slow_load(session_id: str):
        calls.append(session_id)
        time.sleep(30)
        return None

    monkeypatch.setattr(
        "backend.db.forecast_session_pg.load_session_from_pg",
        _slow_load,
    )
    monkeypatch.setattr(
        "backend.db.forecast_session_pg.pg_session_persist_enabled",
        lambda: True,
    )

    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    client.cookies.set("session_id", "stale-session-uuid")

    t0 = time.monotonic()
    r = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "wrong"},
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"login took {elapsed:.1f}s (blocked on PG restore?)"
    assert r.status_code == 401
    assert calls == [], "login must not trigger PostgreSQL session restore"


def test_auth_me_skips_pg_restore(monkeypatch):
    calls = []

    def _slow_load(session_id: str):
        calls.append(session_id)
        time.sleep(30)
        return None

    monkeypatch.setattr(
        "backend.db.forecast_session_pg.load_session_from_pg",
        _slow_load,
    )
    monkeypatch.setattr(
        "backend.db.forecast_session_pg.pg_session_persist_enabled",
        lambda: True,
    )

    from backend.routers.auth import create_token
    from starlette.testclient import TestClient
    from backend.main import app

    token = create_token("tester", role="Admin")
    client = TestClient(app)
    client.cookies.set("auth_token", token)
    client.cookies.set("session_id", "stale-session-uuid")

    t0 = time.monotonic()
    r = client.get("/api/auth/me")
    elapsed = time.monotonic() - t0

    assert elapsed < 3.0, f"/auth/me took {elapsed:.1f}s"
    assert r.status_code == 200
    assert calls == []


def test_pg_restore_times_out_and_uses_empty_session(monkeypatch):
    import backend.session as sess_mod

    monkeypatch.setattr(sess_mod, "_PG_RESTORE_TIMEOUT_SEC", 0.05)

    def _slow_load(session_id: str):
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

    scheduled = []
    monkeypatch.setattr(
        "backend.session._schedule_pg_session_restore",
        lambda sid: scheduled.append(sid),
    )

    sid, sess = sess_mod.store.get_or_create("timed-out-session")
    assert sid == "timed-out-session"
    assert sess.mtr_df.empty
    assert scheduled == ["timed-out-session"]

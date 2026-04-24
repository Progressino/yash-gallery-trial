"""Shared fixtures: FastAPI TestClient with auth bypass."""

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_dashboard_upload_day_gate_by_default(monkeypatch):
    """Intelligence upload-day gating defaults ON in app code; tests expect legacy ungated totals."""
    monkeypatch.setenv("DASHBOARD_UPLOAD_DAY_GATE", "0")


@pytest.fixture
def auth_token(monkeypatch):
    monkeypatch.setattr("backend.main.verify_token", lambda t: "tester" if t else None)


@pytest.fixture
def client(auth_token):
    from backend.main import app

    c = TestClient(app)
    c.cookies.set("auth_token", "test-token")
    return c


@pytest.fixture
def session_for_client(client):
    """Return (session_id, AppSession) after one request activates the session."""
    from backend.session import store

    r = client.get("/api/health")
    assert r.status_code == 200
    sid = client.cookies.get("session_id")
    assert sid
    sess = store.get(sid)
    assert sess is not None
    return sid, sess

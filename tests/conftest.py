"""Shared fixtures: FastAPI TestClient with auth bypass."""

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_dashboard_upload_day_gate_by_default(monkeypatch):
    """Intelligence upload-day gating defaults ON in app code; tests expect legacy ungated totals."""
    monkeypatch.setenv("DASHBOARD_UPLOAD_DAY_GATE", "0")


@pytest.fixture(autouse=True)
def isolated_warm_cache(tmp_path, monkeypatch):
    """Point WARM_CACHE_DIR at an empty per-test dir and clear the in-memory warm cache.

    Without this, ``hydrate_po_session_for_calculate`` (and similar helpers) read the
    developer/CI host's real ``/data/warm_cache`` (production inventory/sales) into
    test sessions, making "no data" tests see real data.
    """
    import backend.main as main_mod

    warm = tmp_path / "warm_cache"
    warm.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    main_mod.clear_warm_cache()
    yield


@pytest.fixture(autouse=True)
def isolated_daily_sales_sqlite(tmp_path, monkeypatch):
    """Tier-3 store path is resolved at import time; point it at an empty per-test DB.

    Without this, ``_restore_daily_if_needed`` merges the developer's real
    ``daily_sales.db`` into TestClient sessions (e.g. platform-summary tests).
    """
    from pathlib import Path

    from backend.services import daily_store

    db = tmp_path / "daily_sales_pytest.db"
    monkeypatch.setattr(daily_store, "_DB_PATH", Path(db))
    yield


@pytest.fixture
def auth_token(monkeypatch):
    def _decode(token: str | None):
        if token == "test-token":
            return {"sub": "tester", "role": "Admin", "permissions": []}
        return None

    monkeypatch.setattr("backend.main.decode_token", _decode)
    monkeypatch.setattr("backend.routers.auth.decode_token", _decode)
    monkeypatch.setattr("backend.routers.auth.verify_token", lambda t: "tester" if t == "test-token" else None)


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


@pytest.fixture
def finance_isolated_db(tmp_path, monkeypatch):
    """Fresh finance SQLite + schema seed for HTTP tests (avoids touching dev finance.db)."""
    path = str(tmp_path / "finance_isolated.db")
    monkeypatch.setattr("backend.db.finance_db.DB_PATH", path)
    from backend.db.finance_db import init_db

    init_db()
    return path

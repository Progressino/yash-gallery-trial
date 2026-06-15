"""Daily upload verify endpoint and fast-ingest lock behaviour."""

from io import BytesIO
from unittest.mock import MagicMock

import pytest

from tests.conftest import bootstrap_test_session


def test_verify_daily_upload_requires_date(client):
    r = client.get("/api/upload/daily-auto/verify")
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_daily_auto_reset_clears_sales_rebuild(client, auth_token, monkeypatch):
    from backend.session import store

    bootstrap_test_session(client)
    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_started = 0.0
    r2 = client.post("/api/upload/daily-auto/reset-stuck")
    assert r2.status_code == 200
    assert sess.sales_rebuild_status == "idle"


def test_fast_ingest_proceeds_without_memory_lock(monkeypatch):
    from backend.concurrency import _UPLOAD_MEMORY_LOCK
    from backend.routers import upload as up

    monkeypatch.setattr(up, "_daily_auto_fast_ingest_enabled", lambda: True)
    while _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
        pass
    sess = MagicMock()
    held = up._acquire_ingest_memory_lock(sess, fnames="Sales.rar")
    assert held is False
    assert "Parsing" in str(sess.daily_auto_ingest_message)


def test_daily_auto_queues_pipeline(client, auth_token, monkeypatch):
    pipeline_called: list[tuple[str, list]] = []

    def _fake_pipeline(session_id: str, file_parts: list[tuple[str, bytes]]):
        pipeline_called.append((session_id, file_parts))

    monkeypatch.setattr(
        "backend.routers.upload._run_daily_auto_ingest_pipeline",
        _fake_pipeline,
    )

    r = client.post(
        "/api/upload/daily-auto",
        files=[("files", ("Sales 2-6-26.rar", BytesIO(b"fake"), "application/octet-stream"))],
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert len(pipeline_called) == 1

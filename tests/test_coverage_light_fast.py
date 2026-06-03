"""Light coverage must not block on Tier-3 restore lock."""

import time
from unittest.mock import patch

from backend.session import AppSession
from backend.routers.data import get_coverage, _hydrate_queued
from backend.routers.upload import clear_stale_background_jobs


def test_clear_stuck_daily_ingest_orphan_started():
    sess = AppSession()
    sess.daily_auto_ingest_status = "running"
    sess.daily_auto_ingest_started = 0.0
    clear_stale_background_jobs(sess)
    assert sess.daily_auto_ingest_status != "running"


def test_clear_stuck_sales_rebuild_orphan_started():
    from backend.routers.upload import _clear_stuck_sales_rebuild

    sess = AppSession()
    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_started = 0.0
    _clear_stuck_sales_rebuild(sess, force=False)
    assert sess.sales_rebuild_status == "idle"


def test_light_coverage_skips_heavy_restore(client, auth_token, monkeypatch):
    """GET /data/coverage?light=1 must not call Tier-3 restore on the request thread."""
    called = {"restore": False, "rebuild": False}

    def _track_restore(*_a, **_k):
        called["restore"] = True

    def _track_rebuild(*_a, **_k):
        called["rebuild"] = True

    monkeypatch.setattr("backend.routers.data._restore_daily_if_needed", _track_restore)
    monkeypatch.setattr("backend.routers.data._ensure_sales_rebuilt", _track_rebuild)
    monkeypatch.setattr("backend.routers.data._maybe_queue_light_session_hydrate", lambda *_a, **_k: None)

    t0 = time.perf_counter()
    r = client.get("/api/data/coverage", params={"light": "1"})
    elapsed = time.perf_counter() - t0

    assert r.status_code == 200, r.text
    assert elapsed < 5.0
    assert called["restore"] is False
    assert called["rebuild"] is False
    _hydrate_queued.clear()

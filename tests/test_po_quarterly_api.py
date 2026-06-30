"""PO /quarterly API — fast cache, warming poll, background job."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import bootstrap_test_session
from backend.services.po_quarterly_warmup import (
    QUARTERLY_CACHE_SCHEMA,
    build_quarterly_payload,
    quarterly_cache_key,
    quarterly_report_window,
)


def test_quarterly_cache_schema_v9():
    assert quarterly_cache_key(False, 8)[0] == QUARTERLY_CACHE_SCHEMA == 13


def test_quarterly_report_window_covers_eight_quarters():
    start, end = quarterly_report_window(8)
    assert len(start) == 10
    assert len(end) == 10
    assert start < end


def test_po_quarterly_returns_cached_without_rebuild(client, monkeypatch):
    from backend.session import store

    monkeypatch.setattr(
        "backend.main.try_attach_shared_frames_fast",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.try_build_quarterly_payload_sync",
        lambda *a, **k: pytest.fail("should not rebuild when cache warm"),
    )

    sid = bootstrap_test_session(client)
    sess = store.get(sid)
    assert sess is not None
    key = quarterly_cache_key(False, 8)
    sess._quarterly_cache[key] = {
        "loaded": True,
        "columns": ["OMS_SKU", "Apr-Jun 2026"],
        "rows": [{"OMS_SKU": "SKU1", "Apr-Jun 2026": 10}],
    }

    r = client.get("/api/po/quarterly?n_quarters=8")
    assert r.status_code == 200
    body = r.json()
    assert body.get("loaded") is True
    assert body["rows"][0]["OMS_SKU"] == "SKU1"


def test_po_quarterly_warming_when_sync_times_out(client, monkeypatch):
    from backend.session import store

    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.try_build_quarterly_payload_sync",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_jobs.start_quarterly_background",
        lambda sid, **k: True,
    )

    sid = bootstrap_test_session(client)
    sess = store.get(sid)
    assert sess is not None
    sess._quarterly_cache.clear()

    r = client.get("/api/po/quarterly?n_quarters=8")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "warming"
    assert body.get("loaded") is False


def test_hydrate_disabled():
    from backend.session import AppSession
    from backend.services import po_quarterly_warmup as mod

    sess = AppSession()
    assert mod.hydrate_platform_frames_for_quarterly(sess, n_quarters=8) is False

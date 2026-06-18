"""Tests for admin performance metrics aggregation."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from backend.services import perf_metrics as pm


@pytest.fixture(autouse=True)
def _reset_perf_state(monkeypatch):
    monkeypatch.setenv("PERF_METRICS", "1")
    with pm._lock:
        pm._events.clear()
        pm._cache_hits = 0
        pm._cache_misses = 0
    pm._table_ready = False
    yield
    with pm._lock:
        pm._events.clear()


def _po_event(day: date, duration_ms: float) -> None:
    pm.record("po_calculate", "PO calculate", duration_ms, meta={"ok": True, "total_rows": 100})


def test_day_summary_po_today_vs_yesterday(monkeypatch):
    today = pm.day_ist()
    yesterday = today - timedelta(days=1)

    def fake_day_ist(_when=None):
        return today

    monkeypatch.setattr(pm, "day_ist", fake_day_ist)

    # Simulate yesterday avg ~21s (3 runs)
    for _ in range(3):
        ev = {
            "recorded_at": datetime.now(timezone.utc) - timedelta(hours=30),
            "day_ist": yesterday,
            "kind": "po_calculate",
            "name": "PO calculate",
            "duration_ms": 21_000.0,
            "meta": {"ok": True},
        }
        pm._events.append(ev)

    # Today avg ~2.4s (2 runs)
    for ms in (2400.0, 2400.0):
        pm._events.append(
            {
                "recorded_at": datetime.now(timezone.utc),
                "day_ist": today,
                "kind": "po_calculate",
                "name": "PO calculate",
                "duration_ms": ms,
                "meta": {"ok": True},
            }
        )

    dash = pm.build_dashboard(hours=48)
    assert dash["po_calculate"]["yesterday"]["avg_sec"] == 21.0
    assert dash["po_calculate"]["today"]["avg_sec"] == 2.4
    assert dash["po_calculate"]["yesterday"]["count"] == 3
    assert dash["po_calculate"]["today"]["count"] == 2


def test_slowest_endpoints_and_cache():
    pm.record_http("GET", "/api/data/coverage", 200, 1.2)
    pm.record_http("GET", "/api/data/coverage", 200, 0.4)
    pm.record_http("POST", "/api/po/calculate", 200, 25.0)
    pm.record_db_query("psycopg", "SELECT * FROM forecast_sales_transactions WHERE sku = %s", 0.8)
    pm.record_cache(hit=True, source="warm_cache", name="hydrate_warm")
    pm.record_cache(hit=False, source="warm_cache", name="hydrate_warm_async")

    dash = pm.build_dashboard(hours=24)
    assert len(dash["slowest_endpoints"]) >= 2
    assert dash["slowest_endpoints"][0]["name"] == "POST /api/po/calculate"
    assert len(dash["slowest_queries"]) == 1
    assert dash["cache"]["hits"] >= 1
    assert dash["cache"]["misses"] >= 1
    assert dash["cache"]["hit_rate"] is not None


def test_session_restore_in_dashboard():
    pm.record_session_restore("pg_snapshot", 3.5, ok=True)
    pm.record_session_restore("full_restore+sales", 45.0, ok=True)

    dash = pm.build_dashboard(hours=24)
    assert dash["session_restore"]["slowest"]
    assert dash["session_restore"]["slowest"][0]["max_sec"] == 45.0


def test_pretty_bytes():
    assert pm._pretty_bytes(512) == "512 B"
    assert pm._pretty_bytes(2048) == "2.0 KB"
    assert pm._pretty_bytes(2 * 1024 * 1024) == "2.0 MB"

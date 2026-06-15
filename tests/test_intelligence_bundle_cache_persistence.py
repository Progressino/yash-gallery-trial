"""Process-wide Intelligence bundle cache must survive a backend restart via
disk persistence, honor a generous TTL, and be invalidated on real data changes."""

from __future__ import annotations

import time

from backend.routers import data as data_router


def _payload(units=10):
    return {
        "sales_summary": {"total_units": units, "total_returns": 0, "net_units": units, "return_rate": 0.0},
        "platform_summary": [],
        "top_skus": [],
        "anomalies": [],
        "dsr_brand_monthly": {"rows": [], "totals": {}, "note": ""},
    }


def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(data_router, "_INTEL_BUNDLE_DISK_DIR", str(tmp_path))
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token", lambda: {}, raising=False
    )
    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()


def test_store_persists_to_disk_and_reload_after_restart(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    cache_key = ("", "", "gross", 10, False)
    sess_cache: dict = {}

    data_router._bundle_cache_store(cache_key, sess_cache, _payload(), ts=time.time())

    # Simulate a restart: process-wide dict wiped, disk survives.
    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()
    assert (
        data_router._bundle_cache_lookup(cache_key, None, start_date=None, end_date=None)
        is None
    )

    data_router._load_intelligence_bundle_cache_from_disk()
    restored = data_router._bundle_cache_lookup(cache_key, None, start_date=None, end_date=None)
    assert restored is not None
    assert restored["sales_summary"]["total_units"] == 10


def test_ttl_is_generous_but_finite(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    cache_key = ("", "", "gross", 10, False)
    sess_cache: dict = {}

    now = time.time()
    data_router._bundle_cache_store(cache_key, sess_cache, _payload(), ts=now)

    # Still fresh well within TTL.
    assert (
        data_router._bundle_cache_lookup(cache_key, sess_cache, start_date=None, end_date=None)
        is not None
    )

    # Older than the TTL → cache miss.
    sess_cache[cache_key]["_ts"] = now - (data_router._INTELLIGENCE_BUNDLE_TTL_SEC + 1)
    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE[
        data_router._bundle_cache_global_key(cache_key)
    ]["_ts"] = now - (data_router._INTELLIGENCE_BUNDLE_TTL_SEC + 1)
    assert (
        data_router._bundle_cache_lookup(cache_key, sess_cache, start_date=None, end_date=None)
        is None
    )


def test_invalidate_clears_memory_and_disk(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    cache_key = ("", "", "gross", 10, False)
    sess_cache: dict = {}

    data_router._bundle_cache_store(cache_key, sess_cache, _payload(), ts=time.time())
    assert (
        data_router._bundle_cache_lookup(cache_key, sess_cache, start_date=None, end_date=None)
        is not None
    )

    data_router._invalidate_intelligence_bundle_cache()

    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()
    data_router._load_intelligence_bundle_cache_from_disk()
    assert data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE == {}

"""Precomputed intelligence disk cache must win over Tier-3 rebuild."""
from __future__ import annotations

from backend.routers import data as data_router
from backend.session import AppSession


def test_cached_bundle_not_stale_for_new_session_when_tier3_has_uploads(monkeypatch):
    monkeypatch.setattr(
        "backend.services.daily_store.platforms_with_uploads_in_range",
        lambda _s, _e: ["amazon"],
    )
    sess = AppSession()
    payload = {
        "sales_summary": {"total_units": 50419},
        "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 27000}],
        "status": "ready",
    }
    assert data_router._cached_bundle_stale_vs_tier3_uploads(
        payload, "2026-05-21", "2026-06-20", sess=sess
    ) is False


def test_cached_bundle_stale_after_session_saw_tier3_upload(monkeypatch):
    monkeypatch.setattr(
        "backend.services.daily_store.platforms_with_uploads_in_range",
        lambda _s, _e: ["amazon"],
    )
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "2:100:2026-06-20"},
    )
    sess = AppSession()
    sess._tier3_sync_token_applied = {"amazon": "1:50:2026-06-19"}
    payload = {"sales_summary": {"total_units": 100}, "platform_summary": [], "status": "ready"}
    assert data_router._cached_bundle_stale_vs_tier3_uploads(
        payload, "2026-05-21", "2026-06-20", sess=sess
    ) is True


def test_global_cache_hit_before_tier3_rebuild(monkeypatch):
    tier3_calls = {"n": 0}

    def _tier3(*_a, **_k):
        tier3_calls["n"] += 1
        return {"sales_summary": {"total_units": 22000}, "platform_summary": [], "status": "ready"}

    monkeypatch.setattr(data_router, "_try_serve_tier3_intelligence_bundle", _tier3)
    monkeypatch.setattr(
        "backend.services.daily_store.platforms_with_uploads_in_range",
        lambda _s, _e: ["amazon"],
    )
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "1:1:x"},
    )

    cache_key = ("2026-05-21", "2026-06-20", "gross", 10, False)
    global_key = data_router._bundle_cache_global_key(cache_key)
    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE[global_key] = {
        "_ts": 1.0,
        "payload": {
            "sales_summary": {"total_units": 50419},
            "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 27000}],
            "status": "ready",
        },
    }

    sess = AppSession()
    hit = data_router._bundle_cache_lookup(
        cache_key,
        {},
        start_date="2026-05-21",
        end_date="2026-06-20",
        allow_sparse=True,
        sess=sess,
    )
    assert hit is not None
    assert hit["sales_summary"]["total_units"] == 50419
    assert tier3_calls["n"] == 0

    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.pop(global_key, None)

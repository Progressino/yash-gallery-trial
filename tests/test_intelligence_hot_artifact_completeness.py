"""Hot intelligence artifacts must not trap the dashboard on permanent 'partial'."""
from __future__ import annotations


def test_hot_payload_marks_full_when_units_cover_window(monkeypatch):
    from backend.services import intelligence_artifacts as arts

    monkeypatch.setattr(
        arts,
        "_build_hot_payload",
        arts._build_hot_payload,
    )

    def fake_tier3(*_a, **_k):
        return {
            "platform_summary": [
                {"platform": "Amazon", "loaded": True, "total_units": 40_000}
            ],
            "top_skus": [],
            "sales_summary": {"total_units": 40_000, "total_returns": 0, "net_units": 40_000},
        }

    monkeypatch.setattr(
        "backend.routers.data._build_intelligence_bundle_payload_from_tier3",
        fake_tier3,
    )
    monkeypatch.setattr(
        "backend.routers.data._tier3_only_undercounts_bulk",
        lambda units, s, e: False,
    )

    out = arts._build_hot_payload(object(), "2026-06-17", "2026-07-17", 10)
    assert out is not None
    assert out["data_completeness"] == "full"
    assert out["sales_summary"]["total_units"] == 40_000


def test_hot_payload_marks_partial_when_undercount(monkeypatch):
    from backend.services import intelligence_artifacts as arts

    def fake_tier3(*_a, **_k):
        return {
            "platform_summary": [
                {"platform": "Amazon", "loaded": True, "total_units": 5_000}
            ],
            "top_skus": [],
            "sales_summary": {"total_units": 5_000, "total_returns": 0, "net_units": 5_000},
        }

    monkeypatch.setattr(
        "backend.routers.data._build_intelligence_bundle_payload_from_tier3",
        fake_tier3,
    )
    monkeypatch.setattr(
        "backend.routers.data._tier3_only_undercounts_bulk",
        lambda units, s, e: True,
    )

    out = arts._build_hot_payload(object(), "2026-06-17", "2026-07-17", 10)
    assert out is not None
    assert out["data_completeness"] == "partial"


def test_load_deep_heals_legacy_partial_hot(monkeypatch):
    from backend.services import intelligence_artifacts as arts

    legacy = {
        "data_completeness": "partial",
        "sales_summary": {"total_units": 55_964},
        "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 55_964}],
        "top_skus": [],
    }

    monkeypatch.setattr(
        arts,
        "load_artifact",
        lambda *a, **k: (legacy, {"source": "disk", "stale": False, "version": "v1"}),
    )
    monkeypatch.setattr(
        "backend.routers.data._tier3_only_undercounts_bulk",
        lambda units, s, e: False,
    )

    bundle, meta = arts.load_deep_bundle_for_request(
        object(), "2026-06-17", "2026-07-17", limit=10, include_extras=False
    )
    assert bundle is not None
    assert bundle["data_completeness"] == "full"
    assert meta.get("source") == "disk"


def test_load_deep_skips_partial_when_require_complete(monkeypatch):
    from backend.services import intelligence_artifacts as arts

    legacy = {
        "data_completeness": "partial",
        "sales_summary": {"total_units": 5_000},
        "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 5_000}],
        "top_skus": [],
    }

    monkeypatch.setattr(
        arts,
        "load_artifact",
        lambda *a, **k: (legacy, {"source": "disk", "stale": False}),
    )
    monkeypatch.setattr(
        "backend.routers.data._tier3_only_undercounts_bulk",
        lambda units, s, e: True,
    )

    bundle, meta = arts.load_deep_bundle_for_request(
        object(),
        "2026-06-17",
        "2026-07-17",
        limit=10,
        include_extras=False,
        require_complete=True,
    )
    assert bundle is None
    assert meta.get("skipped_partial_hot") is True

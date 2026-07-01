"""Dashboard summary must use Tier-1 bulk gap-fill when Tier-3 alone undercounts."""
from __future__ import annotations

import pandas as pd

from backend.services.dashboard_summary import (
    _pick_richer_summary,
    build_dashboard_summary,
)
from backend.session import AppSession


def test_pick_richer_summary_prefers_gapfill():
    tier3 = {
        "sales_summary": {"total_units": 81},
        "data_completeness": "partial",
    }
    gapfill = {
        "sales_summary": {"total_units": 42_000},
        "data_completeness": "full",
    }
    assert _pick_richer_summary(gapfill, tier3) is gapfill
    assert _pick_richer_summary(tier3, gapfill) is gapfill


def test_build_dashboard_summary_skips_gapfill_when_tier3_sufficient(monkeypatch):
    tier3 = {
        "platform_summary": [
            {"platform": "Amazon", "total_units": 30_000, "total_returns": 2000, "net_units": 28_000, "loaded": True},
            {"platform": "Flipkart", "total_units": 8_000, "total_returns": 500, "net_units": 7_500, "loaded": True},
        ],
        "sales_summary": {"total_units": 38_000, "total_returns": 2500, "net_units": 35_500},
        "top_skus": [{"sku": "SKU-1", "units": 100}],
    }
    gapfill_called = {"n": 0}

    def _gapfill(*a, **k):
        gapfill_called["n"] += 1
        return None

    monkeypatch.setattr(
        "backend.services.intelligence_artifacts.load_hot_summary_for_request",
        lambda *a, **k: (None, {}),
    )
    monkeypatch.setattr(
        "backend.routers.data._build_intelligence_bundle_payload_from_tier3",
        lambda *a, **k: tier3,
    )
    monkeypatch.setattr(
        "backend.services.dashboard_summary._summary_from_gapfill",
        _gapfill,
    )
    monkeypatch.setattr(
        "backend.services.dashboard_summary._artifact_undercounts_bulk",
        lambda *a, **k: False,
    )

    out = build_dashboard_summary(
        AppSession(),
        start_date="2026-06-01",
        end_date="2026-06-30",
        limit=10,
    )
    assert gapfill_called["n"] == 0
    assert int(out["sales_summary"]["total_units"]) == 38_000
    assert out.get("data_completeness") == "full"


def test_build_dashboard_summary_uses_gapfill_over_tier3(monkeypatch):
    tier3 = {
        "platform_summary": [
            {"platform": "Flipkart", "total_units": 62, "total_returns": 26, "net_units": 36, "loaded": True},
            {"platform": "Snapdeal", "total_units": 19, "total_returns": 7, "net_units": 12, "loaded": True},
        ],
        "sales_summary": {"total_units": 81, "total_returns": 33, "net_units": 48},
        "top_skus": [],
    }
    gapfill = {
        "platform_summary": [
            {"platform": "Amazon", "total_units": 18_000, "total_returns": 2000, "net_units": 16_000, "loaded": True},
            {"platform": "Flipkart", "total_units": 12_000, "total_returns": 1000, "net_units": 11_000, "loaded": True},
            {"platform": "Myntra", "total_units": 9_000, "total_returns": 800, "net_units": 8_200, "loaded": True},
            {"platform": "Meesho", "total_units": 2_500, "total_returns": 200, "net_units": 2_300, "loaded": True},
            {"platform": "Snapdeal", "total_units": 500, "total_returns": 50, "net_units": 450, "loaded": True},
        ],
        "sales_summary": {"total_units": 42_000, "total_returns": 4050, "net_units": 37_950},
        "top_skus": [],
        "data_completeness": "full",
    }

    monkeypatch.setattr(
        "backend.services.intelligence_artifacts.load_hot_summary_for_request",
        lambda *a, **k: (None, {}),
    )
    monkeypatch.setattr(
        "backend.routers.data._build_intelligence_bundle_payload_from_tier3",
        lambda *a, **k: tier3,
    )
    monkeypatch.setattr(
        "backend.services.dashboard_summary._summary_from_gapfill",
        lambda *a, **k: {
            "source": "gapfill_bulk_history",
            "platform_summary": gapfill["platform_summary"],
            "platforms": {},
            "top_skus": [],
            "sales_summary": gapfill["sales_summary"],
            "data_completeness": "full",
        },
    )
    monkeypatch.setattr(
        "backend.services.dashboard_summary._artifact_undercounts_bulk",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_artifacts.save_artifact",
        lambda *a, **k: "v-test",
    )
    monkeypatch.setattr(
        "backend.services.intelligence_artifacts.schedule_artifact_build",
        lambda *a, **k: None,
    )

    out = build_dashboard_summary(
        AppSession(),
        start_date="2025-01-01",
        end_date="2025-01-31",
        limit=10,
    )
    assert out["source"] == "gapfill_bulk_history"
    assert int(out["sales_summary"]["total_units"]) == 42_000
    assert len(out["platform_summary"]) == 5

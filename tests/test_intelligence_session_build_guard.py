"""Intelligence bundle must not use session unified build on huge catalogs."""
from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_session_build_unsafe_at_750k_rows(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_SESSION_BUILD_MAX_ROWS", "750000")
    sess = AppSession()
    sess.sales_df = pd.DataFrame({"Sku": range(750_000)})
    assert data_router._intelligence_session_build_unsafe(sess) is True


def test_session_build_safe_below_cap(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_SESSION_BUILD_MAX_ROWS", "750000")
    sess = AppSession()
    sess.sales_df = pd.DataFrame({"Sku": range(100_000)})
    assert data_router._intelligence_session_build_unsafe(sess) is False


def test_try_serve_tier3_accepts_headline_units_without_loaded_flags(monkeypatch):
    sess = AppSession()
    cache: dict = {}

    def fake_tier3(*_a, **_k):
        return {
            "sales_summary": {"total_units": 500, "total_returns": 0, "net_units": 500, "return_rate": 0.0},
            "platform_summary": [
                {"platform": "Amazon", "loaded": False, "total_units": 500, "total_returns": 0},
            ],
            "top_skus": [],
            "anomalies": [],
            "dsr_brand_monthly": {"rows": []},
        }

    monkeypatch.setattr(data_router, "_build_intelligence_bundle_payload_from_tier3", fake_tier3)
    out = data_router._try_serve_tier3_intelligence_bundle(
        sess,
        ("2026-06-01", "2026-06-07", "gross", 10, False),
        cache,
        "2026-06-01",
        "2026-06-07",
        10,
        "gross",
        False,
    )
    assert out is not None
    assert out["status"] == "ready"
    assert out["sales_summary"]["total_units"] == 500

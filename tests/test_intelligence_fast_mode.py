"""Intelligence bundle fast mode — partial Tier-3 without undercount rejection."""
from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_fast_mode_serves_tier3_when_undercounts_bulk(monkeypatch):
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "SKU": ["SKU-A", "SKU-A"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [100, 50],
            "Order_Id": ["O1", "O2"],
        }
    )

    monkeypatch.setattr(
        daily_store,
        "platforms_with_uploads_in_range",
        lambda s, e: ["amazon"],
    )
    monkeypatch.setattr(
        daily_store,
        "load_platform_data_for_report_range",
        lambda plat, s, e, dedup=True, **kwargs: amazon.copy() if plat == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)
    monkeypatch.setattr(data_router, "_disk_bulk_history_available", lambda: True)
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)

    sess = AppSession()
    sess.sku_mapping = {"SKU-A": "SKU-A"}

    payload = data_router._serve_intelligence_bundle_fast(
        sess,
        ("2026-05-05", "2026-06-04", "gross", 10, False),
        {},
        "2026-05-05",
        "2026-06-04",
        10,
        "gross",
        False,
    )
    assert payload is not None
    assert payload["sales_summary"]["total_units"] == 150
    assert payload.get("data_completeness") == "partial"


def test_intelligence_precompute_mode_tier3_gapfill(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_PRECOMPUTE_MODE", raising=False)
    monkeypatch.setenv("INTELLIGENCE_PRECOMPUTE", "0")
    assert data_router._intelligence_precompute_mode() == "off"

    monkeypatch.setenv("INTELLIGENCE_PRECOMPUTE_MODE", "tier3_gapfill")
    assert data_router._intelligence_precompute_mode() == "tier3_gapfill"

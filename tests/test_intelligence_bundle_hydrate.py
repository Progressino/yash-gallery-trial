"""Intelligence bundle must hydrate empty sessions and auto-pull Tier-3."""
from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_bundle_session_builder_uses_auto_dashboard_when_sales_empty(monkeypatch):
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-20", "2026-06-01"]),
            "SKU": ["A1", "A2"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [100, 250],
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
        lambda plat, s, e, dedup=True, **kwargs: (
            amazon.copy() if plat == "amazon" else pd.DataFrame()
        ),
    )
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)
    monkeypatch.setattr(
        daily_store,
        "get_upload_report_day_coverage",
        lambda: {"amazon": {"2026-05-20", "2026-06-01"}},
    )

    sess = AppSession()
    sess.sales_df = pd.DataFrame()
    sess.mtr_df = pd.DataFrame()
    sess.sku_mapping = {"A1": "A1", "A2": "A2"}

    payload = data_router._build_intelligence_bundle_payload_from_session(
        sess, "2026-05-05", "2026-06-04", 10, "gross", False
    )
    assert payload is not None
    assert payload["sales_summary"]["total_units"] >= 350
    amazon_card = next(
        p for p in payload["platform_summary"] if p["platform"] == "Amazon"
    )
    assert amazon_card["total_units"] >= 350


def test_intelligence_bundle_warming_when_no_data_anywhere(client, monkeypatch):
    from backend.services import daily_store

    monkeypatch.setattr(
        daily_store, "platforms_with_uploads_in_range", lambda s, e: []
    )
    monkeypatch.setattr(
        daily_store,
        "load_platform_data_for_report_range",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)
    monkeypatch.setattr(data_router, "_hydrate_session_for_intelligence", lambda _s: False)

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-05-05", "end_date": "2026-06-04"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "warming"

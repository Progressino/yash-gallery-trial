"""Auto-pull Tier-3 when Upload has data but Intelligence session sales do not."""

from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_platforms_needing_auto_pull_when_session_sales_empty(monkeypatch):
    from backend.services import daily_store

    monkeypatch.setattr(
        daily_store,
        "platforms_with_uploads_in_range",
        lambda s, e: ["amazon", "meesho"],
    )
    monkeypatch.setattr(
        daily_store,
        "get_upload_report_day_coverage",
        lambda: {"amazon": {"2026-06-01"}, "meesho": {"2026-06-01"}},
    )
    monkeypatch.setattr(
        daily_store,
        "load_platform_data_for_report_range",
        lambda *a, **k: pd.DataFrame(),
    )

    sess = AppSession()
    sess.sales_df = pd.DataFrame()
    need = data_router._platforms_needing_auto_tier3_pull(
        sess, "2026-06-01", "2026-06-04", pd.DataFrame()
    )
    assert "amazon" in need
    assert "meesho" in need


def test_auto_dashboard_builds_from_tier3_when_sales_empty(monkeypatch):
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [50],
            "Order_Id": ["O1"],
        }
    )

    def _load(plat, start, end, dedup=True):
        return amazon.copy() if plat == "amazon" else pd.DataFrame()

    monkeypatch.setattr(daily_store, "platforms_with_uploads_in_range", lambda s, e: ["amazon"])
    monkeypatch.setattr(daily_store, "load_platform_data_for_report_range", _load)
    monkeypatch.setattr(daily_store, "get_upload_report_day_coverage", lambda: {"amazon": {"2026-06-01"}})
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)

    sess = AppSession()
    sess.sales_df = pd.DataFrame()
    sess.sku_mapping = {"SKU-A": "SKU-A"}

    sales, mtr, *_rest, pulled = data_router._auto_dashboard_bundle_data(
        sess, "2026-06-01", "2026-06-04"
    )
    assert "amazon" in pulled
    assert sales is not None and not sales.empty
    assert int(sales[sales["Source"] == "Amazon"]["Quantity"].sum()) == 50
    assert not mtr.empty


def test_intelligence_bundle_serves_june_from_tier3_not_warming(client, monkeypatch):
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data
    from backend.session import store

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [33],
            "Order_Id": ["O1"],
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
        lambda plat, s, e, dedup=True: amazon.copy() if plat == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(
        daily_store,
        "get_upload_report_day_coverage",
        lambda: {"amazon": {"2026-06-01"}},
    )
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)
    monkeypatch.setattr(daily_store, "get_tier3_sync_token", lambda: {"amazon": "1:1:x"})

    client.get("/api/health")
    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    sess.sales_df = pd.DataFrame()
    sess.mtr_df = pd.DataFrame()
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    sess._tier3_sync_token_applied = {}

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-04", "include_extras": "0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ready"
    assert body.get("sales_summary", {}).get("total_units", 0) == 33
    plat = next(p for p in body["platform_summary"] if p["platform"] == "Amazon")
    assert plat.get("loaded") is True
    assert plat.get("total_units", 0) == 33

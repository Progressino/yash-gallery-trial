"""Auto-pull Tier-3 when Upload has data but Intelligence session sales do not."""

from __future__ import annotations

import pandas as pd

from tests.conftest import bootstrap_test_session
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

    def _load(plat, start, end, dedup=True, **kwargs):
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


def test_tier3_direct_preferred_over_empty_session_sales(client, monkeypatch):
    """Session sales empty in window but Tier-3 has rows → bundle uses Tier-3 direct."""
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data
    from backend.session import store

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [99],
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
        lambda plat, s, e, dedup=True, **kwargs: amazon.copy() if plat == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)
    monkeypatch.setattr(daily_store, "get_tier3_sync_token", lambda: {})

    bootstrap_test_session(client)
    sess = store.get(client.cookies.get("session_id"))
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["X"],
            "TxnDate": pd.to_datetime(["2025-01-01"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Units_Effective": [1],
            "Source": ["Amazon"],
            "OrderId": ["O0"],
            "LineKey": [""],
        }
    )
    sess.sku_mapping = {"SKU-A": "SKU-A"}

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-04"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sales_summary"]["total_units"] == 99
    assert body.get("session_fast_path") or body.get("tier3_auto_pull")


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
        lambda plat, s, e, dedup=True, **kwargs: amazon.copy() if plat == "amazon" else pd.DataFrame(),
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

    bootstrap_test_session(client)
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


def test_long_window_uses_session_without_tier3_parquet_load(client, monkeypatch):
    """90D Intelligence must not load Tier-3 parquet blobs when session sales has window data."""
    from backend.services import daily_store
    from backend.session import store

    tier3_loads = {"n": 0}

    def _load(plat, start, end, dedup=True, **kwargs):
        tier3_loads["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(daily_store, "load_platform_data_for_report_range", _load)
    monkeypatch.setattr(daily_store, "platforms_with_uploads_in_range", lambda s, e: [])
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)
    monkeypatch.setattr(data_router, "_maybe_queue_light_session_hydrate", lambda *a, **k: None)
    monkeypatch.setattr(daily_store, "get_tier3_sync_token", lambda: {})
    monkeypatch.setattr(data_router, "_tier3_token_mismatch", lambda _s: False)
    monkeypatch.setattr(
        data_router,
        "_platform_df_for_intelligence_bundle",
        lambda sess, pk, attr, start_date, end_date: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "backend.services.shared_frames.frame_row_count",
        lambda attr, _sess: 2 if attr == "sales_df" else 0,
    )
    monkeypatch.setattr(
        data_router,
        "_resolve_bundle_platform_frames",
        lambda sess, s, e: (
            sess.mtr_df,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(data_router, "_tier3_direct_has_units", lambda *a, **k: None)
    monkeypatch.setattr(data_router, "_load_tier3_frames_for_platforms", lambda *a, **k: {})

    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()
    bootstrap_test_session(client)
    sess = store.get(client.cookies.get("session_id"))
    assert sess is not None
    sess._intelligence_bundle_cache.clear()
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "Transaction Type": ["Shipment", "Refund"],
            "Quantity": [40, 5],
            "Units_Effective": [40, -5],
            "Source": ["Amazon", "Amazon"],
            "OrderId": ["O1", "O2"],
            "LineKey": ["L1", "L2"],
        }
    )
    sess.mtr_df = pd.DataFrame(
        [{"Date": "2026-06-01", "SKU": "SKU-A", "Transaction_Type": "Shipment", "Quantity": 40}]
    )
    sess.sku_mapping = {"SKU-A": "SKU-A"}

    r = client.get(
        "/api/data/intelligence-bundle",
        params={
            "start_date": "2026-03-06",
            "end_date": "2026-06-04",
            "include_extras": "0",
            "mode": "full",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("session_fast_path") is True
    assert body.get("tier3_auto_pull") is not True
    assert tier3_loads["n"] == 0
    # Session sales in fixture: 40 gross Amazon units in the June slice.
    assert body["sales_summary"]["total_units"] >= 40


def test_stale_session_cache_defers_to_tier3_when_uploads_in_window(client, monkeypatch):
    """Cached session bundle must not beat Tier-3 when daily uploads exist for the window."""
    import time as _time

    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data
    from backend.session import store

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
    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)
    monkeypatch.setattr(
        daily_store,
        "get_tier3_sync_token",
        lambda: {"amazon": "2:150:2026-06-02"},
    )
    monkeypatch.setattr(data_router, "_tier3_token_mismatch", lambda _s: True)

    data_router._GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()
    bootstrap_test_session(client)
    sess = store.get(client.cookies.get("session_id"))
    assert sess is not None
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    sess._tier3_sync_token_applied = {}

    stale_payload = {
        "status": "ready",
        "session_fast_path": True,
        "sales_summary": {"total_units": 40, "total_returns": 0, "net_units": 40, "return_rate": 0.0},
        "platform_summary": [
            {
                "platform": "Amazon",
                "loaded": True,
                "total_units": 40,
                "total_returns": 0,
                "net_units": 40,
                "return_rate": 0.0,
                "top_sku": "SKU-A",
                "trend_direction": "flat",
                "monthly": [],
                "daily": [{"date": "2026-06-01", "shipments": 40, "refunds": 0, "net": 40}],
            }
        ],
        "top_skus": [],
        "anomalies": [],
        "dsr_brand_monthly": {"rows": []},
    }
    cache_key = ("2026-06-01", "2026-06-04", "gross", 10, False)
    sess._intelligence_bundle_cache = {
        cache_key: {"_ts": _time.time(), "payload": stale_payload}
    }

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-04", "include_extras": "0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sales_summary"]["total_units"] == 150
    assert body.get("tier3_auto_pull") is True


def test_tier3_intelligence_bundle_applies_return_overlay(monkeypatch):
    """Tier-3 fast path must still merge PO return overlay (net units / return rate)."""
    import pandas as pd

    from backend.routers import data as data_router
    from backend.services import daily_store
    from backend.session import AppSession

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [100],
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
        lambda plat, s, e, dedup=True, **kwargs: amazon.copy() if plat == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(daily_store, "get_tier3_sync_token", lambda: {"amazon": "1:1:x"})

    sess = AppSession()
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    sess.po_return_overlay_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Return_Platform": ["amazon"],
            "Return_Date": ["2026-06-01"],
            "Return_Units": [25],
        }
    )
    sess.return_overlay_as_of = "2026-06-01"

    payload = data_router._build_intelligence_bundle_payload_from_tier3(
        sess, "2026-06-01", "2026-06-04", 10, "gross", False
    )
    assert payload is not None
    assert payload.get("tier3_auto_pull") is True
    assert payload["sales_summary"]["total_units"] == 100
    assert payload["sales_summary"]["total_returns"] == 25
    assert payload["sales_summary"]["net_units"] == 75
    assert payload["sales_summary"]["return_rate"] == 25.0

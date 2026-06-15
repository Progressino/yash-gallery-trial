"""Intelligence bundle must not block on session refresh when sales are loaded."""
from __future__ import annotations

import pandas as pd

from tests.conftest import bootstrap_test_session

from backend.session import AppSession, store


def test_intelligence_bundle_uses_cache_without_refresh(client, monkeypatch):
    """Second identical request returns cached payload (fast path)."""
    called = {"refresh": 0}

    def _track_refresh(sess):
        called["refresh"] += 1

    monkeypatch.setattr("backend.routers.data._ensure_intelligence_session_fresh", _track_refresh)
    monkeypatch.setattr("backend.routers.data._schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {},
    )

    sid = bootstrap_test_session(client)
    assert sid
    sess = store.get(sid)
    assert sess is not None
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU1", "SKU1"],
            "TxnDate": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "Transaction Type": ["Shipment", "Refund"],
            "Quantity": [10, 2],
            "Units_Effective": [10, -2],
            "Source": ["Amazon", "Amazon"],
            "OrderId": ["O1", "O2"],
            "LineKey": ["L1", "L2"],
        }
    )
    sess.mtr_df = pd.DataFrame(
        [{"Date": "2026-06-01", "SKU": "SKU1", "Transaction_Type": "Shipment", "Quantity": 10}]
    )
    sess._tier3_sync_token_applied = {}
    sess.sku_mapping = {"SKU1": "SKU1"}

    r1 = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-03", "include_extras": False},
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1.get("status") == "ready"
    assert body1.get("sales_summary", {}).get("total_units", 0) >= 0

    refresh_after_first = called["refresh"]
    r2 = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-03", "include_extras": False},
    )
    assert r2.status_code == 200
    assert r2.json() == body1
    assert called["refresh"] == refresh_after_first


def test_short_window_bundle_does_not_block_full_refresh(client, monkeypatch):
    """4-day window must not call blocking _ensure_intelligence_session_fresh on GET."""
    called = {"refresh": 0}

    def _track_refresh(sess):
        called["refresh"] += 1

    monkeypatch.setattr("backend.routers.data._ensure_intelligence_session_fresh", _track_refresh)
    monkeypatch.setattr("backend.routers.data._schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(
        "backend.services.daily_store.get_upload_report_day_coverage",
        lambda: {},
    )
    monkeypatch.setattr(
        "backend.services.daily_store.get_tier3_sync_token",
        lambda: {"amazon": "1:1:x"},
    )

    sid = bootstrap_test_session(client)
    sess = store.get(sid)
    assert sess is not None
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU1"],
            "TxnDate": pd.to_datetime(["2026-06-04"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [5],
            "Units_Effective": [5],
            "Source": ["Amazon"],
            "OrderId": ["O1"],
            "LineKey": ["L1"],
        }
    )
    sess.sku_mapping = {"SKU1": "SKU1"}
    sess._tier3_sync_token_applied = {}

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-06-01", "end_date": "2026-06-04", "include_extras": "0"},
    )
    assert r.status_code == 200
    assert called["refresh"] == 0

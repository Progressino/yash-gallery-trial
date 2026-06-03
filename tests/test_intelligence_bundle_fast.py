"""Intelligence bundle must not block on session refresh when sales are loaded."""
from __future__ import annotations

import pandas as pd

from backend.session import AppSession, store


def test_intelligence_bundle_uses_cache_without_refresh(client, monkeypatch):
    """Second identical request returns cached payload (fast path)."""
    called = {"refresh": 0}

    def _track_refresh(sess):
        called["refresh"] += 1

    monkeypatch.setattr("backend.routers.data._ensure_intelligence_session_fresh", _track_refresh)
    monkeypatch.setattr("backend.routers.data._schedule_intelligence_refresh_async", lambda _sid: None)

    r = client.get("/api/health")
    assert r.status_code == 200
    sid = client.cookies.get("session_id")
    assert sid
    sess = store.get(sid)
    assert sess is not None
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU1", "SKU1"],
            "TxnDate": ["2026-06-01", "2026-06-02"],
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

"""Regression: Intelligence bundle must not return empty cards when session has window data."""
from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_build_platform_summary_falls_back_to_session_frames(monkeypatch):
    from backend.services import daily_store

    monkeypatch.setattr(
        daily_store,
        "platforms_with_uploads_in_range",
        lambda s, e: ["amazon", "myntra"],
    )
    monkeypatch.setattr(
        daily_store,
        "load_platform_data_for_report_range",
        lambda *a, **k: pd.DataFrame(),
    )

    sess = AppSession()
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-20", "2026-06-01"]),
            "SKU": ["A1", "A2"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [100, 200],
        }
    )
    sess.myntra_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-22"]),
            "OMS_SKU": ["M1"],
            "TxnType": ["Shipment"],
            "Quantity": [50],
        }
    )
    mtr_b, myntra_b, meesho_b, flipkart_b, snapdeal_b = data_router._resolve_bundle_platform_frames(
        sess, "2026-05-05", "2026-06-04"
    )
    summary = data_router._build_platform_summary_for_bundle(
        sess,
        mtr_b,
        myntra_b,
        meesho_b,
        flipkart_b,
        snapdeal_b,
        "2026-05-05",
        "2026-06-04",
    )
    amazon = next(p for p in summary if p["platform"] == "Amazon")
    myntra = next(p for p in summary if p["platform"] == "Myntra")
    assert amazon["total_units"] == 300
    assert myntra["total_units"] == 50


def test_intelligence_bundle_rejects_empty_cache(client, monkeypatch):
    from backend.session import store

    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)

    client.get("/api/health")
    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    sess._intelligence_bundle_cache.clear()
    sess._intelligence_bundle_cache[
        ("2026-05-05", "2026-06-04", "gross", 10, False)
    ] = {
        "_ts": __import__("time").time(),
        "payload": {
            "status": "ready",
            "sales_summary": {"total_units": 0},
            "platform_summary": [],
        },
    }
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["X"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [77],
        }
    )
    sess.sales_df = pd.DataFrame()

    r = client.get(
        "/api/data/intelligence-bundle",
        params={"start_date": "2026-05-05", "end_date": "2026-06-04", "include_extras": "0"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sales_summary"]["total_units"] >= 77

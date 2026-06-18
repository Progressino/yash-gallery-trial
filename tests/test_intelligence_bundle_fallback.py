"""Regression: Intelligence bundle must not return empty cards when session has window data."""
from __future__ import annotations

import pandas as pd

from tests.conftest import bootstrap_test_session
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


def test_intelligence_bundle_uses_disk_platform_parquets_when_session_empty(monkeypatch, tmp_path):
    """PO-session-only: unified sales may lag; Intelligence reads window from disk parquets."""
    import pandas as pd

    import backend.main as _main

    disk = tmp_path / "warm"
    disk.mkdir()
    monkeypatch.setenv("WARM_CACHE_DIR", str(disk))
    monkeypatch.setenv("WARM_CACHE_PO_SESSION_ONLY", "1")

    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "SKU": ["A1", "A2"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [400, 500],
        }
    )
    amazon.to_parquet(disk / "mtr_df.parquet", index=False)

    sales = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2026-06-01"]),
            "Source": ["Amazon"],
            "Transaction Type": ["Shipment"],
            "Quantity": [10],
            "Sku": ["A1"],
        }
    )

    _main._warm_cache = {"sales_df": sales}
    sess = AppSession()
    _main._copy_warm_cache_to_session(sess)

    frames = data_router._resolve_bundle_platform_frames(sess, "2026-05-18", "2026-06-17")
    mtr_b = frames[0]
    assert len(mtr_b) >= 2
    assert int(mtr_b.loc[mtr_b["Transaction_Type"] == "Shipment", "Quantity"].sum()) == 900


def test_bundle_payload_chart_sparse_rejects_partial_daily_cache():
    payload = {
        "sales_summary": {"total_units": 3191},
        "platform_summary": [
            {
                "platform": "Amazon",
                "loaded": True,
                "total_units": 3191,
                "daily": [
                    {"date": "2026-05-20", "shipments": 1000, "refunds": 0},
                    {"date": "2026-05-21", "shipments": 1000, "refunds": 0},
                    {"date": "2026-06-01", "shipments": 1191, "refunds": 0},
                ],
            }
        ],
    }
    assert data_router._bundle_payload_chart_sparse(
        payload, "2026-05-18", "2026-06-17"
    ) is True
    assert data_router._bundle_payload_has_display_data(payload) is True
    assert data_router._bundle_cache_lookup(
        ("2026-05-18", "2026-06-17", "gross", 10, False),
        {
            ("2026-05-18", "2026-06-17", "gross", 10, False): {
                "_ts": __import__("time").time(),
                "payload": payload,
            }
        },
        start_date="2026-05-18",
        end_date="2026-06-17",
    ) is None


def test_repair_platform_loaded_flags_fixes_legacy_unified_cache():
    payload = {
        "sales_summary": {"total_units": 100},
        "platform_summary": [
            {"platform": "Amazon", "loaded": False, "total_units": 80, "daily": [{"date": "2026-06-01", "shipments": 80}]},
            {"platform": "Flipkart", "loaded": False, "total_units": 0, "daily": []},
        ],
    }
    data_router._repair_platform_loaded_flags(payload)
    assert payload["platform_summary"][0]["loaded"] is True
    assert payload["platform_summary"][1]["loaded"] is False
    assert data_router._bundle_payload_has_display_data(payload) is True


def test_intelligence_bundle_rejects_empty_cache(client, monkeypatch):
    from backend.session import store

    monkeypatch.setattr(data_router, "_schedule_intelligence_refresh_async", lambda _sid: None)
    monkeypatch.setattr(data_router, "_schedule_persist_tier3_window", lambda *a, **k: None)

    sid = bootstrap_test_session(client)
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

"""Regression: PO quarterly build must not crash and must prefer platform history."""
from __future__ import annotations

import pandas as pd

from backend.services.po_engine import calculate_quarterly_history
from backend.services.po_quarterly_warmup import (
    build_quarterly_payload,
    quarterly_cache_key,
    restore_platform_history_for_quarterly,
)
from backend.session import AppSession


def test_quarterly_cache_schema_bumped():
    assert quarterly_cache_key(False, 8)[0] == 13


def test_normalize_quarterly_payload_pads_missing_columns():
    from backend.services.po_quarterly_warmup import (
        expected_quarter_columns,
        normalize_quarterly_payload,
    )

    partial = {
        "loaded": True,
        "columns": ["OMS_SKU", "Apr-Jun 2026", "Jan-Mar 2026"],
        "rows": [{"OMS_SKU": "SKU1", "Apr-Jun 2026": 5, "Jan-Mar 2026": 2}],
    }
    out = normalize_quarterly_payload(partial, n_quarters=8)
    assert len(expected_quarter_columns(8)) == 8
    for col in expected_quarter_columns(8):
        assert col in out["columns"]
        assert out["rows"][0][col] == (5 if col == "Apr-Jun 2026" else 2 if col == "Jan-Mar 2026" else 0)


def test_calculate_quarterly_platform_primary_despite_wide_sales_span():
    """Wide sales_df calendar span must not hide per-SKU history on platform frames."""
    sales = pd.DataFrame(
        {
            "Sku": ["ONLY-IN-SALES"] * 10,
            "TxnDate": pd.date_range("2024-01-01", periods=10, freq="30D"),
            "Transaction Type": ["Shipment"] * 10,
            "Quantity": [1] * 10,
        }
    )
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-11-01", "2025-02-01"]),
            "SKU": ["1001YKBEIGE-M", "1001YKBEIGE-M"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [50, 30],
        }
    )
    pivot = calculate_quarterly_history(
        sales_df=sales,
        mtr_df=mtr,
        sku_mapping=None,
        n_quarters=8,
    )
    row = pivot.loc[pivot["OMS_SKU"] == "1001YKBEIGE-M"].iloc[0]
    assert int(row.get("Oct-Dec 2024", 0)) == 50
    assert int(row.get("Jan-Mar 2025", 0)) == 30


def test_tier3_deep_history_prefers_streaming(monkeypatch):
    """When Tier-3 spans 8 quarters, stream even if session platform span looks wide."""
    from backend.session import AppSession

    sess = AppSession()
    streamed = {"called": False}

    def _fake_stream(*a, **k):
        streamed["called"] = True
        return {
            "loaded": True,
            "columns": ["OMS_SKU", "Oct-Dec 2024"],
            "rows": [{"OMS_SKU": "OLD-SKU", "Oct-Dec 2024": 40}],
        }

    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._ensure_session_operational_frames",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._build_via_streaming",
        lambda *a, **k: _fake_stream(),
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.platform_frames_span_days",
        lambda _s: 900,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.sales_df_span_days",
        lambda _df: 900,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._session_has_platform_rows",
        lambda _s: True,
    )
    monkeypatch.setattr(
        "backend.services.platform_session_window.session_platform_shorter_than_tier3",
        lambda _s: False,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._tier3_covers_quarterly_window",
        lambda _n: True,
    )

    out = build_quarterly_payload(sess, n_quarters=8)
    assert streamed["called"] is True
    assert out.get("loaded") is True


def test_short_session_span_skips_partial_quarterly_and_streams(monkeypatch):
    """A 90-day session must not satisfy quarterly — Tier-3 streaming is used."""
    from backend.session import AppSession

    sess = AppSession()
    sess.sales_df = __import__("pandas").DataFrame(
        {
            "Sku": ["ONLY-IN-SALES"] * 10,
            "TxnDate": __import__("pandas").date_range("2026-03-01", periods=10, freq="7D"),
            "Transaction Type": ["Shipment"] * 10,
            "Quantity": [1] * 10,
        }
    )
    streamed = {"called": False}

    def _fake_stream(*a, **k):
        streamed["called"] = True
        return {
            "loaded": True,
            "columns": ["OMS_SKU", "Apr-Jun 2026"],
            "rows": [{"OMS_SKU": "STREAM-SKU", "Apr-Jun 2026": 9}],
        }

    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._ensure_session_operational_frames",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._build_via_streaming",
        lambda *a, **k: _fake_stream(),
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.platform_frames_span_days",
        lambda _s: 30,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.sales_df_span_days",
        lambda _df: 30,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._session_has_platform_rows",
        lambda _s: True,
    )
    monkeypatch.setattr(
        "backend.services.platform_session_window.session_platform_shorter_than_tier3",
        lambda _s: False,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._tier3_covers_quarterly_window",
        lambda _n: False,
    )

    out = build_quarterly_payload(sess, n_quarters=8)
    assert streamed["called"] is True
    assert out["rows"][0]["OMS_SKU"] == "STREAM-SKU"


def test_tier3_deeper_than_session_uses_streaming(monkeypatch):
    """When Tier-3 has older dates than session frames, stream full history."""
    from backend.session import AppSession

    sess = AppSession()
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-06-01", "2026-06-01"]),
            "SKU": ["165YKRED-M", "165YKRED-M"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [10, 20],
        }
    )
    streamed = {"called": False}

    def _fake_stream(*a, **k):
        streamed["called"] = True
        return {
            "loaded": True,
            "columns": ["OMS_SKU", "Oct-Dec 2024", "Apr-Jun 2026"],
            "rows": [
                {
                    "OMS_SKU": "165YKRED-M",
                    "Oct-Dec 2024": 400,
                    "Apr-Jun 2026": 1200,
                }
            ],
        }

    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._build_via_streaming",
        lambda *a, **k: _fake_stream(),
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.platform_frames_span_days",
        lambda _s: 800,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._session_has_platform_rows",
        lambda _s: True,
    )
    monkeypatch.setattr(
        "backend.services.platform_session_window.session_platform_shorter_than_tier3",
        lambda _s: True,
    )

    out = build_quarterly_payload(sess, n_quarters=8)
    assert streamed["called"] is True
    row = out["rows"][0]
    assert int(row["Oct-Dec 2024"]) == 400


def test_platform_path_merges_unified_sales(monkeypatch):
    """When platform frames exist, quarterly must still include session sales_df."""
    from backend.session import AppSession

    sess = AppSession()
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["UNIFIED-SKU"],
            "TxnDate": pd.to_datetime(["2026-06-15"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [42],
        }
    )
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-10"]),
            "SKU": ["UNIFIED-SKU"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [10],
        }
    )
    seen: dict = {}

    def _fake_quarterly(**kwargs):
        seen["sales_empty"] = kwargs.get("sales_df") is None or kwargs["sales_df"].empty
        return pd.DataFrame({"OMS_SKU": ["UNIFIED-SKU"], "Apr-Jun 2026": [52]})

    monkeypatch.setattr(
        "backend.services.po_engine.calculate_quarterly_history",
        _fake_quarterly,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.platform_frames_span_days",
        lambda _s: 900,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._session_has_platform_rows",
        lambda _s: True,
    )
    monkeypatch.setattr(
        "backend.services.platform_session_window.session_platform_shorter_than_tier3",
        lambda _s: False,
    )

    out = build_quarterly_payload(sess, n_quarters=8)
    assert seen.get("sales_empty") is False
    assert out["rows"][0]["OMS_SKU"] == "UNIFIED-SKU"


def test_hydrate_is_noop(monkeypatch):
    from backend.services import daily_store

    sess = AppSession()
    assert restore_platform_history_for_quarterly(sess, n_quarters=8) is False

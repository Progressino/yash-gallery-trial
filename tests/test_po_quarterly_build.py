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
    assert quarterly_cache_key(False, 8)[0] == 8


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
        "backend.services.po_quarterly_warmup._build_via_streaming",
        lambda *a, **k: _fake_stream(),
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.platform_frames_span_days",
        lambda _s: 30,
    )
    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup._session_has_platform_rows",
        lambda _s: True,
    )

    out = build_quarterly_payload(sess, n_quarters=8)
    assert streamed["called"] is True
    assert out["rows"][0]["OMS_SKU"] == "STREAM-SKU"


def test_hydrate_is_noop(monkeypatch):
    from backend.services import daily_store

    sess = AppSession()
    assert restore_platform_history_for_quarterly(sess, n_quarters=8) is False

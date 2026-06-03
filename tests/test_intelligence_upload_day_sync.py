"""Regression: Tier-3 backfills must appear on Intelligence for the selected calendar day."""

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_tier3_token_mismatch_when_new_backfill_without_max_date_change(monkeypatch):
    """June 1 upload while session max is June 4 must still require Tier-3 merge."""
    from backend.services import daily_store

    store = {"amazon": "3:500:2026-06-04T10:00:00"}
    applied = {"amazon": "2:400:2026-06-03T09:00:00"}
    monkeypatch.setattr(daily_store, "get_tier3_sync_token", lambda: store)

    sess = AppSession()
    sess._tier3_sync_token_applied = applied
    assert data_router._tier3_token_mismatch(sess) is True
    assert data_router._platforms_with_tier3_token_mismatch(sess) == ["amazon"]
    assert data_router._tier3_session_needs_topup(sess) is True


def test_sales_gap_detects_missing_june_1_with_upload_coverage(monkeypatch):
    from backend.services import daily_store

    sales = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2026-06-02", "2026-06-04"]),
            "Transaction Type": ["Shipment", "Shipment"],
            "Quantity": [1, 2],
            "Units_Effective": [1, 2],
            "Source": ["Amazon", "Amazon"],
            "Sku": ["A", "A"],
            "OrderId": ["O1", "O2"],
        }
    )
    sess = AppSession()
    sess.sales_df = sales
    monkeypatch.setattr(
        daily_store,
        "get_upload_report_day_coverage",
        lambda: {"amazon": {"2026-06-01", "2026-06-02", "2026-06-04"}},
    )
    gaps = data_router._platforms_with_sales_gaps_fast(sess, "2026-06-01", "2026-06-01")
    assert gaps == ["amazon"]


def test_merge_report_range_adds_june_1_platform_rows(monkeypatch):
    from backend.services import daily_store
    from backend.services.daily_store import merge_platform_data

    june1 = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [42],
            "OrderId": ["O-j1"],
        }
    )
    sess = AppSession()
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-04"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [1],
            "OrderId": ["O4"],
        }
    )

    monkeypatch.setattr(
        daily_store,
        "load_platform_data_for_report_range",
        lambda platform, start, end, dedup=True: june1 if platform == "amazon" else pd.DataFrame(),
    )
    monkeypatch.setattr(daily_store, "merge_platform_data", merge_platform_data)

    changed = data_router._merge_tier3_for_report_range(
        sess, ["amazon"], "2026-06-01", "2026-06-01"
    )
    assert changed
    j1 = sess.mtr_df[
        pd.to_datetime(sess.mtr_df["Date"]).dt.normalize() == pd.Timestamp("2026-06-01")
    ]
    assert len(j1) == 1
    assert int(j1["Quantity"].iloc[0]) == 42

"""Intelligence must show all platforms when session frames lag Tier-3 uploads."""
from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.services.sales import _platform_summaries_from_unified_bulk
from backend.session import AppSession


def test_empty_unified_sales_uses_raw_platform_frames():
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-30"]),
            "SKU": ["A1"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [500],
        }
    )
    out = _platform_summaries_from_unified_bulk(
        pd.DataFrame(),
        mtr,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        "2026-05-01",
        "2026-06-04",
    )
    amazon = next(p for p in out if p["platform"] == "Amazon")
    assert amazon["loaded"] is True
    assert amazon["total_units"] == 500
    daily = {r["date"]: r["shipments"] for r in amazon.get("daily") or []}
    assert daily.get("2026-05-30") == 500


def test_tier3_gap_fill_when_session_ends_before_window(monkeypatch):
    from backend.services import daily_store

    amazon_t3 = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-30", "2026-06-01"]),
            "SKU": ["A1", "A2"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [100, 200],
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
        lambda plat, s, e, dedup=False, columns_only=False: (
            amazon_t3.copy() if plat == "amazon" else pd.DataFrame()
        ),
    )
    monkeypatch.setattr(
        daily_store,
        "merge_platform_data",
        lambda cur, chunk, plat: pd.concat([cur, chunk], ignore_index=True),
    )

    sess = AppSession()
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-28", "2026-05-29"]),
            "SKU": ["A0", "A0"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [50, 60],
        }
    )

    merged = data_router._platform_df_for_intelligence_bundle(
        sess, "amazon", "mtr_df", "2026-05-01", "2026-06-04"
    )
    assert len(merged) >= 4
    out = _platform_summaries_from_unified_bulk(
        pd.DataFrame(),
        merged,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        "2026-05-01",
        "2026-06-04",
    )
    amazon = next(p for p in out if p["platform"] == "Amazon")
    assert amazon["total_units"] >= 300
    daily = {r["date"]: r["shipments"] for r in amazon.get("daily") or []}
    assert daily.get("2026-05-30") == 100
    assert daily.get("2026-06-01") == 200


def test_unified_sales_lag_uses_newer_tier3_daily():
    """Stale unified sales_df must not zero out chart days that exist in gap-filled frames."""
    from backend.services.sales import _platform_summaries_from_unified_bulk

    sales = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2026-05-28", "2026-05-29"]),
            "Source": ["Amazon", "Amazon"],
            "Transaction Type": ["Shipment", "Shipment"],
            "Quantity": [100, 100],
            "Sku": ["A1", "A1"],
        }
    )
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-28", "2026-05-29", "2026-06-01", "2026-06-02"]),
            "SKU": ["A1", "A1", "A2", "A2"],
            "Transaction_Type": ["Shipment"] * 4,
            "Quantity": [100, 100, 400, 500],
        }
    )
    out = _platform_summaries_from_unified_bulk(
        sales,
        mtr,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        "2026-05-05",
        "2026-06-04",
    )
    amazon = next(p for p in out if p["platform"] == "Amazon")
    daily = {r["date"]: r["shipments"] for r in amazon.get("daily") or []}
    assert daily.get("2026-06-01") == 400
    assert daily.get("2026-06-02") == 500
    assert amazon["total_units"] == 1100

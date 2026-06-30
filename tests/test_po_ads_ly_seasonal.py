"""PO ADS — LY / seasonal must see full Tier-1 history (not 400d trim)."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.services.po_ads_horizon import po_ads_history_horizon_days
from backend.services.po_engine import calculate_po_base


def test_po_ads_horizon_covers_ly_and_seasonal():
    assert po_ads_history_horizon_days(30, use_ly_fallback=True) >= 409
    assert po_ads_history_horizon_days(30, use_seasonality=True) >= 800
    assert po_ads_history_horizon_days(90, use_ly_fallback=True, use_seasonality=True) >= 800


def test_ly_and_seasonal_use_prior_calendar_quarter():
    """SKU with strong Apr–Jun 2025 but weak recent window must not show LY/Season ≈ 0."""
    sku = "1003YKMUSTARD-5XL"
    rows: list[dict] = []
    # Recent 30d (June 2026): moderate
    for d in pd.date_range("2026-06-01", "2026-06-30", freq="D"):
        rows.append(
            {
                "Sku": sku,
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    # Prior calendar Q2 2025: 154 units spread Apr–Jun
    for d in pd.date_range("2025-04-01", "2025-06-30", freq="D"):
        rows.append(
            {
                "Sku": sku,
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": [sku], "Total_Inventory": [50]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=45,
        target_days=60,
        planning_date="2026-06-30",
        demand_basis="Sold",
        use_ly_fallback=True,
        use_seasonality=True,
        seasonal_weight=0.5,
    )
    row = po.loc[po["OMS_SKU"] == sku].iloc[0]
    # Q2 2025 ≈ 91 days → ~1.7/day; parallel LY (May–Jun 2025) should be >> 0.1
    assert float(row["LY_ADS"]) >= 1.0, f"LY_ADS={row['LY_ADS']}"
    assert float(row["Seasonal_Month_ADS"]) >= 0.5, f"Seasonal={row['Seasonal_Month_ADS']}"
    assert float(row["ADS"]) >= 0.77, f"ADS={row['ADS']}"


def test_quarterly_ly_floor_dict_computes_rate_from_prior_year_quarter():
    """quarterly_ly_floor_dict() must convert the prior-year quarter column units
    into a per-day rate, fetched BEFORE calculate_po_base (not post-hoc) so
    Days_Left / Projected_Running_Days stay consistent with the final ADS."""
    from backend.services.po_quarterly_warmup import quarterly_ly_floor_dict
    from backend.session import AppSession

    sku = "1003YKMUSTARD-5XL"
    sess = AppSession()
    payload = {
        "loaded": True,
        "columns": ["OMS_SKU", "Apr-Jun 2025"],
        "rows": [{"OMS_SKU": sku, "Apr-Jun 2025": 154}],
    }

    import backend.services.po_quarterly_warmup as qw

    orig = qw.get_quarterly_payload_for_po
    qw.get_quarterly_payload_for_po = lambda *_a, **_kw: payload
    try:
        floor = quarterly_ly_floor_dict(sess, planning_date="2026-06-30")
    finally:
        qw.get_quarterly_payload_for_po = orig

    # 154 units / 91 days ≈ 1.692/day LY signal.
    assert floor.get(sku, 0) >= 1.5

    sales = pd.DataFrame(
        {
            "Sku": [sku] * 4,
            "TxnDate": pd.date_range("2026-06-01", periods=4, freq="D"),
            "Transaction Type": ["Shipment"] * 4,
            "Quantity": [1, 1, 1, 1],
            "Units_Effective": [1, 1, 1, 1],
        }
    )
    inv = pd.DataFrame({"OMS_SKU": [sku], "Total_Inventory": [100]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=60,
        target_days=180,
        demand_basis="Sold",
        safety_pct=0.0,
        use_seasonality=False,
        use_ly_fallback=True,
        quarterly_ly_floor=floor,
    )
    row = po.iloc[0]
    recent = float(row["Recent_ADS"])
    ads = float(row["ADS"])
    days_left = float(row["Days_Left"])
    # ADS blends recent + the quarterly-floored LY (50/50) when seasonality off.
    expected_blend = round(recent * 0.5 + floor[sku] * 0.5, 3)
    assert ads == pytest.approx(expected_blend, abs=0.01), (
        f"ADS should blend recent={recent} and floored LY={floor[sku]}, got {ads}"
    )
    assert ads > recent, "quarterly floor should lift ADS above the raw recent signal"
    # Days_Left must be derived from this SAME final ADS, not a stale pre-floor one.
    assert days_left == pytest.approx(round(100.0 / ads, 1), abs=0.1)


def test_ly_blend_midpoint_without_seasonality():
    from backend.services.po_engine import _primary_ads_from_signals, _final_ads_from_signals

    prim = _primary_ads_from_signals(0.7, 1.7, use_seasonality=False, use_ly_fallback=True, seasonal_weight=0.5)
    ads = float(_final_ads_from_signals(prim, 1.7, 0.7, use_seasonality=False))
    assert abs(ads - 1.2) < 0.05


def test_ly_fallback_still_lifts_when_only_forward_window_has_sales():
    rows = []
    for d in pd.date_range("2025-11-01", periods=5, freq="D"):
        rows.append(
            {
                "Sku": "ANCHOR-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    for d in pd.date_range("2024-11-01", periods=30, freq="D"):
        rows.append(
            {
                "Sku": "LY-ONLY-SKU-1",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 2,
                "Units_Effective": 2,
                "Source": "Amazon",
            }
        )
    sales = pd.DataFrame(rows)
    inv = pd.DataFrame({"OMS_SKU": ["LY-ONLY-SKU-1", "ANCHOR-SKU-1"], "Total_Inventory": [40, 10]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=7,
        target_days=60,
        demand_basis="Sold",
        safety_pct=0.0,
        group_by_parent=False,
        use_ly_fallback=True,
    )
    row = po[po["OMS_SKU"] == "LY-ONLY-SKU-1"].iloc[0]
    assert float(row["Recent_ADS"]) == 0.0
    assert float(row["LY_ADS"]) > 0.0
    assert float(row["ADS"]) > 0.0

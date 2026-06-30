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

"""Tests for PO dashboard sales windows + payload builder."""

import pandas as pd

from backend.services.po_dashboard import build_dashboard_payload, recent_vs_prev_shipments


def test_recent_vs_prev_shipments_two_windows():
    # 14 days of data: days 1-7 prev window, 8-14 recent (if end is day 14)
    dates = pd.date_range("2026-05-01", periods=14, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["SPIKE-A"] * 14,
            "TxnDate": dates,
            "Transaction Type": ["Shipment"] * 14,
            "Quantity": [1] * 7 + [5] * 7,
            "Units_Effective": [1] * 7 + [5] * 7,
            "Source": ["Amazon"] * 14,
        }
    )
    out = recent_vs_prev_shipments(
        sales, sku_mapping=None, group_by_parent=False, recent_days=7, prev_days=7
    )
    row = out[out["OMS_SKU"] == "SPIKE-A"].iloc[0]
    assert int(row["units_prev"]) == 7
    assert int(row["units_recent"]) == 35


def test_build_dashboard_spike_section():
    po = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B"],
            "Parent_SKU": ["PA", "PB"],
            "PO_Pipeline_Total": [100, 0],
            "Raised_Recently_Units": [0, 0],
            "PO_Qty": [0, 10],
            "ADS": [2.0, 1.0],
            "Total_Inventory": [10, 5],
            "Projected_Running_Days": [55.0, 25.0],
            "Days_Left": [5.0, 5.0],
            "Lead_Time_Days": [45, 45],
            "Priority": ["⚪ OK", "🔴 URGENT"],
            "SKU_Sheet_Status": ["", ""],
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": ["A"] * 14,
            "TxnDate": pd.date_range("2026-05-01", periods=14, freq="D"),
            "Transaction Type": ["Shipment"] * 14,
            "Quantity": [1] * 7 + [4] * 7,
            "Units_Effective": [1] * 7 + [4] * 7,
            "Source": ["X"] * 14,
        }
    )
    dash = build_dashboard_payload(
        po,
        sales,
        recent_days=7,
        prev_days=7,
        spike_ratio=1.5,
        min_recent_units=20,
        low_run_days=50.0,
        max_rows_per_section=20,
        lead_time_default=45,
    )
    assert dash["ok"] is True
    assert dash["summary"]["in_production_skus"] == 1
    assert dash["summary"]["open_po_skus"] == 1
    # A: strong spike (28 vs 7) + days_left 5 < low_run 50 → needs attention
    assert any(r["OMS_SKU"] == "A" for r in dash["spike_attention"])
    # B: tight cover 25 < 50
    assert any(r["OMS_SKU"] == "B" for r in dash["running_tight"])

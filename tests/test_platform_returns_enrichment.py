"""Return data must be picked from uploads without double-counting."""
from __future__ import annotations

import pandas as pd

from backend.services.sales import (
    RETURN_SHEET_ORDER_PLACEHOLDER,
    _compute_platform_metrics,
    enrich_platform_summaries_with_all_returns,
)


def test_upload_blob_counts_refunds_outside_ship_window():
    """Refunds in overlapping daily uploads but dated before window still count."""
    df = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2026-05-20"),
                pd.Timestamp("2026-05-26"),
                pd.Timestamp("2026-05-26"),
            ],
            "Transaction_Type": ["Refund", "Shipment", "Refund"],
            "Quantity": [50, 100, 3],
            "SKU": ["A", "B", "B"],
        }
    )
    cal = _compute_platform_metrics(
        df, "Amazon", "SKU", "Transaction_Type",
        start_date="2026-05-25", end_date="2026-05-30",
        refund_scope="calendar",
    )
    blob = _compute_platform_metrics(
        df, "Amazon", "SKU", "Transaction_Type",
        start_date="2026-05-25", end_date="2026-05-30",
        refund_scope="upload_blob",
    )
    assert cal["total_returns"] == 3
    assert blob["total_returns"] == 53
    assert blob["total_units"] == 100


def test_enrich_uses_unified_sales_without_return_sheet_double_count():
    sales = pd.DataFrame(
        {
            "Source": ["Flipkart", "Flipkart"],
            "Transaction Type": ["Shipment", "Refund"],
            "Quantity": [100, 12],
            "TxnDate": [pd.Timestamp("2026-05-26")] * 2,
            "OrderId": ["O1", "O1"],
            "Sku": ["S1", "S1"],
            "Units_Effective": [100, -12],
        }
    )
    cards = [{"platform": "Flipkart", "loaded": True, "total_units": 100, "total_returns": 0, "net_units": 100, "return_rate": 0.0}]
    out = enrich_platform_summaries_with_all_returns(
        cards, sales, "2026-05-25", "2026-05-30"
    )
    assert out[0]["total_returns"] == 12

    sales_overlay = sales.copy()
    sales_overlay.loc[1, "OrderId"] = RETURN_SHEET_ORDER_PLACEHOLDER
    out2 = enrich_platform_summaries_with_all_returns(
        cards, sales_overlay, "2026-05-25", "2026-05-30"
    )
    assert out2[0]["total_returns"] == 0

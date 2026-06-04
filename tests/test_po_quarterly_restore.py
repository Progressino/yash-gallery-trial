"""PO quarterly must load enough Tier-3 history for 8 FY quarter columns."""
from __future__ import annotations

import pandas as pd

from backend.services.po_quarterly_warmup import (
    platform_frames_span_days,
    quarterly_restore_months,
    sales_df_span_days,
)
from backend.session import AppSession


def test_quarterly_restore_months_default_is_full_history():
    assert quarterly_restore_months(8) is None


def test_platform_frames_span_detects_deep_bulk():
    sess = AppSession()
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-06-01", periods=400, freq="D"),
            "SKU": ["A"] * 400,
            "Transaction_Type": ["Shipment"] * 400,
            "Quantity": [1] * 400,
        }
    )
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["A"] * 30,
            "TxnDate": pd.date_range("2026-05-01", periods=30, freq="D"),
            "Transaction Type": ["Shipment"] * 30,
            "Quantity": [1] * 30,
        }
    )
    assert sales_df_span_days(sess.sales_df) < 60
    assert platform_frames_span_days(sess) > 300

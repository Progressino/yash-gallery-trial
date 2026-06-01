"""PO quarterly warmup — shallow sales_df rebuild."""

import pandas as pd

from backend.services.po_quarterly_warmup import ensure_sales_history_for_quarterly, sales_df_span_days
from backend.session import AppSession


def test_sales_df_span_days():
    df = pd.DataFrame(
        {
            "Sku": ["A"],
            "TxnDate": pd.date_range("2025-01-01", periods=100, freq="D"),
            "Transaction Type": ["Shipment"] * 100,
            "Quantity": [1] * 100,
        }
    )
    assert sales_df_span_days(df) == 99


def test_ensure_sales_history_skips_when_span_wide():
    sess = AppSession()
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["A"] * 600,
            "TxnDate": pd.date_range("2023-01-01", periods=600, freq="D"),
            "Transaction Type": ["Shipment"] * 600,
            "Quantity": [1] * 600,
        }
    )
    sess.mtr_df = pd.DataFrame({"Date": ["2020-01-01"], "SKU": ["X"], "Transaction_Type": ["Shipment"], "Quantity": [1]})
    assert ensure_sales_history_for_quarterly(sess) is False

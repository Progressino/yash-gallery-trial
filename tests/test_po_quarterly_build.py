"""Regression: PO quarterly build must not crash and must prefer platform history."""
from __future__ import annotations

import pandas as pd

from backend.services.po_engine import calculate_quarterly_history
from backend.services.po_quarterly_warmup import (
    quarterly_cache_key,
    restore_platform_history_for_quarterly,
)
from backend.session import AppSession


def test_quarterly_cache_schema_bumped():
    assert quarterly_cache_key(False, 8)[0] == 7


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


def test_hydrate_is_noop(monkeypatch):
    from backend.services import daily_store

    sess = AppSession()
    assert restore_platform_history_for_quarterly(sess, n_quarters=8) is False

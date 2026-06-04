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
    assert quarterly_cache_key(False, 8)[0] == 3


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


def test_restore_platform_history_calls_build_sales_df_with_snapdeal_df(monkeypatch):
    from backend.services import daily_store

    calls: list[dict] = []

    def _fake_build(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr("backend.services.sales.build_sales_df", _fake_build)
    monkeypatch.setattr(
        daily_store,
        "load_platform_data",
        lambda plat, *a, **k: pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-06-01"]),
                "SKU": ["X1"],
                "Transaction_Type": ["Shipment"],
                "Quantity": [1],
            }
        )
        if plat == "amazon"
        else pd.DataFrame(),
    )
    monkeypatch.setattr(
        daily_store,
        "merge_platform_data",
        lambda cur, new, plat: new,
    )

    sess = AppSession()
    sess.mtr_df = pd.DataFrame()
    assert restore_platform_history_for_quarterly(sess, n_quarters=8) is True
    assert calls and "snapdeal_df" in calls[0]
    assert "snapdeal" not in calls[0]

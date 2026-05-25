"""Incremental sales patch after daily upload."""

import pandas as pd

from backend.services.sales import patch_sales_df_after_daily_upload, txn_reporting_naive_ist


def test_patch_sales_df_replaces_window_per_source():
    existing = pd.DataFrame(
        {
            "Sku": ["A", "B", "C"],
            "TxnDate": pd.to_datetime(
                ["2026-05-22", "2026-05-23", "2026-05-24"]
            ),
            "Transaction Type": ["Shipment"] * 3,
            "Quantity": [1, 100, 2],
            "Source": ["Amazon", "Meesho", "Amazon"],
            "OrderId": ["1", "2", "3"],
        }
    )
    fresh = pd.DataFrame(
        {
            "Sku": ["A2"],
            "TxnDate": pd.to_datetime(["2026-05-23"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [50],
            "Source": ["Meesho"],
            "OrderId": ["9"],
        }
    )
    d0 = pd.Timestamp("2026-05-23")
    d1 = pd.Timestamp("2026-05-24")
    out = patch_sales_df_after_daily_upload(existing, fresh, ["meesho"], d0, d1)
    meesho = out[out["Source"] == "Meesho"]
    assert int(meesho["Quantity"].sum()) == 50
    assert int(out.loc[out["TxnDate"].dt.date == pd.Timestamp("2026-05-22").date(), "Quantity"].sum()) == 1
    # Amazon rows are untouched (May 22 + May 24 only).
    assert int(out.loc[out["Source"] == "Amazon", "Quantity"].sum()) == 3

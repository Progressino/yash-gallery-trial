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


def test_patch_sales_df_return_overlay_replaces_synthetic_refunds():
    from backend.services.sales import (
        RETURN_SHEET_ORDER_PLACEHOLDER,
        patch_sales_df_return_overlay,
    )

    existing = pd.DataFrame(
        {
            "Sku": ["SHIP-A", "RET-OLD"],
            "TxnDate": pd.to_datetime(["2026-06-20", "2026-06-21"]),
            "Transaction Type": ["Shipment", "Refund"],
            "Quantity": [10, 3],
            "Units_Effective": [10, -3],
            "Source": ["Amazon", "Return_Sheet"],
            "OrderId": ["ORD-1", RETURN_SHEET_ORDER_PLACEHOLDER],
            "LineKey": ["", "RETURN_SHEET|RET-OLD|Return_Sheet|2026-06-21"],
        }
    )
    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["RET-NEW"],
            "Return_Platform": ["combined"],
            "Return_Units": [3],
        }
    )
    out = patch_sales_df_return_overlay(
        existing,
        overlay,
        {},
        return_overlay_as_of="2026-06-21",
    )
    assert "RET-OLD" not in set(out["Sku"].astype(str))
    assert "SHIP-A" in set(out["Sku"].astype(str))
    ret = out[out["Sku"] == "RET-NEW"]
    assert int(ret["Quantity"].sum()) == 3
    assert (ret["Transaction Type"] == "Refund").all()


def test_patch_sales_df_return_overlay_no_copy_when_only_appending():
    from backend.services.sales import patch_sales_df_return_overlay

    existing = pd.DataFrame(
        {
            "Sku": ["SHIP-A"],
            "TxnDate": pd.to_datetime(["2026-06-20"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [10],
            "Units_Effective": [10],
            "Source": ["Amazon"],
            "OrderId": ["ORD-1"],
            "LineKey": [""],
        }
    )
    overlay = pd.DataFrame(
        {"OMS_SKU": ["RET-NEW"], "Return_Platform": ["combined"], "Return_Units": [3]}
    )
    out = patch_sales_df_return_overlay(
        existing,
        overlay,
        {},
        return_overlay_as_of="2026-06-21",
    )
    assert len(out) == 2
    assert out is not existing

"""Channel split (OMS vs Amazon) and daily sales history matrix."""
import pandas as pd

from backend.services.daily_inventory_history import (
    _channel_from_sheet,
    combine_inventory_channels,
    filter_inventory_history_channel,
    inventory_channel_split_available,
    inventory_history_wide_matrix,
)
from backend.services.daily_sales_history import (
    sales_history_for_sku,
    sales_history_summary,
    sales_history_wide_matrix,
)


def _hist(sku: str, dates: list[str], qtys: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OMS_SKU": [sku] * len(dates),
            "Date": pd.to_datetime(dates),
            "Qty": qtys,
            "Source": ["uploaded"] * len(dates),
        }
    )


def test_channel_from_sheet_names():
    assert _channel_from_sheet("OMS Inventory") == "oms"
    assert _channel_from_sheet("Amazon Inventory") == "amazon"
    assert _channel_from_sheet("FBA Stock") == "amazon"


def test_parse_keeps_oms_and_amazon_separate():
    oms = _hist("SKU-A", ["2026-06-01"], [10.0])
    oms["Channel"] = "oms"
    amz = _hist("SKU-A", ["2026-06-01"], [4.0])
    amz["Channel"] = "amazon"
    raw = pd.concat([oms, amz], ignore_index=True)
    combined = combine_inventory_channels(raw)
    assert float(combined.iloc[0]["Qty"]) == 10.0
    oms_only = filter_inventory_history_channel(raw, "oms")
    assert len(oms_only) == 1
    assert float(oms_only.iloc[0]["Qty"]) == 10.0
    amz_only = filter_inventory_history_channel(raw, "amazon")
    assert float(amz_only.iloc[0]["Qty"]) == 4.0


def test_inventory_matrix_channel_split_flag():
    oms = _hist("SKU-A", ["2026-06-01", "2026-06-02"], [5.0, 5.0])
    oms["Channel"] = "oms"
    amz = _hist("SKU-A", ["2026-06-01", "2026-06-02"], [2.0, 3.0])
    amz["Channel"] = "amazon"
    hist = pd.concat([oms, amz], ignore_index=True)
    assert inventory_channel_split_available(hist) is True
    wide_oms = inventory_history_wide_matrix(hist, days=2, end_date="2026-06-02", channel="oms")
    assert wide_oms["channel_split_available"] is True
    assert wide_oms["rows"][0]["qtys"] == [5.0, 5.0]
    wide_amz = inventory_history_wide_matrix(hist, days=2, end_date="2026-06-02", channel="amazon")
    assert wide_amz["rows"][0]["qtys"] == [2.0, 3.0]


def test_sales_history_wide_matrix_aggregates_net_units():
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A", "SKU-B"],
            "TxnDate": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-02"]),
            "Units_Effective": [3.0, -1.0, 5.0],
            "Source": ["Amazon", "Amazon", "Myntra"],
            "Transaction Type": ["Shipment", "Refund", "Shipment"],
        }
    )
    summary = sales_history_summary(sales, days=2, end_date="2026-06-02")
    assert summary["loaded"] is True
    assert summary["skus"] == 2
    assert summary["total_units"] == 7.0
    wide = sales_history_wide_matrix(sales, days=2, end_date="2026-06-02")
    assert wide["loaded"] is True
    assert wide["dates"] == ["2026-06-01", "2026-06-02"]
    by_sku = {r["sku"]: r["units"] for r in wide["rows"]}
    assert by_sku["SKU-A"] == [2.0, 0.0]
    assert by_sku["SKU-B"] == [0.0, 5.0]
    plat = sales_history_wide_matrix(sales, days=2, end_date="2026-06-02", platform="Amazon")
    assert plat["rows"][0]["units"] == [2.0, 0.0]


def test_sales_history_for_sku_timeline():
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "Units_Effective": [4.0, 1.0],
            "Source": ["Amazon", "Amazon"],
        }
    )
    out = sales_history_for_sku(sales, "SKU-A", window_days=2, end_date="2026-06-02")
    assert out["net_units"] == 5.0
    assert len(out["rows"]) == 2

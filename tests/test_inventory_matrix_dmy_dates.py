"""Wide inventory matrix CSV uses D-M-YY column headers (28-5-26 = 28 May 2026)."""

from io import BytesIO, StringIO

import pandas as pd
import pytest

from backend.services.daily_inventory_history import (
    _parse_inventory_snapshot_date,
    parse_daily_inventory_history_upload,
)
from backend.services.po_engine import calculate_po_base


def test_dmy_header_dates_are_day_first():
    assert _parse_inventory_snapshot_date("28-5-26") == pd.Timestamp("2026-05-28")
    assert _parse_inventory_snapshot_date("1-6-26") == pd.Timestamp("2026-06-01")
    assert _parse_inventory_snapshot_date("5-6-26") == pd.Timestamp("2026-06-05")
    assert _parse_inventory_snapshot_date("6-5-26") == pd.Timestamp("2026-05-06")
    assert _parse_inventory_snapshot_date("25-6-26") == pd.Timestamp("2026-06-25")


def test_inventory_matrix_csv_parses_may_june_window():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
SKU-B,0,0,0,0,0,,0
"""
    df = parse_daily_inventory_history_upload(BytesIO(csv.encode()), "inventory-matrix.csv")
    assert not df.empty
    dates = sorted(df["Date"].dt.strftime("%Y-%m-%d").unique())
    assert dates == [
        "2026-05-28",
        "2026-05-29",
        "2026-06-01",
        "2026-06-05",
        "2026-06-25",
    ]
    a = df[df["OMS_SKU"] == "SKU-A"].set_index("Date")["Qty"]
    assert int(a[pd.Timestamp("2026-06-05")]) == 5
    assert int(a[pd.Timestamp("2026-06-25")]) == 8


def test_nat_cells_do_not_crash_parser():
    import pandas as pd
    from backend.services.daily_inventory_history import _is_date_value, _parse_one_sheet

    assert not _is_date_value(pd.NaT)
    df = pd.DataFrame(
        [
            ["Item SkuCode", "Item", pd.NaT, "2026-06-01", "2026-06-02"],
            ["SKU-A", "PARENT", 5, 6, 7],
        ]
    )
    tall = _parse_one_sheet(df, {})
    assert not tall.empty
    assert set(tall["Date"].dt.strftime("%Y-%m-%d")) == {"2026-06-01", "2026-06-02"}


@pytest.mark.skipif(
    not __import__("pathlib").Path("/Users/samraisinghani/Downloads/inventory-matrix.csv").is_file(),
    reason="user fixture not on disk",
)
def test_user_inventory_matrix_fixture_range():
    from pathlib import Path

    raw = Path("/Users/samraisinghani/Downloads/inventory-matrix.csv").read_bytes()
    df = parse_daily_inventory_history_upload(__import__("io").BytesIO(raw), "inventory-matrix.csv")
    assert df["Date"].min() == pd.Timestamp("2026-05-28")
    assert df["Date"].max() == pd.Timestamp("2026-06-25")
    assert int(df["Date"].nunique()) == 29


def test_po_eff_days_use_june_inventory_window():
    """June in-stock days in the matrix should lift Eff_Days for active sellers."""
    hist_dates = pd.date_range("2026-05-28", "2026-06-25", freq="D")
    inv_hist = pd.DataFrame(
        {
            "OMS_SKU": ["1310YKBLUE-M"] * len(hist_dates),
            "Date": hist_dates,
            "Qty": [10] * len(hist_dates),
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": ["1310YKBLUE-M"] * 4,
            "TxnDate": pd.to_datetime(
                ["2026-06-01", "2026-06-10", "2026-06-15", "2026-06-20"]
            ),
            "Transaction Type": ["Shipment"] * 4,
            "Quantity": [2, 2, 2, 2],
            "Units_Effective": [2, 2, 2, 2],
            "Source": ["Amazon"] * 4,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["1310YKBLUE-M"], "Total_Inventory": [21]})
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=30,
        lead_time=45,
        target_days=180,
        demand_basis="Sold",
        safety_pct=0.0,
        inventory_history_df=inv_hist,
        planning_date="2026-06-25",
        enforce_lead_time_release_gate=False,
    )
    row = po.loc[po["OMS_SKU"] == "1310YKBLUE-M"].iloc[0]
    assert int(row["Sold_Units"]) == 8
    assert int(row.get("Eff_Days_Inventory", 0) or 0) >= 25
    assert float(row["Eff_Days"]) >= 25

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
    sales_history_platform_filters,
    sales_history_summary,
    sales_history_upload_coverage,
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


def test_sales_history_summary_default_end_date_excludes_today_ist():
    """Default window ends yesterday IST — today's sales upload is always next day."""
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    yesterday = today - pd.Timedelta(days=1)
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A"],
            "TxnDate": [today, yesterday],
            "Units_Effective": [99.0, 2.0],
            "Source": ["Amazon", "Amazon"],
            "Transaction Type": ["Shipment", "Shipment"],
        }
    )
    summary = sales_history_summary(sales, days=7)
    assert summary["loaded"] is True
    assert summary["total_units"] == 2.0
    assert summary["max_date"] == str(yesterday.date())


def test_sales_history_summary_explicit_end_includes_today_row():
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"],
            "TxnDate": [today],
            "Units_Effective": [2.0],
            "Source": ["Amazon"],
            "Transaction Type": ["Shipment"],
        }
    )
    summary = sales_history_summary(sales, days=1, end_date=str(today.date()))
    assert summary["loaded"] is True
    assert summary["total_units"] == 2.0


def test_sales_history_upload_coverage_lists_missing_platforms(monkeypatch):
    def _fake_coverage():
        return {
            "amazon": {"2026-06-01", "2026-06-02"},
            "flipkart": {"2026-06-01", "2026-06-02"},
            "meesho": {"2026-06-01"},
            "myntra": {"2026-06-01", "2026-06-02"},
        }

    monkeypatch.setattr(
        "backend.services.daily_store.get_upload_report_day_coverage",
        _fake_coverage,
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-02"]),
            "Units_Effective": [1.0],
            "Source": ["Amazon"],
        }
    )
    out = sales_history_upload_coverage(sales_df=sales, days=2, end_date="2026-06-02")
    gaps = {g["date"]: g["missing_platforms"] for g in out["coverage_gaps"]}
    assert gaps["2026-06-02"] == ["meesho"]
    assert "2026-06-01" not in gaps
    wide = sales_history_wide_matrix(sales, days=2, end_date="2026-06-02")
    assert "coverage_gaps" in wide


def test_sales_history_platform_filters_tier3_amazon_and_no_nan(monkeypatch):
    monkeypatch.setattr(
        "backend.services.daily_store.get_upload_report_day_coverage",
        lambda: {
            "amazon": {"2026-06-02"},
            "flipkart": set(),
            "meesho": set(),
            "myntra": set(),
        },
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-02"]),
            "Units_Effective": [1.0],
            "Source": ["nan"],
        }
    )
    plats = sales_history_platform_filters(sales, days=2, end_date="2026-06-02")
    assert "Amazon" in plats
    assert all(p.strip().lower() not in ("nan", "none") for p in plats)


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


def test_sales_history_excludes_combo_fan_component_rows():
    """Combo fan copies (_Combo_Fan=True) must not inflate Sales History units."""
    sales = pd.DataFrame(
        {
            "Sku": ["1037DPT19WHITE-4XL", "1037YKBLUE-4XL", "DPT19WHITE"],
            "TxnDate": pd.to_datetime(["2026-07-17"] * 3),
            "Transaction Type": ["Shipment"] * 3,
            "Quantity": [1.0, 1.0, 1.0],
            "Units_Effective": [1.0, 1.0, 1.0],
            "Source": ["Amazon"] * 3,
            "_Combo_Fan": [False, True, True],
        }
    )
    wide = sales_history_wide_matrix(sales, days=1, end_date="2026-07-17", platform="Amazon")
    assert wide["date_totals"] == [1.0]


def test_sales_history_meesho_credit_entry_excludes_cancel_and_refund():
    """Meesho CANCELLED/RTO must not inflate Sales History (ops File = sale statuses only)."""
    from backend.services.daily_sales_history import _apply_daily_sales_history_txn_fixup

    sales = pd.DataFrame(
        {
            "Sku": ["A", "B", "C", "D"],
            "TxnDate": pd.to_datetime(["2026-07-20"] * 4),
            "Transaction Type": ["Shipment", "Cancel", "Refund", "Shipment"],
            "Quantity": [100.0, 25.0, 1.0, 98.0],
            "Units_Effective": [100.0, -25.0, -1.0, 98.0],
            "Source": ["Meesho"] * 4,
        }
    )
    fixed = _apply_daily_sales_history_txn_fixup(sales)
    wide = sales_history_wide_matrix(fixed, days=1, end_date="2026-07-20", platform="Meesho")
    assert wide["date_totals"] == [198.0]
    # Myntra Cancel still counts as sale
    myn = pd.DataFrame(
        {
            "Sku": ["M1", "M2"],
            "TxnDate": pd.to_datetime(["2026-07-20"] * 2),
            "Transaction Type": ["Shipment", "Cancel"],
            "Quantity": [10.0, 5.0],
            "Units_Effective": [10.0, 0.0],
            "Source": ["Myntra"] * 2,
        }
    )
    mfixed = _apply_daily_sales_history_txn_fixup(myn)
    assert float(mfixed["Units_Effective"].sum()) == 15.0

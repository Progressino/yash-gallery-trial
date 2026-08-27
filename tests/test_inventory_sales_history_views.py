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


def test_blank_channel_history_served_as_oms():
    """Snapshot/legacy history uses blank Channel — OMS tab must not return empty zeros."""
    hist = _hist("SKU-A", ["2026-07-16", "2026-07-17"], [100.0, 95.0])
    hist["Channel"] = ""
    assert inventory_channel_split_available(hist) is False
    oms = filter_inventory_history_channel(hist, "oms")
    assert len(oms) == 2
    assert float(oms["Qty"].sum()) == 195.0
    amz = filter_inventory_history_channel(hist, "amazon")
    assert amz.empty
    wide = inventory_history_wide_matrix(hist, days=2, end_date="2026-07-17", channel="oms")
    assert wide["loaded"] is True
    assert wide["date_totals"][-1] == 95.0
    assert sum(wide["date_totals"]) > 0


def test_oms_channel_uses_legacy_blank_dates_and_new_explicit_dates():
    legacy = _hist("SKU-A", ["2026-07-15"], [100.0])
    legacy["Channel"] = ""
    oms = _hist("SKU-A", ["2026-07-16"], [90.0])
    oms["Channel"] = "oms"
    amz = _hist("SKU-A", ["2026-07-16"], [20.0])
    amz["Channel"] = "amazon"
    hist = pd.concat([legacy, oms, amz], ignore_index=True)

    oms_only = filter_inventory_history_channel(hist, "oms")
    got = dict(zip(oms_only["Date"].dt.strftime("%Y-%m-%d"), oms_only["Qty"]))
    assert got == {"2026-07-15": 100.0, "2026-07-16": 90.0}
    amazon_only = filter_inventory_history_channel(hist, "amazon")
    assert list(amazon_only["Qty"]) == [20.0]
    combined = combine_inventory_channels(hist)
    totals = combined.groupby("Date")["Qty"].sum()
    assert float(totals[pd.Timestamp("2026-07-15")]) == 100.0
    assert float(totals[pd.Timestamp("2026-07-16")]) == 90.0


def test_oms_channel_keeps_blank_skus_on_hybrid_days():
    """Same-day blank-only SKUs must not be dropped when other SKUs have channel=oms."""
    oms_a = _hist("SKU-A", ["2026-08-05"], [100.0])
    oms_a["Channel"] = "oms"
    blank_b = _hist("SKU-B", ["2026-08-05"], [50.0])
    blank_b["Channel"] = ""
    amz = _hist("SKU-A", ["2026-08-05"], [9.0])
    amz["Channel"] = "amazon"
    hist = pd.concat([oms_a, blank_b, amz], ignore_index=True)

    oms_only = filter_inventory_history_channel(hist, "oms")
    by_sku = dict(zip(oms_only["OMS_SKU"].astype(str), oms_only["Qty"].astype(float)))
    assert by_sku == {"SKU-A": 100.0, "SKU-B": 50.0}
    assert float(oms_only["Qty"].sum()) == 150.0


def test_align_history_day_matches_variant_oms_total():
    from backend.services.daily_inventory_history import (
        align_history_day_to_variant,
        filter_inventory_history_channel,
        inventory_history_wide_matrix,
    )

    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-08-05"], [80.0]),
            _hist("SKU-B", ["2026-08-05"], [20.0]),
            _hist("SKU-A", ["2026-08-04"], [70.0]),
        ],
        ignore_index=True,
    )
    hist["Channel"] = "oms"
    hist["Source"] = "snapshot"
    # Stale history day total 100; Actual / Inventory tab OMS is 198.
    variant = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B", "SKU-C"],
            "OMS_Inventory": [100.0, 50.0, 48.0],
            "Amazon_Inventory": [10.0, 0.0, 0.0],
        }
    )
    out = align_history_day_to_variant(hist, variant, "2026-08-05")
    oms = filter_inventory_history_channel(out, "oms")
    day = oms[pd.to_datetime(oms["Date"]).dt.normalize() == pd.Timestamp("2026-08-05")]
    assert float(day["Qty"].sum()) == 198.0
    wide = inventory_history_wide_matrix(out, days=2, end_date="2026-08-05", channel="oms")
    assert wide["date_totals"][-1] == 198.0
    # Prior day preserved
    prev = oms[pd.to_datetime(oms["Date"]).dt.normalize() == pd.Timestamp("2026-08-04")]
    assert float(prev["Qty"].sum()) == 70.0
    # Amazon zeros are kept so the day counts as uploaded on Amazon FBA tab.
    amz = filter_inventory_history_channel(out, "amazon")
    amz_day = amz[pd.to_datetime(amz["Date"]).dt.normalize() == pd.Timestamp("2026-08-05")]
    assert len(amz_day) == 3
    assert float(amz_day.loc[amz_day["OMS_SKU"] == "SKU-A", "Qty"].iloc[0]) == 10.0
    assert float(amz_day.loc[amz_day["OMS_SKU"] == "SKU-B", "Qty"].iloc[0]) == 0.0


def test_inventory_history_wide_matrix_csv_includes_total_row():
    from backend.services.daily_inventory_history import inventory_history_wide_matrix_csv

    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-07-16", "2026-07-17"], [10.0, 8.0]),
            _hist("SKU-B", ["2026-07-16", "2026-07-17"], [5.0, 4.0]),
        ],
        ignore_index=True,
    )
    hist["Channel"] = ""
    csv_bytes, filename = inventory_history_wide_matrix_csv(
        hist, days=2, end_date="2026-07-17", channel="combined"
    )
    text = csv_bytes.decode("utf-8")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    assert lines[0].startswith("SKU,")
    assert "2026-07-16" in lines[0] and "2026-07-17" in lines[0]
    assert lines[1].startswith("Total inv.,")
    assert "15" in lines[1] and "12" in lines[1]
    assert any(ln.startswith("SKU-A,") for ln in lines)
    assert any(ln.startswith("SKU-B,") for ln in lines)
    assert filename.endswith(".csv")
    assert "2026-07-17" in filename


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


def test_sales_history_fans_combo_like_po_and_drops_dpt_fan():
    """Sales History SKUs must match PO: keep listing + kurta fan, drop DPT fan."""
    from backend.services.daily_sales_history import align_sales_history_skus_to_po

    sales = pd.DataFrame(
        {
            "Sku": ["1037DPT19WHITE-4XL"],
            "TxnDate": pd.to_datetime(["2026-07-17"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [1.0],
            "Units_Effective": [1.0],
            "Source": ["Amazon"],
        }
    )
    combo = {
        "1037DPT19WHITE-4XL": [("1037YKBLUE-4XL", 1.0), ("DPT19WHITE", 1.0)],
    }
    aligned = align_sales_history_skus_to_po(sales, combo_map=combo)
    skus = set(aligned["Sku"].astype(str))
    assert "1037DPT19WHITE-4XL" in skus
    assert "1037YKBLUE-4XL" in skus
    assert "DPT19WHITE" not in skus
    wide = sales_history_wide_matrix(aligned, days=1, end_date="2026-07-17", platform="Amazon")
    assert set(r["sku"] for r in wide["rows"]) == {"1037DPT19WHITE-4XL", "1037YKBLUE-4XL"}
    assert wide["date_totals"] == [2.0]


def test_sales_history_keeps_prebuilt_combo_fan_except_dpt():
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
    assert set(r["sku"] for r in wide["rows"]) == {"1037DPT19WHITE-4XL", "1037YKBLUE-4XL"}
    assert wide["date_totals"] == [2.0]


def test_file_matching_totals_exclude_combo_fan():
    from backend.services.daily_sales_history import filter_sales_history_window

    sales = pd.DataFrame(
        {
            "Sku": ["1037DPT19WHITE-4XL", "1037YKBLUE-4XL"],
            "TxnDate": pd.to_datetime(["2026-07-17"] * 2),
            "Transaction Type": ["Shipment"] * 2,
            "Quantity": [1.0, 1.0],
            "Units_Effective": [1.0, 1.0],
            "Source": ["Amazon"] * 2,
            "_Combo_Fan": [False, True],
        }
    )
    listing = filter_sales_history_window(
        sales, days=1, end_date="2026-07-17", platform="Amazon", include_combo_fan=False
    )
    assert float(listing["Units"].sum()) == 1.0
    po = filter_sales_history_window(
        sales, days=1, end_date="2026-07-17", platform="Amazon", include_combo_fan=True
    )
    assert float(po["Units"].sum()) == 2.0


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

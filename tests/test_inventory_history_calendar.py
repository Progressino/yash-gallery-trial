"""Calendar densification and spike repair for daily inventory history."""
import pandas as pd

from backend.services.daily_inventory_history import (
    densify_inventory_history_for_view,
    inventory_history_wide_matrix,
    project_inventory_calendar,
    repair_inventory_history_spikes,
)


def _hist(sku: str, dates: list[str], qtys: list[float], source: str = "uploaded") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OMS_SKU": [sku] * len(dates),
            "Date": pd.to_datetime(dates),
            "Qty": qtys,
            "Source": [source] * len(dates),
        }
    )


def test_project_inventory_calendar_fills_every_day():
    hist = _hist(
        "SKU-A",
        ["2026-06-01", "2026-06-05"],
        [10.0, 8.0],
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A", "SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-02", "2026-06-03", "2026-06-04"]),
            "Units_Effective": [1.0, 1.0, 0.0],
        }
    )
    out = project_inventory_calendar(
        hist,
        pd.Timestamp("2026-06-01"),
        pd.Timestamp("2026-06-05"),
        sales_df=sales,
    )
    by_day = {str(pd.Timestamp(r["Date"]).date()): float(r["Qty"]) for _, r in out.iterrows()}
    assert len(by_day) == 5
    assert by_day["2026-06-01"] == 10.0
    assert by_day["2026-06-02"] == 9.0
    assert by_day["2026-06-03"] == 8.0
    assert by_day["2026-06-04"] == 8.0
    assert by_day["2026-06-05"] == 8.0


def test_repair_inventory_history_spikes_replaces_bad_column():
    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-06-28"], [80000.0]),
            _hist("SKU-B", ["2026-06-28"], [20000.0]),
            _hist("SKU-A", ["2026-06-29"], [120000.0]),
            _hist("SKU-B", ["2026-06-29"], [80000.0]),
        ],
        ignore_index=True,
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-B"],
            "TxnDate": pd.to_datetime(["2026-06-29", "2026-06-29"]),
            "Units_Effective": [2000.0, 1000.0],
        }
    )
    repaired, actions = repair_inventory_history_spikes(hist, sales)
    assert actions
    totals = repaired.groupby("Date")["Qty"].sum()
    assert float(totals.loc[pd.Timestamp("2026-06-28")]) == 100000.0
    assert float(totals.loc[pd.Timestamp("2026-06-29")]) == 97000.0


def test_repair_inventory_history_spikes_ignores_small_samples():
    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-06-28"], [100.0]),
            _hist("SKU-B", ["2026-06-28"], [50.0]),
            _hist("SKU-A", ["2026-06-29"], [500.0]),
            _hist("SKU-B", ["2026-06-29"], [400.0]),
        ],
        ignore_index=True,
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-B"],
            "TxnDate": pd.to_datetime(["2026-06-29", "2026-06-29"]),
            "Units_Effective": [2.0, 1.0],
        }
    )
    repaired, actions = repair_inventory_history_spikes(hist, sales)
    assert not actions


def test_all_skus_receive_full_calendar_window():
    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-06-01", "2026-06-15"], [10.0, 8.0]),
            _hist("SKU-B", ["2026-06-10"], [5.0]),
            _hist("SKU-C", ["2026-06-28", "2026-06-29"], [80000.0, 120000.0]),
            _hist("SKU-D", ["2026-06-28"], [20000.0]),
        ],
        ignore_index=True,
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-B", "SKU-C", "SKU-D"],
            "TxnDate": pd.to_datetime(["2026-06-29", "2026-06-29", "2026-06-29", "2026-06-29"]),
            "Units_Effective": [1.0, 1.0, 2000.0, 1000.0],
        }
    )
    out = densify_inventory_history_for_view(
        hist, days=30, end_date="2026-06-29", sales_df=sales
    )
    per_sku = out.groupby("OMS_SKU")["Date"].nunique()
    assert set(per_sku.index) == {"SKU-A", "SKU-B", "SKU-C", "SKU-D"}
    assert (per_sku == 30).all()
    wide = inventory_history_wide_matrix(hist, days=30, end_date="2026-06-29", sales_df=sales)
    assert len(wide["dates"]) == 30
    assert wide["total"] == 4
    assert len(wide["rows"][0]["qtys"]) == 30


def test_validate_inventory_history_catalog_flags_spike():
    from backend.services.daily_inventory_history import validate_inventory_history_catalog

    hist = pd.concat(
        [
            _hist("SKU-A", ["2026-06-28"], [80000.0]),
            _hist("SKU-B", ["2026-06-28"], [20000.0]),
            _hist("SKU-A", ["2026-06-29"], [120000.0]),
            _hist("SKU-B", ["2026-06-29"], [80000.0]),
        ],
        ignore_index=True,
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-B"],
            "TxnDate": pd.to_datetime(["2026-06-29", "2026-06-29"]),
            "Units_Effective": [2000.0, 1000.0],
        }
    )
    report = validate_inventory_history_catalog(
        hist, days=2, end_date="2026-06-29", sales_df=sales
    )
    assert report["sku_count"] == 2
    assert report["skus_short_calendar"] == 0
    assert report["day_jump_over_5k"] == 0
    assert report["ok"] is True

    hist = _hist(
        "SKU-A",
        ["2026-06-27", "2026-06-29"],
        [5.0, 3.0],
    )
    out = inventory_history_wide_matrix(hist, q="SKU-A", days=3, end_date="2026-06-29")
    assert out["loaded"] is True
    assert out["dates"] == ["2026-06-27", "2026-06-28", "2026-06-29"]
    assert len(out["rows"][0]["qtys"]) == 3


def test_densify_carries_forward_between_sparse_snapshots():
    hist = _hist(
        "1024YKMUSTARD-8XL",
        ["2026-06-07", "2026-06-19"],
        [1.0, 0.0],
    )
    sales = pd.DataFrame(
        {
            "Sku": ["1024YKMUSTARD-8XL"],
            "TxnDate": pd.to_datetime(["2026-06-10"]),
            "Units_Effective": [1.0],
        }
    )
    out = densify_inventory_history_for_view(
        hist, days=13, end_date="2026-06-19", sales_df=sales
    )
    sub = out[out["OMS_SKU"] == "1024YKMUSTARD-8XL"].sort_values("Date")
    assert len(sub) == 13
    assert float(sub.iloc[0]["Qty"]) == 1.0
    assert float(sub[sub["Date"] == pd.Timestamp("2026-06-10")].iloc[0]["Qty"]) == 0.0

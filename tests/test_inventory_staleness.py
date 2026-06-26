"""Inventory staleness warnings and history browse helpers."""
from datetime import date

import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_summary,
    inventory_rows_for_date,
    list_inventory_history_dates,
)
from backend.services.inventory_staleness import (
    build_inventory_staleness,
    data_is_stale,
    data_lag_days,
)


def test_data_lag_days():
    assert data_lag_days("2026-06-23", "2026-06-22") == 1
    assert data_lag_days("2026-06-23", "2026-06-23") == 0
    assert data_is_stale("2026-06-23", "2026-06-21", max_expected_lag_days=1) is True
    assert data_is_stale("2026-06-23", "2026-06-22", max_expected_lag_days=1) is False


def test_build_inventory_staleness_snapshot_and_matrix():
    out = build_inventory_staleness(
        reference_date="2026-06-23",
        inventory_loaded=True,
        inventory_snapshot_date="2026-06-20",
        daily_inventory_history_loaded=True,
        daily_inventory_history_max_date="2026-06-21",
    )
    assert out["inventory_snapshot_stale"] is True
    assert out["daily_inventory_history_stale"] is True
    assert len(out["inventory_staleness_warnings"]) >= 2


def test_snapshot_not_stale_when_history_matrix_current():
    """Missing snapshot date must not warn when history matrix is through yesterday."""
    out = build_inventory_staleness(
        reference_date="2026-06-26",
        inventory_loaded=True,
        inventory_snapshot_date=None,
        daily_inventory_history_loaded=True,
        daily_inventory_history_max_date="2026-06-25",
    )
    assert out["inventory_snapshot_stale"] is False
    assert out["daily_inventory_history_stale"] is False
    assert out["inventory_staleness_warnings"] == []


def test_build_inventory_staleness_missing_matrix():
    out = build_inventory_staleness(
        reference_date="2026-06-23",
        inventory_loaded=True,
        inventory_snapshot_date="2026-06-23",
        daily_inventory_history_loaded=False,
    )
    assert any("history matrix" in w.lower() for w in out["inventory_staleness_warnings"])


def test_build_inventory_staleness_snapshot_unknown_but_history_current():
    """When wide history matrix is through yesterday, do not warn about missing snapshot date."""
    out = build_inventory_staleness(
        reference_date="2026-06-26",
        inventory_loaded=True,
        inventory_snapshot_date=None,
        daily_inventory_history_loaded=True,
        daily_inventory_history_max_date="2026-06-25",
    )
    assert out["inventory_snapshot_stale"] is False
    assert out["daily_inventory_history_stale"] is False
    assert out["inventory_staleness_warnings"] == []


def test_inventory_history_browse_helpers():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-A", "SKU-B", "SKU-B"],
            "Date": [date(2026, 6, 21), date(2026, 6, 22), date(2026, 6, 21), date(2026, 6, 22)],
            "Qty": [10, 12, 0, 5],
        }
    )
    summary = inventory_history_summary(df)
    assert summary["loaded"] is True
    assert summary["max_date"] == "2026-06-22"
    dates = list_inventory_history_dates(df)
    assert dates[0]["date"] == "2026-06-22"
    by_date = inventory_rows_for_date(df, "2026-06-22")
    assert by_date["total"] == 2
    assert by_date["rows"][0]["sku"] in ("SKU-A", "SKU-B")

    from backend.services.daily_inventory_history import inventory_history_wide_matrix

    wide = inventory_history_wide_matrix(df)
    assert wide["dates"] == ["2026-06-21", "2026-06-22"]
    assert len(wide["rows"]) == 2
    sku_a = next(r for r in wide["rows"] if r["sku"] == "SKU-A")
    assert sku_a["qtys"] == [10.0, 12.0]
    filtered = inventory_history_wide_matrix(df, q="SKU-B")
    assert filtered["total"] == 1
    assert filtered["rows"][0]["qtys"] == [0.0, 5.0]

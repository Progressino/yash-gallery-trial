"""Days without a snapshot upload must be flagged as carried in Inv History."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_wide_matrix,
    non_uploaded_inventory_dates,
)


def test_non_uploaded_dates_are_days_without_snapshot():
    rows = []
    for d, src, qty in [
        ("2026-07-04", "snapshot", 10),
        ("2026-07-05", "derived", 9),  # no file
        ("2026-07-06", "snapshot", 8),
        ("2026-07-07", "derived", 7),  # no file
    ]:
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms",
            }
        )
    df = pd.DataFrame(rows)
    dates = list(pd.date_range("2026-07-04", "2026-07-07"))
    carried = non_uploaded_inventory_dates(df, dates)
    assert carried == ["2026-07-05", "2026-07-07"]


def test_matrix_marks_derived_only_days_as_gap_dates():
    rows = []
    for d, src, qty in [
        ("2026-07-04", "snapshot", 100),
        ("2026-07-05", "derived", 95),
        ("2026-07-06", "snapshot", 90),
    ]:
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms",
            }
        )
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": 0,
                "Source": src if src == "snapshot" else "derived",
                "Channel": "amazon",
            }
        )
    df = pd.DataFrame(rows)
    wide = inventory_history_wide_matrix(df, days=3, end_date="2026-07-06", channel="combined")
    assert "2026-07-05" in (wide.get("gap_dates") or [])
    assert "2026-07-04" not in (wide.get("gap_dates") or [])
    assert "2026-07-06" not in (wide.get("gap_dates") or [])
    assert "2026-07-05" not in (wide.get("uploaded_dates") or [])

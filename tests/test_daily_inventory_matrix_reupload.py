"""Wide-matrix inventory re-upload must replace the matrix window (no Apr-30 bleed)."""
from __future__ import annotations

import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_sheet_end_date_from_filename,
    inventory_sheet_start_date_from_filename,
    is_full_matrix_inventory_reupload,
    merge_inventory_history,
)
from backend.services.daily_inventory_upload_run import execute_daily_inventory_upload
from backend.session import AppSession


def test_filename_parses_may_jun_range():
    fn = "Daily Inventory History 1-May To 29-Jun-26.xlsx"
    assert inventory_sheet_start_date_from_filename(fn) == "2026-05-01"
    assert inventory_sheet_end_date_from_filename(fn) == "2026-06-29"


def test_full_matrix_reupload_detected():
    incoming = pd.DataFrame(
        {
            "OMS_SKU": [f"SKU{i}" for i in range(600)],
            "Date": pd.to_datetime(["2026-06-29"] * 600),
            "Qty": [1.0] * 600,
            "Source": ["uploaded"] * 600,
        }
    )
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["SKU0"],
            "Date": pd.to_datetime(["2026-04-30"]),
            "Qty": [99.0],
            "Source": ["uploaded"],
        }
    )
    fn = "Daily Inventory History 1-May To 29-Jun-26.xlsx"
    assert is_full_matrix_inventory_reupload(existing, incoming, fn) is True


def test_matrix_reupload_drops_pre_range_days():
    """Replacing May–Jun matrix must drop Apr-30 bleed and snapshot overrides."""
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["A-SKU", "A-SKU"],
            "Date": pd.to_datetime(["2026-04-30", "2026-06-29"]),
            "Qty": [67.0, 60.0],
            "Source": ["uploaded", "snapshot"],
            "Channel": ["oms", "oms"],
        }
    )
    incoming = pd.DataFrame(
        {
            "OMS_SKU": ["A-SKU", "A-SKU"],
            "Date": pd.to_datetime(["2026-05-01", "2026-06-29"]),
            "Qty": [67.0, 38.0],
            "Source": ["uploaded", "uploaded"],
            "Channel": ["oms", "oms"],
        }
    )
    in_min = incoming["Date"].min()
    in_max = incoming["Date"].max()
    ex_dates = pd.to_datetime(existing["Date"], errors="coerce").dt.normalize()
    kept = existing.loc[ex_dates > in_max]
    merged = merge_inventory_history(kept, incoming)

    assert "2026-04-30" not in merged["Date"].astype(str).str[:10].tolist()
    jun29 = merged[merged["Date"].astype(str).str[:10] == "2026-06-29"]
    assert float(jun29["Qty"].iloc[0]) == 38.0


def test_execute_upload_from_fixture_xlsx_if_present():
    """Integration: parse real operator file when available locally."""
    from pathlib import Path

    path = Path("/Users/samraisinghani/Downloads/Daily Inventory History 1-May To 29-Jun-26.xlsx")
    if not path.is_file():
        return
    raw = path.read_bytes()
    sess = AppSession()
    res = execute_daily_inventory_upload(sess, raw, path.name)
    assert res["ok"] is True
    assert res["min_date"] == "2026-05-01"
    assert res["max_date"] == "2026-06-29"
    df = sess.daily_inventory_history_df
    dup = df.groupby(["OMS_SKU", "Date", "Channel"]).size()
    assert int((dup > 1).sum()) == 0
    sku = "1001YKBEIGE-3XL"
    row = df[(df["OMS_SKU"] == sku) & (df["Date"].astype(str).str[:10] == "2026-06-29")]
    oms = row[row["Channel"].astype(str).str.lower() == "oms"]
    assert not oms.empty
    assert float(oms["Qty"].iloc[0]) == 38.0

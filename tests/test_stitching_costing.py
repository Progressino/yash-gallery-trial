"""Stitching Costing module smoke tests."""
import pandas as pd

from backend.db.stitching_db import init_db, get_sheet_df, save_sheet_df, DATA_KEYS
from backend.services import stitching_costing as svc


def test_stitching_init_and_dashboard():
    init_db()
    for key in DATA_KEYS:
        df = get_sheet_df(key)
        assert df is not None
    dash = svc.dashboard_summary("2026-05-15")
    assert "metrics" in dash
    assert dash["metrics"]["total_karigar"] >= 1


def test_stitching_save_production_entry():
    init_db()
    out = svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K001",
        karigar_name="Ramesh Kumar",
        challan_no="10220-2526",
        style="1894YKDGREEN",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Cutting", "pieces": 12},
            {"hour_col": "H_10_11", "operation": "Cutting", "pieces": 8},
        ],
    )
    assert out["ok"] is True
    pl = get_sheet_df("production_log")
    assert not pl.empty
    assert int(pl["Total_Pieces"].sum()) >= 20


def test_stitching_style_costing_report():
    init_db()
    rep = svc.style_costing_report(month="All", style="All", party="All")
    assert "summary" in rep
    assert "rows" in rep

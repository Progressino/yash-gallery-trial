"""Stitching Costing module smoke tests."""
import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, get_sheet_df, save_sheet_df
from backend.services import stitching_costing as svc


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_test.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def test_stitching_init_and_dashboard():
    for key in stitching_db.DATA_KEYS:
        df = get_sheet_df(key)
        assert df is not None
    dash = svc.dashboard_summary("2026-05-15")
    assert "metrics" in dash
    assert dash["metrics"]["total_karigar"] >= 1


def test_stitching_save_production_entry():
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


def test_stitching_performance_report():
    from backend.services.stitching_costing import performance_report

    save_sheet_df(
        "production_log",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-10",
                    "Karigar_ID": "K001",
                    "Karigar_Name": "Ramesh",
                    "Total_Pieces": 50,
                    "Piece_Value_Rs": 500,
                    "Efficiency_%": 90,
                }
            ]
        ),
    )
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-10",
                    "E_Code": "K001",
                    "Name": "Ramesh",
                    "Payable_Hrs": 8,
                    "Total_Pay": 400,
                }
            ]
        ),
    )
    out = performance_report("2026-05-01", "2026-05-15")
    assert out["ok"] is True
    assert out["rows"]
    assert out["summary"]["net_surplus"] == 100.0


def test_stitching_admin_and_style_update():
    from backend.db.stitching_db import verify_admin_password, change_admin_password
    from backend.services.stitching_costing import update_style_operation

    assert verify_admin_password("admin123") is True
    out = update_style_operation("1894YKDGREEN", "Cutting", target=130, rate_rs=2.75)
    assert out["ok"] is True
    ch = change_admin_password("admin123", "admin123")
    assert ch["ok"] is True


def test_stitching_style_costing_report():
    rep = svc.style_costing_report(month="All", style="All", party="All")
    assert "summary" in rep
    assert "rows" in rep

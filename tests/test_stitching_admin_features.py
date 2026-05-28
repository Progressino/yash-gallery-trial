"""Stitching admin: production delete, expenses, payroll merge."""
import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, get_sheet_df, save_sheet_df
from backend.services import stitching_costing as svc


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_admin.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def _seed_production():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-A", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K100",
        karigar_name="Ram",
        challan_no="CH-9",
        style="SKU-A",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Stitch", "pieces": 10}],
    )
    svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K100",
        karigar_name="Ram",
        challan_no="CH-9",
        style="SKU-A",
        hour_entries=[{"hour_col": "H_10_11", "operation": "Stitch", "pieces": 5}],
    )


def test_delete_production_syncs_all_reports():
    _seed_production()
    before = svc.production_entry_reports("2026-05-15", "K100")
    assert len(before["report1"]) >= 1
    assert len(before["history"]) >= 1

    out = svc.delete_production_entries(
        date_str="2026-05-15",
        karigar_id="K100",
        challan_no="CH-9",
        style="SKU-A",
        operation="Stitch",
    )
    assert out["ok"] is True
    assert out["removed"] >= 1

    after = svc.production_entry_reports("2026-05-15", "K100")
    assert after["report1"] == []
    assert after["history"] == []
    assert after["report2_summary"] == []
    assert after["report2_hourly"] == []


def test_karigar_expense_and_payroll():
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-15",
                    "E_Code": "K200",
                    "Name": "Ali",
                    "Payable_Hrs": 8,
                    "Normal_Pay": 400,
                    "OT_Hours": 0,
                    "OT_Pay": 0,
                    "Total_Pay": 400,
                }
            ]
        ),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "K200", "Name": "Ali", "Daily_Rate_Rs": 400}]),
    )
    exp = svc.upsert_karigar_expense(
        date_str="2026-05-15",
        karigar_id="K200",
        work_type="Part Change",
        challan_no="CH-1",
        style="SKU-X",
        amount_rs=150,
        notes="part change on challan",
    )
    assert exp["ok"] is True

    listed = svc.list_karigar_expenses("2026-05-15", "2026-05-15", "K200")
    assert len(listed) == 1
    assert listed[0]["Work_Type"] == "Part Change"

    payroll = svc.payroll_report("2026-05-15", "2026-05-15")
    assert payroll["total_payroll"] == 550.0
    assert payroll["total_attendance"] == 400.0
    assert payroll["total_other_work"] == 150.0
    row = payroll["rows"][0]
    assert row["Karigar_ID"] == "K200"
    assert float(row["Total"]) == 550.0

    eid = listed[0]["Expense_ID"]
    del_out = svc.delete_karigar_expense(eid)
    assert del_out["ok"] is True
    payroll2 = svc.payroll_report("2026-05-15", "2026-05-15")
    assert payroll2["total_payroll"] == 400.0


def test_ltl_setup_lists_overrides():
    svc.upsert_ltl_override("SKU-Z", "Cut", "K1", 42, notes="test")
    table = svc.get_ltl_setup_table()
    assert table["ok"] is True
    assert any(o["Manual_LTL"] == 42 for o in table["overrides"])


def test_delete_single_hour_recalculates_related_rows():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-HR", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    svc.save_production_entry(
        date_str="2026-05-25",
        karigar_id="K300",
        karigar_name="Test",
        challan_no="CH-300",
        style="SKU-HR",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Stitch", "pieces": 12},
            {"hour_col": "H_10_11", "operation": "Stitch", "pieces": 8},
        ],
    )
    out = svc.delete_production_hour_entry(
        date_str="2026-05-25",
        karigar_id="K300",
        challan_no="CH-300",
        style="SKU-HR",
        operation="Stitch",
        hour_label="9-10",
    )
    assert out["ok"] is True
    rep = svc.production_entry_reports("2026-05-25", "K300")
    assert rep["report2_hourly"]
    hours = [str(r["Hour"]) for r in rep["report2_hourly"]]
    assert "9-10" not in hours
    assert "10-11" in hours

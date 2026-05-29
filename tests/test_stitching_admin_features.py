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


def test_list_karigar_directory_merges_sources():
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "K1", "Name": "Master One"}]),
    )
    save_sheet_df(
        "employee_master",
        pd.DataFrame(
            [
                {"E_Code": "K2", "Name": "Employee Karigar", "Type": "Karigar"},
                {"E_Code": "E9", "Name": "Office Staff", "Type": "Admin"},
            ]
        ),
    )
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-P", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    save_sheet_df(
        "production_log",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-20",
                    "Karigar_ID": "K3",
                    "Karigar_Name": "From Production",
                    "Challan_No": "CH-P",
                    "Style": "SKU-P",
                    "Operation": "Stitch",
                    "H_09_10": 1,
                    "Save_Time": "2026-05-20 10:00:00",
                }
            ]
        ),
    )

    directory = svc.list_karigar_directory()
    ids = {d["Karigar_ID"] for d in directory}
    assert ids == {"K1", "K2", "K3"}
    by_id = {d["Karigar_ID"]: d["Name"] for d in directory}
    assert by_id["K3"] == "From Production"


def test_delete_hour_uses_latest_duplicate_production_row():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-DUP", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    svc.save_production_entry(
        date_str="2026-05-26",
        karigar_id="K400",
        karigar_name="Dup",
        challan_no="CH-400",
        style="SKU-DUP",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Stitch", "pieces": 20},
            {"hour_col": "H_10_11", "operation": "Stitch", "pieces": 10},
        ],
    )
    pl = get_sheet_df("production_log")
    dup = pl.iloc[-1].copy()
    dup["Save_Time"] = "2099-01-01 00:00:00"
    dup["H_09_10"] = 99
    save_sheet_df("production_log", pd.concat([pl, pd.DataFrame([dup])], ignore_index=True))

    out = svc.delete_production_hour_entry(
        date_str="2026-05-26",
        karigar_id="K400",
        challan_no="CH-400",
        style="SKU-DUP",
        operation="Stitch",
        hour_label="9-10",
    )
    assert out["ok"] is True
    rep = svc.production_entry_reports("2026-05-26", "K400")
    hours = [str(r["Hour"]) for r in rep["report2_hourly"]]
    assert "9-10" not in hours
    assert "10-11" in hours


def test_karigar_rate_change_recalculates_production():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-R", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "K500", "Name": "Rate Test", "Skill": "Stitching", "Daily_Rate_Rs": 400}]),
    )
    svc.save_production_entry(
        date_str="2026-05-18",
        karigar_id="K500",
        karigar_name="Rate Test",
        challan_no="CH-500",
        style="SKU-R",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Stitch", "pieces": 15}],
    )
    pl_before = get_sheet_df("production_log")
    old_rate = float(pl_before.iloc[-1].get("Daily_Rate_Rs", 0))

    svc.update_karigar_master("K500", daily_rate_rs=550.0, effective_from="2026-05-18")
    pl_after = get_sheet_df("production_log")
    assert not pl_after.empty
    new_rate = float(pl_after.iloc[-1].get("Daily_Rate_Rs", 0))
    assert old_rate == 400.0
    assert new_rate == 550.0


def test_ltl_tolerance_bands_from_sheet():
    save_sheet_df(
        "ltl_tolerance_bands",
        pd.DataFrame([{"Min_Rs": 200, "Max_Rs": 300, "Tolerance_Pct": 20}]),
    )
    assert svc.ltl_tolerance_pct_for_rate(250) == 20.0
    assert svc.ltl_tolerance_factor(250) == 0.8


def test_bulk_ltl_all_styles():
    save_sheet_df(
        "style_master",
        pd.DataFrame(
            [
                {"Style": "SKU-A", "Operation": "Cut", "Target": 100, "Rate_Rs": 2},
                {"Style": "SKU-B", "Operation": "Cut", "Target": 100, "Rate_Rs": 2},
            ]
        ),
    )
    out = svc.bulk_upsert_ltl_override_all_styles("Cut", "K1", 55, notes="all styles")
    assert out["ok"] is True
    assert out["styles_updated"] == 2
    ov = get_sheet_df("target_ltl_override")
    assert len(ov) == 2


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

"""Stitching reports hub — karigar profitability and challan labour payroll."""
import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, save_sheet_df
from backend.services import stitching_costing as svc


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_reports.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def _seed_karigar_day():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-A", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "K300", "Name": "Ravi", "Daily_Rate_Rs": 480}]),
    )
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-20",
                    "E_Code": "K300",
                    "Name": "Ravi",
                    "Payable_Hrs": 8,
                    "Normal_Pay": 480,
                    "OT_Hours": 0,
                    "OT_Pay": 0,
                    "Total_Pay": 480,
                }
            ]
        ),
    )
    svc.upsert_karigar_expense(
        date_str="2026-05-20",
        karigar_id="K300",
        work_type="Alter",
        challan_no="CH-55",
        style="SKU-A",
        amount_rs=50,
    )
    svc.save_production_entry(
        date_str="2026-05-20",
        karigar_id="K300",
        karigar_name="Ravi",
        challan_no="CH-55",
        style="SKU-A",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Stitch", "pieces": 20}],
    )


def test_karigar_profitability_includes_full_payroll():
    _seed_karigar_day()
    out = svc.karigar_profitability_report("2026-05-20", "2026-05-20")
    assert out["rows"]
    row = next(r for r in out["rows"] if r["Karigar_ID"] == "K300")
    assert float(row["Total_Payroll_Paid"]) == 530.0
    assert float(row["Attendance_Pay"]) == 480.0
    assert float(row["Other_Work_Pay"]) == 50.0
    assert row["Profitable_On_Payroll"] in ("Yes", "No")
    assert "Net_PL_Benchmark" in row


def test_challan_labour_payroll_report():
    _seed_karigar_day()
    out = svc.challan_labour_payroll_report("2026-05-20", "2026-05-20")
    assert len(out["rows"]) >= 1
    row = next(r for r in out["rows"] if r["Challan_No"] == "CH-55")
    assert float(row["Expense_On_Challan_Rs"]) == 50.0
    assert float(row["Total_Payroll_Paid"]) >= 50.0
    assert "Budgeted_Labour_Rs" in row


def test_performance_uses_full_payroll():
    _seed_karigar_day()
    perf = svc.performance_report("2026-05-20", "2026-05-20")
    assert perf["ok"] is True
    row = next(r for r in perf["rows"] if str(r.get("E_Code")) == "K300")
    assert float(row["Total_Payroll_Paid"]) == 530.0
    assert float(row["Other_Work_Pay"]) == 50.0


def test_reports_hub_sections():
    _seed_karigar_day()
    hub = svc.stitching_reports_hub("2026-05-20", "2026-05-20")
    assert hub["payroll"]["total_payroll"] == 530.0
    assert hub["karigar_profitability"]["rows"]
    assert hub["challan_labour"]["rows"]
    assert hub["comparison"]["summary"]

"""Karigar attendance policy and biometric import tests."""

import io
from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import get_sheet_df, init_db, save_sheet_df
from backend.services import karigar_attendance as att
from backend.services import stitching_costing as svc


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "attendance_test.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def test_full_day_present_pays_daily_rate():
    out = att.calc_salary_from_punches(
        [(time(8, 43), time(18, 0))],
        460.0,
        on_date="2026-05-20",
        status="P",
    )
    assert out["Payable_Hrs"] == 8.0
    assert out["Normal_Pay"] == 460.0
    assert out["Hourly_Rate_Rs"] == 57.5


def test_early_punch_in_counts_from_nine():
    out = att.calc_salary_from_punches(
        [(time(8, 30), time(18, 0))],
        480.0,
        on_date="2026-05-20",
        status="P",
    )
    assert out["Payable_Hrs"] == 8.0
    assert out["Normal_Pay"] == 480.0


def test_overtime_same_hourly_rate_after_grace():
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(19, 0))],
        460.0,
        on_date="2026-05-20",
        status="P",
    )
    assert out["OT_Hours"] == 1.0
    assert out["OT_Pay"] == 57.5
    assert out["Total_Pay"] == round(out["Normal_Pay"] + out["OT_Pay"], 2)


def test_overtime_manoj_807_9_to_2059():
    """09:00–20:59 → 8h regular (520) + 3h OT at hourly (520/8) = 715."""
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(20, 59))],
        520.0,
        on_date="2026-05-22",
        status="P",
    )
    assert out["Normal_Pay"] == 520.0
    assert out["Payable_Hrs"] == 8.0
    assert out["OT_Hours"] == 3.0
    assert out["OT_Pay"] == 195.0
    assert out["Total_Pay"] == 715.0
    assert out["Hourly_Rate_Rs"] == 65.0


def test_overtime_manoj_807_two_minute_late_still_full_day():
    """09:02–20:59: grace late-in, full ₹520 day + 3h OT = ₹715 (manual salary sheet)."""
    out = att.calc_salary_from_punches(
        [(time(9, 2), time(20, 59))],
        520.0,
        on_date="2026-06-01",
        status="P",
    )
    assert out["Late_Deduction_Hrs"] == 0.02
    assert out["Late_Deduction_Rs"] == 0.0
    assert out["Normal_Pay"] == 520.0
    assert out["Payable_Hrs"] == 8.0
    assert out["OT_Hours"] == 3.0
    assert out["OT_Pay"] == 195.0
    assert out["Total_Pay"] == 715.0


def test_late_out_grace_2059_counts_as_21():
    """20:59 should get OT-cap grace to 21:00 (policy)."""
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(20, 59))],
        520.0,
        on_date="2026-06-01",
        status="P",
    )
    assert out["OT_Hours"] == 3.0
    assert out["Total_Pay"] == 715.0


def test_sunday_9_to_4_full_day_payable_and_hourly_basis():
    """Sunday: shift 09:00–16:00 with 13:00–14:00 lunch => 6h payable = full daily rate."""
    # 2026-06-07 is a Sunday.
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(16, 0))],
        400.0,
        on_date="2026-06-07",
        status="P",
    )
    assert out["Payable_Hrs"] == 6.0
    assert out["Normal_Pay"] == 400.0
    assert out["Hourly_Rate_Rs"] == round(400 / 6, 2)
    assert out["Early_Deduction_Hrs"] == 0.0


def test_sunday_production_hours_end_at_15_16():
    cols = att.production_hour_cols_for_date("2026-06-07")
    assert "H_15_16" in cols
    assert "H_16_17" not in cols
    assert "H_13_14" in cols


def test_early_leave_before_16_adds_30_min_penalty():
    """Leaving at 16:00 triggers extra 30m deduction in addition to early minutes."""
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(16, 0))],
        480.0,
        on_date="2026-06-01",
        status="P",
    )
    assert out["Payable_Hrs"] < 7.0  # 7h block minus lunch/tea plus 30m penalty


def test_needs_miss_punch_single_in():
    assert att.needs_miss_punch([(time(9, 0), None)]) is True
    assert att.needs_miss_punch([(time(9, 0), time(18, 0))]) is False
    assert att.needs_miss_punch([(time(9, 0), time(12, 0)), (time(13, 0), None)]) is True


def test_multiple_punch_pairs_tracked():
    """Biometric IN-1…OUT-3 style day — hours from all segments."""
    pairs = [
        (time(9, 3), time(12, 28)),
        (time(13, 7), time(15, 53)),
        (time(16, 18), time(20, 48)),
    ]
    out = att.calc_salary_from_punches(pairs, 520.0, on_date="2026-05-22")
    assert out["Punch_Count"] == 3
    assert out["Needs_Miss_Punch"] is False
    assert out["Total_Presence_Hrs"] > 8.0
    assert out["OT_Hours"] >= 2.0
    assert out["In_Punch"] == "09:03"  # actual first punch; work blocks use 09:00 when within grace
    assert out["Out_Punch"] == "20:48"
    stored = att.deserialize_punch_pairs(out["Punch_Pairs"])
    assert len(stored) == 3


def test_two_tea_breaks_15_min_each_when_not_taken():
    pairs = [
        (time(9, 0), time(11, 0)),
        (time(12, 0), time(15, 30)),
        (time(17, 0), time(18, 0)),
    ]
    out = att.calc_salary_from_punches(pairs, 480.0, on_date="2026-05-20")
    assert out["Lunch_Deduction_Hrs"] == 0.5
    assert out["Tea_Deduction_Hrs"] == 0.5


def test_update_row_rejects_incomplete_pairs():
    save_sheet_df(
        "employee_master",
        pd.DataFrame(
            [{"E_Code": "921", "Name": "Worker 921", "Type": "Karigar", "Daily_Rate_Rs": 520, "Hourly_Rate_Rs": 65}]
        ),
    )
    out = att.update_karigar_attendance_row(
        on_date="2026-05-22",
        e_code="921",
        punch_pairs=[(time(9, 0), None)],
    )
    assert out["ok"] is False


def test_absent_shift_pays_zero():
    out = att.calc_salary_from_punches([], 460.0, on_date="2026-05-20", status="A")
    assert out["Total_Pay"] == 0.0
    assert out["Payable_Hrs"] == 0.0


def test_employee_828_on_time_near_full_day_payable_hrs():
    """09:00–17:58 within grace: full daily rate; lunch tracked but not prorated down."""
    out = att.calc_salary_from_punches(
        [(time(9, 0), time(17, 58))],
        330.0,
        on_date="2026-05-22",
    )
    assert out["Payable_Hrs"] == 8.0
    assert out["Normal_Pay"] == 330.0
    assert out["Lunch_Deduction_Hrs"] == 0.5
    assert out["Early_Deduction_Hrs"] == 0.0


def test_employee_804_on_time_near_full_and_late():
    """Sohan E804 @ ₹460/day — on-time near-full uses block hours; late keeps lunch+late cuts."""
    near_full = att.calc_salary_from_punches(
        [(time(9, 0), time(17, 58))],
        460.0,
        on_date="2026-05-20",
    )
    assert near_full["Payable_Hrs"] == 8.0
    assert near_full["Lunch_Deduction_Hrs"] == 0.5
    assert near_full["Late_Deduction_Hrs"] == 0.0
    assert near_full["Normal_Pay"] == 460.0

    full = att.calc_salary_from_punches(
        [(time(9, 0), time(18, 0))],
        460.0,
        on_date="2026-05-20",
    )
    assert full["Payable_Hrs"] == 8.0
    assert full["Normal_Pay"] == 460.0

    late = att.calc_salary_from_punches(
        [(time(9, 21), time(18, 0))],
        460.0,
        on_date="2026-05-20",
    )
    assert late["Payable_Hrs"] == 7.15
    assert late["Late_Deduction_Hrs"] == 0.35
    assert late["Lunch_Deduction_Hrs"] == 0.5


def test_late_within_17_min_grace_pays_full_daily():
    """17 minutes late (e.g. E_Code 822) must not reduce Normal_Pay."""
    out = att.calc_salary_from_punches(
        [(time(9, 17), time(18, 0))],
        450.0,
        on_date="2026-06-03",
    )
    assert out["Normal_Pay"] == 450.0
    assert out["Late_Deduction_Rs"] == 0.0
    assert out["Payable_Hrs"] == 8.0


def test_extract_report_date_from_header_label():
    raw = pd.DataFrame(
        [
            ["Daily Attendance IN/OUT Punch Report", "", ""],
            ["Date", "03-Jun-2026", ""],
            ["", "", ""],
        ]
    )
    assert att._extract_report_date(raw) == "2026-06-03"


def test_cell_to_report_date_rejects_clock_times():
    assert att._cell_to_report_date("08:58") == ""
    assert att._cell_to_report_date("21:00") == ""
    assert att._cell_to_report_date("Jun 08 2026  To  Jun 08 2026") == "2026-06-08"


def test_extract_report_date_from_jun_range_header_not_punch_times():
    """Biometric export: title row has the day; punch times must not become the report date."""
    raw = pd.DataFrame(
        [
            ["Jun 08 2026  To  Jun 08 2026", None, None, None, None, None, None],
            ["SNo", "E. Code", "Name", "Department", "Shift", "IN-1", "OUT-1"],
            [1, 800, "Worker", "Default", "GS", "08:58", "18:29"],
            [2, 803, "Absent", "Default", "NS", None, None],
        ]
    )
    assert att._extract_report_date(raw) == "2026-06-08"


def test_extract_report_date_ignores_generated_timestamp_row():
    """Saturday sheet uploaded Monday must use attendance date, not print/export date."""
    raw = pd.DataFrame(
        [
            ["Daily Attendance IN/OUT Punch Report", "", ""],
            ["Generated on", "09-Jun-2026", ""],
            ["Date", "07-Jun-2026", ""],
            ["", "", ""],
        ]
    )
    assert att._extract_report_date(raw) == "2026-06-07"


def test_import_uses_report_date_override_not_sheet_header():
    save_sheet_df(
        "employee_master",
        pd.DataFrame([{"E_Code": "901", "Name": "Test", "Type": "Karigar", "Daily_Rate_Rs": 400, "Hourly_Rate_Rs": 50}]),
    )
    raw = pd.DataFrame(
        [
            ["Date", "09-Jun-2026"],
            ["E. Code", "Name", "IN-1", "OUT-1"],
            ["901", "Test", "09:00", "18:00"],
        ]
    )
    bio = io.BytesIO()
    raw.to_excel(bio, index=False, header=False)
    out = att.import_karigar_attendance_bytes(
        bio.getvalue(),
        "attendance.xlsx",
        report_date_override="2026-06-07",
    )
    assert out["ok"] is True
    assert out["date"] == "2026-06-07"
    pl = get_sheet_df("karigar_attendance")
    assert pl.iloc[0]["Date"] == "2026-06-07"


def test_employee_845_late_arrival_reduces_normal_pay():
    """09:21–18:30 on ₹330/day: late 21m + lunch-through 30m → ~₹295 normal + ₹41.25 OT = ~₹336."""
    out = att.calc_salary_from_punches(
        [(time(9, 21), time(18, 30))],
        330.0,
        on_date="2026-05-20",
    )
    assert out["Hourly_Rate_Rs"] == 41.25
    assert out["Late_Deduction_Hrs"] == 0.35
    assert out["Late_Deduction_Rs"] == 14.44
    assert out["Normal_Pay"] == 294.94
    assert out["OT_Pay"] == 41.25
    assert out["Total_Pay"] == 336.19
    assert out["Payable_Hrs"] == 7.15


def test_recalculate_attendance_for_date():
    save_sheet_df(
        "employee_master",
        pd.DataFrame(
            [{"E_Code": "807", "Name": "Manoj", "Type": "Karigar", "Daily_Rate_Rs": 520, "Hourly_Rate_Rs": 65}]
        ),
    )
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-06-01",
                    "E_Code": "807",
                    "Name": "Manoj",
                    "Status": "P",
                    "Daily_Rate_Rs": 520,
                    "Punch_Pairs": att.serialize_punch_pairs([(time(9, 2), time(20, 59))]),
                    "Normal_Pay": 485.33,
                    "Total_Pay": 680.33,
                }
            ]
        ),
    )
    out = att.recalculate_attendance_for_date("2026-06-01")
    assert out["ok"] is True
    assert out["updated"] == 1
    row = get_sheet_df("karigar_attendance").iloc[0]
    assert float(row["Total_Pay"]) == 715.0
    assert float(row["Normal_Pay"]) == 520.0


def test_calc_salary_wrapper():
    out = svc.calc_salary("09:00", "18:00", 400.0)
    assert out["Payable_Hrs"] == 8.0
    assert out["Normal_Pay"] == 400.0


def test_import_8_jun_26_xlsx_absent_workers_not_present():
    """Regression: 8-6-26.xlsx — NS / blank punches absent; date is 2026-06-08 not punch day."""
    save_sheet_df(
        "employee_master",
        pd.DataFrame(
            [
                {"E_Code": "803", "Name": "Sunita", "Type": "Karigar", "Daily_Rate_Rs": 400, "Hourly_Rate_Rs": 50},
                {"E_Code": "804", "Name": "Sohan", "Type": "Karigar", "Daily_Rate_Rs": 460, "Hourly_Rate_Rs": 57.5},
            ]
        ),
    )
    fixture = Path("/Users/samraisinghani/Downloads/8-6-26.xlsx")
    if not fixture.is_file():
        raw = pd.DataFrame(
            [
                ["Jun 08 2026  To  Jun 08 2026", None, None, None, None, None, None],
                ["SNo", "E. Code", "Name", "Department", "Shift", " IN-1", "OUT-1"],
                [1, 804, "Sohan S/o Bihari", "Default", "GS", "08:56", "20:59"],
                [2, 803, "Sunita W/O Harjeet Singh", "Default", "NS", None, None],
            ]
        )
        bio = io.BytesIO()
        raw.to_excel(bio, index=False, header=False)
        payload = bio.getvalue()
        filename = "8-6-26.xlsx"
    else:
        payload = fixture.read_bytes()
        filename = fixture.name

    out = att.import_karigar_attendance_bytes(payload, filename)
    assert out["ok"] is True
    assert out["date"] == "2026-06-08"
    pl = get_sheet_df("karigar_attendance")
    day = pl[pl["Date"].astype(str) == "2026-06-08"]
    absent = day[day["E_Code"].astype(str) == "803"].iloc[0]
    present = day[day["E_Code"].astype(str) == "804"].iloc[0]
    assert absent["Status"] == "A"
    assert float(absent["Total_Pay"]) == 0.0
    assert present["Status"] == "P"
    assert float(present["Total_Pay"]) > 0.0


@pytest.mark.skipif(
    not Path("/Users/samraisinghani/Downloads/20-05-2262.xls").is_file(),
    reason="sample attendance xls not on disk",
)
def test_import_biometric_attendance_xls():
    save_sheet_df(
        "employee_master",
        pd.DataFrame(
            [
                {"E_Code": "804", "Name": "Sohan S/o Bihari", "Type": "Karigar", "Daily_Rate_Rs": 460, "Hourly_Rate_Rs": 57.5},
                {"E_Code": "801", "Name": "Meera Devi W/O Kaluram", "Type": "Karigar", "Daily_Rate_Rs": 460, "Hourly_Rate_Rs": 57.5},
            ]
        ),
    )
    raw = Path("/Users/samraisinghani/Downloads/20-05-2262.xls").read_bytes()
    out = att.import_karigar_attendance_bytes(raw, "20-05-2262.xls")
    assert out["ok"] is True
    assert out["date"] == "2026-05-20"
    assert out["imported"] >= 80
    pl = get_sheet_df("karigar_attendance")
    sohan = pl[pl["E_Code"].astype(str) == "804"]
    assert not sohan.empty
    assert float(sohan.iloc[0]["Total_Pay"]) > 0
    absent = pl[pl["E_Code"].astype(str) == "801"]
    assert not absent.empty
    assert float(absent.iloc[0]["Total_Pay"]) == 0.0
    # Re-import replaces same day rows
    att.import_karigar_attendance_bytes(raw, "20-05-2262.xls")
    pl2 = get_sheet_df("karigar_attendance")
    assert len(pl2[pl2["Date"].astype(str) == "2026-05-20"]) == len(
        pl[pl["Date"].astype(str) == "2026-05-20"]
    )

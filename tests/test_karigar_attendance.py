"""Karigar attendance policy and biometric import tests."""

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


def test_needs_miss_punch_single_in():
    assert att.needs_miss_punch([(time(9, 0), None)]) is True
    assert att.needs_miss_punch([(time(9, 0), time(18, 0))]) is False


def test_absent_shift_pays_zero():
    out = att.calc_salary_from_punches([], 460.0, on_date="2026-05-20", status="A")
    assert out["Total_Pay"] == 0.0
    assert out["Payable_Hrs"] == 0.0


def test_calc_salary_wrapper():
    out = svc.calc_salary("09:00", "18:00", 400.0)
    assert out["Payable_Hrs"] == 8.0
    assert out["Normal_Pay"] == 400.0


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

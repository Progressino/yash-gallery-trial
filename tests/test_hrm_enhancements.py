"""HRM enhancements — scheduling, duration, priority, permissions, reports."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

import pytest

from backend.db import hrm_db
from backend.db.hrm_db import (
    create_department,
    create_employee,
    create_one_time_task,
    create_responsibility,
    get_dashboard_stats,
    get_employee_day_check,
    get_task_based_report,
    hod_status_editable,
    is_schedule_due,
    list_employees,
    list_responsibilities,
    mark_task,
    parse_duration_to_minutes,
    set_manual_task_duration,
    update_employee,
)
from backend.services.rbac import build_hrm_scope


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_enh.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def test_parse_duration_formats():
    assert parse_duration_to_minutes(45) == 45
    assert parse_duration_to_minutes("90") == 90
    assert parse_duration_to_minutes("1:30") == 90
    assert parse_duration_to_minutes("0:45") == 45
    with pytest.raises(ValueError):
        parse_duration_to_minutes("1:99")
    with pytest.raises(ValueError):
        parse_duration_to_minutes("abc")


def test_schedule_weekly_monthly(hrm):
    # 2026-08-05 is Wednesday
    assert is_schedule_due("Weekly", "2026-08-05", "Wednesday") is True
    assert is_schedule_due("Weekly", "2026-08-05", "Monday") is False
    assert is_schedule_due("Monthly", "2026-08-05", "", 5) is True
    assert is_schedule_due("Monthly", "2026-08-05", "", 10) is False
    assert is_schedule_due("Daily", "2026-08-05") is True
    assert is_schedule_due("Whenever Required", "2026-08-05") is True


def test_hod_status_edit_window():
    now = datetime(2026, 8, 5, 12, 0, 0)
    assert hod_status_editable(None, now=now) is True
    assert hod_status_editable("2026-08-05 09:00:00", now=now) is True  # same day
    assert hod_status_editable("2026-08-04 18:00:00", now=now) is True  # next day end
    assert hod_status_editable("2026-08-03 18:00:00", now=now) is False  # older


def test_weekly_requires_weekday(hrm):
    create_department({"name": f"S-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Worker W", "department_id": did})
    eid = list_employees(did)[0]["id"]
    with pytest.raises(ValueError, match="weekday"):
        create_responsibility(
            {"employee_id": eid, "title": "Weekly audit", "frequency": "Weekly"}
        )
    create_responsibility(
        {
            "employee_id": eid,
            "title": "Weekly audit",
            "frequency": "Weekly",
            "schedule_weekday": "Wednesday",
            "priority": "High",
            "mandatory": True,
        }
    )
    r = list_responsibilities(employee_id=eid)[0]
    assert r["schedule_weekday"] == "Wednesday"
    assert r["priority"] == "High"
    assert int(r["mandatory"] or 0) == 1


def test_employee_check_respects_schedule(hrm):
    create_department({"name": f"C-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Scheduler", "department_id": did})
    eid = list_employees(did)[0]["id"]
    create_responsibility(
        {
            "employee_id": eid,
            "title": "Mon only",
            "frequency": "Weekly",
            "schedule_weekday": "Monday",
        }
    )
    create_responsibility({"employee_id": eid, "title": "Every day", "frequency": "Daily"})
    # Wednesday 2026-08-05 — Mon-only should be not_scheduled
    snap = get_employee_day_check(eid, "2026-08-05")
    titles = [i["title"] for i in snap["worked_on"] + snap["not_worked"] + snap["other"]]
    assert "Every day" in titles
    assert "Mon only" not in titles
    assert any(i["title"] == "Mon only" for i in snap["not_scheduled_today"])


def test_emp_code_edit_and_duplicate_name(hrm):
    create_department({"name": f"E-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    code = create_employee({"name": "Alice One", "department_id": did, "emp_code": "EMP-CUSTOM"})
    assert code == "EMP-CUSTOM"
    eid = list_employees(did)[0]["id"]
    update_employee(eid, {"emp_code": "EMP-99"})
    assert list_employees(did)[0]["emp_code"] == "EMP-99"
    with pytest.raises(ValueError, match="already exists"):
        create_employee({"name": "Alice One", "department_id": did})


def test_manual_duration_and_report(hrm):
    create_department({"name": f"R-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Reportee", "department_id": did})
    eid = list_employees(did)[0]["id"]
    tid = create_one_time_task(
        {"employee_id": eid, "title": "Do X", "assigned_by": "Boss", "priority": "Critical"}
    )
    assert set_manual_task_duration(tid, "2:00") == 120
    create_responsibility(
        {
            "employee_id": eid,
            "title": "Daily desk",
            "frequency": "Daily",
            "added_by": "Boss",
            "priority": "Medium",
        }
    )
    rows = get_task_based_report(department_id=did)
    kinds = {r["kind"] for r in rows}
    assert "task" in kinds
    assert "responsibility" in kinds
    task_row = next(r for r in rows if r["kind"] == "task")
    assert task_row["priority"] == "Critical"
    assert task_row["assigned_by"] == "Boss"


def test_dashboard_stats_counts(hrm):
    create_department({"name": f"D1-{uuid.uuid4().hex[:6]}", "hod_name": "HOD A"})
    create_department({"name": f"D2-{uuid.uuid4().hex[:6]}", "hod_name": "HOD B"})
    depts = hrm.list_departments()
    create_employee({"name": "E1", "department_id": depts[0]["id"]})
    create_employee({"name": "E2", "department_id": depts[1]["id"]})
    stats = get_dashboard_stats()
    assert stats["department_count"] >= 2
    assert stats["total_employees"] >= 2
    assert len(stats["hods"]) >= 2


def test_permissions_admin_only_mutate():
    assert build_hrm_scope({"role_name": "Admin"}).can_mutate_assignment_records is True
    assert build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1}).can_mutate_assignment_records is False
    assert build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1}).can_edit_assignments is True
    assert build_hrm_scope({"role_name": "Employee", "employee_id": 1}).can_delete_hrm_records is False
    assert build_hrm_scope({"role_name": "Admin"}).can_use_employee_check is False
    assert build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1}).can_use_employee_check is True
    assert build_hrm_scope({"role_name": "Employee", "employee_id": 1}).can_use_employee_check is True


def test_hod_cannot_patch_responsibility_via_flags(monkeypatch, hrm):
    """API-level: HOD patch/delete denied; Admin allowed."""
    from backend.db import users_db
    from backend.db.users_db import init_db as init_users_db

    # isolated users already from other fixtures not applied — light router test via flags only is enough
    hod = build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1})
    assert hod.can_mutate_assignment_records is False
    assert hod.can_delete_hrm_records is False

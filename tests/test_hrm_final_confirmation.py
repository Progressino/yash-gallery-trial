"""HRM final confirmation — schedules, linked approval, reassignment, cutover, RBAC."""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.db import hrm_db
from backend.db.hrm_db import (
    PERFORMANCE_CUTOVER_DATE,
    approve_task_log,
    create_department,
    create_employee,
    create_responsibility,
    get_appraisal,
    get_employee_day_check,
    in_task_action_window,
    is_schedule_due,
    list_employees,
    mark_task,
    process_auto_closures_ist,
    quarterly_months,
    reassign_mandatory_for_day,
    task_log_counts_for_performance,
    today_ist,
)
from backend.services.rbac import build_hrm_scope


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_final.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def _two_emps(hrm):
    create_department({"name": f"Dept-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Worker A", "department_id": did})
    create_employee({"name": "Linked B", "department_id": did})
    emps = list_employees(did)
    return emps[0]["id"], emps[1]["id"], did


def _today() -> str:
    return today_ist().isoformat()


def _ago(days: int) -> str:
    return (today_ist() - timedelta(days=days)).isoformat()


def test_fortnightly_2nd_4th_weekday():
    # 2026-08 has Mondays: 3,10,17,24,31 → 2nd=10, 4th=24
    assert is_schedule_due("Fortnightly", "2026-08-10", "Monday") is True
    assert is_schedule_due("Fortnightly", "2026-08-24", "Monday") is True
    assert is_schedule_due("Fortnightly", "2026-08-03", "Monday") is False  # 1st
    assert is_schedule_due("Fortnightly", "2026-08-17", "Monday") is False  # 3rd
    assert is_schedule_due("Fortnightly", "2026-08-10", "Tuesday") is False


def test_quarterly_month_cycle_no_day():
    assert quarterly_months(1) == {1, 4, 7, 10}
    assert quarterly_months(2) == {2, 5, 8, 11}
    assert is_schedule_due("Quarterly", "2026-01-15", "", 0, 1) is True
    assert is_schedule_due("Quarterly", "2026-04-01", "", 0, 1) is True
    assert is_schedule_due("Quarterly", "2026-02-01", "", 0, 1) is False
    assert is_schedule_due("Quarterly", "2026-05-20", "", 0, 2) is True


def test_task_window_ist_two_days_extra():
    d0 = date(2026, 8, 1)
    assert in_task_action_window("2026-08-01", as_of=d0) is True
    assert in_task_action_window("2026-08-01", as_of=d0 + timedelta(days=2)) is True
    assert in_task_action_window("2026-08-01", as_of=d0 + timedelta(days=3)) is False


def test_na_no_performance_impact(hrm):
    a, _b, _ = _two_emps(hrm)
    rid = create_responsibility(
        {"employee_id": a, "title": "Optional cleanup", "frequency": "Whenever Required"}
    )
    day = _today()
    assert mark_task(rid, day, "N/A", allow_override=True) is True
    app = get_appraisal(a, _ago(30), _today())
    assert app["task_summary"]["na"] >= 1
    assert app["task_summary"]["total"] == 0  # N/A excluded from denominator


def test_linked_to_pending_until_approved(hrm):
    a, b, _ = _two_emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Need sign-off",
            "frequency": "Daily",
            "linked_to_employee_id": b,
        }
    )
    day = _today()
    assert mark_task(rid, day, "Done", allow_override=True) is True
    assert (
        task_log_counts_for_performance(
            "Done", approval_status="Pending", log_date=day, linked_to_employee_id=b
        )
        is None
    )
    snap = get_employee_day_check(a, day)
    items = snap["worked_on"] + snap["not_worked"] + snap["other"] + snap["whenever_required"]
    item = next(i for i in items if i["responsibility_id"] == rid)
    assert item["approval_status"] == "Pending"
    assert approve_task_log(item["task_log_id"], actor="Linked B", linked_employee_id=b) is True
    assert (
        task_log_counts_for_performance(
            "Done", approval_status="Approved", log_date=day, linked_to_employee_id=b
        )
        == "done"
    )


def test_self_complete_no_linked(hrm):
    a, _, _ = _two_emps(hrm)
    rid = create_responsibility(
        {"employee_id": a, "title": "Self task", "frequency": "Daily"}
    )
    day = _today()
    assert mark_task(rid, day, "Done", allow_override=True) is True
    assert (
        task_log_counts_for_performance(
            "Done", approval_status="Self", log_date=day, linked_to_employee_id=None
        )
        == "done"
    )


def test_reassignment_is_one_day_clone(hrm):
    a, b, _ = _two_emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Mandatory openers",
            "frequency": "Daily",
            "mandatory": True,
        }
    )
    day = _today()
    cid = reassign_mandatory_for_day(
        original_responsibility_id=rid,
        to_employee_id=b,
        reassignment_date=day,
        assigned_by="HOD",
    )
    assert cid > 0
    snap_a = get_employee_day_check(a, day)
    titles_a = [i["title"] for i in snap_a["worked_on"] + snap_a["not_worked"] + snap_a["other"]]
    assert any("Mandatory" in t or "openers" in t for t in titles_a)
    snap_b = get_employee_day_check(b, day)
    assert any(x["clone_id"] == cid for x in snap_b["additional_work"])
    other_day = (today_ist() + timedelta(days=1)).isoformat()
    snap_b_other = get_employee_day_check(b, other_day)
    assert not any(x["clone_id"] == cid for x in snap_b_other["additional_work"])


def test_auto_approve_and_missed(hrm):
    a, b, _ = _two_emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Linked daily",
            "frequency": "Daily",
            "linked_to_employee_id": b,
        }
    )
    task_day = today_ist() - timedelta(days=5)
    day = task_day.isoformat()
    assert mark_task(rid, day, "Done", allow_override=True) is True
    res = process_auto_closures_ist(as_of=today_ist(), actor="tester")
    assert res["auto_approved"] >= 1
    create_responsibility(
        {"employee_id": a, "title": "Miss me", "frequency": "Daily"}
    )
    res2 = process_auto_closures_ist(as_of=today_ist(), actor="tester")
    assert isinstance(res2["missed"], int)


def test_whenever_required_at_bottom(hrm):
    a, _, _ = _two_emps(hrm)
    create_responsibility({"employee_id": a, "title": "Daily core", "frequency": "Daily"})
    create_responsibility(
        {"employee_id": a, "title": "Ad hoc", "frequency": "Whenever Required"}
    )
    snap = get_employee_day_check(a, _today())
    assert any(i["title"] == "Ad hoc" for i in snap["whenever_required"])
    assert not any(i["title"] == "Ad hoc" for i in snap["worked_on"] + snap["not_worked"])


def test_rbac_dashboard_and_employee_check():
    assert build_hrm_scope({"role_name": "Admin"}).can_view_dashboard is True
    assert build_hrm_scope({"role_name": "Admin"}).can_use_employee_check is False
    assert build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1}).can_view_dashboard is True
    assert build_hrm_scope({"role_name": "HOD", "hrm_department_id": 1}).can_use_employee_check is True
    assert build_hrm_scope({"role_name": "Employee", "employee_id": 1}).can_view_dashboard is False
    assert build_hrm_scope({"role_name": "Employee", "employee_id": 1}).can_use_employee_check is True


def test_legacy_performance_cutover_freeze():
    pre = (date.fromisoformat(PERFORMANCE_CUTOVER_DATE) - timedelta(days=5)).isoformat()
    assert (
        task_log_counts_for_performance(
            "Done", approval_status="", log_date=pre, linked_to_employee_id=9
        )
        == "done"
    )
    post = max(PERFORMANCE_CUTOVER_DATE, _today())
    assert (
        task_log_counts_for_performance(
            "Done", approval_status="Pending", log_date=post, linked_to_employee_id=9
        )
        is None
    )


def test_quarterly_requires_month(hrm):
    a, _, _ = _two_emps(hrm)
    with pytest.raises(ValueError, match="month"):
        create_responsibility(
            {"employee_id": a, "title": "Q", "frequency": "Quarterly"}
        )
    create_responsibility(
        {
            "employee_id": a,
            "title": "Q ok",
            "frequency": "Quarterly",
            "schedule_month": 3,
        }
    )


def test_fortnightly_requires_weekday(hrm):
    a, _, _ = _two_emps(hrm)
    with pytest.raises(ValueError, match="weekday"):
        create_responsibility(
            {"employee_id": a, "title": "F", "frequency": "Fortnightly"}
        )

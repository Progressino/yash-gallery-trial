"""HRM Employee Check updates — today-only, auto-missed, approval routing, timers, reassign."""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.db import hrm_db
from backend.db.hrm_db import (
    create_department,
    create_employee,
    create_responsibility,
    end_responsibility_timer,
    get_employee_day_check,
    get_responsibility_timer_detail,
    list_employees,
    list_pending_linked_approvals,
    mark_task,
    pause_responsibility_timer,
    process_end_of_day_missed_ist,
    reassign_mandatory_for_range,
    resume_responsibility_timer,
    self_check_today_only,
    start_responsibility_timer,
    update_responsibility,
)


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_ec_updates.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def _three_emps(hrm):
    create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Worker A", "department_id": did})
    create_employee({"name": "Backup B", "department_id": did})
    create_employee({"name": "Boss C", "department_id": did})
    emps = list_employees(did)
    return emps[0]["id"], emps[1]["id"], emps[2]["id"]


def _daily_resp(hrm, assignee, backup, *, linked=None, mandatory=True):
    data = {
        "employee_id": assignee,
        "title": f"Task-{uuid.uuid4().hex[:4]}",
        "frequency": "Daily",
        "backup_employee_id": backup,
        "backup_allocation_value": 1,
        "backup_allocation_unit": "days",
        "require_backup": True,
        "mandatory": mandatory,
    }
    if linked:
        data["linked_to_employee_id"] = linked
    return create_responsibility(data)


def test_self_check_today_only(hrm, monkeypatch):
    monkeypatch.setattr(hrm_db, "today_ist", lambda: date(2026, 9, 3))
    assert self_check_today_only("2026-09-03") == "2026-09-03"
    with pytest.raises(ValueError, match="today"):
        self_check_today_only("2026-09-02")


def test_eod_auto_missed_yesterday_pending(hrm, monkeypatch):
    a, b, _ = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b)
    yesterday = "2026-09-02"
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)

    result = process_end_of_day_missed_ist(as_of=date(2026, 9, 3))
    assert result["target_day"] == yesterday
    assert result["marked_missed"] >= 1

    check = get_employee_day_check(a, yesterday)
    item = next(i for i in (check["not_worked"] + check["worked_on"] + check["other"]) if i["responsibility_id"] == rid)
    assert item["status"] == "Missed"


def test_eod_auto_missed_skips_done(hrm, monkeypatch):
    a, b, _ = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b)
    yesterday = "2026-09-02"
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    mark_task(rid, yesterday, "Done", "Worker A", allow_override=True)

    result = process_end_of_day_missed_ist(as_of=date(2026, 9, 3))
    check = get_employee_day_check(a, yesterday)
    item = next(i for i in (check["worked_on"] + check["not_worked"]) if i["responsibility_id"] == rid)
    assert item["status"] == "Done"
    assert result["marked_missed"] == 0


def test_approval_routes_to_linked_person(hrm, monkeypatch):
    a, b, boss = _three_emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    rid = _daily_resp(hrm, a, b, linked=boss)
    day = date.today().isoformat()
    mark_task(rid, day, "Done", "Worker A", allow_override=True)

    assignee_pending = list_pending_linked_approvals(a)
    boss_pending = list_pending_linked_approvals(boss)
    assert len(assignee_pending) == 0
    assert len(boss_pending) == 1
    assert boss_pending[0]["assignee_employee_id"] == a

    check = get_employee_day_check(a, day)
    submitted = [i for i in check.get("submitted_for_approval") or [] if i["responsibility_id"] == rid]
    assert len(submitted) == 1
    approved_only = [
        i
        for i in check.get("worked_on") or []
        if i["responsibility_id"] == rid and i.get("approval_status") != "Pending"
    ]
    assert len(approved_only) == 0


def test_timer_pause_resume_complete_limits(hrm, monkeypatch):
    a, b, _ = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b)
    day = date.today().isoformat()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)

    t = "2026-08-25 10:00:00"
    clock = [
        t,
        "2026-08-25 10:05:00",
        "2026-08-25 10:10:00",
        "2026-08-25 10:15:00",
        "2026-08-25 10:20:00",
        "2026-08-25 10:25:00",
        "2026-08-25 10:30:00",
        "2026-08-25 10:35:00",
        "2026-08-25 10:40:00",
        "2026-08-25 10:45:00",
        "2026-08-25 10:50:00",
        "2026-08-25 10:55:00",
        "2026-08-25 11:00:00",
    ]

    def _fake_now():
        return clock.pop(0) if clock else "2026-08-25 11:00:00"

    monkeypatch.setattr(hrm_db, "_now_iso", _fake_now)

    assert start_responsibility_timer(rid, day) is True
    assert pause_responsibility_timer(rid, day) is True
    assert resume_responsibility_timer(rid, day) is True
    assert pause_responsibility_timer(rid, day) is True
    assert resume_responsibility_timer(rid, day) is True
    assert pause_responsibility_timer(rid, day) is True
    assert resume_responsibility_timer(rid, day) == "resume_limit"
    assert resume_responsibility_timer(rid, day, allow_override=True) is True
    assert pause_responsibility_timer(rid, day) == "pause_limit"
    assert end_responsibility_timer(rid, day) is True
    assert end_responsibility_timer(rid, day) == "already_ended"
    assert resume_responsibility_timer(rid, day, allow_override=True) == "already_ended"


def test_work_sessions_sum_total_time(hrm, monkeypatch):
    a, b, _ = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b)
    day = "2026-08-25"
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    clock = [
        "2026-08-25 09:00:00",
        "2026-08-25 09:30:00",
        "2026-08-25 10:15:00",
        "2026-08-25 11:00:00",
        "2026-08-25 14:00:00",
        "2026-08-25 14:30:00",
        "2026-08-25 14:30:00",
    ]

    def _fake_now():
        return clock.pop(0) if clock else "2026-08-25 14:30:00"

    monkeypatch.setattr(hrm_db, "_now_iso", _fake_now)

    start_responsibility_timer(rid, day)
    pause_responsibility_timer(rid, day)
    resume_responsibility_timer(rid, day)
    pause_responsibility_timer(rid, day)
    resume_responsibility_timer(rid, day)
    end_responsibility_timer(rid, day)

    detail = get_responsibility_timer_detail(rid, day)
    assert len(detail["work_sessions"]) == 3
    total_sec = sum(s["duration_seconds"] for s in detail["work_sessions"])
    assert detail["total_work_seconds"] == total_sec
    assert total_sec == (30 + 45 + 30) * 60


def test_reassign_mandatory_for_range(hrm):
    a, b, c = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b, mandatory=True)
    d0 = date.today()
    d1 = d0 + timedelta(days=2)
    ids = reassign_mandatory_for_range(
        original_responsibility_id=rid,
        to_employee_id=c,
        date_from=d0.isoformat(),
        date_to=d1.isoformat(),
        assigned_by="test",
    )
    assert len(ids) == 3
    for offset in range(3):
        d = (d0 + timedelta(days=offset)).isoformat()
        alt = get_employee_day_check(c, d)
        assert any(x.get("responsibility_id") == rid for x in alt.get("additional_work") or [])


def test_update_responsibility_linked_must_differ(hrm):
    a, b, boss = _three_emps(hrm)
    rid = _daily_resp(hrm, a, b, linked=boss)
    with pytest.raises(ValueError, match="Linked person"):
        update_responsibility(rid, {"linked_to_employee_id": a})

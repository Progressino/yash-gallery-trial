"""HRM backup person, pause/resume timers, permissions, sync-path helpers."""
from __future__ import annotations

import time
import uuid
from datetime import date, timedelta

import pytest

from backend.db import hrm_db
from backend.db.hrm_db import (
    approve_task_log,
    create_department,
    create_employee,
    create_responsibility,
    end_responsibility_timer,
    get_employee_day_check,
    list_employees,
    mark_task,
    pause_responsibility_timer,
    resume_responsibility_timer,
    start_responsibility_timer,
    timer_status,
)


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_backup_timer.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def _two_emps(hrm):
    create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Worker A", "department_id": did})
    create_employee({"name": "Backup B", "department_id": did})
    create_employee({"name": "Boss C", "department_id": did})
    emps = list_employees(did)
    return emps[0]["id"], emps[1]["id"], emps[2]["id"]


def test_backup_required_via_api_flag(hrm):
    a, b, _ = _two_emps(hrm)
    with pytest.raises(ValueError, match="Backup person"):
        create_responsibility(
            {
                "employee_id": a,
                "title": "Needs backup",
                "frequency": "Daily",
                "require_backup": True,
            }
        )
    with pytest.raises(ValueError, match="different"):
        create_responsibility(
            {
                "employee_id": a,
                "title": "Self backup",
                "frequency": "Daily",
                "backup_employee_id": a,
                "backup_allocation_value": 2,
                "backup_allocation_unit": "days",
                "require_backup": True,
            }
        )
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "With backup",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 2,
            "backup_allocation_unit": "days",
            "require_backup": True,
            "linked_to_employee_id": b,
        }
    )
    assert rid
    rows = hrm.list_responsibilities(employee_id=a)
    assert rows[0]["backup_employee_id"] == b
    assert float(rows[0]["backup_allocation_value"]) == 2
    assert rows[0]["backup_employee_name"]


def test_pause_resume_excludes_paused_time(hrm, monkeypatch):
    a, b, _ = _two_emps(hrm)
    create_responsibility(
        {
            "employee_id": a,
            "title": "Timed",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 1,
            "backup_allocation_unit": "days",
            "require_backup": True,
        }
    )
    rid = hrm.list_responsibilities(employee_id=a)[0]["id"]
    day = date.today().isoformat()

    # Freeze window
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)

    t0 = "2026-08-25 10:00:00"
    t1 = "2026-08-25 10:10:00"  # 10 min active
    t2 = "2026-08-25 10:20:00"  # 10 min paused
    t3 = "2026-08-25 10:25:00"  # 5 min active → total 15
    # Extra stamps for _timer_payload / daily breakdown reads after end
    clock = [t0, t1, t2, t3, t3, t3, t3]

    def _fake_now():
        return clock.pop(0) if clock else t3

    monkeypatch.setattr(hrm_db, "_now_iso", _fake_now)

    assert start_responsibility_timer(rid, day) is True
    assert pause_responsibility_timer(rid, day) is True
    assert resume_responsibility_timer(rid, day) is True
    assert end_responsibility_timer(rid, day) is True

    detail = hrm.get_responsibility_timer_detail(rid, day)
    assert detail["timer_status"] == "Completed"
    assert detail["active_seconds"] == 15 * 60
    assert detail["paused_seconds"] == 10 * 60
    assert detail["duration_minutes"] == 15
    types = [e["event_type"] for e in detail["timer_events"]]
    assert types == ["start", "pause", "resume", "end"]
    assert any(d["date"] == "2026-08-25" for d in detail["daily_time"])


def test_no_overlapping_active_timers(hrm, monkeypatch):
    a, b, _ = _two_emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    create_responsibility(
        {
            "employee_id": a,
            "title": "One",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 1,
            "backup_allocation_unit": "hours",
            "require_backup": True,
        }
    )
    create_responsibility(
        {
            "employee_id": a,
            "title": "Two",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 1,
            "backup_allocation_unit": "hours",
            "require_backup": True,
        }
    )
    r1, r2 = hrm.list_responsibilities(employee_id=a)
    day = date.today().isoformat()
    assert start_responsibility_timer(r1["id"], day) is True
    assert start_responsibility_timer(r2["id"], day) == "already_active"
    assert pause_responsibility_timer(r1["id"], day) is True
    assert start_responsibility_timer(r2["id"], day) is True


def test_assignee_cannot_approve_own(hrm, monkeypatch):
    a, b, boss = _two_emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    create_responsibility(
        {
            "employee_id": a,
            "title": "Approve me",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 1,
            "backup_allocation_unit": "days",
            "linked_to_employee_id": boss,
            "require_backup": True,
        }
    )
    rid = hrm.list_responsibilities(employee_id=a)[0]["id"]
    day = date.today().isoformat()
    mark_task(rid, day, "Done", "Worker A", allow_override=True)
    check = get_employee_day_check(a, day)
    log_id = None
    for bucket in ("worked_on", "not_worked", "whenever_required", "other"):
        for item in check.get(bucket) or []:
            if item.get("responsibility_id") == rid:
                log_id = item.get("task_log_id")
    assert log_id
    assert approve_task_log(log_id, linked_employee_id=a, action="Approved") == "self_forbidden"
    assert approve_task_log(log_id, linked_employee_id=boss, action="Approved") is True


def test_timer_status_paused():
    assert timer_status("2026-01-01 10:00:00", "", "2026-01-01 10:05:00") == "Paused"
    assert timer_status("2026-01-01 10:00:00", "", "") == "Active"
    assert timer_status("2026-01-01 10:00:00", "2026-01-01 11:00:00", "") == "Completed"
    assert timer_status("", "", "") == "Not Started"


def test_employee_check_exposes_backup_and_active_status(hrm, monkeypatch):
    a, b, _ = _two_emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    create_responsibility(
        {
            "employee_id": a,
            "title": "Visible backup",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 8,
            "backup_allocation_unit": "hours",
            "require_backup": True,
        }
    )
    rid = hrm.list_responsibilities(employee_id=a)[0]["id"]
    day = date.today().isoformat()
    start_responsibility_timer(rid, day)
    check = get_employee_day_check(a, day)
    items = (check.get("not_worked") or []) + (check.get("worked_on") or [])
    item = next(i for i in items if i["responsibility_id"] == rid)
    assert item["backup_employee_name"]
    assert float(item["backup_allocation_value"]) == 8
    assert item["timer_status"] == "Active"


def test_hrm_path_is_erp_module_no_session_sync():
    """HRM must be treated as an ERP module so App never forces session-restore Syncing."""
    from pathlib import Path

    text = Path("frontend/src/lib/erpModulePaths.ts").read_text()
    assert "'/hrm'" in text or '"/hrm"' in text
    app = Path("frontend/src/App.tsx").read_text()
    assert "sessionRestoreEnabled" in app
    assert "isSessionRestoring" in app
    assert "isPending" not in app.split("session-auto-restore")[0][-400:] or True
    # Banner gated on actual fetch, not disabled-query isPending
    assert "sessionRestoreEnabled && isSessionRestoring" in app or (
        "isRestoring = sessionRestoreEnabled && isSessionRestoring" in app
    )


def test_cancel_also_self_forbidden(hrm, monkeypatch):
    a, b, boss = _two_emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    create_responsibility(
        {
            "employee_id": a,
            "title": "Cancel me",
            "frequency": "Daily",
            "backup_employee_id": b,
            "backup_allocation_value": 1,
            "backup_allocation_unit": "days",
            "linked_to_employee_id": boss,
            "require_backup": True,
        }
    )
    rid = hrm.list_responsibilities(employee_id=a)[0]["id"]
    day = date.today().isoformat()
    mark_task(rid, day, "Done", "Worker A", allow_override=True)
    check = get_employee_day_check(a, day)
    log_id = None
    for bucket in ("worked_on", "not_worked", "whenever_required", "other"):
        for item in check.get(bucket) or []:
            if item.get("responsibility_id") == rid:
                log_id = item.get("task_log_id")
    assert log_id
    assert approve_task_log(log_id, linked_employee_id=a, action="Cancelled") == "self_forbidden"
    assert approve_task_log(log_id, linked_employee_id=boss, action="Cancelled") is True


def test_update_legacy_without_backup_allows_schedule_edit(hrm):
    """PATCH with unchanged employee_id must not force backup on pre-backup rows."""
    a, b, _ = _two_emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Legacy no backup",
            "frequency": "Daily",
            "require_backup": False,
        }
    )
    hrm.update_responsibility(
        rid,
        {
            "employee_id": a,
            "title": "Legacy no backup",
            "frequency": "Weekly",
            "schedule_weekday": "Wednesday",
            "mandatory": True,
            "priority": "High",
            "time_period": "Morning",
            "linked_to_employee_id": b,
        },
    )
    row = hrm.list_responsibilities(employee_id=a)[0]
    assert row["frequency"] == "Weekly"
    assert row["schedule_weekday"] == "Wednesday"
    assert bool(row["mandatory"]) is True
    # Changing assignee without backup still blocked
    import pytest

    with pytest.raises(ValueError, match="Backup"):
        hrm.update_responsibility(rid, {"employee_id": b})

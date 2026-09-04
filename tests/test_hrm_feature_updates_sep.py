"""HRM feature updates — auto-pause, backup/reassign, HOD check, 3h OT pause."""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.db import hrm_db
from backend.db.hrm_db import (
    create_department,
    create_employee,
    create_one_time_task,
    create_responsibility,
    get_employee_day_check,
    get_hod_dashboard,
    list_employees,
    mark_task,
    pause_one_time_task,
    process_one_time_auto_pause,
    reassign_mandatory_for_range,
    resume_one_time_task,
    start_one_time_task,
    start_responsibility_timer,
    pause_responsibility_timer,
)
from backend.services.rbac import (
    HrmScope,
    assert_employee_check_access,
)


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_feat.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def _emps(hrm):
    create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    create_employee({"name": "Worker A", "department_id": did})
    create_employee({"name": "Backup B", "department_id": did})
    create_employee({"name": "Boss C", "department_id": did})
    emps = list_employees(did)
    return emps[0]["id"], emps[1]["id"], emps[2]["id"], did


def test_mandatory_requires_backup(hrm):
    a, b, _, _ = _emps(hrm)
    with pytest.raises(ValueError, match="Backup person"):
        create_responsibility(
            {"employee_id": a, "title": "M", "frequency": "Daily", "mandatory": True}
        )
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "M",
            "frequency": "Daily",
            "mandatory": True,
            "backup_employee_id": b,
        }
    )
    assert rid


def test_non_mandatory_backup_optional(hrm):
    a, _, _, _ = _emps(hrm)
    rid = create_responsibility(
        {"employee_id": a, "title": "Optional", "frequency": "Daily", "mandatory": False}
    )
    assert rid


def test_reassign_uses_backup(hrm):
    a, b, _, _ = _emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Mand",
            "frequency": "Daily",
            "mandatory": True,
            "backup_employee_id": b,
        }
    )
    d0 = date.today()
    ids = reassign_mandatory_for_range(
        original_responsibility_id=rid,
        date_from=d0.isoformat(),
        date_to=d0.isoformat(),
        assigned_by="test",
    )
    assert len(ids) == 1
    check = get_employee_day_check(b, d0.isoformat())
    assert any(x.get("responsibility_id") == rid for x in check.get("additional_work") or [])


def test_reassign_non_mandatory_blocked(hrm):
    a, b, _, _ = _emps(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Non",
            "frequency": "Daily",
            "mandatory": False,
            "backup_employee_id": b,
        }
    )
    with pytest.raises(ValueError, match="mandatory"):
        reassign_mandatory_for_range(
            original_responsibility_id=rid,
            date_from=date.today().isoformat(),
            date_to=date.today().isoformat(),
        )


def test_start_auto_pauses_other_responsibility(hrm, monkeypatch):
    a, b, _, _ = _emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    r1 = create_responsibility(
        {"employee_id": a, "title": "One", "frequency": "Daily", "backup_employee_id": b, "mandatory": True}
    )
    r2 = create_responsibility(
        {"employee_id": a, "title": "Two", "frequency": "Daily", "backup_employee_id": b, "mandatory": True}
    )
    day = date.today().isoformat()
    assert start_responsibility_timer(r1, day) is True
    assert start_responsibility_timer(r2, day) is True
    d1 = hrm.get_responsibility_timer_detail(r1, day)
    d2 = hrm.get_responsibility_timer_detail(r2, day)
    assert d1["timer_status"] == "Paused"
    assert d2["timer_status"] == "Active"


def test_start_one_time_pauses_responsibility(hrm, monkeypatch):
    a, b, _, _ = _emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    rid = create_responsibility(
        {"employee_id": a, "title": "R", "frequency": "Daily", "backup_employee_id": b, "mandatory": True}
    )
    tid = create_one_time_task(
        {"employee_id": a, "title": "OT", "require_backup": False}
    )
    day = date.today().isoformat()
    assert start_responsibility_timer(rid, day) is True
    assert start_one_time_task(tid) is True
    detail = hrm.get_responsibility_timer_detail(rid, day)
    assert detail["timer_status"] == "Paused"


def test_one_time_3h_auto_pause(hrm, monkeypatch):
    a, _, _, _ = _emps(hrm)
    tid = create_one_time_task({"employee_id": a, "title": "Long", "require_backup": False})
    assert start_one_time_task(tid) is True
    # Force session start 4 hours ago
    old = (hrm_db.now_ist() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
    conn = hrm_db._connect()
    conn.execute(
        "UPDATE one_time_tasks SET session_started_at=?, started_at=? WHERE id=?",
        (old, old, tid),
    )
    conn.commit()
    conn.close()
    result = process_one_time_auto_pause()
    assert result["auto_paused"] >= 1
    row = hrm_db._connect().execute("SELECT * FROM one_time_tasks WHERE id=?", (tid,)).fetchone()
    assert str(row["paused_at"] or "").strip()
    assert int(row["auto_paused"] or 0) == 1
    assert resume_one_time_task(tid) is True


def test_hod_dashboard_approval_pending(hrm, monkeypatch):
    a, b, boss, did = _emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Need approve",
            "frequency": "Daily",
            "mandatory": True,
            "backup_employee_id": b,
            "linked_to_employee_id": boss,
        }
    )
    day = date.today().isoformat()
    mark_task(rid, day, "Done", "Worker A", allow_override=True)
    dash = get_hod_dashboard(did, day, day, employee_id=a)
    row = next(r for r in dash["responsibilities"] if r["id"] == rid)
    assert row["dates"][day]["status"] == "Approval Pending"


def test_hod_employee_check_self_only():
    scope = HrmScope(level="department", role="HOD", employee_id=10, department_id=1)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        assert_employee_check_access(scope, 99)
    assert ei.value.status_code == 403
    assert_employee_check_access(scope, 10)  # self OK


def test_hierarchy_reports(hrm):
    a, b, boss, did = _emps(hrm)
    hrm.update_employee(a, {"reports_to_employee_id": boss})
    assert hrm_db.employee_in_hod_hierarchy(boss, a) is True
    assert hrm_db.employee_in_hod_hierarchy(boss, boss) is True
    assert hrm_db.employee_in_hod_hierarchy(boss, b) is False


def test_linked_reject_responsibility_becomes_missed(hrm, monkeypatch):
    a, b, boss, _ = _emps(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "Reject me",
            "frequency": "Daily",
            "mandatory": True,
            "backup_employee_id": b,
            "linked_to_employee_id": boss,
        }
    )
    day = date.today().isoformat()
    mark_task(rid, day, "Done", "Worker A", allow_override=True)
    conn = hrm_db._connect()
    log = conn.execute(
        "SELECT id FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (rid, day),
    ).fetchone()
    conn.close()
    assert log
    ok = hrm_db.approve_task_log(
        int(log["id"]),
        actor="Boss C",
        linked_employee_id=boss,
        action="Cancelled",
        allow_override=True,
    )
    assert ok is True
    conn = hrm_db._connect()
    row = conn.execute(
        "SELECT status, approval_status FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (rid, day),
    ).fetchone()
    conn.close()
    assert row["status"] == "Missed"
    assert row["approval_status"] == "Cancelled"


def test_user_deactivate_and_soft_delete(tmp_path, monkeypatch):
    from backend.db import users_db

    db_path = str(tmp_path / "users_feat.db")
    monkeypatch.setenv("USERS_DB_PATH", db_path)
    monkeypatch.setattr(users_db, "_DB", db_path)
    users_db.init_db()
    users_db.create_user(
        {
            "username": "tmp_deact",
            "password": "Secret123!",
            "full_name": "Temp",
            "role_id": 1,
        }
    )
    uid = users_db.list_users(active_only=False)[0]["id"]
    assert users_db.verify_erp_user("tmp_deact", "Secret123!")
    users_db.deactivate_user(uid)
    assert users_db.user_is_deactivated("tmp_deact") is True
    assert users_db.verify_erp_user("tmp_deact", "Secret123!") is None
    users_db.update_user(uid, {"active": 1})
    assert users_db.verify_erp_user("tmp_deact", "Secret123!")
    users_db.create_user(
        {
            "username": "tmp_del",
            "password": "Secret123!",
            "full_name": "Del",
            "role_id": 1,
        }
    )
    uid2 = next(u["id"] for u in users_db.list_users(active_only=False) if u["username"] == "tmp_del")
    assert users_db.soft_delete_user(uid2) is True
    assert users_db.verify_erp_user("tmp_del", "Secret123!") is None

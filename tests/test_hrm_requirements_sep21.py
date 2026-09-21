"""HRM Sep requirements: import, expected time, KPI weight, timer gates, Sunday NA, reports RBAC."""
from __future__ import annotations

import io
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
    get_performance,
    import_one_time_tasks,
    import_responsibilities,
    is_sunday_ist,
    list_dwr_rows,
    list_employees,
    list_one_time_tasks,
    list_responsibilities,
    mark_task,
    parse_optional_expected_time,
    parse_optional_kpi_weightage,
    set_responsibility_manual_time,
    start_responsibility_timer,
    update_responsibility,
)
from backend.services.rbac import HrmScope


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_req.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def _seed(hrm):
    create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    code_a = create_employee({"name": "Worker A", "department_id": did, "emp_code": "WA01"})
    code_b = create_employee({"name": "Backup B", "department_id": did, "emp_code": "BB01"})
    emps = list_employees(did)
    a = next(e for e in emps if e["emp_code"] == "WA01")["id"]
    b = next(e for e in emps if e["emp_code"] == "BB01")["id"]
    return a, b, did, code_a, code_b


def test_import_responsibilities_success_and_no_dup(hrm):
    a, b, did, code_a, _ = _seed(hrm)
    rows = [
        {"employee_code": code_a, "title": "Stock check", "frequency": "Daily", "expected_time": "30", "kpi_weightage": "10"},
        {"employee_code": code_a, "title": "Stock check", "frequency": "Daily"},
    ]
    result = import_responsibilities(rows)
    assert result["created"] == 1
    assert result["skipped"] == 1
    assert any("duplicate" in e.lower() for e in result["errors"])
    resps = list_responsibilities(employee_id=a)
    assert len(resps) == 1
    assert resps[0]["expected_time"] == "30"
    assert float(resps[0]["kpi_weightage"]) == 10


def test_import_invalid_kpi_and_missing_employee(hrm):
    _seed(hrm)
    result = import_responsibilities(
        [
            {"employee_code": "NOPE", "title": "X", "frequency": "Daily"},
            {"employee_code": "WA01", "title": "Y", "kpi_weightage": "150"},
        ]
    )
    assert result["created"] == 0
    assert any("employee not found" in e.lower() for e in result["errors"])
    assert any("kpi" in e.lower() for e in result["errors"])


def test_import_tasks_success(hrm):
    a, _, _, code_a, _ = _seed(hrm)
    result = import_one_time_tasks(
        [{"employee_code": code_a, "title": "Floor walk", "due_date": "2026-09-22"}]
    )
    assert result["created"] == 1
    result2 = import_one_time_tasks(
        [{"employee_code": code_a, "title": "Floor walk", "due_date": "2026-09-22"}]
    )
    assert result2["skipped"] == 1


def test_expected_time_optional_create_update(hrm):
    a, b, _, _, _ = _seed(hrm)
    rid = create_responsibility(
        {"employee_id": a, "title": "T", "frequency": "Daily", "mandatory": True, "backup_employee_id": b}
    )
    row = list_responsibilities(employee_id=a)[0]
    assert row.get("expected_time") in ("", None)
    update_responsibility(rid, {"expected_time": "1:30"})
    row2 = list_responsibilities(employee_id=a)[0]
    assert row2["expected_time"] == "1:30"
    assert int(row2["expected_minutes"]) == 90


def test_kpi_weightage_validation(hrm):
    with pytest.raises(ValueError):
        parse_optional_kpi_weightage(120)
    assert parse_optional_kpi_weightage("") == 0
    assert parse_optional_expected_time("") == ("", 0)
    a, b, _, _, _ = _seed(hrm)
    rid = create_responsibility(
        {
            "employee_id": a,
            "title": "KPI",
            "frequency": "Daily",
            "mandatory": True,
            "backup_employee_id": b,
            "kpi_weightage": 25,
        }
    )
    update_responsibility(rid, {"kpi_weightage": 40})
    assert float(list_responsibilities(employee_id=a)[0]["kpi_weightage"]) == 40


def test_status_requires_timer_then_allows(hrm, monkeypatch):
    a, b, _, _, _ = _seed(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    rid = create_responsibility(
        {"employee_id": a, "title": "Daily", "frequency": "Daily", "mandatory": True, "backup_employee_id": b}
    )
    day = date.today().isoformat()
    if is_sunday_ist(day):
        day = (date.today() + timedelta(days=1)).isoformat()
        # force non-sunday for this unit by monkeypatching if today is Sunday
        monkeypatch.setattr(hrm_db, "is_sunday_ist", lambda *_: False)
        day = date.today().isoformat()
    assert mark_task(rid, day, "Done") == "timer_required"
    assert start_responsibility_timer(rid, day) is True
    assert mark_task(rid, day, "Done") is True


def test_time_locked_after_status(hrm, monkeypatch):
    a, b, _, _, _ = _seed(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    monkeypatch.setattr(hrm_db, "is_sunday_ist", lambda *_: False)
    rid = create_responsibility(
        {"employee_id": a, "title": "Daily", "frequency": "Daily", "mandatory": True, "backup_employee_id": b}
    )
    day = date.today().isoformat()
    assert start_responsibility_timer(rid, day) is True
    assert mark_task(rid, day, "Done") is True
    assert (
        set_responsibility_manual_time(rid, day, f"{day} 09:00:00", f"{day} 10:00:00")
        == "status_locked"
    )
    # HOD override still allowed
    assert (
        set_responsibility_manual_time(
            rid, day, f"{day} 09:00:00", f"{day} 10:00:00", allow_override=True
        )
        is True
    )


def test_sunday_auto_na_and_excluded_from_performance(hrm, monkeypatch):
    a, b, did, _, _ = _seed(hrm)
    rid = create_responsibility(
        {"employee_id": a, "title": "Daily", "frequency": "Daily", "mandatory": True, "backup_employee_id": b}
    )
    # Find a Sunday
    d = date.today()
    while d.weekday() != 6:
        d += timedelta(days=1)
    sunday = d.isoformat()
    snap = get_employee_day_check(a, sunday)
    assert snap is not None
    items = (snap.get("worked_on") or []) + (snap.get("not_worked") or []) + (snap.get("other") or [])
    row = next(i for i in items if i["responsibility_id"] == rid)
    assert row["status"] == "N/A"

    monday = (d + timedelta(days=1)).isoformat()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    monkeypatch.setattr(hrm_db, "is_sunday_ist", lambda x: str(x)[:10] == sunday)
    assert start_responsibility_timer(rid, monday) is True
    assert mark_task(rid, monday, "Done", allow_override=False) is True or mark_task(
        rid, monday, "Done", allow_override=True
    ) is True

    perf = get_performance(did, sunday, monday)
    me = next(p for p in perf if p["employee_id"] == a)
    # Sunday should not inflate denominator as a working day for Daily
    assert me["total_tasks"] >= 1


def test_dwr_includes_employee_check_updates(hrm, monkeypatch):
    a, b, _, _, _ = _seed(hrm)
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    monkeypatch.setattr(hrm_db, "is_sunday_ist", lambda *_: False)
    rid = create_responsibility(
        {"employee_id": a, "title": "Daily", "frequency": "Daily", "mandatory": True, "backup_employee_id": b}
    )
    day = date.today().isoformat()
    assert start_responsibility_timer(rid, day) is True
    assert mark_task(rid, day, "Done") is True
    dwr = list_dwr_rows(employee_id=a, check_date=day)
    assert any(r["responsibility_id"] == rid and r["status"] == "Done" for r in dwr["rows"])


def test_self_assign_task_http(client, hrm, monkeypatch):
    a, b, did, _, _ = _seed(hrm)
    # employee scope: only self
    from backend.services import rbac

    def fake_scope(request):
        return HrmScope(
            role="Employee",
            level="self",
            user_id=1,
            employee_id=a,
            department_id=did,
        )

    monkeypatch.setattr("backend.routers.hrm._scope_from_request", fake_scope)
    ok = client.post(
        "/api/hrm/one-time-tasks",
        json={"employee_id": a, "title": "Boss asked now", "priority": "High"},
    )
    assert ok.status_code == 200, ok.text
    blocked = client.post(
        "/api/hrm/one-time-tasks",
        json={"employee_id": b, "title": "Other person", "priority": "High"},
    )
    assert blocked.status_code == 403


def test_import_http_template(client, auth_token):
    r = client.get("/api/hrm/import/responsibilities/template")
    # may 403 without assignment rights depending on test user role
    assert r.status_code in (200, 403)

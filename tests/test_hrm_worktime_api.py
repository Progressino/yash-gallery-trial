"""HTTP-level checks for office time, leave, hold, slots, breaks, holidays and reports (RBAC)."""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from backend.db import hrm_db
from backend.services.rbac import HrmScope

DAY = "2026-08-25"


class Clock:
    def __init__(self, t: str):
        self.t = t

    def __call__(self) -> str:
        return self.t


@pytest.fixture()
def env(tmp_path, monkeypatch, client):
    db_path = str(tmp_path / "hrm_api.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    clock = Clock(f"{DAY} 09:00:00")
    monkeypatch.setattr(hrm_db, "_now_iso", clock)
    monkeypatch.setattr(hrm_db, "today_ist", lambda: date.fromisoformat(clock.t[:10]))
    monkeypatch.setattr("backend.routers.hrm.today_ist", lambda: date.fromisoformat(clock.t[:10]))

    hrm_db.create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm_db.list_departments()[0]["id"]
    for n in ("Worker A", "Backup B"):
        hrm_db.create_employee({"name": n, "department_id": did})
    ids = {e["name"]: e["id"] for e in hrm_db.list_employees(did)}
    a, b = ids["Worker A"], ids["Backup B"]

    state = {"scope": HrmScope(level="self", role="Employee", user_id=1, employee_id=a, department_id=did)}
    monkeypatch.setattr("backend.routers.hrm._scope_from_request", lambda request: state["scope"])

    def as_admin():
        state["scope"] = HrmScope(level="all", role="Admin", user_id=9)

    def as_employee(emp=a):
        state["scope"] = HrmScope(level="self", role="Employee", user_id=1, employee_id=emp, department_id=did)

    return {"client": client, "a": a, "b": b, "did": did, "clock": clock,
            "as_admin": as_admin, "as_employee": as_employee}


def test_office_start_close_blocked_then_ok(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = hrm_db.create_responsibility({"employee_id": a, "title": "Daily MIS", "frequency": "Daily"})
    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    assert c.post("/api/hrm/office/start", json={}).status_code == 400
    clock.t = f"{DAY} 18:00:00"
    r = c.post("/api/hrm/office/close", json={})
    assert r.status_code == 409
    assert r.json()["pending"][0]["title"] == "Daily MIS"
    hrm_db.mark_task(rid, DAY, "Done", "Worker A", allow_override=True)
    r = c.post("/api/hrm/office/close", json={})
    assert r.status_code == 200, r.text
    assert r.json()["office_seconds"] == 9 * 3600
    assert c.post("/api/hrm/office/start", json={"employee_id": env["b"]}).status_code == 403


def test_no_timer_start_or_resume_after_office_close(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    running = hrm_db.create_one_time_task({"employee_id": a, "title": "Running"})
    fresh = hrm_db.create_one_time_task({"employee_id": a, "title": "Fresh"})
    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    clock.t = f"{DAY} 10:00:00"
    assert c.post(f"/api/hrm/one-time-tasks/{running}/start").status_code == 200
    clock.t = f"{DAY} 18:00:00"
    r = c.post("/api/hrm/office/close", json={})
    assert r.status_code == 200 and r.json()["auto_paused_tasks"] == 1
    rid = hrm_db.create_responsibility({"employee_id": a, "title": "Late item", "frequency": "Daily"})
    clock.t = f"{DAY} 18:30:00"
    assert c.post(f"/api/hrm/one-time-tasks/{running}/resume").status_code == 409
    assert c.post(f"/api/hrm/one-time-tasks/{fresh}/start").status_code == 409
    assert c.post(f"/api/hrm/tasks/{rid}/start", json={"log_date": DAY}).status_code == 409
    task = next(t for t in hrm_db.list_one_time_tasks(employee_id=a) if t["id"] == running)
    assert task["total_work_seconds"] == 8 * 3600
    env["as_admin"]()
    assert c.post(f"/api/hrm/one-time-tasks/{running}/resume").status_code == 200


def test_break_confirmation_over_http(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = hrm_db.create_responsibility({"employee_id": a, "title": "Lunch overlap", "frequency": "Daily"})
    clock.t = f"{DAY} 12:50:00"
    assert c.post(f"/api/hrm/tasks/{rid}/start", json={"log_date": DAY}).status_code == 200
    clock.t = f"{DAY} 13:40:00"
    r = c.post(f"/api/hrm/tasks/{rid}/end", json={"log_date": DAY})
    assert r.status_code == 200 and r.json()["needs_break_confirmation"] is True
    assert r.json()["breaks"][0]["overlap_minutes"] == 30
    assert c.post(f"/api/hrm/tasks/{rid}/end", json={"log_date": DAY, "break_decision": "nope"}).status_code == 400
    r = c.post(f"/api/hrm/tasks/{rid}/end", json={"log_date": DAY, "break_decision": "deduct"})
    assert r.json() == {"ok": True}
    assert hrm_db.get_responsibility_timer_detail(rid, DAY)["total_work_seconds"] == 20 * 60


def test_slot_edit_rbac(env):
    c, a, b, clock = env["client"], env["a"], env["b"], env["clock"]
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Mine"})
    other = hrm_db.create_one_time_task({"employee_id": b, "title": "Theirs"})
    for t, emp in ((tid, a), (other, b)):
        env["as_employee"](emp)
        clock.t = f"{DAY} 10:00:00"
        assert c.post(f"/api/hrm/one-time-tasks/{t}/start").status_code == 200
        clock.t = f"{DAY} 10:30:00"
        assert c.post(f"/api/hrm/one-time-tasks/{t}/pause").status_code == 200
    env["as_employee"](a)
    mine = hrm_db.list_one_time_tasks(employee_id=a)[0]["time_slots"][0]["id"]
    theirs = hrm_db.list_one_time_tasks(employee_id=b)[0]["time_slots"][0]["id"]
    r = c.patch(f"/api/hrm/time-slots/{mine}", json={"started_at": "09:50", "notes": "prep"})
    assert r.status_code == 200 and r.json()["slot"]["manual_edited"] == 1
    assert c.patch(f"/api/hrm/time-slots/{theirs}", json={"notes": "x"}).status_code == 403
    assert c.get(f"/api/hrm/time-slots/{mine}/audit").json()["audit"]
    assert c.post("/api/hrm/time-slots", json={"entity_type": "one_time", "task_id": tid,
                                                "started_at": "11:00", "ended_at": "11:15"}).status_code == 400
    clock.t = f"{DAY} 12:00:00"
    r = c.post("/api/hrm/time-slots", json={"entity_type": "one_time", "task_id": tid,
                                             "started_at": "11:00", "ended_at": "11:15", "notes": "call"})
    assert r.status_code == 200, r.text
    assert hrm_db.list_one_time_tasks(employee_id=a)[0]["total_work_seconds"] == (40 + 15) * 60


def test_hold_requires_hod_or_admin(env):
    c, a = env["client"], env["a"]
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Hold"})
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/hold", json={"resume_date": "2026-09-01"}).status_code == 403
    env["as_admin"]()
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/hold", json={"resume_date": DAY}).status_code == 400
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/hold", json={"resume_date": "2026-09-01"}).status_code == 200
    assert not c.get("/api/hrm/one-time-tasks").json()
    assert c.get("/api/hrm/one-time-tasks", params={"status": "On Hold"}).json()[0]["id"] == tid
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/unhold").status_code == 200
    assert c.get("/api/hrm/one-time-tasks").json()[0]["status"] == "Pending"


def test_leave_self_only_and_holidays_admin(env):
    c, a, b = env["client"], env["a"], env["b"]
    r = c.post("/api/hrm/leaves", json={"from_date": "2026-08-29", "to_date": "2026-08-31"})
    assert r.status_code == 200 and r.json()["sundays_included"] == 1
    assert c.post("/api/hrm/leaves", json={"employee_id": b, "from_date": "2026-09-02", "to_date": "2026-09-02"}).status_code == 403
    leaves = c.get("/api/hrm/leaves").json()
    assert [lv["employee_id"] for lv in leaves] == [a]
    assert c.post(f"/api/hrm/leaves/{leaves[0]['id']}/cancel").status_code == 200

    assert c.post("/api/hrm/holidays", json={"holiday_date": "2026-10-02", "name": "Gandhi Jayanti"}).status_code == 403
    env["as_admin"]()
    assert c.post("/api/hrm/holidays", json={"holiday_date": "2026-10-02", "name": "Gandhi Jayanti"}).status_code == 200
    assert c.get("/api/hrm/holidays").json()[0]["holiday_date"] == "2026-10-02"
    assert c.delete("/api/hrm/holidays/2026-10-02").status_code == 200


def test_reports_scope_and_range(env):
    c, a, b, clock = env["client"], env["a"], env["b"], env["clock"]
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Report me"})
    clock.t = f"{DAY} 10:00:00"
    c.post("/api/hrm/office/start", json={})
    c.post(f"/api/hrm/one-time-tasks/{tid}/start")
    clock.t = f"{DAY} 12:00:00"
    c.post(f"/api/hrm/one-time-tasks/{tid}/pause")

    r = c.get("/api/hrm/dwr", params={"from_date": DAY, "to_date": DAY})
    assert r.status_code == 200
    assert r.json()["rows"][0]["title"] == "Report me"
    assert c.get("/api/hrm/dwr", params={"employee_id": b}).status_code == 403
    assert c.get("/api/hrm/dwr", params={"from_date": DAY, "to_date": "2026-08-01"}).status_code == 400
    wh = c.get("/api/hrm/reports/working-hours", params={"from_date": DAY, "to_date": DAY}).json()
    assert wh["totals"]["working_seconds"] == 2 * 3600
    assert wh["totals"]["office_seconds"] == 2 * 3600

    meta = c.get("/api/hrm/meta").json()
    assert "Last Working Day" in meta["schedule_rule_examples"]

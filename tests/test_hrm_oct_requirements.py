"""Updated requirements: office-first timers, Missed/Blocked without time, HOD status
correction audit, date-specific durations, simplified import template and scheduling."""
from __future__ import annotations

import io
import uuid
from datetime import date

import pytest

from backend.db import hrm_db, hrm_worktime
from backend.services.rbac import HrmScope

MON, TUE, WED = "2026-10-05", "2026-10-06", "2026-10-07"
OFFICE_MSG = "Please start Office Time before starting a Responsibility or Task."


class Clock:
    def __init__(self, t: str):
        self.t = t

    def __call__(self) -> str:
        return self.t


@pytest.fixture()
def env(tmp_path, monkeypatch, client):
    db_path = str(tmp_path / "hrm_oct.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    clock = Clock(f"{MON} 09:00:00")
    monkeypatch.setattr(hrm_db, "_now_iso", clock)
    monkeypatch.setattr(hrm_db, "today_ist", lambda: date.fromisoformat(clock.t[:10]))
    monkeypatch.setattr("backend.routers.hrm.today_ist", lambda: date.fromisoformat(clock.t[:10]))

    hrm_db.create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    hrm_db.create_department({"name": f"X-{uuid.uuid4().hex[:6]}"})
    did, other_did = [d["id"] for d in hrm_db.list_departments()][:2]
    for n in ("Worker A", "Hod H"):
        hrm_db.create_employee({"name": n, "department_id": did})
    hrm_db.create_employee({"name": "Outsider O", "department_id": other_did})
    ids = {e["name"]: e["id"] for e in hrm_db.list_employees()}
    a, h, o = ids["Worker A"], ids["Hod H"], ids["Outsider O"]

    state = {"scope": None}
    monkeypatch.setattr("backend.routers.hrm._scope_from_request", lambda request: state["scope"])

    def as_employee(emp=a):
        state["scope"] = HrmScope(level="self", role="Employee", user_id=1, employee_id=emp, department_id=did)

    def as_hod(dept=did, emp=h):
        state["scope"] = HrmScope(level="department", role="HOD", user_id=5, employee_id=emp, department_id=dept)

    def as_admin():
        state["scope"] = HrmScope(level="all", role="Admin", user_id=9)

    as_employee()
    return {"client": client, "a": a, "h": h, "o": o, "did": did, "other_did": other_did,
            "clock": clock, "as_employee": as_employee, "as_hod": as_hod, "as_admin": as_admin}


def _resp(emp, **kw):
    data = {"employee_id": emp, "title": f"Task-{uuid.uuid4().hex[:4]}", "frequency": "Daily"}
    data.update(kw)
    return hrm_db.create_responsibility(data)


# ── Scenario B: Office Time is required before starting work ────────────────


def test_office_time_required_before_start_and_resume(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = _resp(a)
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Quote"})

    r = c.post(f"/api/hrm/tasks/{rid}/start", json={"log_date": MON})
    assert r.status_code == 409 and r.json()["detail"] == OFFICE_MSG
    r = c.post(f"/api/hrm/one-time-tasks/{tid}/start")
    assert r.status_code == 409 and r.json()["detail"] == OFFICE_MSG

    # HOD acting on the employee's item (correction) is not gated by the employee's office
    env["as_hod"]()
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/start").status_code == 200
    clock.t = f"{MON} 09:30:00"
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/pause").status_code == 200
    # …but the HOD's own item needs the HOD's own Office Time
    own = _resp(env["h"])
    r = c.post(f"/api/hrm/tasks/{own}/start", json={"log_date": MON})
    assert r.status_code == 409 and r.json()["detail"] == OFFICE_MSG

    env["as_employee"]()
    r = c.post(f"/api/hrm/one-time-tasks/{tid}/resume")
    assert r.status_code == 409 and r.json()["detail"] == OFFICE_MSG

    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    clock.t = f"{MON} 10:00:00"
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/resume").status_code == 200
    assert c.post(f"/api/hrm/tasks/{rid}/start", json={"log_date": MON}).status_code == 200


def test_office_required_is_per_day(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = _resp(a)
    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    clock.t = f"{TUE} 09:00:00"  # yesterday's office does not count today
    r = c.post(f"/api/hrm/tasks/{rid}/start", json={"log_date": TUE})
    assert r.status_code == 409 and r.json()["detail"] == OFFICE_MSG


# ── Scenario C: Missed / Blocked need no Start/Complete time ────────────────


def test_missed_and_blocked_without_timer(env):
    c, a, h = env["client"], env["a"], env["h"]
    missed, blocked, done = _resp(a), _resp(a), _resp(a)
    r = c.post("/api/hrm/tasks/mark", json={"responsibility_id": missed, "log_date": MON, "status": "Missed"})
    assert r.status_code == 200, r.text
    r = c.post("/api/hrm/tasks/mark", json={
        "responsibility_id": _resp(a), "log_date": MON, "status": "Blocked",
        "blocker_employee_id": env["o"], "blocker_reason": "Other department"})
    assert r.status_code == 403  # blocker must be a department peer
    r = c.post("/api/hrm/tasks/mark", json={
        "responsibility_id": blocked, "log_date": MON, "status": "Blocked",
        "blocker_employee_id": h, "blocker_reason": "Waiting for data"})
    assert r.status_code == 200, r.text
    r = c.post("/api/hrm/tasks/mark", json={"responsibility_id": missed, "log_date": MON, "status": "Done"})
    assert r.status_code == 409  # saved status is locked for the employee (not a timer message)
    r = c.post("/api/hrm/tasks/mark", json={"responsibility_id": done, "log_date": MON, "status": "Done"})
    assert r.status_code == 400 and "Start time tracking" in r.json()["detail"]
    assert hrm_db.mark_task(_resp(a), MON, "Partial") == "timer_required"


# ── Scenario D: Office Close after Missed (no time) is allowed ──────────────


def test_office_close_after_missed_without_time(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = _resp(a)
    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    clock.t = f"{MON} 18:00:00"
    assert c.post("/api/hrm/office/close", json={}).status_code == 409
    assert c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Missed"}).status_code == 200
    assert c.post("/api/hrm/office/close", json={}).status_code == 200


# ── Scenario E: HOD status correction — audit + flag across repeat edits ────


def test_hod_status_correction_audit_and_flag(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    rid = _resp(a)
    assert c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Missed"}).status_code == 200
    log_id = hrm_db.get_responsibility_timer_detail(rid, MON)["task_log_id"]

    env["as_hod"]()
    clock.t = f"{MON} 15:00:00"
    r = c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Done"})
    assert r.status_code == 200, r.text
    clock.t = f"{MON} 16:30:00"
    assert c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Partial"}).status_code == 200

    audit = c.get(f"/api/hrm/task-logs/{log_id}/status-audit").json()["audit"]
    assert [(x["old_value"], x["new_value"], x["actor_role"]) for x in audit] == [
        ("Missed", "Done", "HOD"), ("Done", "Partial", "HOD")]
    assert audit[0]["task_log_id"] == log_id and audit[0]["created_at"] == f"{MON} 15:00:00"
    assert audit[0]["actor"] and f"employee_id={a}" in audit[0]["notes"]

    rows = c.get("/api/hrm/dwr", params={"employee_id": a, "from_date": MON, "to_date": MON}).json()["rows"]
    row = next(x for x in rows if x["responsibility_id"] == rid)
    assert row["hod_edited"] is True and row["hod_edit_count"] == 2
    assert row["hod_original_status"] == "Missed" and row["status"] == "Partial"
    assert row["hod_edited_role"] == "HOD" and row["hod_edited_at"] == f"{MON} 16:30:00"

    # HOD of another department cannot correct or read this record
    env["as_hod"](dept=env["other_did"], emp=env["o"])
    assert c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Done"}).status_code == 403
    assert c.get(f"/api/hrm/task-logs/{log_id}/status-audit").status_code == 403


def test_employee_own_mark_is_not_flagged(env):
    c, a = env["client"], env["a"]
    rid = _resp(a)
    assert c.post("/api/hrm/tasks/mark", json={"responsibility_id": rid, "log_date": MON, "status": "Missed"}).status_code == 200
    rows = c.get("/api/hrm/dwr", params={"from_date": MON, "to_date": MON}).json()["rows"]
    assert next(x for x in rows if x["responsibility_id"] == rid)["hod_edited"] is False


# ── Scenario A: date-specific durations for a multi-day resumed task ────────


def test_multi_day_resumed_task_is_date_specific(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Long project"})
    plan = [(MON, "10:00", "12:00"), (TUE, "10:00", "11:30"), (WED, "09:00", "12:00")]
    for i, (day, s, e) in enumerate(plan):
        clock.t = f"{day} 08:55:00"
        assert c.post("/api/hrm/office/start", json={}).status_code == 200
        clock.t = f"{day} {s}:00"
        verb = "start" if i == 0 else "resume"
        assert c.post(f"/api/hrm/one-time-tasks/{tid}/{verb}").status_code == 200, day
        clock.t = f"{day} {e}:00"
        assert c.post(f"/api/hrm/one-time-tasks/{tid}/pause").status_code == 200, day

    dwr = c.get("/api/hrm/dwr", params={"from_date": MON, "to_date": WED}).json()
    per_day = {r["check_date"]: r["duration_seconds"] for r in dwr["rows"] if r.get("task_id") == tid}
    assert per_day == {MON: 2 * 3600, TUE: 90 * 60, WED: 3 * 3600}
    assert dwr["total_seconds"] == int(6.5 * 3600)

    wh = c.get("/api/hrm/reports/working-hours", params={"from_date": MON, "to_date": WED}).json()
    assert {r["work_date"]: r["working_seconds"] for r in wh["rows"]} == per_day

    check = hrm_db.get_employee_day_check(a, WED)
    item = next(t for t in check["one_time_all"] if t["id"] == tid)
    assert item["day_work_seconds"] == 3 * 3600
    assert item["total_work_seconds"] == int(6.5 * 3600)
    assert check["time_summary"]["working_seconds"] == 3 * 3600


def test_cross_midnight_slot_split_between_dates(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    tid = hrm_db.create_one_time_task({"employee_id": a, "title": "Night shift"})
    assert c.post("/api/hrm/office/start", json={}).status_code == 200
    clock.t = f"{MON} 23:00:00"
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/start").status_code == 200
    clock.t = f"{TUE} 01:00:00"
    assert c.post(f"/api/hrm/one-time-tasks/{tid}/pause").status_code == 200

    dwr = c.get("/api/hrm/dwr", params={"from_date": MON, "to_date": TUE}).json()
    per_day = {r["check_date"]: r["duration_seconds"] for r in dwr["rows"] if r.get("task_id") == tid}
    assert per_day == {MON: 3600, TUE: 3600}
    for d in (MON, TUE):
        assert hrm_worktime.day_time_summary(a, d)["working_seconds"] == 3600


def test_dwr_matches_working_hours_for_carried_responsibility(env):
    """Monday's responsibility worked on Tuesday counts on Tuesday, not Monday."""
    a, clock = env["a"], env["clock"]
    rid = _resp(a)
    clock.t = f"{TUE} 10:00:00"
    assert hrm_db.start_responsibility_timer(rid, MON, allow_override=True) is True
    clock.t = f"{TUE} 10:45:00"
    assert hrm_db.pause_responsibility_timer(rid, MON, allow_override=True) is True

    rep = hrm_worktime.list_dwr_report(employee_ids=[a], from_date=MON, to_date=TUE)
    by_day: dict[str, int] = {}
    for r in rep["rows"]:
        by_day[r["check_date"]] = by_day.get(r["check_date"], 0) + r["duration_seconds"]
    assert by_day.get(MON, 0) == 0
    assert by_day[TUE] == 45 * 60
    carried = next(r for r in rep["rows"] if r["check_date"] == TUE and r.get("carried_from_date") == MON)
    assert carried["responsibility_id"] == rid
    for d in (MON, TUE):
        assert by_day.get(d, 0) == hrm_worktime.day_time_summary(a, d)["working_seconds"]


# ── Scenario G: scheduling + simplified import template ─────────────────────


def _due_days(r: dict, month_days: range, year=2026, month=10) -> list[int]:
    conn = hrm_db._connect()
    try:
        ctx = hrm_db.DueContext(conn)
        return [d for d in month_days
                if hrm_db.responsibility_due_on(r, date(year, month, d).isoformat(), ctx, employee_id=r.get("employee_id"))]
    finally:
        conn.close()


def test_schedule_monthly_third_thursday_fixed_25_and_fortnightly_second_monday(env):
    a = env["a"]
    third_thu = {"frequency": "Monthly", "schedule_rule": "3rd Thursday", "employee_id": a}
    assert _due_days(third_thu, range(1, 32)) == [15]
    fixed = {"frequency": "Monthly", "schedule_month_day": 25, "employee_id": a}
    assert _due_days(fixed, range(1, 32)) == [26]  # 25 Oct 2026 is a Sunday → Monday
    fortnight = {"frequency": "Fortnightly", "schedule_rule": "2nd Monday", "employee_id": a}
    assert _due_days(fortnight, range(1, 32)) == [12, 26]
    legacy_fortnight = {"frequency": "Fortnightly", "schedule_weekday": "Monday", "employee_id": a}
    assert _due_days(legacy_fortnight, range(1, 32)) == [12, 26]
    twice = {"frequency": "Twice a Week", "schedule_weekday": "Monday,Thursday", "employee_id": a}
    assert _due_days(twice, range(1, 15)) == [1, 5, 8, 12]


def test_schedule_holiday_and_leave_chain_to_next_working_day(env):
    a = env["a"]
    hrm_db.upsert_holiday("2026-10-12", "Holiday 1")
    hrm_db.upsert_holiday("2026-10-13", "Holiday 2")
    fortnight = {"frequency": "Fortnightly", "schedule_rule": "2nd Monday", "employee_id": a}
    assert _due_days(fortnight, range(1, 32)) == [14, 26]
    hrm_db.create_leave(a, "2026-10-14", "2026-10-15")
    assert _due_days(fortnight, range(1, 32)) == [16, 26]


def test_template_resolver_matrix_and_errors():
    R = hrm_db.resolve_template_schedule
    assert R("Daily", "Daily", "N/A")["frequency"] == "Daily"
    assert R("Twice in a week", "Monday/Thursday", "N/A")["schedule_weekday"] == "Monday,Thursday"
    assert R("Weekly", "All", "Monday")["schedule_weekday"] == "Monday"
    fn = R("Fortnightly", "2nd", "Monday")
    assert (fn["schedule_rule"], fn["frequency"]) == ("2nd Monday", "Fortnightly")
    assert R("Fortnightly", "1st", "Saturday")["schedule_rule"] == "1st Saturday"
    assert R("Monthly", "3rd", "Thursday")["schedule_rule"] == "3rd Thursday"
    assert R("Monthly", "Fixed Date", "25")["schedule_month_day"] == 25
    assert R("Monthly", "Last Working Day", "N/A")["schedule_rule"] == "Last Working Day"
    assert R("Quarterly", "March", "N/A")["schedule_month"] == 3
    y = R("Yearly", "April", "N/A")
    assert (y["schedule_month"], y["schedule_month_day"]) == (4, 1)
    for bad in [("Weekly", "All", "N/A"), ("Monthly", "Fixed Date", "32"), ("Monthly", "3rd", "N/A"),
                ("Fortnightly", "4th", "Monday"), ("Fortnightly", "Last", "Friday"),
                ("Twice in a week", "Monday", "N/A"), ("Quarterly", "Smarch", "N/A"),
                ("Quarterly", "March", "5"), ("Daily", "Daily", "Monday"), ("Hourly", "", "")]:
        with pytest.raises(ValueError):
            R(*bad)


def test_fortnightly_pairing_logic_preserved():
    p = hrm_db.parse_schedule_rule("1st Saturday")
    sats = [d for d in range(1, 32) if date(2026, 10, d).weekday() == 5]  # 3,10,17,24,31
    hits = [d for d in sats if hrm_db._schedule_rule_matches(p, date(2026, 10, d), holidays=set(), frequency="Fortnightly")]
    assert hits == [3, 17]  # 1st & 3rd
    p3 = hrm_db.parse_schedule_rule("3rd Thursday")
    thus = [d for d in range(1, 32) if date(2026, 10, d).weekday() == 3]  # 1,8,15,22,29
    hits = [d for d in thus if hrm_db._schedule_rule_matches(p3, date(2026, 10, d), holidays=set(), frequency="Fortnightly")]
    assert hits == [15, 29]  # existing rule: 3rd & 5th


def test_import_simplified_template_over_http(env):
    c, a, h = env["client"], env["a"], env["h"]
    env["as_admin"]()
    tpl = c.get("/api/hrm/import/responsibilities/template").text
    header = tpl.splitlines()[0]
    assert "Frequency,Schedule Value,Occurrence Day,Linked Person" in header

    csv = (
        "employee_name,title,Frequency,Schedule Value,Occurrence Day,Linked Person\n"
        "Worker A,Daily MIS,Daily,Daily,N/A,\n"
        "Worker A,Supplier calls,Twice in a week,Monday/Thursday,N/A,Hod H\n"
        "Worker A,Weekly review,Weekly,All,Monday,\n"
        "Worker A,Vendor follow-up,Fortnightly,2nd,Monday,\n"
        "Worker A,Stock recon,Monthly,3rd,Thursday,\n"
        "Worker A,Salary sheet,Monthly,Fixed Date,25,\n"
        "Worker A,GST review,Quarterly,March,N/A,\n"
        "Worker A,Audit prep,Yearly,April,N/A,\n"
        "Worker A,Bad weekly,Weekly,All,N/A,\n"
        "Worker A,Bad fortnight,Fortnightly,Last,Friday,\n"
    )
    r = c.post("/api/hrm/import/responsibilities",
               files={"file": ("resp.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 8, body
    assert len(body["errors"]) == 2
    assert body["errors"][0].startswith("Row 9:") and "Occurrence Day" in body["errors"][0]
    assert body["errors"][1].startswith("Row 10:") and "Fortnightly" in body["errors"][1]

    by_title = {r["title"]: r for r in hrm_db.list_responsibilities(employee_id=a)}
    assert by_title["Supplier calls"]["frequency"] == "Twice a Week"
    assert by_title["Supplier calls"]["schedule_weekday"] == "Monday,Thursday"
    assert by_title["Supplier calls"]["linked_to_employee_id"] == h
    assert by_title["Vendor follow-up"]["schedule_rule"] == "2nd Monday"
    assert by_title["Stock recon"]["schedule_rule"] == "3rd Thursday"
    assert by_title["Salary sheet"]["schedule_month_day"] == 25
    assert by_title["GST review"]["schedule_month"] == 3
    assert (by_title["Audit prep"]["schedule_month"], by_title["Audit prep"]["schedule_month_day"]) == (4, 1)


def test_import_legacy_columns_still_work(env):
    c = env["client"]
    env["as_admin"]()
    csv = (
        "employee_name,title,frequency,schedule_weekday,schedule_month_day,schedule_rule\n"
        "Worker A,Old weekly,Weekly,Tuesday,,\n"
        "Worker A,Old monthly,Monthly,,,Last Working Day\n"
    )
    r = c.post("/api/hrm/import/responsibilities",
               files={"file": ("old.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.json()["created"] == 2, r.text

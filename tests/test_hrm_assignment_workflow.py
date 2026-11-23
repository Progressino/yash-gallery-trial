"""Cross-department assignment workflow: HOD review -> employee acceptance, trackers, approvals."""
from __future__ import annotations

import uuid

import pytest

from backend.db import hrm_db
from backend.services import hrm_assignment_workflow as wf
from backend.services.rbac import HrmScope


@pytest.fixture()
def env(tmp_path, monkeypatch, client):
    db_path = str(tmp_path / "hrm_assign.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)

    def dept(name):
        hrm_db.create_department({"name": f"{name}-{uuid.uuid4().hex[:4]}"})
        return next(d["id"] for d in hrm_db.list_departments() if d["name"].startswith(name))

    sales, prod, empty = dept("Sales"), dept("Production"), dept("NoHod")

    def emp(name, dept_id):
        hrm_db.create_employee({"name": name, "department_id": dept_id})
        return next(e["id"] for e in hrm_db.list_employees() if e["name"] == name)

    ids = {
        "vijay": emp("Vijay Sales", sales),          # Sales HOD
        "sales_emp": emp("Sana Sales", sales),
        "prod_hod": emp("Pooja Production", prod),   # Production HOD
        "prod_emp": emp("Pranav Production", prod),
        "lone": emp("Lata Lone", empty),
        "lone2": emp("Leo Lone", empty),
    }
    hods = {sales: [ids["vijay"]], prod: [ids["prod_hod"]]}
    monkeypatch.setattr(wf, "_hod_users_for_department", lambda d: hods.get(int(d), []))

    def act_as(role, emp_key, dept_id=None):
        eid = ids[emp_key]
        level = {"HOD": "department", "Employee": "self"}.get(role, "all")
        scope = HrmScope(level=level, role=role, user_id=eid, employee_id=eid, department_id=dept_id)
        monkeypatch.setattr("backend.routers.hrm._scope_from_request", lambda request: scope)

    return {"client": client, "ids": ids, "dept": {"sales": sales, "prod": prod, "empty": empty}, "act_as": act_as}


def _assign_task(env, to, **extra):
    body = {"employee_id": env["ids"][to], "title": f"T-{uuid.uuid4().hex[:4]}", "due_date": "2026-12-31"}
    body.update(extra)
    r = env["client"].post("/api/hrm/one-time-tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _action(env, name, entity_type, entity_id, remarks=""):
    return env["client"].post(
        f"/api/hrm/assignments/{name}",
        json={"entity_type": entity_type, "entity_id": entity_id, "remarks": remarks},
    )


def _tracker(env, view, **params):
    r = env["client"].get("/api/hrm/assignments/tracker", params={"view": view, **params})
    assert r.status_code == 200, r.text
    return r.json()


def _visible_to_assignee(emp_id, task_id):
    return any(t["id"] == task_id for t in hrm_db.list_one_time_tasks(employee_id=emp_id))


def test_cross_department_create_goes_to_hod_review(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    out = _assign_task(env, "prod_emp")
    assert out["workflow_status"] == wf.PENDING_HOD_REVIEW
    assert out["assignee_hod_employee_id"] == env["ids"]["prod_hod"]
    # not actionable work for the assignee yet
    assert not _visible_to_assignee(env["ids"]["prod_emp"], out["id"])
    # HOD gets a notification
    notes = wf.list_notifications(env["ids"]["prod_hod"])
    assert notes and notes[0]["event_type"] == "hod_review_requested"


def test_assigner_is_assignee_hod_skips_to_employee_acceptance(env):
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    out = _assign_task(env, "prod_emp")
    assert out["workflow_status"] == wf.PENDING_EMPLOYEE_ACCEPTANCE
    notes = wf.list_notifications(env["ids"]["prod_emp"])
    assert notes[0]["event_type"] == "acceptance_requested"


def test_no_hod_goes_straight_to_employee_acceptance(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    out = _assign_task(env, "lone")
    assert out["workflow_status"] == wf.PENDING_EMPLOYEE_ACCEPTANCE
    assert out["assignee_hod_employee_id"] is None


def test_org_manager_assignment_is_accepted_immediately(env):
    env["act_as"]("Admin", "vijay")
    out = _assign_task(env, "prod_emp")
    assert out["workflow_status"] == wf.ACCEPTED
    assert _visible_to_assignee(env["ids"]["prod_emp"], out["id"])


def test_forward_then_accept_appears_in_assigned_to_me(env):
    ids = env["ids"]
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    tid = _assign_task(env, "prod_emp")["id"]

    # only the assignee's HOD may forward
    assert _action(env, "hod-forward", "task", tid).status_code == 403
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    inbox = _tracker(env, "hod_inbox")
    assert [r["id"] for r in inbox["rows"]] == [tid]
    assert inbox["rows"][0]["actions"]["can_forward"] is True
    assert env["client"].get("/api/hrm/approvals/pending").json()["assignment_reviews"][0]["id"] == tid
    assert _action(env, "hod-forward", "task", tid).status_code == 200

    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    to_me = _tracker(env, "assigned_to_me")
    row = next(r for r in to_me["rows"] if r["id"] == tid)
    assert row["workflow_status"] == wf.PENDING_EMPLOYEE_ACCEPTANCE
    assert row["pending_with"] == "Pranav Production"
    assert not _visible_to_assignee(ids["prod_emp"], tid)

    # only the assignee accepts
    env["act_as"]("Employee", "sales_emp", env["dept"]["sales"])
    assert _action(env, "accept", "task", tid).status_code == 403
    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    ok = _action(env, "accept", "task", tid)
    assert ok.status_code == 200
    assert ok.json()["acceptance_mode"] == wf.ACCEPTED_WITHOUT_CLARIFICATION
    assert _visible_to_assignee(ids["prod_emp"], tid)
    assert _tracker(env, "assigned_to_me")["summary"]["accepted"] >= 1

    # assigner is told it was accepted
    kinds = [n["event_type"] for n in wf.list_notifications(ids["vijay"])]
    assert "hod_forwarded" in kinds and "accepted" in kinds


def test_hod_reject_goes_back_to_assigner(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    tid = _assign_task(env, "prod_emp")["id"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    assert _action(env, "hod-reject", "task", tid).status_code == 400  # reason required
    assert _action(env, "hod-reject", "task", tid, "Team is full").status_code == 200
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    row = next(r for r in _tracker(env, "assigned_by_me")["rows"] if r["id"] == tid)
    assert row["workflow_status"] == wf.REJECTED_BY_HOD
    assert row["cancel_reason"] == "Team is full"
    assert not _visible_to_assignee(env["ids"]["prod_emp"], tid)


def test_clarification_loop_with_employee(env):
    ids = env["ids"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    tid = _assign_task(env, "prod_emp")["id"]

    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    assert _action(env, "clarify", "task", tid).status_code == 400  # question required
    assert _action(env, "clarify", "task", tid, "What format?").status_code == 200
    row = next(r for r in _tracker(env, "assigned_to_me")["rows"] if r["id"] == tid)
    assert row["workflow_status"] == wf.CLARIFICATION_REQUIRED
    assert row["pending_with_role"] == "Assigner"

    env["act_as"]("Employee", "sales_emp", env["dept"]["sales"])
    assert _action(env, "clarify-reply", "task", tid, "x").status_code == 403
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    assert _action(env, "clarify-reply", "task", tid, "PDF please").status_code == 200

    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    ok = _action(env, "accept", "task", tid)
    assert ok.json()["acceptance_mode"] == wf.ACCEPTED_WITH_CLARIFICATION
    events = [e["event_type"] for e in env["client"].get(
        "/api/hrm/assignments/history", params={"entity_type": "task", "entity_id": tid}).json()["events"]]
    assert events == ["created", "clarification_requested", "clarification_replied", "accepted"]
    assert _visible_to_assignee(ids["prod_emp"], tid)


def test_hod_clarification_returns_to_hod_review(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    tid = _assign_task(env, "prod_emp")["id"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    out = _action(env, "clarify", "task", tid, "Why my team?")
    assert out.status_code == 200 and out.json()["clarification_from"] == "hod"
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    assert _action(env, "clarify-reply", "task", tid, "Production owns it").status_code == 200
    row = next(r for r in _tracker(env, "assigned_by_me")["rows"] if r["id"] == tid)
    assert row["workflow_status"] == wf.PENDING_HOD_REVIEW


def test_assigned_by_me_sees_cross_dept_item_only(env):
    ids = env["ids"]
    # another person's work for the same employee, plus legacy work
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    other = _assign_task(env, "prod_emp")["id"]
    hrm_db.create_one_time_task({"employee_id": ids["prod_emp"], "title": "Legacy"})

    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    mine = _assign_task(env, "prod_emp")["id"]
    resp = env["client"].post(
        "/api/hrm/responsibilities",
        json={"employee_id": ids["prod_emp"], "title": "Weekly MIS", "frequency": "Daily"},
    )
    assert resp.status_code == 200, resp.text
    rid = resp.json()["id"]

    tasks = env["client"].get("/api/hrm/one-time-tasks", params={"assigned_by_me": 1}).json()
    assert [t["id"] for t in tasks] == [mine]
    assert tasks[0]["workflow_status"] == wf.PENDING_HOD_REVIEW
    resps = env["client"].get("/api/hrm/responsibilities", params={"assigned_by_me": 1}).json()
    assert [r["id"] for r in resps] == [rid]
    assert other not in [t["id"] for t in tasks]

    tracker = _tracker(env, "assigned_by_me")
    assert {(r["entity_type"], r["id"]) for r in tracker["rows"]} == {("task", mine), ("responsibility", rid)}
    assert tracker["summary"]["pending_hod_review"] == 2
    # the normal list stays department-scoped: Vijay (Sales HOD) cannot see Production's records
    assert env["client"].get("/api/hrm/one-time-tasks", params={"employee_id": ids["prod_emp"]}).status_code == 403


def test_attachment_links_and_remarks_stored_and_validated(env):
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    bad = env["client"].post(
        "/api/hrm/one-time-tasks",
        json={"employee_id": env["ids"]["prod_emp"], "title": "x", "attachment_links": ["not a url"]},
    )
    assert bad.status_code == 400
    out = _assign_task(
        env, "prod_emp",
        attachment_links=["https://example.com/brief.pdf"],
        assign_remarks="Urgent for audit",
        voice_note_url="https://example.com/v.mp3",
    )
    row = next(r for r in _tracker(env, "assigned_by_me")["rows"] if r["id"] == out["id"])
    assert row["attachment_links"] == ["https://example.com/brief.pdf"]
    assert row["assign_remarks"] == "Urgent for audit"
    assert row["voice_note_url"] == "https://example.com/v.mp3"


def test_employee_role_can_assign_to_others_but_not_self(env):
    env["act_as"]("Employee", "sales_emp", env["dept"]["sales"])
    ok = env["client"].post(
        "/api/hrm/responsibilities",
        json={"employee_id": env["ids"]["prod_emp"], "title": "Daily report", "frequency": "Daily"},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["workflow_status"] == wf.PENDING_HOD_REVIEW
    task = env["client"].post(
        "/api/hrm/one-time-tasks", json={"employee_id": env["ids"]["lone"], "title": "Send samples"}
    )
    assert task.status_code == 200
    assert task.json()["workflow_status"] == wf.PENDING_EMPLOYEE_ACCEPTANCE
    mine = env["client"].post(
        "/api/hrm/responsibilities",
        json={"employee_id": env["ids"]["sales_emp"], "title": "Self", "frequency": "Daily"},
    )
    assert mine.status_code == 403
    scope = env["client"].get("/api/hrm/scope").json()
    assert scope["can_assign_work"] is True


def test_completion_approval_goes_to_assigner_when_linked_blank(env):
    ids = env["ids"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    tid = _assign_task(env, "prod_emp")["id"]
    rid = env["client"].post(
        "/api/hrm/responsibilities",
        json={"employee_id": ids["prod_emp"], "title": "Daily stock", "frequency": "Daily"},
    ).json()["id"]
    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    _action(env, "accept", "task", tid)
    _action(env, "accept", "responsibility", rid)

    # task: assignee completes -> assigner's Approvals
    assert hrm_db.start_one_time_task(tid) is True
    assert hrm_db.complete_one_time_task(tid, "done") is True
    # responsibility: N/A mark waits for approval
    day = hrm_db.today_ist().isoformat()
    assert hrm_db.mark_task(rid, day, "N/A", "Pranav", allow_override=True) is True

    bucket = hrm_db.list_linked_approval_bucket(ids["prod_hod"])
    kinds = {(p["kind"], p.get("title")) for p in bucket}
    assert any(k == "one_time" for k, _ in kinds)
    assert ("responsibility", "Daily stock") in kinds
    # not the employee's own bucket, nor an unrelated person's
    assert not hrm_db.list_linked_approval_bucket(ids["prod_emp"])
    assert not hrm_db.list_linked_approval_bucket(ids["vijay"])
    assert hrm_db.list_approval_notifications(ids["prod_hod"])

    # the assigner (an Employee-role login here) can approve the task
    env["act_as"]("Employee", "prod_hod", env["dept"]["prod"])
    ok = env["client"].post(f"/api/hrm/one-time-tasks/{tid}/approve", json={"achievement_pct": 90})
    assert ok.status_code == 200, ok.text
    # but an unrelated employee cannot
    env["act_as"]("Employee", "sales_emp", env["dept"]["sales"])
    assert env["client"].post(
        f"/api/hrm/one-time-tasks/{tid}/approve", json={"achievement_pct": 90}).status_code == 403


def test_linked_person_still_wins_over_assigner(env):
    ids = env["ids"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    tid = _assign_task(env, "prod_emp", linked_to_employee_id=ids["sales_emp"])["id"]
    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    _action(env, "accept", "task", tid)
    hrm_db.start_one_time_task(tid)
    hrm_db.complete_one_time_task(tid, "done")
    assert any(p.get("task_id") == tid for p in hrm_db.list_linked_approval_bucket(ids["sales_emp"]))
    assert not any(p.get("task_id") == tid for p in hrm_db.list_linked_approval_bucket(ids["prod_hod"]))


def test_responsibility_accountable_required_without_assigner_identity(env, monkeypatch):
    scope = HrmScope(level="all", role="Admin", user_id=1, employee_id=None)
    monkeypatch.setattr("backend.routers.hrm._scope_from_request", lambda request: scope)
    r = env["client"].post(
        "/api/hrm/responsibilities",
        json={"employee_id": env["ids"]["prod_emp"], "title": "No owner", "frequency": "Daily"},
    )
    assert r.status_code == 400 and "Accountable" in r.json()["detail"]


def test_reminder_and_cancel(env, monkeypatch):
    class Clock:
        t = "2026-11-10 09:00:00"

        def __call__(self):
            return self.t

    clock = Clock()
    monkeypatch.setattr(hrm_db, "_now_iso", clock)
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    tid = _assign_task(env, "prod_emp")["id"]
    assert _action(env, "remind", "task", tid).status_code == 200
    assert _action(env, "remind", "task", tid).status_code == 429
    clock.t = "2026-11-10 10:30:00"
    assert _action(env, "remind", "task", tid).status_code == 200
    assert [n["event_type"] for n in wf.list_notifications(env["ids"]["prod_hod"])].count("reminder") == 2

    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    assert _action(env, "remind", "task", tid).status_code == 403
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    assert _action(env, "cancel", "task", tid, "").status_code == 400
    assert _action(env, "cancel", "task", tid, "No longer needed").status_code == 200
    assert _action(env, "remind", "task", tid).status_code == 409
    events = [e["event_type"] for e in env["client"].get(
        "/api/hrm/assignments/history", params={"entity_type": "task", "entity_id": tid}).json()["events"]]
    assert events[-1] == "cancelled" and events.count("reminder") == 2


def test_history_is_private_to_involved_people(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    tid = _assign_task(env, "prod_emp")["id"]
    env["act_as"]("Employee", "lone2", env["dept"]["empty"])
    r = env["client"].get("/api/hrm/assignments/history", params={"entity_type": "task", "entity_id": tid})
    assert r.status_code == 403


def test_legacy_rows_count_as_accepted_work(env):
    ids = env["ids"]
    tid = hrm_db.create_one_time_task({"employee_id": ids["prod_emp"], "title": "Old task"})
    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    rows = _tracker(env, "assigned_to_me")["rows"]
    row = next(r for r in rows if r["id"] == tid)
    assert row["workflow_status"] == wf.ACCEPTED
    assert _visible_to_assignee(ids["prod_emp"], tid)


def test_notifications_and_summary_endpoints(env):
    env["act_as"]("HOD", "vijay", env["dept"]["sales"])
    _assign_task(env, "prod_emp")
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    summary = env["client"].get("/api/hrm/assignments/summary").json()
    assert summary["hod_inbox"]["pending_hod_review"] == 1
    assert summary["hod_inbox"]["needs_my_action"] == 1
    assert summary["unread_notifications"] == 1
    data = env["client"].get("/api/hrm/assignments/notifications").json()
    assert data["unread"] == 1
    assert env["client"].post("/api/hrm/assignments/notifications/read").json()["marked"] == 1
    assert env["client"].get("/api/hrm/assignments/notifications").json()["unread"] == 0


def test_assigned_by_selected_person_not_login(env):
    """Create with Assigned By = another employee must land in that person's Assigned By Me."""
    ids = env["ids"]
    # sales_emp (Sana Sales) is the selected Assigned By; login is prod_hod
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    r = env["client"].post(
        "/api/hrm/one-time-tasks",
        json={
            "employee_id": ids["lone"],
            "title": "For Sana tracking",
            "assigned_by": "Sana Sales",
            "assigned_by_employee_id": ids["sales_emp"],
        },
    )
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    env["act_as"]("Employee", "sales_emp", env["dept"]["sales"])
    by_me = _tracker(env, "assigned_by_me")["rows"]
    assert any(row["id"] == tid for row in by_me)
    hist = env["client"].get(
        "/api/hrm/assignments/history", params={"entity_type": "task", "entity_id": tid}
    ).json()
    events = hist.get("events") if isinstance(hist, dict) else hist
    assert isinstance(events, list)
    assert any(e.get("event_type") == "created_by_login" for e in events), hist
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    creator_rows = [row for row in _tracker(env, "assigned_by_me")["rows"] if row["id"] == tid]
    assert not creator_rows


def test_reject_one_time_without_achieved_pct_requires_reason(env):
    ids = env["ids"]
    env["act_as"]("HOD", "prod_hod", env["dept"]["prod"])
    tid = _assign_task(env, "prod_emp")["id"]
    env["act_as"]("Employee", "prod_emp", env["dept"]["prod"])
    _action(env, "accept", "task", tid)
    hrm_db.start_one_time_task(tid)
    hrm_db.complete_one_time_task(tid, "done")
    env["act_as"]("Employee", "prod_hod", env["dept"]["prod"])
    bad = env["client"].post(f"/api/hrm/one-time-tasks/{tid}/reject", json={})
    assert bad.status_code == 400
    ok = env["client"].post(
        f"/api/hrm/one-time-tasks/{tid}/reject", json={"notes": "Incomplete evidence"}
    )
    assert ok.status_code == 200, ok.text
    notes = hrm_db.list_approval_notifications(ids["prod_emp"])
    assert any("Incomplete evidence" in str(n.get("message") or "") for n in notes)

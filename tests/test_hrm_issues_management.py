"""HRM Issues Management — lifecycle, audit, permissions, search, voice log."""
from __future__ import annotations

import uuid

import pytest

from backend.db import hrm_db, users_db
from backend.db.hrm_db import init_db as init_hrm_db
from backend.db.users_db import init_db as init_users_db, create_user, list_roles
from backend.services.rbac import build_hrm_scope


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    users_path = str(tmp_path / "users.db")
    hrm_path = str(tmp_path / "hrm.db")
    monkeypatch.setenv("USERS_DB_PATH", users_path)
    monkeypatch.setenv("HRM_DB_PATH", hrm_path)
    monkeypatch.setattr(users_db, "_DB", users_path)
    monkeypatch.setattr(hrm_db, "_DB", hrm_path)
    init_users_db()
    init_hrm_db()
    yield


def _seed_emp(name="Worker"):
    hrm_db.create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    dept_id = hrm_db.list_departments()[-1]["id"]
    hrm_db.create_employee({"name": name, "department_id": dept_id, "email": f"{name}@x.com", "phone": "9999999999"})
    emp = hrm_db.list_employees(dept_id)[0]
    return dept_id, emp["id"], emp


def _make_client(monkeypatch, username: str, profile: dict):
    def _decode(token: str | None):
        if token == "tok":
            return {
                "sub": username,
                "role": profile.get("role_name", "Admin"),
                "full_name": profile.get("full_name") or username,
            }
        return None

    def _profile_fn(name: str):
        return profile if name == username else None

    monkeypatch.setattr("backend.main.decode_token", _decode)
    monkeypatch.setattr("backend.routers.auth.decode_token", _decode)
    monkeypatch.setattr("backend.routers.hrm.get_user_auth_profile", _profile_fn)
    monkeypatch.setattr("backend.db.users_db.get_user_auth_profile", _profile_fn)

    from starlette.testclient import TestClient
    from backend.main import app

    c = TestClient(app)
    c.cookies.set("auth_token", "tok")
    return c


def _admin_profile(**extra):
    base = {
        "id": 10,
        "role_name": "Admin",
        "full_name": "HR Admin",
        "username": "admin1",
        "employee_id": None,
        "hrm_department_id": None,
        "module_access": None,
    }
    base.update(extra)
    return base


# 1 manual create
def test_create_issue_manual_and_recorded_by(monkeypatch):
    dept_id, emp_id, _ = _seed_emp("John")
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    r = client.post(
        "/api/hrm/issues",
        json={"employee_id": emp_id, "title": "Late Attendance", "description": "15 min late"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recorded_by"] == "HR Admin"
    assert body["recorded_by_user_id"] == 10
    rows = hrm_db.list_issues(employee_id=emp_id)
    assert len(rows) == 1
    assert rows[0]["title"] == "Late Attendance"
    assert rows[0]["recorded_by"] == "HR Admin"
    assert rows[0]["status"] == "Open"
    hist = hrm_db.list_issue_history(rows[0]["id"])
    assert any(h["action"] == "Issue Created" for h in hist)


def test_recorded_by_cannot_be_forged_on_create(monkeypatch):
    _, emp_id, _ = _seed_emp("Jane")
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    r = client.post(
        "/api/hrm/issues",
        json={
            "employee_id": emp_id,
            "title": "Policy",
            "recorded_by": "Fake Person",
            "recorded_by_user_id": 999,
        },
    )
    assert r.status_code == 200
    row = hrm_db.list_issues(employee_id=emp_id)[0]
    assert row["recorded_by"] == "HR Admin"
    assert row["recorded_by_user_id"] == 10


def test_recorded_by_hidden_when_same_as_employee(monkeypatch):
    _, emp_id, emp = _seed_emp("John")
    hrm_db.create_issue(
        {
            "employee_id": emp_id,
            "title": "Late",
            "recorded_by": "John",
            "subject_user_name": "John",
        }
    )
    rows = hrm_db.list_issues(employee_id=emp_id)
    assert rows[0]["show_recorded_by"] is False


def test_recorded_by_shown_when_different(monkeypatch):
    _, emp_id, _ = _seed_emp("John")
    hrm_db.create_issue(
        {
            "employee_id": emp_id,
            "title": "Late",
            "recorded_by": "HR Admin",
            "subject_user_name": "John",
        }
    )
    rows = hrm_db.list_issues(employee_id=emp_id)
    assert rows[0]["show_recorded_by"] is True


# Status lifecycle
def test_status_lifecycle_open_hold_resolve(monkeypatch):
    _, emp_id, _ = _seed_emp()
    iid = hrm_db.create_issue({"employee_id": emp_id, "title": "QA miss", "recorded_by": "H"})
    hrm_db.update_issue_status(iid, "Hold", user_name="mgr")
    hrm_db.update_issue_status(iid, "Resolve", resolution="Fixed", user_name="mgr")
    row = hrm_db.get_issue_raw(iid)
    assert row["status"] == "Resolve"
    hist = hrm_db.list_issue_history(iid)
    actions = [h["action"] for h in hist]
    assert actions.count("Status Changed") >= 2


def test_cancel_issue(monkeypatch):
    _, emp_id, _ = _seed_emp()
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    iid = hrm_db.create_issue({"employee_id": emp_id, "title": "Withdraw", "recorded_by": "H"})
    r = client.patch(f"/api/hrm/issues/{iid}/status", json={"status": "Cancel"})
    assert r.status_code == 200
    assert hrm_db.get_issue_raw(iid)["status"] == "Cancel"


def test_edit_issue_and_audit(monkeypatch):
    dept_id, emp_id, _ = _seed_emp("A")
    hrm_db.create_employee({"name": "B", "department_id": dept_id})
    emp_b = [e for e in hrm_db.list_employees(dept_id) if e["name"] == "B"][0]["id"]
    iid = hrm_db.create_issue(
        {"employee_id": emp_id, "title": "Old", "description": "d", "recorded_by": "HR"}
    )
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    r = client.patch(
        f"/api/hrm/issues/{iid}",
        json={"title": "New Title", "employee_id": emp_b, "description": "updated"},
    )
    assert r.status_code == 200, r.text
    raw = hrm_db.get_issue_raw(iid)
    assert raw["title"] == "New Title"
    assert raw["employee_id"] == emp_b
    # recorded_by immutable via update
    r2 = client.patch(f"/api/hrm/issues/{iid}", json={"recorded_by": "Hacker"})
    assert r2.status_code == 200
    assert hrm_db.get_issue_raw(iid)["recorded_by"] == "HR"
    hist = hrm_db.list_issue_history(iid)
    assert any(h["action"] == "Employee Changed" for h in hist)
    assert any(h["action"] == "Issue Updated" or h["field_name"] == "title" for h in hist)


def test_employee_cannot_resolve(monkeypatch):
    dept_id, emp_id, _ = _seed_emp()
    iid = hrm_db.create_issue({"employee_id": emp_id, "title": "X", "recorded_by": "H"})
    client = _make_client(
        monkeypatch,
        "emp1",
        {
            "id": 2,
            "role_name": "Employee",
            "full_name": "Emp",
            "employee_id": emp_id,
            "hrm_department_id": dept_id,
        },
    )
    assert client.patch(f"/api/hrm/issues/{iid}/resolve", json={"resolution": "ok"}).status_code == 403
    assert client.patch(f"/api/hrm/issues/{iid}/status", json={"status": "Hold"}).status_code == 403


def test_hod_can_resolve(monkeypatch):
    dept_id, emp_id, _ = _seed_emp()
    iid = hrm_db.create_issue({"employee_id": emp_id, "title": "X", "recorded_by": "H"})
    client = _make_client(
        monkeypatch,
        "hod1",
        {
            "id": 3,
            "role_name": "HOD",
            "full_name": "Hod",
            "hrm_department_id": dept_id,
        },
    )
    r = client.patch(f"/api/hrm/issues/{iid}/resolve", json={"resolution": "Done"})
    assert r.status_code == 200
    assert hrm_db.get_issue_raw(iid)["status"] == "Resolve"


def test_search_and_status_filter(monkeypatch):
    _, emp_id, _ = _seed_emp("Searchable")
    hrm_db.create_issue(
        {"employee_id": emp_id, "title": "Alpha Problem", "description": "misc", "recorded_by": "HR"}
    )
    hrm_db.create_issue(
        {"employee_id": emp_id, "title": "Beta", "status": "Hold", "recorded_by": "HR"}
    )
    # hold needs update — create always Open then status
    holds = hrm_db.list_issues(employee_id=emp_id, q="Beta")
    hrm_db.update_issue_status(holds[0]["id"], "Hold")
    all_s = hrm_db.list_issues(employee_id=emp_id, q="Alpha")
    assert len(all_s) == 1
    assert all_s[0]["title"] == "Alpha Problem"
    held = hrm_db.list_issues(employee_id=emp_id, status="Hold")
    assert len(held) == 1


def test_active_users_endpoint(monkeypatch):
    roles = list_roles()
    admin_rid = next(r["id"] for r in roles if r["role_name"] == "Admin")
    create_user(
        {
            "username": "act1",
            "password": "x",
            "role_id": admin_rid,
            "full_name": "Active One",
            "email": "a@test.com",
            "phone": "9111111111",
        }
    )
    # deactivate second user
    create_user(
        {
            "username": "inact1",
            "password": "x",
            "role_id": admin_rid,
            "full_name": "Inactive One",
        }
    )
    u = next(u for u in users_db.list_users(active_only=False) if u["username"] == "inact1")
    users_db.update_user(u["id"], {"active": 0})

    client = _make_client(monkeypatch, "admin1", _admin_profile())
    rows = client.get("/api/hrm/issues/users").json()
    names = [r.get("full_name") or r.get("username") for r in rows]
    assert "Active One" in names
    assert "Inactive One" not in names
    found = client.get("/api/hrm/issues/users", params={"q": "Active"}).json()
    assert any("Active" in (r.get("display_name") or "") for r in found)


def test_comments_attachments_voice_audit(monkeypatch):
    _, emp_id, _ = _seed_emp()
    iid = hrm_db.create_issue({"employee_id": emp_id, "title": "T", "recorded_by": "HR"})
    hrm_db.add_issue_comment(iid, "Looking into it", user_name="HR")
    hrm_db.add_issue_attachment(iid, {"file_name": "note.pdf"}, user_name="HR")
    hrm_db.log_voice_transcription(
        {"issue_id": iid, "transcript": "late again", "target_field": "description", "status": "success"}
    )
    hrm_db.log_voice_transcription(
        {"transcript": "", "status": "failed", "error_message": "mic denied"}
    )
    hist = hrm_db.list_issue_history(iid)
    actions = {h["action"] for h in hist}
    assert "Comments Added" in actions
    assert "Attachments Added" in actions
    assert "Voice Transcription Created" in actions


def test_notifications_on_create_and_status():
    _, emp_id, _ = _seed_emp()
    iid = hrm_db.create_issue(
        {"employee_id": emp_id, "title": "Notify", "recorded_by": "HR", "subject_user_id": 5}
    )
    hrm_db.update_issue_status(iid, "Hold")
    notes = hrm_db.list_issue_notifications(limit=20)
    events = {n["event_type"] for n in notes}
    assert "created" in events
    assert "status_hold" in events


def test_permission_flags():
    assert build_hrm_scope({"role_name": "Admin", "id": 1}).can_edit_issues is True
    assert build_hrm_scope({"role_name": "HOD", "id": 1, "hrm_department_id": 2}).can_change_issue_status is True
    assert build_hrm_scope({"role_name": "Employee", "id": 1, "employee_id": 3}).can_edit_issues is False
    assert build_hrm_scope({"role_name": "Employee", "id": 1, "employee_id": 3}).can_create_issues is True


def test_voice_log_api_does_not_block(monkeypatch):
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    r = client.post(
        "/api/hrm/issues/voice-log",
        json={"transcript": "", "status": "failed", "error_message": "network"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_same_user_employee_and_caused_by(monkeypatch):
    dept_id, emp_id, _ = _seed_emp("Self")
    roles = list_roles()
    rid = next(r["id"] for r in roles if r["role_name"] == "Employee")
    create_user(
        {
            "username": "selfu",
            "password": "x",
            "role_id": rid,
            "full_name": "Self User",
            "employee_id": emp_id,
        }
    )
    uid = next(u for u in users_db.list_users() if u["username"] == "selfu")["id"]
    client = _make_client(monkeypatch, "admin1", _admin_profile())
    r = client.post(
        "/api/hrm/issues",
        json={
            "subject_user_id": uid,
            "caused_by_user_id": uid,
            "title": "Self-caused case",
        },
    )
    assert r.status_code == 200, r.text
    row = hrm_db.list_issues(employee_id=emp_id)[0]
    assert row["subject_user_id"] == uid
    assert row["caused_by_user_id"] == uid


def test_list_column_order_fields_present(monkeypatch):
    _, emp_id, _ = _seed_emp("John")
    hrm_db.create_issue(
        {
            "employee_id": emp_id,
            "title": "Late Attendance",
            "recorded_by": "Manager X",
            "subject_user_name": "John",
            "caused_by_user_name": "Manager",
        }
    )
    row = hrm_db.list_issues(employee_id=emp_id)[0]
    for key in (
        "display_employee",
        "title",
        "display_caused_by",
        "display_recorded_by",
        "status",
        "created_at",
        "updated_at",
        "show_recorded_by",
    ):
        assert key in row

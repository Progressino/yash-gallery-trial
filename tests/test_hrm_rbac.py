"""HRM RBAC — scope filtering by Admin / HOD / Employee."""
from __future__ import annotations

import uuid

import pytest

from backend.db import hrm_db, users_db
from backend.db.hrm_db import init_db as init_hrm_db
from backend.db.users_db import init_db as init_users_db, create_user, list_roles
from backend.services.rbac import build_hrm_scope, resolve_module_access


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


def _profile(role_name: str, **extra):
    base = {
        "role_name": role_name,
        "id": 1,
        "employee_id": None,
        "hrm_department_id": None,
        "reporting_hod_user_id": None,
        "module_access": None,
    }
    base.update(extra)
    return base


def test_rbac_modules_by_role():
    assert "hrm" in resolve_module_access("Employee")
    assert len(resolve_module_access("Employee")) == 1
    assert "intelligence" in resolve_module_access("Admin")
    assert "hrm" in resolve_module_access("Sir")
    mods = resolve_module_access("Employee", '["hrm","sales"]')
    assert "sales" in mods and "hrm" in mods


def test_ensure_user_has_modules_grants_hrm():
    import json

    roles = list_roles()
    karigar_id = next(r["id"] for r in roles if r["role_name"] == "Karigar")
    create_user({"username": "harsh", "password": "secret", "role_id": karigar_id})
    conn = users_db._connect()
    users_db.ensure_user_has_modules(conn, "harsh", ["hrm"])
    conn.commit()
    conn.close()
    harsh = next(u for u in users_db.list_users() if u["username"] == "harsh")
    mods = json.loads(harsh["module_access"] or "[]")
    assert "hrm" in mods
    assert "stitching" in mods


def test_hrm_scope_levels():
    assert build_hrm_scope(_profile("Admin")).level == "all"
    assert build_hrm_scope(_profile("Sir")).level == "all"
    assert build_hrm_scope(_profile("HOD", hrm_department_id=3)).level == "department"
    assert build_hrm_scope(_profile("Employee", employee_id=9)).level == "self"


def test_hrm_edit_assignment_permission_flags():
    assert build_hrm_scope(_profile("Admin")).can_edit_assignments is True
    assert build_hrm_scope(_profile("Super Admin")).can_edit_assignments is True
    assert build_hrm_scope(_profile("HOD", hrm_department_id=2)).can_edit_assignments is True
    assert build_hrm_scope(_profile("Employee", employee_id=1)).can_edit_assignments is False


def _make_client(monkeypatch, username: str, profile: dict):
    def _decode(token: str | None):
        if token == "tok":
            return {"sub": username, "role": profile.get("role_name", "Admin")}
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


def test_employee_sees_only_own_employees_list(monkeypatch):
    hrm_db.create_department({"name": f"D-{uuid.uuid4().hex[:6]}", "description": "", "hod_name": ""})
    depts = hrm_db.list_departments()
    did = depts[-1]["id"]
    hrm_db.create_employee({"name": "Alice", "department_id": did})
    hrm_db.create_employee({"name": "Bob", "department_id": did})
    emps = hrm_db.list_employees(did)
    alice_id = next(e["id"] for e in emps if e["name"] == "Alice")

    client = _make_client(
        monkeypatch,
        "emp1",
        _profile("Employee", employee_id=alice_id, hrm_department_id=did),
    )
    rows = client.get("/api/hrm/employees").json()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_hod_sees_department_employees_only(monkeypatch):
    d1 = f"D1-{uuid.uuid4().hex[:6]}"
    d2 = f"D2-{uuid.uuid4().hex[:6]}"
    hrm_db.create_department({"name": d1})
    hrm_db.create_department({"name": d2})
    depts = {d["name"]: d["id"] for d in hrm_db.list_departments()}
    hrm_db.create_employee({"name": "In Dept", "department_id": depts[d1]})
    hrm_db.create_employee({"name": "Other Dept", "department_id": depts[d2]})

    client = _make_client(
        monkeypatch,
        "hod1",
        _profile("HOD", hrm_department_id=depts[d1]),
    )
    rows = client.get("/api/hrm/employees").json()
    assert len(rows) == 1
    assert rows[0]["name"] == "In Dept"


def test_admin_sees_all_departments(monkeypatch):
    hrm_db.create_department({"name": f"A-{uuid.uuid4().hex[:6]}"})
    hrm_db.create_department({"name": f"B-{uuid.uuid4().hex[:6]}"})
    client = _make_client(monkeypatch, "admin1", _profile("Admin"))
    assert len(client.get("/api/hrm/departments").json()) >= 2


def test_employee_cannot_create_department(monkeypatch):
    client = _make_client(monkeypatch, "emp2", _profile("Employee", employee_id=1))
    r = client.post("/api/hrm/departments", json={"name": "Hack"})
    assert r.status_code == 403


def _seed_resp_and_task():
    dept_name = f"D-{uuid.uuid4().hex[:6]}"
    hrm_db.create_department({"name": dept_name})
    dept_id = hrm_db.list_departments()[-1]["id"]
    hrm_db.create_employee({"name": "Worker", "department_id": dept_id})
    emp_id = hrm_db.list_employees(dept_id)[0]["id"]
    hrm_db.create_responsibility({"employee_id": emp_id, "title": "Daily check", "frequency": "Daily"})
    rid = hrm_db.list_responsibilities(employee_id=emp_id)[0]["id"]
    tid = hrm_db.create_one_time_task({"employee_id": emp_id, "title": "One-off audit"})
    return dept_id, emp_id, rid, tid


def test_hod_can_edit_responsibility_and_task(monkeypatch):
    dept_id, emp_id, rid, tid = _seed_resp_and_task()
    client = _make_client(
        monkeypatch,
        "hod_edit",
        _profile("HOD", hrm_department_id=dept_id),
    )
    assert client.patch(f"/api/hrm/responsibilities/{rid}", json={"title": "Updated daily check"}).status_code == 200
    assert client.patch(f"/api/hrm/one-time-tasks/{tid}", json={"title": "Updated audit"}).status_code == 200
    assert hrm_db.list_responsibilities(employee_id=emp_id)[0]["title"] == "Updated daily check"
    assert hrm_db.list_one_time_tasks(employee_id=emp_id)[0]["title"] == "Updated audit"


def test_employee_cannot_edit_responsibility_or_task(monkeypatch):
    dept_id, emp_id, rid, tid = _seed_resp_and_task()
    client = _make_client(
        monkeypatch,
        "emp_edit",
        _profile("Employee", employee_id=emp_id, hrm_department_id=dept_id),
    )
    assert client.patch(f"/api/hrm/responsibilities/{rid}", json={"title": "Hack"}).status_code == 403
    assert client.patch(f"/api/hrm/one-time-tasks/{tid}", json={"title": "Hack"}).status_code == 403
    assert client.delete(f"/api/hrm/responsibilities/{rid}").status_code == 403


def test_admin_can_edit_responsibility(monkeypatch):
    _, emp_id, rid, _ = _seed_resp_and_task()
    client = _make_client(monkeypatch, "admin_edit", _profile("Admin"))
    assert client.patch(f"/api/hrm/responsibilities/{rid}", json={"title": "Admin edit"}).status_code == 200
    assert hrm_db.list_responsibilities(employee_id=emp_id)[0]["title"] == "Admin edit"


def test_create_hod_user_seeded_roles():
    roles = {r["role_name"] for r in list_roles()}
    assert "HOD" in roles
    assert "Employee" in roles
    assert "Sir" in roles
    hod_id = next(r["id"] for r in list_roles() if r["role_name"] == "HOD")
    create_user({"username": "hod_u", "password": "x", "role_id": hod_id, "department": "Production"})
    assert users_db.verify_erp_user("hod_u", "x") is not None

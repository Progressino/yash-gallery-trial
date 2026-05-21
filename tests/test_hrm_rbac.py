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


def test_hrm_scope_levels():
    assert build_hrm_scope(_profile("Admin")).level == "all"
    assert build_hrm_scope(_profile("Sir")).level == "all"
    assert build_hrm_scope(_profile("HOD", hrm_department_id=3)).level == "department"
    assert build_hrm_scope(_profile("Employee", employee_id=9)).level == "self"


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


def test_create_hod_user_seeded_roles():
    roles = {r["role_name"] for r in list_roles()}
    assert "HOD" in roles
    assert "Employee" in roles
    assert "Sir" in roles
    hod_id = next(r["id"] for r in list_roles() if r["role_name"] == "HOD")
    create_user({"username": "hod_u", "password": "x", "role_id": hod_id, "department": "Production"})
    assert users_db.verify_erp_user("hod_u", "x") is not None

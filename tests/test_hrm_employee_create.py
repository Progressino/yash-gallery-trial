"""HRM New Employee create — empty optional ints must not 422."""
from __future__ import annotations

import uuid

import pytest

from backend.db import hrm_db, users_db
from backend.db.hrm_db import init_db as init_hrm_db
from backend.db.users_db import init_db as init_users_db
from backend.routers.hrm import EmployeeIn
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


def test_employee_in_accepts_empty_reports_to():
    body = EmployeeIn(
        name="Suresh Kundnani",
        department_id=1,
        designation="Ceo",
        email="suresh@yashgallery.com",
        join_date="",
        emp_code="",
        reports_to_employee_id="",  # type: ignore[arg-type]
    )
    assert body.reports_to_employee_id is None
    assert body.emp_code == ""


def test_create_employee_api_with_empty_optional_ints(monkeypatch):
    hrm_db.create_department({"name": f"Office-{uuid.uuid4().hex[:6]}"})
    did = hrm_db.list_departments()[0]["id"]
    client = _make_client(monkeypatch, "admin1", _profile("Admin"))
    r = client.post(
        "/api/hrm/employees",
        json={
            "name": "Suresh Kundnani",
            "emp_code": "",
            "department_id": did,
            "designation": "Ceo",
            "phone": "",
            "email": "suresh@yashgallery.com",
            "join_date": "",
            "reports_to_employee_id": "",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    emps = hrm_db.list_employees(did)
    assert any(e["name"] == "Suresh Kundnani" for e in emps)

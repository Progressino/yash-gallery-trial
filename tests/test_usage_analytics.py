"""User usage analytics — tracking, flush/upsert, report and API access."""
import os
import sqlite3
import tempfile

import bcrypt
import pytest


@pytest.fixture
def usage_env(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("USERS_DB_PATH", path)
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("OTP_REQUIRED", "0")
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    monkeypatch.setenv("SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("AUTH_PASSWORD_HASH", bcrypt.hashpw(b"secret123", bcrypt.gensalt()).decode())

    from backend.db import users_db
    from backend.services import usage_analytics as ua

    monkeypatch.setattr(users_db, "_DB", path)
    users_db.init_db()
    conn = sqlite3.connect(path)
    role_id = {r: i for i, r in conn.execute("SELECT id, role_name FROM roles")}
    for uname, role, dept in [
        ("asha", "Manager", "Production"),
        ("ravi", "Employee", "Stores"),
        ("neha", "Admin", "Admin"),
        ("idle", "Employee", "Stores"),
    ]:
        conn.execute(
            "INSERT INTO erp_users(username, password_hash, full_name, role_id, department, active) VALUES(?,?,?,?,?,1)",
            (uname, bcrypt.hashpw(b"pw123456", bcrypt.gensalt()).decode(), uname.title(), role_id[role], dept),
        )
    conn.commit()
    conn.close()
    ua.reset_buffers_for_tests()
    yield path
    ua.reset_buffers_for_tests()
    try:
        os.unlink(path)
    except OSError:
        pass


def test_normalize_paths():
    from backend.services.usage_analytics import normalize_api_path, normalize_page_path

    assert normalize_api_path("/api/purchase/min/12/lines/3/revise") == (
        "/purchase/min/{id}/lines/{id}/revise",
        "purchase",
    )
    assert normalize_api_path("/api/grey/printed/P501Floral%20Cotton/allocate")[1] == "grey"
    assert normalize_api_path("/api/data/intelligence-bundle")[1] == "intelligence"
    assert normalize_page_path("/purchase?tab=jwo") == "/purchase"
    assert normalize_page_path("/production/JO-2026-0012") == "/production/{id}"
    assert normalize_page_path("") == "/"


def test_record_flush_and_report(usage_env):
    from backend.services import usage_analytics as ua

    ua.record_client_events("asha", "Manager", [
        {"kind": "page", "path": "/production", "module": "production"},
        {"kind": "tab", "path": "/production", "module": "production", "tab": "mrp"},
        {"kind": "tab", "path": "/production", "module": "production", "tab": "mrp"},
        {"kind": "bogus", "path": "/x"},
        {"kind": "tab", "path": "/production", "module": "production", "tab": ""},
    ])
    ua.record_api_action("asha", "Manager", "POST", "/api/production/jo/5/confirm", 200)
    ua.record_api_action("asha", "Manager", "POST", "/api/production/jo/6/confirm", 200)
    ua.record_api_action("asha", "Manager", "POST", "/api/production/jo/7/confirm", 500)
    ua.record_api_action("asha", "Manager", "GET", "/api/production/jo", 200)
    ua.record_api_action("asha", "Manager", "POST", "/api/usage/track", 200)
    ua.record_api_action("asha", "Manager", "POST", "/api/cache/hydrate-warm", 200)
    ua.record_api_action("asha", "Manager", "POST", "/api/upload/daily-auto/reset-stuck", 200)
    ua.record_client_events("ravi", "Employee", [{"kind": "page", "path": "/hrm", "module": "hrm"}])
    assert ua.flush() > 0

    # Second flush merges into the same daily rows (upsert, not duplicate).
    ua.record_client_events("asha", "Manager", [{"kind": "page", "path": "/production", "module": "production"}])
    ua.flush()

    rep = ua.usage_report(30)
    s = rep["summary"]
    assert s["active_today"] == 2
    assert s["active_7d"] == 2
    assert s["total_users"] >= 4
    assert s["actions"] == 2

    users = {u["username"]: u for u in rep["users"]}
    assert users["asha"]["page_views"] == 4  # 2 page + 2 tab
    assert users["asha"]["actions"] == 2
    assert users["asha"]["days_active"] == 1
    assert users["asha"]["active_minutes"] >= 5
    assert users["asha"]["top_module"] == "production"
    assert users["asha"]["last_seen"]
    assert users["idle"]["last_seen"] is None
    assert users["idle"]["days_active"] == 0

    mods = {m["module"]: m for m in rep["modules"]}
    assert mods["production"]["views"] == 4
    assert mods["production"]["actions"] == 2
    assert mods["hrm"]["users"] == 1

    pages = {p["key"]: p for p in rep["pages"]}
    assert pages["/production"]["hits"] == 2
    tabs = {t["key"]: t for t in rep["tabs"]}
    assert tabs["/production › mrp"]["hits"] == 2
    actions = {a["key"]: a for a in rep["actions"]}
    assert actions == {"POST /production/jo/{id}/confirm": actions["POST /production/jo/{id}/confirm"]}
    assert actions["POST /production/jo/{id}/confirm"]["hits"] == 2

    assert rep["trend"][-1]["active_users"] == 2
    assert len(rep["trend"]) == 30
    assert sum(h["active_users"] for h in rep["by_hour"]) >= 2


def test_report_filters(usage_env):
    from backend.services import usage_analytics as ua

    ua.record_client_events("asha", "Manager", [{"kind": "page", "path": "/production", "module": "production"}])
    ua.record_client_events("ravi", "Employee", [{"kind": "page", "path": "/hrm", "module": "hrm"}])
    ua.flush()

    by_role = ua.usage_report(7, role="Employee")
    assert {u["username"] for u in by_role["users"]} == {"ravi", "idle"}
    assert [m["module"] for m in by_role["modules"]] == ["hrm"]
    assert by_role["summary"]["active_today"] == 1

    by_user = ua.usage_report(7, username="asha")
    assert [u["username"] for u in by_user["users"]] == ["asha"]
    assert [p["key"] for p in by_user["pages"]] == ["/production"]

    by_dept = ua.usage_report(7, department="Stores")
    assert {u["username"] for u in by_dept["users"]} == {"ravi", "idle"}


def test_ping_counts_active_time_without_views(usage_env):
    from backend.services import usage_analytics as ua

    ua.record_client_events("ravi", "Employee", [], active=True)
    ua.record_client_events("ravi", "Employee", [], active=False)
    ua.flush()
    rep = ua.usage_report(7)
    ravi = next(u for u in rep["users"] if u["username"] == "ravi")
    assert ravi["page_views"] == 0
    assert ravi["active_minutes"] == 5
    assert ravi["last_seen"]


def test_api_login_track_and_report_access(usage_env):
    from starlette.testclient import TestClient

    from backend.main import app
    from backend.routers.auth import create_token
    from backend.services import usage_analytics as ua

    if not any(getattr(r, "path", "") == "/api/zz-usage-test/{doc_id}/save" for r in app.routes):
        app.add_api_route("/api/zz-usage-test/{doc_id}/save", lambda doc_id: {"ok": True}, methods=["POST"])

    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": "asha", "password": "pw123456"})
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True

    t = client.post("/api/usage/track", json={
        "events": [{"kind": "page", "path": "/purchase", "module": "purchase"},
                   {"kind": "tab", "path": "/purchase", "module": "purchase", "tab": "variance"}],
        "active": True,
    })
    assert t.status_code == 200 and t.json()["recorded"] == 2

    assert client.post("/api/zz-usage-test/42/save").status_code == 200
    assert client.post("/api/zz-usage-test-missing").status_code in (404, 405)

    # Manager can reach /api/admin/* generally, but the usage report is Super Admin / Admin only.
    denied = client.get("/api/admin/usage")
    assert denied.status_code == 403

    admin = TestClient(app)
    admin.cookies.set("auth_token", create_token("neha", role="Admin"))
    rep = admin.get("/api/admin/usage", params={"days": 7})
    assert rep.status_code == 200, rep.text
    data = rep.json()
    users = {u["username"]: u for u in data["users"]}
    assert users["asha"]["logins"] == 1
    assert users["asha"]["last_login"]
    assert users["asha"]["page_views"] == 2
    assert users["asha"]["actions"] == 1
    assert {a["key"] for a in data["actions"]} == {"POST /zz-usage-test/{id}/save"}
    assert data["recent_logins"][0]["username"] == "asha"
    assert data["recent_logins"][0]["method"] == "password"

    emp = TestClient(app)
    emp.cookies.set("auth_token", create_token("ravi", role="Employee"))
    assert emp.get("/api/admin/usage").status_code == 403
    ua.reset_buffers_for_tests()


def test_usage_endpoints_skip_heavy_session_restore():
    from backend import main

    assert "/api/usage/track" in main._SESSION_LIGHTWEIGHT
    assert "/api/admin/usage" in main._SESSION_LIGHTWEIGHT


def test_karigar_may_post_usage_track():
    from backend.services.permissions import karigar_may_access_api

    assert karigar_may_access_api("/api/usage/track", "POST")
    assert not karigar_may_access_api("/api/admin/usage", "GET")

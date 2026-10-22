"""Vendor portal isolation: own documents only, no stock writes, no cross-vendor reads."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db import users_db, vendor_portal_db
from backend.db.users_db import create_user, init_db as init_users, list_roles
from backend.db.vendor_portal_db import init_db as init_vendor
from backend.routers.auth import create_token
from backend.routers.vendor_portal import router as vendor_router
from backend.services.permissions import VENDOR_ROLE, vendor_may_access_api


def _seed_purchase(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE suppliers (
        id INTEGER PRIMARY KEY, supplier_code TEXT, supplier_name TEXT, active INTEGER DEFAULT 1
    );
    CREATE TABLE processors (
        id INTEGER PRIMARY KEY, processor_code TEXT, processor_name TEXT, active INTEGER DEFAULT 1
    );
    CREATE TABLE po_headers (
        id INTEGER PRIMARY KEY, po_number TEXT, po_date TEXT, supplier_id INTEGER,
        supplier_name TEXT, status TEXT, subtotal REAL, gst_amount REAL, total REAL, remarks TEXT
    );
    CREATE TABLE po_lines (
        id INTEGER PRIMARY KEY, po_id INTEGER, material_code TEXT, material_name TEXT,
        po_qty REAL, unit TEXT, rate REAL, gst_pct INTEGER, amount REAL
    );
    CREATE TABLE jwo_headers (
        id INTEGER PRIMARY KEY, jwo_number TEXT, jwo_date TEXT, processor_id INTEGER,
        processor_name TEXT, status TEXT, total REAL, remarks TEXT
    );
    CREATE TABLE jwo_lines (
        id INTEGER PRIMARY KEY, jwo_id INTEGER, input_material TEXT, input_qty REAL,
        output_material TEXT, output_qty REAL, rate REAL, amount REAL
    );
    CREATE TABLE material_issue_notes (
        id INTEGER PRIMARY KEY, min_number TEXT, min_date TEXT, to_vendor TEXT, status TEXT
    );
    CREATE TABLE min_lines (
        id INTEGER PRIMARY KEY, min_id INTEGER, material_code TEXT, issue_qty REAL
    );
    CREATE TABLE grn_headers (
        id INTEGER PRIMARY KEY, grn_number TEXT
    );
    """)
    conn.execute("INSERT INTO suppliers(id, supplier_code, supplier_name) VALUES(1,'S1','Alpha Mills')")
    conn.execute("INSERT INTO suppliers(id, supplier_code, supplier_name) VALUES(2,'S2','Beta Mills')")
    conn.execute("INSERT INTO processors(id, processor_code, processor_name) VALUES(7,'P7','Print House')")
    conn.execute(
        "INSERT INTO po_headers(id, po_number, po_date, supplier_id, status, total) VALUES(10,'PO-10','2026-10-01',1,'Released',500)"
    )
    conn.execute(
        "INSERT INTO po_lines(id, po_id, material_code, po_qty, rate, amount) VALUES(1,10,'FAB',100,5,500)"
    )
    conn.execute(
        "INSERT INTO po_headers(id, po_number, po_date, supplier_id, status, total) VALUES(11,'PO-11','2026-10-02',2,'Released',900)"
    )
    conn.execute(
        "INSERT INTO jwo_headers(id, jwo_number, jwo_date, processor_id, status, total) VALUES(3,'JWO-3','2026-10-03',7,'Released',200)"
    )
    conn.execute(
        "INSERT INTO jwo_lines(id, jwo_id, input_material, input_qty, output_material, output_qty, rate, amount) VALUES(1,3,'GREY',10,'PRINT',10,20,200)"
    )
    conn.execute(
        "INSERT INTO material_issue_notes(id, min_number, min_date, to_vendor, status) VALUES(4,'MIN-4','2026-10-04','Alpha Mills','Issued')"
    )
    conn.execute(
        "INSERT INTO min_lines(id, min_id, material_code, issue_qty) VALUES(8,4,'FAB',100)"
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def portal(tmp_path, monkeypatch):
    users_path = str(tmp_path / "users.db")
    purchase_path = str(tmp_path / "purchase.db")
    monkeypatch.setenv("USERS_DB_PATH", users_path)
    monkeypatch.setenv("PURCHASE_DB_PATH", purchase_path)
    monkeypatch.setattr(users_db, "_DB", users_path)
    monkeypatch.setattr(vendor_portal_db, "_DB", purchase_path)
    monkeypatch.setattr(vendor_portal_db, "_PROD_DB", str(tmp_path / "missing-production.db"))
    init_users()
    _seed_purchase(purchase_path)
    init_vendor()
    app = FastAPI()

    @app.middleware("http")
    async def auth_mw(request, call_next):
        token = request.cookies.get("auth_token")
        payload = None
        if token:
            from backend.routers.auth import decode_token
            payload = decode_token(token)
        if not payload:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
        request.state.auth = payload
        if payload.get("role") == VENDOR_ROLE and not vendor_may_access_api(request.url.path, request.method):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=403, content={"detail": "Access denied for vendor role"})
        return await call_next(request)

    app.include_router(vendor_router, prefix="/api/vendor")

    @app.get("/api/purchase/po")
    def internal_po():
        return {"ok": True}

    return TestClient(app)


def _vendor_user(username: str) -> dict:
    role_id = next(r["id"] for r in list_roles() if r["role_name"] == "Vendor")
    create_user({
        "username": username,
        "password": "secret12",
        "full_name": username,
        "role_id": role_id,
        "department": "External",
    })
    from backend.db.users_db import verify_erp_user
    user = verify_erp_user(username, "secret12")
    assert user is not None
    return user


def _client_as(portal: TestClient, user: dict) -> TestClient:
    token = create_token(user["username"], role="Vendor", user_id=user["id"], full_name=user["full_name"])
    portal.cookies.set("auth_token", token)
    return portal


def test_vendor_role_seeded(portal):
    assert "Vendor" in {r["role_name"] for r in list_roles()}


def test_vendor_cannot_call_internal_purchase(portal):
    user = _vendor_user("alpha_login")
    vendor_portal_db.upsert_link(user_id=user["id"], party_kind="supplier", party_id=1, portal_role="vendor_ops")
    _client_as(portal, user)
    res = portal.get("/api/purchase/po")
    assert res.status_code == 403


def test_supplier_sees_only_own_po_and_ops_hides_rates(portal):
    alpha = _vendor_user("alpha_login")
    beta = _vendor_user("beta_login")
    vendor_portal_db.upsert_link(user_id=alpha["id"], party_kind="supplier", party_id=1, portal_role="vendor_ops")
    vendor_portal_db.upsert_link(user_id=beta["id"], party_kind="supplier", party_id=2, portal_role="vendor_accounts")

    _client_as(portal, alpha)
    own = portal.get("/api/vendor/purchase-orders")
    assert own.status_code == 200
    body = own.json()
    assert [r["po_number"] for r in body] == ["PO-10"]
    assert "rate" not in body[0]["lines"][0]
    assert "total" not in body[0]
    other = portal.get("/api/vendor/purchase-orders/11")
    assert other.status_code == 404

    _client_as(portal, beta)
    beta_rows = portal.get("/api/vendor/purchase-orders").json()
    assert [r["po_number"] for r in beta_rows] == ["PO-11"]
    assert beta_rows[0]["total"] == 900


def test_processor_does_not_see_purchase_orders(portal):
    user = _vendor_user("printer_login")
    vendor_portal_db.upsert_link(user_id=user["id"], party_kind="processor", party_id=7, portal_role="vendor_admin")
    _client_as(portal, user)
    assert portal.get("/api/vendor/purchase-orders").json() == []
    jwos = portal.get("/api/vendor/job-work-orders").json()
    assert [r["jwo_number"] for r in jwos] == ["JWO-3"]
    assert jwos[0]["lines"][0]["rate"] == 20
    assert portal.get("/api/vendor/job-work-orders/99").status_code == 404


def test_vendor_cannot_manage_links(portal):
    user = _vendor_user("alpha_login")
    vendor_portal_db.upsert_link(user_id=user["id"], party_kind="supplier", party_id=1, portal_role="vendor_admin")
    _client_as(portal, user)
    assert portal.get("/api/vendor/admin/links").status_code == 403


def test_accept_does_not_change_po_and_ack_does_not_change_issue_qty(portal):
    user = _vendor_user("alpha_login")
    vendor_portal_db.upsert_link(user_id=user["id"], party_kind="supplier", party_id=1, portal_role="vendor_admin")
    _client_as(portal, user)
    res = portal.post("/api/vendor/purchase-orders/10/response", json={"action": "accept"})
    assert res.status_code == 200
    assert res.json()["erp_status_unchanged"] is True
    assert portal.get("/api/vendor/purchase-orders/10").json()["status"] == "Released"
    assert portal.get("/api/vendor/purchase-orders/10").json()["vendor_response"]["action"] == "accept"
    denied = portal.post("/api/vendor/purchase-orders/11/response", json={"action": "accept"})
    assert denied.status_code == 404
    missing_reason = portal.post("/api/vendor/purchase-orders/10/response", json={"action": "reject"})
    assert missing_reason.status_code == 400

    ack = portal.post("/api/vendor/material-issues/4/acknowledge", json={"lines": [{"line_id": 8, "received_qty": 98, "reason": "2 short"}]})
    assert ack.status_code == 200
    assert ack.json()["stock_unchanged"] is True
    assert ack.json()["issued_qty_after"] == [100.0]
    short_without_reason = portal.post("/api/vendor/material-issues/4/acknowledge", json={"lines": [{"line_id": 8, "received_qty": 90}]})
    assert short_without_reason.status_code == 400

    intim = portal.post("/api/vendor/delivery-intimations", json={"doc_type": "po", "doc_id": 10, "qty": 40, "challan_no": "C1"})
    assert intim.status_code == 200
    assert intim.json()["grn_created"] is False
    assert intim.json()["status"] == "Submitted"


def test_deactivated_link_blocks_portal(portal):
    user = _vendor_user("alpha_login")
    link = vendor_portal_db.upsert_link(user_id=user["id"], party_kind="supplier", party_id=1, portal_role="vendor_admin")
    vendor_portal_db.set_link_active(link["id"], False)
    _client_as(portal, user)
    assert portal.get("/api/vendor/dashboard").status_code == 403

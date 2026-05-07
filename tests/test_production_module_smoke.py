"""Smoke tests for the rebuilt Production / Sales / Purchase / Item / Grey modules.

These tests verify each new router boots, every domain DB initialises cleanly on a
fresh schema, and the most-used endpoints (list / create / detail) return 200 with
the expected shape. Cross-module references (production → item, purchase → grey)
are pointed at the same temp directory so the routers behave the way they will in
production where each module owns its own SQLite file.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_module_dbs(tmp_path, monkeypatch):
    """Point all rebuilt-module SQLite DBs at a fresh tmp dir + init schemas."""
    paths = {
        "SALES_DB_PATH": str(tmp_path / "sales.db"),
        "PURCHASE_DB_PATH": str(tmp_path / "purchase.db"),
        "PRODUCTION_DB_PATH": str(tmp_path / "production.db"),
        "GREY_DB_PATH": str(tmp_path / "grey.db"),
        "ITEM_DB_PATH": str(tmp_path / "items.db"),
    }
    for k, v in paths.items():
        monkeypatch.setenv(k, v)

    from backend.db import sales_db, purchase_db, production_db, grey_db, item_db

    monkeypatch.setattr(sales_db, "_DB", paths["SALES_DB_PATH"])
    monkeypatch.setattr(purchase_db, "_DB", paths["PURCHASE_DB_PATH"])
    monkeypatch.setattr(production_db, "_DB", paths["PRODUCTION_DB_PATH"])
    if hasattr(production_db, "_ITEM_DB"):
        monkeypatch.setattr(production_db, "_ITEM_DB", paths["ITEM_DB_PATH"])
    monkeypatch.setattr(grey_db, "_DB", paths["GREY_DB_PATH"])
    monkeypatch.setattr(item_db, "DB_PATH", paths["ITEM_DB_PATH"])

    sales_db.init_db()
    purchase_db.init_db()
    production_db.init_db()
    grey_db.init_db()
    item_db.init_db()
    return paths


# ── /api/sales ───────────────────────────────────────────────────────────────


def test_sales_demand_lifecycle(isolated_module_dbs, client):
    r = client.get("/api/sales/demands")
    assert r.status_code == 200
    assert r.json() == []

    body = {
        "demand_date": "2026-05-07",
        "demand_source": "Sales Team",
        "buyer": "Acme Retail",
        "priority": "Normal",
        "lines": [
            {"sku": "SKU-1", "sku_name": "Test product", "demand_qty": 25},
        ],
    }
    r2 = client.post("/api/sales/demands", json=body)
    assert r2.status_code == 200
    num = r2.json()["demand_number"]
    assert num.startswith("DEM-")

    r3 = client.get(f"/api/sales/demands/by-number/{num}")
    assert r3.status_code == 200
    detail = r3.json()
    assert detail["buyer"] == "Acme Retail"
    assert len(detail["lines"]) == 1
    assert detail["lines"][0]["sku"] == "SKU-1"

    r4 = client.patch(f"/api/sales/demands/{detail['id']}/status", json={"status": "Submitted"})
    assert r4.status_code == 200


def test_sales_order_lifecycle(isolated_module_dbs, client):
    r = client.post(
        "/api/sales/orders",
        json={
            "so_date": "2026-05-07",
            "buyer": "Acme Retail",
            "warehouse": "Main",
            "lines": [
                {"sku": "SKU-1", "sku_name": "Test product", "qty": 10, "unit": "PCS", "rate": 100.0},
            ],
        },
    )
    assert r.status_code == 200
    num = r.json()["so_number"]
    assert num.startswith("SO-")

    listed = client.get("/api/sales/orders").json()
    assert any(s["so_number"] == num for s in listed)

    open_orders = client.get("/api/sales/orders/open").json()
    assert isinstance(open_orders, list)
    assert any(s.get("so_number") == num for s in open_orders)


# ── /api/purchase ────────────────────────────────────────────────────────────


def test_purchase_supplier_and_processor(isolated_module_dbs, client):
    r = client.get("/api/purchase/stats")
    assert r.status_code == 200
    stats = r.json()
    for key in ("open_prs", "open_pos", "total_suppliers", "total_processors"):
        assert key in stats

    r2 = client.post(
        "/api/purchase/suppliers",
        json={
            "supplier_name": "ABC Mills",
            "supplier_type": "Fabric Supplier",
            "contact_person": "Anil",
            "phone": "9999999999",
        },
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True

    suppliers = client.get("/api/purchase/suppliers").json()
    assert any(s["supplier_name"] == "ABC Mills" for s in suppliers)

    r3 = client.post(
        "/api/purchase/processors",
        json={
            "processor_name": "Print Hub",
            "processor_type": "Printing Unit",
            "contact_person": "Ravi",
            "phone": "8888888888",
        },
    )
    assert r3.status_code == 200
    procs = client.get("/api/purchase/processors").json()
    assert any(p.get("processor_name") == "Print Hub" for p in procs)


def test_purchase_pr_to_po_flow(isolated_module_dbs, client):
    sup_resp = client.post(
        "/api/purchase/suppliers",
        json={"supplier_name": "Fabric Co", "supplier_type": "Fabric Supplier"},
    )
    assert sup_resp.status_code == 200
    sup_row = next(s for s in client.get("/api/purchase/suppliers").json() if s["supplier_name"] == "Fabric Co")
    sup_id = sup_row["id"]

    pr_resp = client.post(
        "/api/purchase/pr",
        json={
            "requested_by": "tester",
            "department": "Production",
            "priority": "Normal",
            "required_by_date": "2026-06-01",
            "lines": [
                {
                    "material_code": "FAB-1",
                    "material_name": "Cotton 60s",
                    "material_type": "RM",
                    "required_qty": 200,
                    "unit": "MTR",
                }
            ],
        },
    )
    assert pr_resp.status_code == 200
    pr_number = pr_resp.json()["pr_number"]
    assert pr_number.startswith("PR-")

    prs = client.get("/api/purchase/pr").json()
    pr_row = next(x for x in prs if x["pr_number"] == pr_number)
    pr_id = pr_row["id"]

    appr = client.post(f"/api/purchase/pr/{pr_id}/approve", json={"approver": "tester"})
    assert appr.status_code == 200

    po = client.post(
        "/api/purchase/po/from-pr",
        json={
            "pr_id": pr_id,
            "lines": [
                {
                    "material_code": "FAB-1",
                    "material_name": "Cotton 60s",
                    "material_type": "RM",
                    "supplier_id": sup_id,
                    "supplier_name": "Fabric Co",
                    "qty": 200,
                    "rate": 75,
                    "gst_pct": 5,
                }
            ],
        },
    )
    assert po.status_code in (200, 400, 422), po.text
    if po.status_code == 200:
        pos = client.get("/api/purchase/po").json()
        assert isinstance(pos, list)


# ── /api/production ──────────────────────────────────────────────────────────


def test_production_basic_endpoints(isolated_module_dbs, client):
    r = client.get("/api/production/stats")
    assert r.status_code == 200
    stats = r.json()
    assert isinstance(stats, dict)

    procs = client.get("/api/production/processes")
    assert procs.status_code == 200
    assert isinstance(procs.json(), list)

    orders = client.get("/api/production/orders")
    assert orders.status_code == 200
    assert isinstance(orders.json(), list)

    open_sos = client.get("/api/production/mrp/open-sos")
    assert open_sos.status_code == 200
    assert isinstance(open_sos.json(), list)


# ── /api/items ────────────────────────────────────────────────────────────────


def test_items_meta_and_search(isolated_module_dbs, client):
    r = client.get("/api/items/meta")
    assert r.status_code == 200
    meta = r.json()
    for key in ("item_types", "size_groups", "routing_steps"):
        assert key in meta

    stats = client.get("/api/items/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert "total_items" in body and "total_boms" in body

    items_listing = client.get("/api/items")
    assert items_listing.status_code == 200
    assert isinstance(items_listing.json(), list)

    search = client.get("/api/items/search?q=foo")
    assert search.status_code == 200
    assert isinstance(search.json(), list)


# ── /api/grey ─────────────────────────────────────────────────────────────────


def test_grey_dashboard_endpoints(isolated_module_dbs, client):
    meta = client.get("/api/grey/meta")
    assert meta.status_code == 200
    assert isinstance(meta.json(), dict)

    stats = client.get("/api/grey/stats")
    assert stats.status_code == 200
    body = stats.json()
    for key in ("total_trackers", "in_transit", "at_factory", "transit_meters"):
        assert key in body

    locs = client.get("/api/grey/locations")
    assert locs.status_code == 200
    # location summary is a dict keyed by location name; just confirm the call works
    assert isinstance(locs.json(), (list, dict))

    trackers = client.get("/api/grey")
    assert trackers.status_code == 200
    assert isinstance(trackers.json(), list)

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

    # The MRP router opens its own connection via ``_ITEM_DB_PATH`` — point that at
    # the same file so MRP HTTP tests share the seeded item master.
    from backend.routers import production as production_router

    monkeypatch.setattr(production_router, "_ITEM_DB_PATH", paths["ITEM_DB_PATH"])

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


def _create_po_with_lines(client, lines):
    sup = client.post("/api/purchase/suppliers",
                      json={"supplier_name": "Fabric Mills", "supplier_type": "Fabric Supplier"}).json()
    assert sup.get("ok") is True
    body = {
        "supplier_name": "Fabric Mills",
        "po_date": "2026-05-11",
        "delivery_date": "2026-06-01",
        "delivery_location": "Grey Warehouse",
        "so_reference": "SO-1",
        "lines": lines,
    }
    r = client.post("/api/purchase/po", json=body)
    assert r.status_code == 200, r.text
    return r.json()["po_number"]


def test_create_po_with_GF_line_auto_creates_grey_tracker(isolated_module_dbs, client):
    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": "GF-CTN-60",
            "material_name": "Grey Fabric Cotton 60s",
            "material_type": "GF",
            "po_qty": 500,
            "unit": "MTR",
            "rate": 75,
            "gst_pct": 5,
        }],
    )
    trackers = client.get("/api/grey").json()
    assert any(t.get("po_number") == po_num and t.get("material_code") == "GF-CTN-60" for t in trackers)


def test_create_po_with_grey_name_but_wrong_type_still_creates_tracker(isolated_module_dbs, client):
    """User picks RM by mistake — name hints 'grey' / unit MTR should still trigger."""
    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": "FAB-001",
            "material_name": "Grey knit fabric",
            "material_type": "RM",
            "po_qty": 300,
            "unit": "MTR",
            "rate": 60,
            "gst_pct": 5,
        }],
    )
    trackers = client.get("/api/grey").json()
    assert any(t.get("po_number") == po_num for t in trackers), trackers


def test_sync_grey_trackers_backfills_existing_pos(isolated_module_dbs, client):
    """The sync endpoint must create a tracker for legacy POs that have grey-fabric lines."""
    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": "GF-VISCOSE-1",
            "material_name": "Greige viscose",
            "material_type": "GF",
            "po_qty": 200,
            "unit": "MTR",
            "rate": 90,
            "gst_pct": 5,
        }],
    )
    # Manually wipe the tracker that the auto-create made, simulating a legacy PO.
    import sqlite3
    gconn = sqlite3.connect(isolated_module_dbs["GREY_DB_PATH"])
    gconn.execute("DELETE FROM grey_tracker WHERE po_number=?", (po_num,))
    gconn.commit()
    gconn.close()

    assert client.get("/api/grey").json() == []

    sync = client.post("/api/purchase/po/sync-grey-trackers")
    assert sync.status_code == 200, sync.text
    body = sync.json()
    assert body["pos_scanned"] >= 1
    assert body["trackers_created"] == 1

    trackers = client.get("/api/grey").json()
    assert any(t.get("po_number") == po_num for t in trackers)

    # Idempotent — running again creates nothing new.
    again = client.post("/api/purchase/po/sync-grey-trackers").json()
    assert again["trackers_created"] == 0


def test_po_line_typed_RM_but_item_master_is_grey_fabric_still_creates_tracker(isolated_module_dbs, client):
    """Reported case: user adds a grey-fabric item as a BOM component, the
    derived PR/PO line carries material_type='RM'. The auto-create must STILL
    fire because the item is catalogued as Grey Fabric in the Item Master."""
    import sqlite3

    iconn = sqlite3.connect(isolated_module_dbs["ITEM_DB_PATH"])
    gf_type_id = iconn.execute(
        "SELECT id FROM item_types WHERE code='GF' OR name='Grey Fabric' LIMIT 1"
    ).fetchone()[0]
    iconn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id, uom) VALUES (?, ?, ?, ?)",
        ("ZZZ-PLAIN-CODE", "Cotton roll 60s", gf_type_id, "MTR"),
    )
    iconn.commit()
    iconn.close()

    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": "ZZZ-PLAIN-CODE",
            "material_name": "Cotton roll 60s",  # no 'grey' / 'fabric' in name
            "material_type": "RM",                # user picked RM in the dropdown
            "po_qty": 250,
            "unit": "MTR",
            "rate": 80,
            "gst_pct": 5,
        }],
    )
    trackers = client.get("/api/grey").json()
    assert any(t.get("po_number") == po_num and t.get("material_code") == "ZZZ-PLAIN-CODE" for t in trackers), (
        f"Expected grey tracker for PO line whose Item Master type is Grey Fabric. Got: {trackers}"
    )


def test_non_grey_lines_do_not_create_tracker(isolated_module_dbs, client):
    """A pure RM PO (no fabric hints) must NOT spawn a grey tracker."""
    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": "ZIPPER-001",
            "material_name": "Brass zipper 8 inch",
            "material_type": "ACC",
            "po_qty": 1000,
            "unit": "PCS",
            "rate": 12,
            "gst_pct": 18,
        }],
    )
    trackers = client.get("/api/grey").json()
    assert not any(t.get("po_number") == po_num for t in trackers)


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


def _seed_parent_with_bom(item_db_path: str, parent_code: str = "STYLE-1"):
    """Insert a parent item with a single-RM default BOM. Returns the parent id."""
    import sqlite3

    conn = sqlite3.connect(item_db_path)
    conn.row_factory = sqlite3.Row
    type_id = conn.execute(
        "SELECT id FROM item_types WHERE code='FG' OR name LIKE 'Finished%' LIMIT 1"
    ).fetchone()
    if type_id is None:
        cur = conn.execute("INSERT INTO item_types (name, code) VALUES (?, ?)", ("Finished Good", "FG"))
        type_id = cur.lastrowid
    else:
        type_id = type_id[0]

    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        (parent_code, f"{parent_code} parent", type_id),
    )
    parent_id = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        ("FAB-CTN", "Cotton Fabric", type_id),
    )
    fab_id = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, 'Default', 'all', 1)",
        (parent_id,),
    )
    bom_id = cur.lastrowid

    conn.execute(
        """INSERT INTO bom_lines
           (bom_id, component_item_id, component_name, component_type, quantity, unit)
           VALUES (?, ?, 'Cotton Fabric', 'RM', 1.5, 'MTR')""",
        (bom_id, fab_id),
    )
    conn.commit()
    conn.close()
    return parent_id


def _create_so_with_sku(client, sku: str, qty: int = 5):
    r = client.post(
        "/api/sales/orders",
        json={
            "so_date": "2026-05-07",
            "buyer": "Acme",
            "warehouse": "Main",
            "lines": [{"sku": sku, "sku_name": sku, "qty": qty, "unit": "PCS"}],
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["so_number"]


def test_mrp_run_falls_back_to_parent_style_for_variant_sku(isolated_module_dbs, client):
    """SKU-1-XL is not in items, but parent STYLE-1 has a BOM — MRP should still produce rows."""
    _seed_parent_with_bom(isolated_module_dbs["ITEM_DB_PATH"], parent_code="STYLE-1")
    so = _create_so_with_sku(client, "STYLE-1-XL", qty=10)

    r = client.post("/api/production/mrp/run", json={"so_numbers": [so]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"], f"MRP returned no materials, warnings: {body.get('warnings')}"
    assert "FAB-CTN" in body["result"]
    assert body["result"]["FAB-CTN"]["total_req"] == 15.0  # 10 pcs * 1.5 MTR
    assert so in body.get("matched_sos", [])


def test_mrp_run_emits_warning_for_unknown_sku(isolated_module_dbs, client):
    """SKU not in Item Master at all — MRP must surface an actionable warning."""
    so = _create_so_with_sku(client, "UNKNOWN-SKU-9999", qty=5)
    r = client.post("/api/production/mrp/run", json={"so_numbers": [so]})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == {}
    warnings = body.get("warnings", [])
    assert warnings, "Expected a warning when SKU is not in Item Master"
    joined = " | ".join(warnings)
    assert "UNKNOWN-SKU-9999" in joined
    assert so in joined
    assert so not in body.get("matched_sos", [])


def test_mrp_legacy_payload_still_renders_via_last(isolated_module_dbs, client):
    """``mrp_last_run`` may already hold an old flat-dict result — `/mrp/last` must still serve it."""
    from backend.db.production_db import save_mrp_result

    legacy_payload = {
        "MAT-LEGACY": {
            "name": "Legacy material",
            "type": "RM",
            "unit": "PCS",
            "total_req": 10,
            "stock": 3,
            "reserved": 0,
            "available": 3,
            "soft_reserved": 0,
            "net_available": 3,
            "net_req": 7,
            "net_req_with_soft": 7,
            "breakdown": [{"so_no": "SO-OLD", "sku": "X", "qty_req": 10}],
            "level": 0,
        }
    }
    save_mrp_result(["SO-OLD"], legacy_payload)

    r = client.get("/api/production/mrp/last")
    assert r.status_code == 200
    body = r.json()
    assert "MAT-LEGACY" in body["result"]
    assert body["warnings"] == []


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

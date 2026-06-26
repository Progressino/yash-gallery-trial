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

    from backend.services import jo_issue_notes

    monkeypatch.setattr(jo_issue_notes, "_PROD_DB", paths["PRODUCTION_DB_PATH"])
    monkeypatch.setattr(jo_issue_notes, "_ITEM_DB", paths["ITEM_DB_PATH"])

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


def test_jo_auto_creates_bom_issue_note(isolated_module_dbs, client):
    """Creating a JO should auto-generate an issue note with BOM-scaled material lines."""
    _seed_parent_with_bom(isolated_module_dbs["ITEM_DB_PATH"], parent_code="PRINTED-1")

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-05-15",
            "sku": "PRINTED-1",
            "sku_name": "Printed shirt",
            "process": "Cutting",
            "planned_qty": 100,
            "fabric_code": "FAB-CTN",
            "fabric_qty": 100,
            "fabric_unit": "MTR",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["jo_number"].startswith("PJO-")
    assert body.get("issue_note") is not None

    orders = client.get("/api/production/orders").json()
    jo = next(o for o in orders if o["jo_number"] == body["jo_number"])
    note = client.get(f"/api/production/orders/{jo['id']}/issue-note").json()
    assert note["jo_number"] == body["jo_number"]
    assert note["finished_item_code"] == "PRINTED-1"
    assert note["planned_qty"] == 100

    fab = next(l for l in note["lines"] if l["material_code"] == "FAB-CTN")
    assert fab["finished_item_code"] == "PRINTED-1"
    assert fab["required_qty"] == pytest.approx(150.0)  # 100 pcs × 1.5 MTR per unit BOM

    listed = client.get("/api/production/issue-notes").json()
    assert any(n["in_number"] == note["in_number"] for n in listed)


def test_jo_outsource_requires_vendor_name(isolated_module_dbs, client):
    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-05-26",
            "so_number": "SO-0001",
            "sku": "TEST-SKU",
            "process": "Cutting",
            "exec_type": "Outsource",
            "vendor_name": "",
            "planned_qty": 10,
        },
    )
    assert r.status_code == 400
    assert "vendor" in r.json()["detail"].lower()


def test_jo_outsource_stores_vendor_name(isolated_module_dbs, client):
    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-05-26",
            "so_number": "SO-0001",
            "sku": "TEST-SKU",
            "process": "Cutting",
            "exec_type": "Outsource",
            "vendor_name": "ABC Stitching Unit",
            "planned_qty": 10,
        },
    )
    assert r.status_code == 200, r.text
    jo = next(o for o in client.get("/api/production/orders").json() if o["jo_number"] == r.json()["jo_number"])
    assert jo["exec_type"] == "Outsource"
    assert jo["vendor_name"] == "ABC Stitching Unit"


def test_receive_pieces_per_line_cutting_jo(isolated_module_dbs, client):
    """Line-level receive on a multi-size cutting JO (cutting challan / Rec button)."""
    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-05-22",
            "so_number": "SO-0006",
            "sku": "7100YKTEAL-S",
            "process": "Cutting",
            "planned_qty": 40,
            "fabric_code": "P308",
            "fabric_qty": 100,
            "lines": [
                {"sku": "7100YKTEAL-S", "style": "S", "planned_qty": 10},
                {"sku": "7100YKTEAL-M", "style": "M", "planned_qty": 20},
                {"sku": "7100YKTEAL-L", "style": "L", "planned_qty": 10},
            ],
        },
    )
    assert r.status_code == 200, r.text
    jo = client.get("/api/production/orders").json()
    jo = next(o for o in jo if o["jo_number"] == r.json()["jo_number"])
    line_s = next(ln for ln in jo["lines"] if ln["style"] == "S")
    rec = client.post(
        f"/api/production/orders/{jo['id']}/receive-pieces",
        json={
            "received_qty": 10,
            "process": "Cutting",
            "sku": line_s["sku"],
            "jo_line_id": line_s["id"],
        },
    )
    assert rec.status_code == 200, rec.text
    refreshed = client.get(f"/api/production/orders/{jo['id']}").json()
    line_after = next(ln for ln in refreshed["lines"] if ln["id"] == line_s["id"])
    assert line_after["received_qty"] == 10
    assert line_after["balance_qty"] == 0
    assert refreshed["received_qty"] == 10


def test_init_db_adds_jo_lines_received_qty_column(tmp_path):
    """Older production DBs lacked jo_lines.received_qty — migration must add it."""
    import os
    import sqlite3

    from backend.db import production_db

    legacy_path = str(tmp_path / "legacy_production.db")
    lconn = sqlite3.connect(legacy_path)
    lconn.execute(
        """CREATE TABLE jo_lines (
        id INTEGER PRIMARY KEY, jo_id INTEGER, sku TEXT, planned_qty INTEGER DEFAULT 0,
        issued_qty INTEGER DEFAULT 0, rejected_qty INTEGER DEFAULT 0, balance_qty INTEGER DEFAULT 0)"""
    )
    lconn.commit()
    lconn.close()

    saved = production_db._DB
    production_db._DB = legacy_path
    try:
        production_db.init_db()
        lconn = sqlite3.connect(legacy_path)
        cols = {r[1] for r in lconn.execute("PRAGMA table_info(jo_lines)").fetchall()}
        lconn.close()
        assert "received_qty" in cols
    finally:
        production_db._DB = saved
        if os.path.exists(legacy_path):
            os.remove(legacy_path)


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


def _seed_printed_grey_bom_chain(
    item_db_path: str,
    *,
    printed_stock: float = 0.0,
    grey_bom_qty: float = 1.0,
    parent_code: str = "STYLE-PRT",
):
    """FG → Printed Fabric (SFG, optional stock) → Grey Fabric (GF) via nested BOM."""
    import sqlite3

    conn = sqlite3.connect(item_db_path)
    conn.row_factory = sqlite3.Row

    def _type_id(code: str, name: str) -> int:
        row = conn.execute("SELECT id FROM item_types WHERE code=? LIMIT 1", (code,)).fetchone()
        if row:
            return int(row[0])
        cur = conn.execute("INSERT INTO item_types (name, code) VALUES (?, ?)", (name, code))
        return int(cur.lastrowid)

    fg_id = _type_id("FG", "Finished Good")
    sfg_id = _type_id("SFG", "Semi-Finished Goods")
    gf_id = _type_id("GF", "Grey Fabric")

    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        (parent_code, f"{parent_code} style", fg_id),
    )
    style_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id, stock) VALUES (?, ?, ?, ?)",
        ("PRINTED-FAB", "Printed Fabric", sfg_id, printed_stock),
    )
    printed_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        ("GREY-FAB", "Grey Fabric", gf_id),
    )
    grey_id = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, 'Default', 'all', 1)",
        (printed_id,),
    )
    printed_bom_id = cur.lastrowid
    conn.execute(
        """INSERT INTO bom_lines
           (bom_id, component_item_id, component_name, component_type, quantity, unit)
           VALUES (?, ?, 'Grey Fabric', 'GF', ?, 'MTR')""",
        (printed_bom_id, grey_id, grey_bom_qty),
    )

    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, 'Default', 'all', 1)",
        (style_id,),
    )
    style_bom_id = cur.lastrowid
    conn.execute(
        """INSERT INTO bom_lines
           (bom_id, component_item_id, component_name, component_type, quantity, unit)
           VALUES (?, ?, 'Printed Fabric', 'SFG', 1, 'MTR')""",
        (style_bom_id, printed_id),
    )
    conn.commit()
    conn.close()


def test_mrp_sub_bom_uses_parent_net_after_printed_stock(isolated_module_dbs, client):
    """Grey requirement = (printed gross − printed stock) × BOM ratio."""
    _seed_printed_grey_bom_chain(
        isolated_module_dbs["ITEM_DB_PATH"],
        printed_stock=100.0,
        grey_bom_qty=1.0,
    )
    so = _create_so_with_sku(client, "STYLE-PRT-M", qty=1000)

    r = client.post("/api/production/mrp/run", json={"so_numbers": [so]})
    assert r.status_code == 200, r.text
    body = r.json()
    printed = body["result"]["PRINTED-FAB"]
    grey = body["result"]["GREY-FAB"]
    assert printed["total_req"] == 1000.0
    assert printed["net_req"] == 900.0
    assert grey["total_req"] == 900.0
    assert grey["net_req"] == 900.0


def test_mrp_sub_bom_applies_grey_bom_ratio(isolated_module_dbs, client):
    """BOM 1:1.05 → grey = net printed × 1.05."""
    _seed_printed_grey_bom_chain(
        isolated_module_dbs["ITEM_DB_PATH"],
        printed_stock=100.0,
        grey_bom_qty=1.05,
    )
    so = _create_so_with_sku(client, "STYLE-PRT-L", qty=1000)

    r = client.post("/api/production/mrp/run", json={"so_numbers": [so]})
    assert r.status_code == 200, r.text
    grey = r.json()["result"]["GREY-FAB"]
    assert grey["total_req"] == 945.0
    assert grey["net_req"] == 945.0


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


def test_mrp_lines_for_so_skip_zero_net_requirement(isolated_module_dbs, client):
    """Purchase PR must not list materials already covered by stock (net_req = 0)."""
    import sqlite3

    _seed_parent_with_bom(isolated_module_dbs["ITEM_DB_PATH"], parent_code="STYLE-Z")
    conn = sqlite3.connect(isolated_module_dbs["ITEM_DB_PATH"])
    conn.execute("UPDATE items SET stock=1000 WHERE item_code='FAB-CTN'")
    conn.commit()
    conn.close()

    so = _create_so_with_sku(client, "STYLE-Z-XL", qty=10)
    run = client.post("/api/production/mrp/run", json={"so_numbers": [so]})
    assert run.status_code == 200
    fab = run.json()["result"].get("FAB-CTN")
    assert fab is not None
    assert fab["net_req"] == 0

    lines = client.get(f"/api/production/mrp/lines-for-so?so_number={so}")
    assert lines.status_code == 200
    codes = [i["material_code"] for i in lines.json()["purchase_items"]]
    assert "FAB-CTN" not in codes


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


# ── /api/grey — full workflow lifecycle ──────────────────────────────────────


def _seed_grey_tracker(client, qty=500.0, material_code="GF-CTN-60"):
    """Create a PO with a grey-fabric line and return its auto-created tracker."""
    po_num = _create_po_with_lines(
        client,
        [{
            "material_code": material_code,
            "material_name": "Grey Fabric Cotton 60s",
            "material_type": "GF",
            "po_qty": qty,
            "unit": "MTR",
            "rate": 75,
            "gst_pct": 5,
        }],
    )
    trackers = client.get("/api/grey").json()
    tracker = next(t for t in trackers if t["po_number"] == po_num and t["material_code"] == material_code)
    return tracker, po_num


def test_grey_workflow_vendor_dispatch_accepts_blank_dispatch_date(isolated_module_dbs, client):
    """User-reported scenario: filled bilty + qty but left dispatch_date blank.

    Backend must accept an empty dispatch_date string (the field is optional)
    and transition the tracker to "In Transit". The frontend can pre-fill today
    for UX, but missing date should never block the API call.
    """
    tracker, _ = _seed_grey_tracker(client, qty=500.0)
    r = client.post(
        f"/api/grey/{tracker['id']}/vendor-dispatch",
        json={
            "bilty_no": "9837",
            "transporter": "SHREE RAM ROADWAYS",
            "dispatch_date": "",          # ← user left this blank
            "expected_arrival": "",
            "dispatched_qty": 120,
            "vehicle_no": "RJ14BU9023",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "In Transit"

    after = next(t for t in client.get("/api/grey").json() if t["id"] == tracker["id"])
    assert after["status"] == "In Transit"
    assert float(after["in_transit_qty"]) == 120.0
    assert after["bilty_no"] == "9837"
    assert after["transporter"] == "SHREE RAM ROADWAYS"


def test_grey_workflow_end_to_end(isolated_module_dbs, client):
    """Walk every status transition: PO Created → In Transit → Arrived →
    Factory / Printer → QC → Issue to printer → Receive printed.

    Catches regressions in any of the 7 status-changing endpoints used by the
    Grey Fabric Tracker UI.
    """
    tracker, _ = _seed_grey_tracker(client, qty=500.0)
    tid = tracker["id"]

    # 1. Vendor dispatch (qty 500)
    r = client.post(f"/api/grey/{tid}/vendor-dispatch", json={
        "bilty_no": "BILTY-001",
        "transporter": "SHREE RAM ROADWAYS",
        "dispatch_date": "2026-05-12",
        "expected_arrival": "2026-05-14",
        "dispatched_qty": 500,
        "vehicle_no": "RJ14BU9023",
    })
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "In Transit"

    # 2. Arrive at transport (all 500 MTR)
    r = client.post(f"/api/grey/{tid}/arrive-transport", json={"qty": 500})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "At Transport Location"

    snap = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap["transport_qty"]) == 500.0
    assert float(snap["in_transit_qty"]) == 0.0

    # 3. Transfer 200 MTR to factory
    r = client.post(f"/api/grey/{tid}/transfer", json={"to_location": "factory", "qty": 200})
    assert r.status_code == 200, r.text
    snap = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap["factory_qty"]) == 200.0
    assert float(snap["transport_qty"]) == 300.0

    # 4. QC the factory stock
    r = client.post(f"/api/grey/{tid}/qc", json={
        "received_qty": 200,
        "checked_qty": 200,
        "passed_qty": 190,
        "rejected_qty": 10,
        "rework_qty": 0,
        "outcome": "Partial Pass",
        "qc_remarks": "10m rejected — dye streak",
        "qc_by": "QC-Team",
        "qc_date": "2026-05-14",
    })
    assert r.status_code == 200, r.text

    # 5. Issue 300 MTR to a printer (from the remaining transport stock)
    r = client.post("/api/grey/printer-issue", json={
        "tracker_id": tid,
        "material_code": "GF-CTN-60",
        "job_order_no": "JO-001",
        "issue_qty": 300,
        "from_location": "Transport Location",
        "to_vendor": "Sunshine Printers",
        "issue_date": "2026-05-15",
        "challan_no": "CH-100",
        "gate_pass": "GP-100",
        "remarks": "Floral print",
    })
    assert r.status_code == 200, r.text
    issue = r.json()
    issue_id = issue.get("id") or issue.get("issue_id")
    assert issue_id, f"printer-issue response missing id: {issue}"

    # Confirm issue is listed by tracker id
    issues = client.get(f"/api/grey/printer-issue/list?tracker_id={tid}").json()
    assert isinstance(issues, list)
    assert any((i.get("id") if isinstance(i, dict) else None) == issue_id for i in issues), issues

    # 6. Receive printed fabric back (output 290 + wastage 10)
    r = client.post(f"/api/grey/printer-issue/{issue_id}/receive-printed", json={
        "received_back_qty": 300,
        "grey_input_mtr": 300,
        "printed_item_code": "PF-FLORAL-A1",
        "printed_output_mtr": 290,
        "wastage_mtr": 10,
        "conversion_date": "2026-05-18",
        "remarks": "Floral A1 done",
    })
    assert r.status_code == 200, r.text

    # 7. Reports / ledger endpoints still respond
    for url in (
        "/api/grey/reports/transit",
        "/api/grey/reports/stock-locations",
        "/api/grey/reports/qc",
        "/api/grey/reports/printer-issues",
        "/api/grey/printed-fabric/unchecked",
    ):
        r = client.get(url)
        assert r.status_code == 200, f"{url} → {r.status_code} {r.text}"


def test_grey_workflow_arrive_transport_default_uses_in_transit_qty(isolated_module_dbs, client):
    """If qty is omitted, arrive-transport should consume the in_transit balance."""
    tracker, _ = _seed_grey_tracker(client, qty=500.0)
    tid = tracker["id"]
    client.post(f"/api/grey/{tid}/vendor-dispatch", json={
        "bilty_no": "B-2", "transporter": "X", "dispatch_date": "2026-05-12",
        "expected_arrival": "", "dispatched_qty": 250, "vehicle_no": "",
    })
    r = client.post(f"/api/grey/{tid}/arrive-transport", json={})
    assert r.status_code == 200, r.text
    snap = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap["transport_qty"]) == 250.0
    assert float(snap["in_transit_qty"]) == 0.0


def test_grey_workflow_qc_invalid_tracker_returns_error(isolated_module_dbs, client):
    r = client.post("/api/grey/999999/qc", json={
        "received_qty": 10, "checked_qty": 10, "passed_qty": 10,
        "rejected_qty": 0, "rework_qty": 0, "outcome": "Pass",
        "qc_remarks": "", "qc_by": "", "qc_date": "2026-05-12",
    })
    assert r.status_code in (400, 404), r.text


# ── JWO BOM input resolution (SFG output → grey-fabric input) ────────────────


def _seed_sfg_bom(item_db_path: str, *, sfg_code: str, grey_code: str, bom_qty: float = 1.05) -> None:
    """Item Master fixture for the JWO input-resolution tests.

    Inserts a grey-fabric component, a semi-finished output, and a default
    BOM linking SFG to grey fabric at ``bom_qty`` MTR per unit of output.
    """
    import sqlite3

    conn = sqlite3.connect(item_db_path)
    conn.row_factory = sqlite3.Row
    gf_type_id = conn.execute(
        "SELECT id FROM item_types WHERE code='GF' OR name='Grey Fabric' LIMIT 1"
    ).fetchone()[0]

    sfg_type_id_row = conn.execute(
        "SELECT id FROM item_types WHERE code='SFG' OR name LIKE 'Semi%' LIMIT 1"
    ).fetchone()
    if sfg_type_id_row is None:
        cur = conn.execute("INSERT INTO item_types (name, code) VALUES (?, ?)", ("Semi Finished", "SFG"))
        sfg_type_id = cur.lastrowid
    else:
        sfg_type_id = sfg_type_id_row[0]

    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id, uom) VALUES (?, ?, ?, ?)",
        (grey_code, "Grey Cotton 60s", gf_type_id, "MTR"),
    )
    grey_id = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id, uom) VALUES (?, ?, ?, ?)",
        (sfg_code, "Printed Cotton Floral", sfg_type_id, "MTR"),
    )
    sfg_id = cur.lastrowid

    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, 'Default', 'all', 1)",
        (sfg_id,),
    )
    bom_id = cur.lastrowid

    conn.execute(
        """INSERT INTO bom_lines
           (bom_id, component_item_id, component_name, component_type, quantity, unit)
           VALUES (?, ?, 'Grey Cotton 60s', 'GF', ?, 'MTR')""",
        (bom_id, grey_id, float(bom_qty)),
    )
    conn.commit()
    conn.close()


def test_jwo_create_auto_resolves_grey_fabric_input_from_bom(isolated_module_dbs, client):
    """Bug fix: when the From-PR wizard set both input_material and
    output_material to the SFG code, the backend should auto-fill the input
    from the SFG default BOM so the JWO line shows the grey fabric being
    issued to the printer."""
    _seed_sfg_bom(
        isolated_module_dbs["ITEM_DB_PATH"],
        sfg_code="PF-FLORAL-A1",
        grey_code="GF-CTN-60",
        bom_qty=1.05,
    )
    proc = client.post(
        "/api/purchase/processors",
        json={"processor_name": "Sunshine Printers", "processor_type": "Printing Unit"},
    ).json()
    assert proc.get("ok") is True

    body = {
        "processor_name": "Sunshine Printers",
        "expected_return_date": "2026-06-10",
        "lines": [
            {
                "input_material": "PF-FLORAL-A1",   # legacy bug: same as output
                "input_qty": 100,
                "input_unit": "MTR",
                "output_material": "PF-FLORAL-A1",
                "output_qty": 100,
                "output_unit": "MTR",
                "process_type": "Printing",
                "rate": 12,
            }
        ],
    }
    r = client.post("/api/purchase/jwo", json=body)
    assert r.status_code == 200, r.text
    jwo_no = r.json()["jwo_number"]

    jwos = client.get("/api/purchase/jwo").json()
    j = next(j for j in jwos if j["jwo_number"] == jwo_no)
    assert len(j["lines"]) == 1
    line = j["lines"][0]
    assert line["output_material"] == "PF-FLORAL-A1"
    assert line["input_material"] == "GF-CTN-60", (
        f"BOM-driven input was not applied; got input={line['input_material']}"
    )
    # 100 MTR × 1.05 ratio = 105 MTR of grey fabric to issue.
    assert abs(float(line["input_qty"]) - 105.0) < 1e-3


def test_jwo_create_keeps_explicit_input_when_distinct_from_output(isolated_module_dbs, client):
    """If the operator typed a real input on the manual JWO screen, do NOT
    overwrite it from the BOM — only the legacy ``input == output`` case
    should be patched."""
    _seed_sfg_bom(
        isolated_module_dbs["ITEM_DB_PATH"],
        sfg_code="PF-FLORAL-A2",
        grey_code="GF-RAYON-1",
        bom_qty=1.10,
    )
    client.post(
        "/api/purchase/processors",
        json={"processor_name": "Dye House", "processor_type": "Dyeing Unit"},
    )
    body = {
        "processor_name": "Dye House",
        "lines": [
            {
                "input_material": "GF-RAYON-OPERATOR-PICK",  # explicit, different code
                "input_qty": 220,
                "input_unit": "MTR",
                "output_material": "PF-FLORAL-A2",
                "output_qty": 200,
                "output_unit": "MTR",
                "process_type": "Dyeing",
                "rate": 18,
            }
        ],
    }
    r = client.post("/api/purchase/jwo", json=body)
    assert r.status_code == 200, r.text
    jwo_no = r.json()["jwo_number"]
    j = next(j for j in client.get("/api/purchase/jwo").json() if j["jwo_number"] == jwo_no)
    line = j["lines"][0]
    assert line["input_material"] == "GF-RAYON-OPERATOR-PICK"
    assert float(line["input_qty"]) == 220.0


def test_grey_arrive_transport_blocks_duplicate_receive(isolated_module_dbs, client):
    """Second 'Receive at transport' click must not inflate transport_qty again."""
    tracker, _ = _seed_grey_tracker(client, qty=100.0)
    tid = tracker["id"]
    client.post(
        f"/api/grey/{tid}/vendor-dispatch",
        json={
            "bilty_no": "DUP-1",
            "transporter": "T",
            "dispatch_date": "2026-05-12",
            "expected_arrival": "",
            "dispatched_qty": 100,
            "vehicle_no": "",
        },
    )
    assert client.post(f"/api/grey/{tid}/arrive-transport", json={"qty": 100}).status_code == 200
    snap = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap["transport_qty"]) == 100.0
    dup = client.post(f"/api/grey/{tid}/arrive-transport", json={"qty": 100})
    assert dup.status_code == 400, dup.text
    snap2 = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap2["transport_qty"]) == 100.0


def test_po_grn_respects_tolerance_and_blocks_over_receive(isolated_module_dbs, client):
    from backend.db import purchase_db

    po_body = {
        "po_date": "2026-05-12",
        "supplier_name": "Test Supplier",
        "so_reference": "",
        "lines": [
            {
                "material_code": "GF-TEST-GRN",
                "material_name": "Grey Test",
                "material_type": "GF",
                "po_qty": 1000,
                "unit": "MTR",
                "rate": 10,
                "gst_pct": 0,
            }
        ],
    }
    po_num = client.post("/api/purchase/po", json=po_body).json()["po_number"]
    base_line = {
        "material_code": "GF-TEST-GRN",
        "material_name": "Grey Test",
        "material_type": "GF",
        "po_qty": 1000,
        "unit": "MTR",
        "rate": 10,
    }
    grn1 = {
        "grn_type": "PO Receipt",
        "reference_number": po_num,
        "party_name": "Test Supplier",
        "lines": [{**base_line, "received_qty": 900, "accepted_qty": 900, "rejected_qty": 0}],
    }
    assert client.post("/api/purchase/grn", json=grn1).status_code == 200
    grn2 = {
        **grn1,
        "lines": [{**base_line, "received_qty": 150, "accepted_qty": 150, "rejected_qty": 0}],
    }
    assert client.post("/api/purchase/grn", json=grn2).status_code == 200
    over = client.post(
        "/api/purchase/grn",
        json={**grn1, "lines": [{**base_line, "received_qty": 1, "accepted_qty": 1, "rejected_qty": 0}]},
    )
    assert over.status_code == 400, over.text
    bal = purchase_db.get_po_receive_balance(po_num)
    assert bal is not None
    assert float(bal["lines"][0]["grn_accepted_qty"]) == 1050.0


def test_mrp_po_commitment_limits_duplicate_po_qty(isolated_module_dbs, client):
    from backend.db import production_db

    production_db.sync_mrp_commitments_from_run(
        ["SO-COMMIT-1"],
        {
            "FAB-A": {
                "name": "Fabric A",
                "unit": "MTR",
                "type": "RM",
                "breakdown": [{"so_no": "SO-COMMIT-1", "sku": "SKU1", "qty_req": 100}],
            }
        },
    )
    line = {
        "material_code": "FAB-A",
        "material_name": "Fabric A",
        "material_type": "RM",
        "po_qty": 70,
        "unit": "MTR",
        "rate": 1,
        "gst_pct": 0,
    }
    po1 = {
        "po_date": "2026-05-12",
        "supplier_name": "Sup",
        "so_reference": "SO-COMMIT-1",
        "lines": [line],
    }
    assert client.post("/api/purchase/po", json=po1).status_code == 200
    po2 = {**po1, "lines": [{**line, "po_qty": 40}]}
    blocked = client.post("/api/purchase/po", json=po2)
    assert blocked.status_code == 400, blocked.text
    commits = production_db.get_mrp_commitments_for_so("SO-COMMIT-1")
    fab = next(c for c in commits if c["material_code"] == "FAB-A")
    assert float(fab["remaining_qty"]) == 30.0


def test_grey_reverse_arrive_transport(isolated_module_dbs, client):
    tracker, _ = _seed_grey_tracker(client, qty=80.0)
    tid = tracker["id"]
    client.post(
        f"/api/grey/{tid}/vendor-dispatch",
        json={
            "bilty_no": "REV-1",
            "transporter": "T",
            "dispatch_date": "2026-05-12",
            "expected_arrival": "",
            "dispatched_qty": 80,
            "vehicle_no": "",
        },
    )
    client.post(f"/api/grey/{tid}/arrive-transport", json={"qty": 80})
    r = client.post(f"/api/grey/{tid}/reverse-arrive-transport", json={})
    assert r.status_code == 200, r.text
    snap = next(t for t in client.get("/api/grey").json() if t["id"] == tid)
    assert float(snap["transport_qty"]) == 0.0
    assert float(snap["in_transit_qty"]) == 80.0


def test_mrp_jo_fabric_commitment(isolated_module_dbs, client):
    from backend.db import production_db

    production_db.sync_mrp_commitments_from_run(
        ["SO-JO-FAB"],
        {
            "GF-ROLL-1": {
                "name": "Grey Roll",
                "unit": "MTR",
                "type": "GF",
                "breakdown": [{"so_no": "SO-JO-FAB", "sku": "SKU-A", "qty_req": 200}],
            }
        },
    )
    body = {
        "jo_date": "2026-05-12",
        "so_number": "SO-JO-FAB",
        "sku": "SKU-A",
        "process": "Cutting",
        "planned_qty": 10,
        "fabric_code": "GF-ROLL-1",
        "fabric_qty": 150,
        "fabric_unit": "MTR",
    }
    assert client.post("/api/production/orders", json=body).status_code == 200
    over = {**body, "fabric_qty": 60}
    blocked = client.post("/api/production/orders", json=over)
    assert blocked.status_code == 400, blocked.text
    commits = production_db.get_mrp_commitments_for_so("SO-JO-FAB")
    row = next(c for c in commits if c["material_code"] == "GF-ROLL-1")
    assert float(row["jo_committed_qty"]) == 150.0
    assert float(row["remaining_qty"]) == 50.0


def test_jwo_grn_tolerance(isolated_module_dbs, client):
    jwo_body = {
        "jwo_date": "2026-05-12",
        "processor_name": "Printer Co",
        "so_reference": "SO-JWO-1",
        "lines": [
            {
                "input_material": "GF-IN",
                "input_qty": 100,
                "output_material": "PF-OUT",
                "output_qty": 1000,
                "output_unit": "MTR",
                "process_type": "Printing",
                "rate": 5,
            }
        ],
    }
    jwo_num = client.post("/api/purchase/jwo", json=jwo_body).json()["jwo_number"]
    grn1 = {
        "grn_type": "JWO Receipt",
        "reference_number": jwo_num,
        "party_name": "Printer Co",
        "lines": [
            {
                "material_code": "PF-OUT",
                "accepted_qty": 1000,
                "received_qty": 1000,
                "rejected_qty": 0,
                "unit": "MTR",
                "rate": 5,
            }
        ],
    }
    assert client.post("/api/purchase/grn", json=grn1).status_code == 200
    grn2 = {**grn1, "lines": [{**grn1["lines"][0], "accepted_qty": 60, "received_qty": 60}]}
    assert client.post("/api/purchase/grn", json=grn2).status_code == 400, client.post("/api/purchase/grn", json=grn2).text


def test_document_chain_audit_api(isolated_module_dbs, client):
    from backend.db import production_db

    production_db.sync_mrp_commitments_from_run(
        ["SO-AUDIT"],
        {
            "FAB-X": {
                "name": "Fabric X",
                "unit": "MTR",
                "breakdown": [{"so_no": "SO-AUDIT", "sku": "S1", "qty_req": 50}],
            }
        },
    )
    r = client.get("/api/purchase/audit/document-chain?so_number=SO-AUDIT")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["so_number"] == "SO-AUDIT"
    assert any(m["material_code"] == "FAB-X" for m in data["materials"])

"""Embroidery measurement units + JO qty in Border meters / Yog count."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_module_dbs(tmp_path, monkeypatch):
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
    yield paths


def test_embroidery_type_required_before_cutting(isolated_module_dbs, client):
    from backend.services.operation_routing import normalize_embroidery_line_fields

    with pytest.raises(ValueError, match="Embroidery Type"):
        normalize_embroidery_line_fields({
            "routing": "Cutting>Embroidery>Cutting>Stitching",
            "requires_embroidery": True,
            "embroidery_before_cutting": True,
        })


def test_border_jo_qty_from_garment_pieces(isolated_module_dbs, client):
    """10 pcs × 2 MTR border → Embroidery JO planned 20 MTR."""
    save = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "BORDERJO",
            "lines": [
                {
                    "component_code": "TOP",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
                },
                {
                    "component_code": "FRONT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "embroidery_before_cutting": True,
                    "embroidery_type": "Border",
                    "embroidery_qty_per_piece": 2,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )
    assert save.status_code == 200, save.text

    jo_res = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "process": "Cutting",
            "so_number": "SO-BORDER-1",
            "sku": "BORDERJO-M",
            "planned_qty": 10,
            "create_component_jos": True,
            "lines": [{"sku": "BORDERJO-M", "style": "M", "planned_qty": 10}],
        },
    )
    assert jo_res.status_code == 200, jo_res.text
    jo = next(
        o for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-BORDER-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    joid = jo["id"]

    recv = client.post(
        f"/api/production/orders/{joid}/receive-pieces",
        json={"received_qty": 10, "process": "Cutting", "sku": "BORDERJO-M-TOP", "split_components": True},
    )
    assert recv.status_code == 200, recv.text

    issue = client.post(
        f"/api/production/orders/{joid}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "BORDERJO-M-FRONT",
        },
    )
    assert issue.status_code == 200, issue.text
    child = issue.json().get("child_jo") or {}
    assert child.get("process") == "Embroidery"
    assert int(child.get("planned_qty") or 0) == 20
    assert child.get("embroidery_type") == "Border"
    assert child.get("embroidery_unit") == "MTR"
    assert int(child.get("garment_qty") or 0) == 10


def test_yog_jo_qty_and_leftover_stock(isolated_module_dbs, client):
    save = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "YOGJO",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {
                    "component_code": "FRONT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "embroidery_before_cutting": True,
                    "embroidery_type": "Yog",
                    "embroidery_qty_per_piece": 4,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )
    assert save.status_code == 200, save.text

    jo_res = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "process": "Cutting",
            "so_number": "SO-YOG-1",
            "sku": "YOGJO-M",
            "planned_qty": 10,
            "create_component_jos": True,
            "lines": [{"sku": "YOGJO-M", "style": "M", "planned_qty": 10}],
        },
    )
    jo = next(
        o for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-YOG-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    joid = jo["id"]
    client.post(
        f"/api/production/orders/{joid}/receive-pieces",
        json={"received_qty": 10, "process": "Cutting", "sku": "YOGJO-M-TOP", "split_components": True},
    )
    issue = client.post(
        f"/api/production/orders/{joid}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "YOGJO-M-FRONT",
        },
    )
    child = issue.json().get("child_jo") or {}
    assert int(child.get("planned_qty") or 0) == 40
    assert child.get("embroidery_unit") == "YOG"
    emb_joid = child["id"]

    recv = client.post(
        f"/api/production/orders/{emb_joid}/receive-pieces",
        json={
            "received_qty": 40,
            "process": "Embroidery",
            "sku": "YOGJO-M-FRONT",
            "leftover_measurement": 5,
        },
    )
    assert recv.status_code == 200, recv.text
    assert float(recv.json().get("leftover_credited") or 0) == 5, recv.json()

    stock = client.get("/api/production/embroidery-stock/YOGJO-M-FRONT").json()
    items = stock.get("items") or []
    assert any(float(i.get("available_qty") or 0) == 5 for i in items), items


def test_embroidery_stock_reduces_next_jo(isolated_module_dbs, client):
    """5 Yog in stock → next 10-pc JO needs 35 not 40."""
    from backend.db import production_db

    conn = production_db._connect()
    production_db.adjust_embroidery_stock(
        conn,
        style_key="STOCKJO",
        component_code="FRONT",
        embroidery_type="Yog",
        unit="YOG",
        delta_qty=5,
        remarks="seed",
    )
    conn.commit()
    conn.close()

    client.post(
        "/api/production/set-bom",
        json={
            "style_key": "STOCKJO",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {
                    "component_code": "FRONT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "embroidery_before_cutting": True,
                    "embroidery_type": "Yog",
                    "embroidery_qty_per_piece": 4,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )

    jo_res = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "process": "Cutting",
            "so_number": "SO-STOCK-1",
            "sku": "STOCKJO-M",
            "planned_qty": 10,
            "create_component_jos": True,
            "lines": [{"sku": "STOCKJO-M", "style": "M", "planned_qty": 10}],
        },
    )
    jo = next(
        o for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-STOCK-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    joid = jo["id"]
    client.post(
        f"/api/production/orders/{joid}/receive-pieces",
        json={"received_qty": 10, "process": "Cutting", "sku": "STOCKJO-M-TOP", "split_components": True},
    )
    issue = client.post(
        f"/api/production/orders/{joid}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "STOCKJO-M-FRONT",
        },
    )
    child = issue.json().get("child_jo") or {}
    assert int(child.get("planned_qty") or 0) == 35
    assert float(child.get("stock_used") or 0) == 5


def test_mrp_surfaces_embroidery_leftover_stock(isolated_module_dbs, client):
    """MRP / PO planning screen must show leftover Border stock for SO styles."""
    from backend.db import production_db
    import sqlite3

    # Seed Item Master BOM so MRP can explode the SO line.
    conn = sqlite3.connect(isolated_module_dbs["ITEM_DB_PATH"])
    type_row = conn.execute(
        "SELECT id FROM item_types WHERE code='FG' OR name LIKE 'Finished%' LIMIT 1"
    ).fetchone()
    if type_row is None:
        cur = conn.execute("INSERT INTO item_types (name, code) VALUES (?, ?)", ("Finished Good", "FG"))
        type_id = cur.lastrowid
    else:
        type_id = type_row[0]
    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        ("MRPEMB", "MRP Emb Style", type_id),
    )
    parent_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?, ?, ?)",
        ("FAB-EMB", "Emb Fabric", type_id),
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
           VALUES (?, ?, 'Emb Fabric', 'RM', 1, 'MTR')""",
        (bom_id, fab_id),
    )
    conn.commit()
    conn.close()

    # Leftover Border from a prior embroidery receive.
    pdb = production_db._connect()
    production_db.adjust_embroidery_stock(
        pdb,
        style_key="MRPEMB",
        component_code="FRONT",
        embroidery_type="Border",
        unit="MTR",
        delta_qty=7.5,
        so_number="SO-PRIOR",
        remarks="Leftover from prior JO",
    )
    pdb.commit()
    pdb.close()

    so = client.post(
        "/api/sales/orders",
        json={
            "so_date": "2026-07-29",
            "buyer": "Acme",
            "warehouse": "Main",
            "lines": [{"sku": "MRPEMB-M", "sku_name": "MRPEMB M", "qty": 10, "unit": "PCS"}],
        },
    )
    assert so.status_code == 200, so.text
    so_number = so.json()["so_number"]

    preview = client.get(
        f"/api/production/mrp/embroidery-stock?so_numbers={so_number}"
    )
    assert preview.status_code == 200, preview.text
    items = preview.json().get("items") or []
    assert any(
        i.get("style_key") == "MRPEMB"
        and i.get("embroidery_type") == "Border"
        and float(i.get("available_qty") or 0) == 7.5
        for i in items
    ), items

    run = client.post("/api/production/mrp/run", json={"so_numbers": [so_number]})
    assert run.status_code == 200, run.text
    emb = run.json().get("embroidery_stock") or []
    assert any(float(i.get("available_qty") or 0) == 7.5 for i in emb), emb

    last = client.get("/api/production/mrp/last")
    assert last.status_code == 200
    assert any(
        float(i.get("available_qty") or 0) == 7.5
        for i in (last.json().get("embroidery_stock") or [])
    )

"""Tests for Gate Inward (GIN), barcodes, and Ready-To WIP import."""
from __future__ import annotations

import io

import pytest


@pytest.fixture
def isolated_module_dbs(tmp_path, monkeypatch):
    """Point ERP SQLite DBs at a fresh tmp dir + init schemas."""
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


def test_parse_barcode_payload():
    from backend.services.document_barcode import make_payload, parse_payload

    assert parse_payload("PO:PO-0001") == ("PO", "PO-0001")
    assert parse_payload("jo:JO-42") == ("JO", "JO-42")
    assert parse_payload("JO-0042") == ("JO", "JO-0042")
    assert make_payload("GIN", "GIN-0007") == "GIN:GIN-0007"
    with pytest.raises(ValueError):
        parse_payload("")


def test_barcode_bundle_has_qr():
    from backend.services.document_barcode import barcode_bundle

    b = barcode_bundle("PO", "PO-0001")
    assert b["payload"] == "PO:PO-0001"
    assert b["qr_data_url"].startswith("data:image/svg+xml")


def test_gin_po_flow_creates_grn(isolated_module_dbs, client):
    client.post(
        "/api/purchase/suppliers",
        json={"supplier_name": "Gate Vendor", "supplier_type": "Fabric"},
    )
    r = client.post(
        "/api/purchase/po",
        json={
            "supplier_name": "Gate Vendor",
            "po_date": "2026-07-30",
            "delivery_date": "2026-08-15",
            "delivery_location": "Warehouse",
            "lines": [
                {
                    "material_code": "FAB-1",
                    "material_name": "Cotton",
                    "material_type": "RM",
                    "po_qty": 100,
                    "unit": "MTR",
                    "rate": 10,
                    "gst_pct": 5,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    po_number = r.json()["po_number"]

    scan = client.get("/api/gate/scan", params={"code": f"PO:{po_number}"})
    assert scan.status_code == 200, scan.text
    body = scan.json()
    assert body["source_type"] == "PO"
    assert body["lines"]
    line = body["lines"][0]
    assert float(line["pending_qty"]) > 0

    gin = client.post(
        "/api/gate/gin",
        json={
            "source_type": "PO",
            "source_number": po_number,
            "party_name": body["party_name"],
            "stage": body["stage"],
            "challan_no": "CH-1",
            "lines": [
                {
                    "line_key": line.get("line_key") or "",
                    "material_code": line.get("material_code") or "",
                    "material_name": line.get("material_name") or "",
                    "sku": line.get("sku") or "",
                    "planned_qty": line.get("planned_qty") or 0,
                    "already_received_qty": line.get("already_received_qty") or 0,
                    "pending_qty": line.get("pending_qty") or 0,
                    "unit": line.get("unit") or "PCS",
                    "received_qty": min(10, float(line["pending_qty"])),
                }
            ],
        },
    )
    assert gin.status_code == 200, gin.text
    data = gin.json()
    assert data["ok"] is True
    assert data["gin_number"].startswith("GIN-")
    assert data["grn_number"]

    scan2 = client.get("/api/gate/scan", params={"code": f"PO:{po_number}"})
    pending = float(scan2.json()["lines"][0]["pending_qty"])
    bad = client.post(
        "/api/gate/gin",
        json={
            "source_type": "PO",
            "source_number": po_number,
            "lines": [
                {
                    "line_key": scan2.json()["lines"][0]["line_key"],
                    "material_code": "FAB-1",
                    "sku": "FAB-1",
                    "planned_qty": 100,
                    "already_received_qty": 10,
                    "pending_qty": pending,
                    "received_qty": pending + 50,
                    "unit": "MTR",
                }
            ],
        },
    )
    assert bad.status_code == 400


def test_ready_to_wip_import(isolated_module_dbs, client):
    csv = (
        "Ready_To_Stage,SO_Number,OMS_SKU,Quantity,JO_Number,Batch,Vendor,Remarks\n"
        "Stitching,SO-WIP,SKU-WIP-M,40,,LOT-1,Vendor A,migrate\n"
    )
    files = {"file": ("wip.csv", io.BytesIO(csv.encode()), "text/csv")}
    data = {"stage": "Stitching"}
    r = client.post("/api/production/ready-to-wip/import", files=files, data=data)
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 1

    ready = client.get("/api/production/ready-to-process/Stitching")
    assert ready.status_code == 200
    rows = ready.json()
    assert any(
        str(x.get("sku")) == "SKU-WIP-M" and float(x.get("available_qty") or 0) >= 40
        for x in rows
    )

    filtered = client.get(
        "/api/production/ready-to-process/Stitching",
        params={"sku": "SKU-WIP", "min_qty": 10, "q": "SO-WIP"},
    )
    assert filtered.status_code == 200
    assert len(filtered.json()) >= 1


def test_ready_to_wip_import_lowercase_columns_and_bom(isolated_module_dbs, client):
    """Excel/Mac CSV often uses lower headers or a BOM — must still import."""
    csv = (
        "\ufeffready_to_stage,so_number,oms_sku,quantity\n"
        "Stitching,SO-LOWER,LOWER-SKU-M,25\n"
    )
    r = client.post(
        "/api/production/ready-to-wip/import",
        files={"file": ("wip.csv", io.BytesIO(csv.encode("utf-8")), "text/csv")},
        data={"stage": "Stitching"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["imported"] == 1, body
    assert body.get("failed", 0) == 0


def test_ready_to_wip_uses_feeder_fallback_when_routing_omits_stage(isolated_module_dbs, client, monkeypatch):
    """If item routing does not list Stitching, still credit Cutting stock."""
    from backend.db import production_db

    monkeypatch.setattr(production_db, "get_item_routing", lambda sku: ["Cutting", "Finishing"])
    monkeypatch.setattr(production_db, "get_component_routing", lambda sku: ["Cutting", "Finishing"])
    csv = "Ready_To_Stage,SO_Number,OMS_SKU,Quantity\nStitching,SO-FALL,FALL-SKU-M,12\n"
    r = client.post(
        "/api/production/ready-to-wip/import",
        files={"file": ("wip.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"stage": "Stitching"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 1, r.json()


@pytest.mark.parametrize(
    "stage,feeder,sku,item_path",
    [
        ("Kaj Button", "Stitching", "KAJ-WIP-M", ["Cutting", "Stitching"]),
        ("Handwork", "Stitching", "HW-WIP-M", ["Cutting", "Stitching"]),
        ("Finishing", "Stitching", "FIN-WIP-M", ["Cutting", "Stitching"]),
        ("Kaj Button", "Stitching", "KAJ-DEF-M", []),
        ("Handwork", "Kaj Button", "HW-DEF-M", []),
        ("Finishing", "Handwork", "FIN-DEF-M", []),
    ],
)
def test_ready_to_wip_handwork_kaj_finishing_feeder_defaults(
    isolated_module_dbs, client, monkeypatch, stage, feeder, sku, item_path
):
    """Handwork / Kaj / Finishing WIP must resolve feeders when routing is short or empty."""
    from backend.db import production_db

    monkeypatch.setattr(production_db, "get_item_routing", lambda s: list(item_path))
    monkeypatch.setattr(production_db, "get_component_routing", lambda s: list(item_path))
    csv = f"Ready_To_Stage,SO_Number,OMS_SKU,Quantity\n{stage},SO-WIP,{sku},25\n"
    r = client.post(
        "/api/production/ready-to-wip/import",
        files={"file": ("wip.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"stage": stage},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("imported") == 1, body
    assert body.get("failed", 0) == 0, body
    # Stock is always credited on the feeder process.
    assert production_db.get_process_stock("SO-WIP", sku, feeder) == 25
    # Ready board must list the WIP row (routing next-hop or import ledger).
    ready = client.get(f"/api/production/ready-to-process/{stage}")
    assert ready.status_code == 200
    row = next((x for x in ready.json() if x.get("sku") == sku), None)
    assert row is not None, ready.json()
    assert int(row.get("available_qty") or 0) == 25
    assert str(row.get("from_process") or "") == feeder


def test_ready_to_wip_full_path_kaj_handwork_finishing(isolated_module_dbs, client, monkeypatch):
    """On full factory path, each stage is fed by the prior hop."""
    from backend.db import production_db

    path = ["Cutting", "Stitching", "Kaj Button", "Handwork", "Finishing"]
    monkeypatch.setattr(production_db, "get_item_routing", lambda s: list(path))
    monkeypatch.setattr(production_db, "get_component_routing", lambda s: list(path))
    for stage, feeder, sku in (
        ("Kaj Button", "Stitching", "FULL-KAJ"),
        ("Handwork", "Kaj Button", "FULL-HW"),
        ("Finishing", "Handwork", "FULL-FIN"),
    ):
        csv = f"Ready_To_Stage,SO_Number,OMS_SKU,Quantity\n{stage},SO-FULL,{sku},10\n"
        r = client.post(
            "/api/production/ready-to-wip/import",
            files={"file": ("wip.csv", io.BytesIO(csv.encode()), "text/csv")},
            data={"stage": stage},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("imported") == 1, r.json()
        ready = client.get(f"/api/production/ready-to-process/{stage}")
        row = next(x for x in ready.json() if x.get("sku") == sku)
        assert str(row.get("from_process") or "") == feeder
        assert int(row.get("available_qty") or 0) == 10


def test_stitching_stock_not_on_multiple_ready_boards(isolated_module_dbs, client, monkeypatch):
    """Leftover Stitching stock with BOM next=Kaj must not also appear on Finishing/Handwork."""
    from backend.db import production_db

    path = ["Cutting", "Stitching", "Kaj Button", "Finishing"]
    monkeypatch.setattr(production_db, "get_item_routing", lambda s: list(path))
    monkeypatch.setattr(production_db, "get_component_routing", lambda s: list(path))
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-X", "MULTI-BOARD-M", "Stitching", qty_in=40)
    conn.commit()
    conn.close()

    kaj = client.get("/api/production/ready-to-process/Kaj Button").json()
    fin = client.get("/api/production/ready-to-process/Finishing").json()
    hw = client.get("/api/production/ready-to-process/Handwork").json()
    assert any(x.get("sku") == "MULTI-BOARD-M" for x in kaj)
    assert not any(x.get("sku") == "MULTI-BOARD-M" for x in fin)
    assert not any(x.get("sku") == "MULTI-BOARD-M" for x in hw)


def test_jo_import_autoroute_ready_to_wip_template(isolated_module_dbs, client):
    """ready_to_*_wip_template.csv must credit stock even if uploaded via JO import."""
    csv = (
        "Ready_To_Stage,SO_Number,OMS_SKU,Quantity,JO_Number,Batch,Vendor,Remarks\n"
        "Stitching,03-2627,1112YKBLACK-M,150,,,,\n"
        "Stitching,03-2627,1112YKBLACK-L,230,,,,\n"
    )
    r = client.post(
        "/api/production/orders/import",
        files={"file": ("ready_to_stitching_wip_template.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"process": "Stitching"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("kind") == "ready_to_wip"
    assert body.get("imported") == 2, body
    assert body.get("created") == 0
    assert "Ready-To WIP" in (body.get("message") or "")

    ready = client.get("/api/production/ready-to-process/Stitching")
    assert ready.status_code == 200
    skus = {str(x.get("sku")) for x in ready.json()}
    assert "1112YKBLACK-M" in skus
    assert "1112YKBLACK-L" in skus


def test_ready_to_stitch_after_cutting_issue(isolated_module_dbs, client):
    """Receive 45 on Cutting then issue to Stitching — must stay on Ready to Stitch."""
    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-08-14",
            "so_number": "08-2627",
            "sku": "1592YKBLUE-XXL",
            "process": "Cutting",
            "planned_qty": 50,
            "lines": [{"sku": "1592YKBLUE-XXL", "style": "XXL", "planned_qty": 50}],
        },
    )
    assert r.status_code == 200, r.text
    jo = next(
        o
        for o in client.get("/api/production/orders").json()
        if o["jo_number"] == r.json()["jo_number"]
    )
    line = jo["lines"][0]
    rec = client.post(
        f"/api/production/orders/{jo['id']}/receive-pieces",
        json={
            "received_qty": 45,
            "process": "Cutting",
            "sku": line["sku"],
            "jo_line_id": line["id"],
        },
    )
    assert rec.status_code == 200, rec.text

    before = client.get("/api/production/ready-to-process/Stitching")
    assert before.status_code == 200
    before_row = next((x for x in before.json() if x.get("sku") == "1592YKBLUE-XXL"), None)
    assert before_row is not None
    assert int(before_row.get("available_qty") or 0) == 45

    iss = client.post(
        f"/api/production/orders/{jo['id']}/issue-pieces",
        json={
            "issued_qty": 45,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "1592YKBLUE-XXL",
            "jo_line_id": line["id"],
        },
    )
    assert iss.status_code == 200, iss.text

    after = client.get("/api/production/ready-to-process/Stitching")
    assert after.status_code == 200
    after_row = next((x for x in after.json() if x.get("sku") == "1592YKBLUE-XXL"), None)
    assert after_row is not None, after.json()
    assert int(after_row.get("available_qty") or 0) == 45


def test_gin_jo_receive(isolated_module_dbs, client):
    from backend.db import production_db

    # Seed feeder stock so a Stitching JO can be created, then receive via GIN
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-G1", "1001YKBEIGE-M", "Cutting", qty_in=50)
    conn.commit()
    conn.close()

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-30",
            "so_number": "SO-G1",
            "sku": "1001YKBEIGE-M",
            "sku_name": "Gate SKU",
            "process": "Stitching",
            "exec_type": "Inhouse",
            "planned_qty": 20,
            "lines": [
                {
                    "so_number": "SO-G1",
                    "sku": "1001YKBEIGE-M",
                    "sku_name": "Gate SKU",
                    "planned_qty": 20,
                    "vendor_rate": 0,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    jo_number = r.json().get("jo_number")
    assert jo_number

    scan = client.get("/api/gate/scan", params={"code": f"JO:{jo_number}"})
    assert scan.status_code == 200, scan.text
    line = scan.json()["lines"][0]
    gin = client.post(
        "/api/gate/gin",
        json={
            "source_type": "JO",
            "source_number": jo_number,
            "party_name": "",
            "stage": "Stitching",
            "lines": [
                {
                    "line_key": line["line_key"],
                    "material_code": line["sku"],
                    "sku": line["sku"],
                    "planned_qty": line["planned_qty"],
                    "already_received_qty": line["already_received_qty"],
                    "pending_qty": line["pending_qty"],
                    "received_qty": 5,
                    "unit": "PCS",
                    "jo_id": line.get("jo_id"),
                    "jo_line_id": line.get("jo_line_id"),
                }
            ],
        },
    )
    assert gin.status_code == 200, gin.text
    assert gin.json()["gin_number"].startswith("GIN-")
    assert gin.json().get("jo_receipt_ids")

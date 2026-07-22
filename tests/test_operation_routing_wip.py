"""Operation-based routing with partial WIP — embroidery child ops + stitching bundle gate."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_module_dbs(tmp_path, monkeypatch):
    """Point module SQLite DBs at a fresh tmp dir (same as production smoke tests)."""
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


def test_operation_routing_helpers():
    from backend.services.operation_routing import (
        embroidery_is_child_of_cutting,
        next_process_in_path,
        parse_routing_path,
        resolve_component_routing,
        compute_bundle_readiness,
    )

    path = parse_routing_path("Cutting>Embroidery>Cutting>Stitching")
    assert path == ["Cutting", "Embroidery", "Cutting", "Stitching"]
    assert embroidery_is_child_of_cutting(path)
    assert next_process_in_path(path, "Cutting") == "Embroidery"
    assert next_process_in_path(path, "Embroidery") == "Cutting"
    assert next_process_in_path(path, "Cutting", after_process="Embroidery") == "Stitching"

    # Prefer explicit routing over default_next
    resolved = resolve_component_routing(
        routing="Cutting>Embroidery>Cutting>Stitching",
        default_next_process="Stitching",
    )
    assert resolved[1] == "Embroidery"

    ready = compute_bundle_readiness(
        [
            {
                "component_code": "FRONT",
                "qty_per_set": 1,
                "available_at_gate": 10,
                "embroidery_outstanding": 0,
                "routing": path,
                "location": "Cutting",
            },
            {
                "component_code": "BACK",
                "qty_per_set": 1,
                "available_at_gate": 10,
                "embroidery_outstanding": 0,
                "routing": ["Cutting", "Stitching"],
                "location": "Cutting",
            },
        ]
    )
    assert ready["bundle_complete"] is True
    assert ready["complete_sets"] == 10

    blocked = compute_bundle_readiness(
        [
            {
                "component_code": "FRONT",
                "qty_per_set": 1,
                "available_at_gate": 0,
                "embroidery_outstanding": 10,
                "routing": path,
                "location": "Embroidery",
            },
            {
                "component_code": "BACK",
                "qty_per_set": 1,
                "available_at_gate": 10,
                "embroidery_outstanding": 0,
                "routing": ["Cutting", "Stitching"],
                "location": "Cutting",
            },
        ]
    )
    assert blocked["bundle_complete"] is False
    assert any("Embroidery" in b for b in blocked["blockers"])


def test_embroidery_after_cutting_partial_wip_and_stitch_gate(isolated_module_dbs, client):
    """Scenario 2: Front → Embroidery; Back stays Cutting; Stitching blocked until bundle complete."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "PANELSET",
            "style_name": "Panel Set",
            "stitching_requires_complete_set": True,
            "bundle_gate_process": "Cutting",
            "lines": [
                {
                    "component_code": "FRONT",
                    "component_name": "Front",
                    "qty_per_set": 1,
                    "default_next_process": "Embroidery",
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                },
                {
                    "component_code": "BACK",
                    "component_name": "Back",
                    "qty_per_set": 1,
                    "default_next_process": "Stitching",
                    "routing": "Cutting>Stitching",
                },
                {
                    "component_code": "SLEEVE",
                    "component_name": "Sleeve",
                    "qty_per_set": 2,
                    "default_next_process": "Stitching",
                    "routing": "Cutting>Stitching",
                },
            ],
        },
    )
    assert bom.status_code == 200, bom.text
    assert bom.json()["stitching_requires_complete_set"] in (1, True)

    route = client.get("/api/production/item-routing/PANELSET-XS-FRONT").json()
    assert route["routing"][:3] == ["Cutting", "Embroidery", "Cutting"]
    assert route["next_after_cutting"] == "Embroidery"

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "so_number": "SO-PANEL-1",
            "sku": "PANELSET-XS",
            "process": "Cutting",
            "planned_qty": 10,
            "create_component_jos": False,
            "lines": [{"sku": "PANELSET-XS", "style": "XS", "planned_qty": 10}],
        },
    )
    assert r.status_code == 200, r.text
    jo = next(o for o in client.get("/api/production/orders").json() if o["jo_number"] == r.json()["jo_number"])
    line = jo["lines"][0]

    rec = client.post(
        f"/api/production/orders/{jo['id']}/receive-pieces",
        json={
            "received_qty": 10,
            "process": "Cutting",
            "sku": "PANELSET-XS",
            "jo_line_id": line["id"],
            "split_components": True,
        },
    )
    assert rec.status_code == 200, rec.text

    # Partial issue: only Front to Embroidery
    iss_front = client.post(
        f"/api/production/orders/{jo['id']}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "PANELSET-XS-FRONT",
        },
    )
    assert iss_front.status_code == 200, iss_front.text

    board = client.get(
        "/api/production/wip-board",
        params={"so_number": "SO-PANEL-1", "main_sku": "PANELSET-XS"},
    ).json()
    by_code = {i["component_code"]: i for i in board["items"]}
    assert by_code["FRONT"]["current_location"] == "Embroidery"
    assert by_code["FRONT"]["status"] == "In Process"
    assert by_code["BACK"]["current_location"] == "Cutting"
    assert board["bundle_complete"] is False

    # Stitching blocked while Front is under embroidery
    blocked = client.post(
        f"/api/production/orders/{jo['id']}/issue-pieces",
        json={
            "issued_qty": 1,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "PANELSET-XS-BACK",
        },
    )
    assert blocked.status_code == 400
    assert "Stitching blocked" in blocked.json()["detail"]

    # Return embroidered Front to Cutting (child op complete)
    ret = client.post(
        f"/api/production/orders/{jo['id']}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Embroidery",
            "to_process": "Cutting",
            "sku": "PANELSET-XS-FRONT",
        },
    )
    assert ret.status_code == 200, ret.text

    ready = client.get(
        "/api/production/bundle-ready",
        params={"so_number": "SO-PANEL-1", "main_sku": "PANELSET-XS"},
    ).json()
    assert ready["bundle_complete"] is True
    assert ready["complete_sets"] == 10

    # Now Back may proceed to Stitching
    ok_stitch = client.post(
        f"/api/production/orders/{jo['id']}/issue-pieces",
        json={
            "issued_qty": 10,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "PANELSET-XS-BACK",
        },
    )
    assert ok_stitch.status_code == 200, ok_stitch.text


def test_embroidery_before_cutting_routing_resolution(isolated_module_dbs, client):
    """Scenario 1 route shape: Embroidery before final Cutting hop for Front fabric/panel."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "EMBBEFORE",
            "lines": [
                {
                    "component_code": "FRONT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                },
            ],
        },
    )
    assert bom.status_code == 200, bom.text
    from backend.db.production_db import get_next_process, get_component_routing

    assert get_component_routing("EMBBEFORE-M-FRONT") == [
        "Cutting",
        "Embroidery",
        "Cutting",
        "Stitching",
    ]
    assert get_next_process("EMBBEFORE-M-FRONT", "Cutting") == "Embroidery"
    assert get_next_process("EMBBEFORE-M-FRONT", "Embroidery") == "Cutting"
    assert get_next_process("EMBBEFORE-M-BACK", "Cutting") == "Stitching"

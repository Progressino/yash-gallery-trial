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
                    "component_code": "TOP",
                    "component_name": "Top",
                    "qty_per_set": 1,
                    "default_next_process": "Stitching",
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
                },
                {
                    "component_code": "FRONT",
                    "component_name": "Front",
                    "qty_per_set": 1,
                    "default_next_process": "Embroidery",
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "component_name": "Back",
                    "qty_per_set": 1,
                    "default_next_process": "Stitching",
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BOTTOM",
                    "component_name": "Bottom",
                    "qty_per_set": 1,
                    "default_next_process": "Stitching",
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
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
                    "embroidery_qty_per_piece": 1,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
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

    route_api = client.get("/api/production/item-routing/EMBBEFORE-M-FRONT").json()
    assert route_api["embroidery_before_cutting"] is True
    assert route_api["requires_embroidery"] is True


def test_set_bom_embroidery_before_cutting_persisted(isolated_module_dbs, client):
    """embroidery_before_cutting flag is saved and returned on Set BOM lines."""
    save = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "EMBTIMING",
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
                    "embroidery_qty_per_piece": 1,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )
    assert save.status_code == 200, save.text
    bom = client.get("/api/production/set-bom/EMBTIMING").json()
    front = next(ln for ln in bom["lines"] if ln["component_code"] == "FRONT")
    assert front["embroidery_before_cutting"] in (1, True)
    assert front["requires_embroidery"] in (1, True)

    after = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "EMBTIMING",
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
                    "embroidery_before_cutting": False,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )
    assert after.status_code == 200, after.text
    bom2 = client.get("/api/production/set-bom/EMBTIMING").json()
    front2 = next(ln for ln in bom2["lines"] if ln["component_code"] == "FRONT")
    assert front2["embroidery_before_cutting"] in (0, False)


def test_embroidery_issue_label_helpers():
    from backend.services.operation_routing import embroidery_issue_label, embroidery_timing_label

    assert embroidery_timing_label(before_cutting=True) == "Before cutting (fabric)"
    assert embroidery_timing_label(before_cutting=False) == "After cutting (panel)"
    assert embroidery_issue_label(issue_from="Cutting", issue_to="Embroidery", before_cutting=True) == "Embroidery (fabric)"
    assert embroidery_issue_label(issue_from="Cutting", issue_to="Embroidery", before_cutting=False) == "Embroidery (panel)"
    assert embroidery_issue_label(issue_from="Cutting", issue_to="Stitching", before_cutting=True) == "Stitching"


def test_parent_component_jo_receive_explodes_panel_stock(isolated_module_dbs, client):
    """Default flow: TOP Cutting JO receive creates FRONT/BACK panel stock under parent."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "COMPJOSET",
            "stitching_requires_complete_set": True,
            "bundle_gate_process": "Cutting",
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
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BOTTOM",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
                },
            ],
        },
    )
    assert bom.status_code == 200, bom.text

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "so_number": "SO-COMPJO-1",
            "sku": "COMPJOSET-M",
            "process": "Cutting",
            "planned_qty": 4,
            "create_component_jos": True,
            "lines": [{"sku": "COMPJOSET-M", "style": "M", "planned_qty": 4}],
        },
    )
    assert r.status_code == 200, r.text
    orders = client.get("/api/production/orders").json()
    top_jo = next(
        o
        for o in orders
        if o.get("so_number") == "SO-COMPJO-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    line = top_jo["lines"][0]

    rec = client.post(
        f"/api/production/orders/{top_jo['id']}/receive-pieces",
        json={
            "received_qty": 4,
            "process": "Cutting",
            "sku": "COMPJOSET-M-TOP",
            "jo_line_id": line["id"],
            "split_components": True,
        },
    )
    assert rec.status_code == 200, rec.text
    split = rec.json().get("split") or {}
    assert split.get("panels"), split
    panel_codes = {p["component_code"] for p in split["panels"]}
    assert panel_codes == {"FRONT", "BACK"}

    iss_front = client.post(
        f"/api/production/orders/{top_jo['id']}/issue-pieces",
        json={
            "issued_qty": 4,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "COMPJOSET-M-FRONT",
        },
    )
    assert iss_front.status_code == 200, iss_front.text

    board = client.get(
        "/api/production/wip-board",
        params={"so_number": "SO-COMPJO-1", "main_sku": "COMPJOSET-M"},
    ).json()
    by_code = {i["component_code"]: i for i in board["items"]}
    assert by_code["FRONT"]["current_location"] == "Embroidery"
    assert board["bundle_complete"] is False


def test_jo_panel_wip_endpoint_before_and_after_receive(isolated_module_dbs, client):
    """Cutting JO detail exposes FRONT/BACK panel rows for the Production UI."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "PANELUI",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "FRONT", "qty_per_set": 1, "routing": "Cutting>Embroidery>Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
                {"component_code": "BACK", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
            ],
        },
    )
    assert bom.status_code == 200, bom.text

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "so_number": "SO-PANELUI",
            "sku": "PANELUI-M",
            "process": "Cutting",
            "planned_qty": 3,
            "create_component_jos": True,
            "lines": [{"sku": "PANELUI-M", "style": "M", "planned_qty": 3}],
        },
    )
    assert r.status_code == 200, r.text
    top_jo = next(
        o for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-PANELUI" and str(o.get("sku") or "").endswith("-TOP")
    )

    before = client.get(f"/api/production/orders/{top_jo['id']}/panel-wip").json()
    assert before["has_panels"] is True
    assert {p["component_code"] for p in before["panels"]} == {"FRONT", "BACK"}
    assert all(p["issueable_qty"] == 0 for p in before["panels"])

    line = top_jo["lines"][0]
    rec = client.post(
        f"/api/production/orders/{top_jo['id']}/receive-pieces",
        json={
            "received_qty": 3,
            "process": "Cutting",
            "sku": "PANELUI-M-TOP",
            "jo_line_id": line["id"],
            "split_components": True,
        },
    )
    assert rec.status_code == 200, rec.text

    after = client.get(f"/api/production/orders/{top_jo['id']}/panel-wip").json()
    by_code = {p["component_code"]: p for p in after["panels"]}
    assert by_code["FRONT"]["issueable_qty"] == 3
    assert by_code["BACK"]["issueable_qty"] == 3
    assert by_code["FRONT"]["issue_to_process"] == "Embroidery"
    assert by_code["BACK"]["issue_to_process"] == "Stitching"


def test_issue_panel_to_embroidery_auto_creates_embroidery_jo(isolated_module_dbs, client):
    """Cutting → Embroidery issue for FRONT must spawn an Embroidery work order."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "EMBJO",
            "stitching_requires_complete_set": True,
            "bundle_gate_process": "Cutting",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {
                    "component_code": "FRONT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Embroidery>Cutting>Stitching",
                    "requires_embroidery": True,
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
            ],
        },
    )
    assert bom.status_code == 200, bom.text

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-22",
            "so_number": "SO-EMBJO-1",
            "sku": "EMBJO-M",
            "process": "Cutting",
            "planned_qty": 5,
            "create_component_jos": True,
            "lines": [{"sku": "EMBJO-M", "style": "M", "planned_qty": 5}],
        },
    )
    assert r.status_code == 200, r.text
    top_jo = next(
        o for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-EMBJO-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    line = top_jo["lines"][0]
    rec = client.post(
        f"/api/production/orders/{top_jo['id']}/receive-pieces",
        json={
            "received_qty": 5,
            "process": "Cutting",
            "sku": "EMBJO-M-TOP",
            "jo_line_id": line["id"],
            "split_components": True,
        },
    )
    assert rec.status_code == 200, rec.text

    iss = client.post(
        f"/api/production/orders/{top_jo['id']}/issue-pieces",
        json={
            "issued_qty": 5,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "EMBJO-M-FRONT",
        },
    )
    assert iss.status_code == 200, iss.text
    child = iss.json().get("child_jo") or {}
    assert child.get("created") is True
    assert child.get("process") == "Embroidery"
    assert child.get("sku") == "EMBJO-M-FRONT"
    assert child.get("planned_qty") == 5
    emb_id = child["id"]

    emb_jos = client.get("/api/production/orders?process=Embroidery").json()
    emb = next(o for o in emb_jos if o["id"] == emb_id)
    assert emb["status"] in ("Created", "In Progress")
    assert emb["parent_jo_id"] == top_jo["id"]
    assert emb["planned_qty"] == 5

    # Embroidery receive acknowledges preloaded stock (no double-count).
    emb_line = emb["lines"][0]
    recv_emb = client.post(
        f"/api/production/orders/{emb_id}/receive-pieces",
        json={
            "received_qty": 5,
            "process": "Embroidery",
            "sku": "EMBJO-M-FRONT",
            "jo_line_id": emb_line["id"],
        },
    )
    assert recv_emb.status_code == 200, recv_emb.text

    # Return embroidered front to Cutting via Embroidery JO.
    ret = client.post(
        f"/api/production/orders/{emb_id}/issue-pieces",
        json={
            "issued_qty": 5,
            "from_process": "Embroidery",
            "to_process": "Cutting",
            "sku": "EMBJO-M-FRONT",
        },
    )
    assert ret.status_code == 200, ret.text
    # Returning to Cutting must NOT spawn a new Cutting JO for the panel.
    assert not (ret.json().get("child_jo") or {}).get("created")

    panel_wip = client.get(f"/api/production/orders/{top_jo['id']}/panel-wip").json()
    front = next(p for p in panel_wip["panels"] if p["component_code"] == "FRONT")
    assert front["embroidery_jo"]["jo_number"] == child["jo_number"]
    assert front["current_location"] == "Cutting"
    assert front["embroidery_outstanding"] == 0


def test_top_stitching_independent_of_pant_dupatta(isolated_module_dbs, client):
    """TOP FRONT/BACK may go to Stitching without PANT or DUPATTA at Cutting."""
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "INDEPSET",
            "stitching_requires_complete_set": True,
            "bundle_gate_process": "Cutting",
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
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "BACK",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "PANEL",
                    "parent_component_code": "TOP",
                },
                {
                    "component_code": "PANT",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
                },
                {
                    "component_code": "DUPATTA",
                    "qty_per_set": 1,
                    "routing": "Cutting>Stitching",
                    "component_role": "SET_COMPONENT",
                },
            ],
        },
    )
    assert bom.status_code == 200, bom.text

    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-07-28",
            "so_number": "SO-INDEP-1",
            "sku": "INDEPSET-M",
            "process": "Cutting",
            "planned_qty": 1,
            "create_component_jos": True,
            "lines": [{"sku": "INDEPSET-M", "style": "M", "planned_qty": 1}],
        },
    )
    assert r.status_code == 200, r.text
    orders = client.get("/api/production/orders").json()
    top_jo = next(
        o for o in orders
        if o.get("so_number") == "SO-INDEP-1" and str(o.get("sku") or "").endswith("-TOP")
    )
    # Do not receive PANT/DUPATTA JOs — they stay at 0 Cutting stock.

    line = top_jo["lines"][0]
    rec = client.post(
        f"/api/production/orders/{top_jo['id']}/receive-pieces",
        json={
            "received_qty": 1,
            "process": "Cutting",
            "sku": "INDEPSET-M-TOP",
            "jo_line_id": line["id"],
            "split_components": True,
        },
    )
    assert rec.status_code == 200, rec.text

    # FRONT → Embroidery → back to Cutting
    iss_front = client.post(
        f"/api/production/orders/{top_jo['id']}/issue-pieces",
        json={
            "issued_qty": 1,
            "from_process": "Cutting",
            "to_process": "Embroidery",
            "sku": "INDEPSET-M-FRONT",
        },
    )
    assert iss_front.status_code == 200, iss_front.text
    child = (iss_front.json().get("child_jo") or {})
    assert child.get("created"), iss_front.json()
    emb_id = child["id"]
    emb_jo = client.get(f"/api/production/orders/{emb_id}").json()
    emb_line = emb_jo["lines"][0]
    assert client.post(
        f"/api/production/orders/{emb_id}/receive-pieces",
        json={
            "received_qty": 1,
            "process": "Embroidery",
            "sku": "INDEPSET-M-FRONT",
            "jo_line_id": emb_line["id"],
        },
    ).status_code == 200
    assert client.post(
        f"/api/production/orders/{emb_id}/issue-pieces",
        json={
            "issued_qty": 1,
            "from_process": "Embroidery",
            "to_process": "Cutting",
            "sku": "INDEPSET-M-FRONT",
        },
    ).status_code == 200

    # Full-style board still incomplete without PANT/DUPATTA.
    full = client.get(
        "/api/production/bundle-ready",
        params={"so_number": "SO-INDEP-1", "main_sku": "INDEPSET-M"},
    ).json()
    assert full["bundle_complete"] is False
    assert any("PANT" in b or "DUPATTA" in b for b in full["blockers"])

    # TOP-scoped gate is ready (panels only).
    top_ready = client.get(
        "/api/production/bundle-ready",
        params={
            "so_number": "SO-INDEP-1",
            "main_sku": "INDEPSET-M",
            "parent_component_code": "TOP",
        },
    ).json()
    assert top_ready["bundle_complete"] is True
    assert top_ready["parent_component_code"] == "TOP"

    panel_wip = client.get(f"/api/production/orders/{top_jo['id']}/panel-wip").json()
    assert panel_wip["bundle_complete"] is True
    msg = (panel_wip.get("bundle_message") or "").upper()
    assert "PANT" not in msg and "DUPATTA" not in msg

    # BACK → Stitching must succeed despite PANT/DUPATTA still at 0.
    ok = client.post(
        f"/api/production/orders/{top_jo['id']}/issue-pieces",
        json={
            "issued_qty": 1,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "INDEPSET-M-BACK",
        },
    )
    assert ok.status_code == 200, ok.text

    # FRONT → Stitching also allowed for the TOP panel bundle.
    ok_front = client.post(
        f"/api/production/orders/{top_jo['id']}/issue-pieces",
        json={
            "issued_qty": 1,
            "from_process": "Cutting",
            "to_process": "Stitching",
            "sku": "INDEPSET-M-FRONT",
        },
    )
    assert ok_front.status_code == 200, ok_front.text

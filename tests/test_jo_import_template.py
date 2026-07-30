"""Bulk JO import template — main SKU vs panels vs set-components."""
from __future__ import annotations

import io
import math

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


def test_jo_import_template_has_component_and_panel_guidance(client):
    r = client.get("/api/production/orders/import-template")
    assert r.status_code == 200
    text = r.text
    assert "component_code" in text
    assert "create_component_jos" in text
    assert "FRONT/BACK" in text or "Do NOT add FRONT" in text
    assert "TEST SKU-M" in text


def test_import_main_sku_creates_component_jos_not_panels(isolated_module_dbs, client):
    bom = client.post(
        "/api/production/set-bom",
        json={
            "style_key": "IMPSET",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "FRONT", "qty_per_set": 1, "routing": "Cutting>Embroidery>Cutting>Stitching", "requires_embroidery": True, "component_role": "PANEL", "parent_component_code": "TOP"},
                {"component_code": "BACK", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
                {"component_code": "PANT", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
            ],
        },
    )
    assert bom.status_code == 200, bom.text

    csv = (
        "so_number,sku,planned_qty,process,create_component_jos\n"
        "SO-IMP-1,IMPSET-M,5,Cutting,yes\n"
    )
    res = client.post(
        "/api/production/orders/import",
        files={"file": ("jos.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"process": "Cutting"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 2, body
    skus = {
        o["sku"]
        for o in client.get("/api/production/orders").json()
        if o.get("so_number") == "SO-IMP-1"
    }
    assert skus == {"IMPSET-M-TOP", "IMPSET-M-PANT"}
    assert not any("FRONT" in s or "BACK" in s for s in skus)


def test_import_rejects_front_panel_row(isolated_module_dbs, client):
    client.post(
        "/api/production/set-bom",
        json={
            "style_key": "IMPPANEL",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "FRONT", "qty_per_set": 1, "routing": "Cutting>Embroidery>Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
                {"component_code": "BACK", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
            ],
        },
    )
    csv = (
        "so_number,sku,component_code,planned_qty,process\n"
        "SO-IMP-2,IMPPANEL-M,FRONT,5,Cutting\n"
    )
    res = client.post(
        "/api/production/orders/import",
        files={"file": ("jos.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"process": "Cutting"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 0
    assert body["errors"]
    assert any("panel" in e.lower() for e in body["errors"])


def test_import_component_code_top_only(isolated_module_dbs, client):
    client.post(
        "/api/production/set-bom",
        json={
            "style_key": "IMPTOP",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "FRONT", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "PANEL", "parent_component_code": "TOP"},
                {"component_code": "PANT", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
            ],
        },
    )
    csv = (
        "so_number,sku,component_code,planned_qty,process\n"
        "SO-IMP-3,IMPTOP-M,TOP,3,Cutting\n"
    )
    res = client.post(
        "/api/production/orders/import",
        files={"file": ("jos.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"process": "Cutting"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["created"] == 1
    jos = [o for o in client.get("/api/production/orders").json() if o.get("so_number") == "SO-IMP-3"]
    assert len(jos) == 1
    assert jos[0]["sku"] == "IMPTOP-M-TOP"
    assert jos[0].get("component_code") == "TOP"


def test_import_empty_component_code_nan_creates_set_component_jos(isolated_module_dbs, client):
    """Blank component_code cells from Excel/CSV must not be treated as literal NAN."""
    client.post(
        "/api/production/set-bom",
        json={
            "style_key": "IMPNAN",
            "lines": [
                {"component_code": "TOP", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "PANT", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
                {"component_code": "DUPATTA", "qty_per_set": 1, "routing": "Cutting>Stitching", "component_role": "SET_COMPONENT"},
            ],
        },
    )
    csv = (
        "so_number,sku,component_code,planned_qty,process,create_component_jos\n"
        "SO-IMP-NAN,IMPNAN-M,,3,Cutting,\n"
    )
    res = client.post(
        "/api/production/orders/import",
        files={"file": ("jos.csv", io.BytesIO(csv.encode()), "text/csv")},
        data={"process": "Cutting"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 3, body
    assert not body["errors"]


def test_build_jo_payload_treats_pandas_nan_component_code_as_empty(isolated_module_dbs):
    from backend.services.jo_import import build_jo_payload_from_import_row

    payload = build_jo_payload_from_import_row(
        {
            "so_number": "SO-1",
            "sku": "IMPNAN-M",
            "component_code": math.nan,
            "planned_qty": 2,
            "process": "Cutting",
        }
    )
    assert payload["sku"] == "IMPNAN-M"
    assert payload.get("component_code", "") == ""
    assert payload.get("create_component_jos") is None

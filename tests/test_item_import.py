"""Item Master bulk import — routing column support."""

import pytest


@pytest.fixture()
def item_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "items_test.db")
    monkeypatch.setenv("ITEM_DB_PATH", db_path)
    from backend.db import item_db

    monkeypatch.setattr(item_db, "DB_PATH", db_path)
    item_db.init_db()
    return item_db


def test_parse_item_import_with_routing():
    from backend.services.item_import import parse_item_import

    csv = (
        "item_code,item_name,item_type,routing\n"
        "SKU1,Style One,FG,\"Cutting,Stitching,Finishing\"\n"
        "SKU2,Style Two,FG,Cutting>Printing>Stitching\n"
    ).encode()
    rows = parse_item_import(csv, "items.csv")
    assert len(rows) == 2
    assert rows[0]["routing_steps"] == ["Cutting", "Stitching", "Finishing"]
    assert rows[1]["routing_steps"] == ["Cutting", "Printing", "Stitching"]


def test_bulk_create_items_sets_routing(item_db):
    from backend.services.item_import import parse_item_import

    csv = (
        "item_code,item_name,routing\n"
        "IMP1,Imported Style,Cutting,Stitching,Packing\n"
    ).encode()
    # fix csv - routing with commas needs quoting
    csv = (
        "item_code,item_name,routing\n"
        "IMP1,Imported Style,\"Cutting,Stitching,Packing\"\n"
    ).encode()
    rows = parse_item_import(csv, "items.csv")
    result = item_db.bulk_create_items(rows)
    assert result["created"] == 1
    assert result["routing_set"] == 1

    items = item_db.list_items(parent_only=True)
    imp = next(i for i in items if i["item_code"] == "IMP1")
    detail = item_db.get_item(imp["id"])
    names = [r["name"] for r in detail["routing"]]
    assert names == ["Cutting", "Stitching", "Packing"]

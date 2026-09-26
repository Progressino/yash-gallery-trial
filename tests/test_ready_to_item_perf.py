"""Ready-To / Item Master caching must stay correct right after entries."""
from __future__ import annotations

import pytest

from backend.db import grey_db, item_db, production_db, sales_db
from backend.services import so_production_path


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    grey = str(tmp_path / "grey.db")
    sales = str(tmp_path / "sales.db")
    items = str(tmp_path / "items.db")
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setenv("GREY_DB_PATH", grey)
    monkeypatch.setenv("SALES_DB_PATH", sales)
    monkeypatch.setenv("ITEM_DB_PATH", items)
    monkeypatch.setenv("ITEM_MEDIA_DIR", str(media))
    monkeypatch.setattr(production_db, "_DB", prod)
    monkeypatch.setattr(production_db, "_ITEM_DB", items)
    monkeypatch.setattr(grey_db, "_DB", grey)
    monkeypatch.setattr(sales_db, "_DB", sales)
    monkeypatch.setattr(item_db, "DB_PATH", items)
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    item_db.init_db()
    production_db._READY_CACHE.clear()
    production_db.clear_item_routing_cache()
    yield media


def _wip(sku: str, qty: int, so: str = "SO-RT1") -> None:
    res = production_db.import_ready_to_wip(
        [{"Ready_To_Stage": "Stitching", "OMS_SKU": sku, "SO_Number": so, "Quantity": qty}]
    )
    assert res["imported"] == 1, res


def _ready_qty(sku: str) -> int:
    rows = production_db.get_ready_to_process("Stitching")
    return sum(int(r["available_qty"]) for r in rows if r["sku"] == sku)


def test_ready_to_reflects_new_stock_immediately(iso):
    _wip("RTSTYLE-M", 10)
    assert _ready_qty("RTSTYLE-M") == 10
    _wip("RTSTYLE-M", 5)
    assert _ready_qty("RTSTYLE-M") == 15
    _wip("RTSTYLE-L", 3)
    assert _ready_qty("RTSTYLE-L") == 3


def test_ready_to_rows_are_copies(iso):
    _wip("RTCOPY-M", 7)
    rows = production_db.get_ready_to_process("Stitching")
    rows[0]["available_qty"] = 999
    assert _ready_qty("RTCOPY-M") == 7


def test_ready_to_filters_apply_on_cached_rows(iso):
    _wip("RTFILT-M", 4, so="SO-AAA")
    _wip("RTOTHER-M", 6, so="SO-BBB")
    hits = production_db.get_ready_to_process("Stitching", q="aaa")
    assert [r["sku"] for r in hits] == ["RTFILT-M"]
    assert production_db.get_ready_to_process("Stitching", min_qty=5)[0]["sku"] == "RTOTHER-M"


def test_so_production_mode_refreshes_after_update(iso):
    conn = sales_db._connect()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sales_orders)")}
    assert "production_mode" in cols
    conn.execute(
        "INSERT INTO sales_orders (so_number, so_date, buyer, production_mode) VALUES (?,?,?,?)",
        ("SO-MODE1", "2026-09-26", "Buyer", "inhouse"),
    )
    conn.commit()
    conn.close()
    assert so_production_path.get_so_production_mode("SO-MODE1") == "inhouse"
    conn = sales_db._connect()
    conn.execute("UPDATE sales_orders SET production_mode='cut_to_pack' WHERE so_number='SO-MODE1'")
    conn.commit()
    conn.close()
    assert so_production_path.get_so_production_mode("SO-MODE1") == "cut_to_pack"
    assert so_production_path.get_so_production_mode("SO-MISSING") == "inhouse"


def _type_id() -> int:
    return int(item_db.list_item_types()[0]["id"])


def test_item_search_ranks_exact_then_prefix_and_limits(iso):
    tid = _type_id()
    item_db.create_item("ZZ-YK10", "contains yk", tid)
    item_db.create_item("YK10-RED", "prefix", tid)
    item_db.create_item("YK10", "exact", tid)
    ranked = item_db.list_items(search="yk10", rank_search=True)
    assert [r["item_code"] for r in ranked] == ["YK10", "YK10-RED", "ZZ-YK10"]
    assert len(item_db.list_items(search="yk10", rank_search=True, limit=2)) == 2


def test_item_list_variant_inherits_parent_image(iso):
    media = iso
    (media / "PARENT1.jpg").write_bytes(b"x")
    tid = _type_id()
    parent_id = item_db.create_item("PARENT1", "Parent", tid)
    conn = item_db._connect()
    conn.execute("UPDATE items SET image_path='PARENT1.jpg' WHERE id=?", (parent_id,))
    conn.commit()
    conn.close()
    item_db.create_size_variants(parent_id, ["M"])
    by_code = {r["item_code"]: r for r in item_db.list_items()}
    assert by_code["PARENT1"]["has_image"] is True
    assert by_code["PARENT1-M"]["has_image"] is True
    assert by_code["PARENT1-M"]["image_url"].endswith("/PARENT1-M/image")
    assert by_code["PARENT1"]["variant_count"] == 1
    assert "_parent_image_path" not in by_code["PARENT1-M"]

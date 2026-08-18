"""Routing-based next process: Style/SKU path, not a hardcoded Finishing hop."""
from __future__ import annotations

import pytest

from backend.db import grey_db, item_db, production_db, sales_db


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    grey = str(tmp_path / "grey.db")
    sales = str(tmp_path / "sales.db")
    items = str(tmp_path / "items.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setenv("GREY_DB_PATH", grey)
    monkeypatch.setenv("SALES_DB_PATH", sales)
    monkeypatch.setenv("ITEM_DB_PATH", items)
    monkeypatch.setattr(production_db, "_DB", prod)
    monkeypatch.setattr(production_db, "_ITEM_DB", items)
    monkeypatch.setattr(grey_db, "_DB", grey)
    monkeypatch.setattr(sales_db, "_DB", sales)
    monkeypatch.setattr(item_db, "DB_PATH", items)
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    item_db.init_db()
    yield


def _fg_type_id() -> int:
    conn = item_db._connect()
    row = conn.execute("SELECT id FROM item_types WHERE code='FG'").fetchone()
    conn.close()
    return int(row["id"])


def _style(code: str, steps: list[str], sizes: list[str] | None = None) -> int:
    parent_id = item_db.create_item(code, code, _fg_type_id())
    if sizes:
        item_db.create_size_variants(parent_id, sizes)
    item_db.set_item_routing(parent_id, item_db.resolve_routing_step_names(steps))
    return parent_id


MULTI = ["Cutting", "Stitching", "Kaj Button", "Handwork", "Finishing"]
NO_KAJ = ["Cutting", "Stitching", "Handwork", "Finishing"]
DIRECT = ["Cutting", "Stitching", "Finishing"]


def test_multi_step_next_hops(iso):
    _style("RT-A", MULTI, ["M"])
    sku = "RT-A-M"
    assert production_db.get_item_routing(sku) == MULTI
    assert production_db.get_next_process(sku, "Cutting") == "Stitching"
    assert production_db.get_next_process(sku, "Stitching") == "Kaj Button"
    assert production_db.get_next_process(sku, "Kaj Button") == "Handwork"
    assert production_db.get_next_process(sku, "Handwork") == "Finishing"
    assert production_db.get_next_process(sku, "Finishing") is None


def test_routing_without_kaj_button(iso):
    _style("RT-B", NO_KAJ, ["S"])
    assert production_db.get_next_process("RT-B-S", "Stitching") == "Handwork"


def test_routing_directly_to_finishing(iso):
    _style("RT-C", DIRECT, ["L"])
    assert production_db.get_next_process("RT-C-L", "Stitching") == "Finishing"


def test_different_styles_do_not_share_routing(iso):
    _style("STYLE-A", MULTI, ["M"])
    _style("STYLE-B", DIRECT, ["M"])
    assert production_db.get_next_process("STYLE-A-M", "Stitching") == "Kaj Button"
    assert production_db.get_next_process("STYLE-B-M", "Stitching") == "Finishing"


def test_last_process_cannot_issue(iso):
    _style("RT-END", DIRECT, ["M"])
    num = production_db.create_jo(
        {
            "so_number": "SO-END",
            "so_source": "manual",
            "sku": "RT-END-M",
            "process": "Finishing",
            "planned_qty": 10,
            "create_component_jos": False,
            "lines": [{"sku": "RT-END-M", "planned_qty": 10}],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-END", "RT-END-M", "Finishing", qty_in=10)
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="No next process"):
        production_db.issue_pieces(jo["id"], {"issued_qty": 5, "from_process": "Finishing"})


def test_missing_routing_no_invented_next(iso):
    item_db.create_item("RT-EMPTY", "Empty", _fg_type_id())
    assert production_db.get_item_routing("RT-EMPTY") == []
    assert production_db.get_next_process("RT-EMPTY", "Stitching") is None


def test_current_process_missing_from_routing(iso):
    _style("RT-MISS", ["Cutting", "Stitching", "Finishing"], ["M"])
    assert production_db.get_next_process("RT-MISS-M", "Kaj Button") is None
    ok, msg = production_db.is_valid_routing_hop("RT-MISS-M", "Kaj Button", "Handwork")
    assert ok is False
    assert "not on the configured routing" in msg


def test_backend_rejects_skip_to_finishing(iso):
    _style("RT-SKIP", MULTI, ["M"])
    num = production_db.create_jo(
        {
            "so_number": "SO-SKIP",
            "so_source": "manual",
            "sku": "RT-SKIP-M",
            "process": "Stitching",
            "planned_qty": 20,
            "create_component_jos": False,
            "lines": [{"sku": "RT-SKIP-M", "planned_qty": 20}],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-SKIP", "RT-SKIP-M", "Stitching", qty_in=20)
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="Kaj Button"):
        production_db.issue_pieces(
            jo["id"],
            {"issued_qty": 10, "from_process": "Stitching", "to_process": "Finishing"},
        )
    stock = production_db.get_process_stock("SO-SKIP", "RT-SKIP-M", "Stitching")
    assert stock == 20
    production_db.issue_pieces(
        jo["id"],
        {"issued_qty": 10, "from_process": "Stitching", "to_process": "Kaj Button"},
    )
    assert production_db.get_process_stock("SO-SKIP", "RT-SKIP-M", "Stitching") == 10
    assert production_db.get_process_stock("SO-SKIP", "RT-SKIP-M", "Kaj Button") == 10


def test_kajh_button_alias_normalizes(iso):
    from backend.services.operation_routing import normalize_process_name

    assert normalize_process_name("Kajh Button") == "Kaj Button"
    assert normalize_process_name("kaj button") == "Kaj Button"

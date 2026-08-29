"""Process tab alias coalescing — Embroidary→Embroidery, Kaj Button→Kajh Button."""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_module_dbs(tmp_path, monkeypatch):
    paths = {
        "PRODUCTION_DB_PATH": str(tmp_path / "production.db"),
        "ITEM_DB_PATH": str(tmp_path / "items.db"),
    }
    for k, v in paths.items():
        monkeypatch.setenv(k, v)

    from backend.db import production_db, item_db

    monkeypatch.setattr(production_db, "_DB", paths["PRODUCTION_DB_PATH"])
    monkeypatch.setattr(production_db, "_ITEM_DB", paths["ITEM_DB_PATH"])
    monkeypatch.setattr(item_db, "DB_PATH", paths["ITEM_DB_PATH"])

    production_db.init_db()
    item_db.init_db()
    return paths


def test_process_tab_aliases_map_misspellings():
    from backend.db.production_db import _PROCESS_TAB_ALIASES, _HIDDEN_PROCESS_TABS

    assert _PROCESS_TAB_ALIASES["Embroidary"] == "Embroidery"
    assert _PROCESS_TAB_ALIASES["Kaj Button"] == "Kajh Button"
    assert "Embroidary" in _HIDDEN_PROCESS_TABS
    assert "Kaj Button" in _HIDDEN_PROCESS_TABS
    assert "Embroidery" not in _HIDDEN_PROCESS_TABS


def test_coalesce_embroidary_into_embroidery(isolated_module_dbs):
    from backend.db import item_db, production_db

    item_db.create_routing_step("Embroidary", "typo", 2)
    steps = {s["name"]: s["id"] for s in item_db.list_routing_steps()}
    assert "Embroidary" in steps
    if "Embroidery" not in steps:
        item_db.create_routing_step("Embroidery", "correct", 3)

    conn = production_db._connect()
    conn.execute(
        """INSERT INTO process_stock(so_number, sku, process, available_qty, total_in, total_out)
           VALUES(?,?,?,?,?,?)""",
        ("SO-E1", "SKU-E1", "Embroidary", 5, 5, 0),
    )
    conn.execute(
        """INSERT INTO job_orders(
             jo_number, jo_date, so_number, sku, process, stage, planned_qty, balance_qty, status)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        ("JO-E1", "2026-08-29", "SO-E1", "SKU-E1", "Embroidary", "Embroidary", 5, 5, "Created"),
    )
    conn.commit()
    conn.close()

    report = production_db.coalesce_duplicate_process_tabs()
    assert not report.get("errors"), report

    names = production_db.get_all_routing_steps()
    assert "Embroidary" not in names
    assert "Embroidery" in names

    conn = production_db._connect()
    stock = conn.execute(
        "SELECT process, available_qty FROM process_stock WHERE sku=?",
        ("SKU-E1",),
    ).fetchall()
    assert len(stock) == 1
    assert stock[0][0] == "Embroidery"
    assert int(stock[0][1]) == 5
    jo = conn.execute(
        "SELECT process FROM job_orders WHERE jo_number=?",
        ("JO-E1",),
    ).fetchone()
    assert jo[0] == "Embroidery"
    conn.close()

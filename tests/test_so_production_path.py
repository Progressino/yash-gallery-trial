"""SO-level Cut-to-Pack / Stitch-to-Pack vs in-house, and Ready-To JO netting."""
from __future__ import annotations

import pytest

from backend.db import production_db, sales_db, grey_db


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    grey = str(tmp_path / "grey.db")
    sales = str(tmp_path / "sales.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setenv("GREY_DB_PATH", grey)
    monkeypatch.setenv("SALES_DB_PATH", sales)
    monkeypatch.setattr(production_db, "_DB", prod)
    monkeypatch.setattr(grey_db, "_DB", grey)
    monkeypatch.setattr(sales_db, "_DB", sales)
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    yield


def test_same_style_two_sos_use_different_paths(iso, monkeypatch):
    """Style routing stays in-house; each SO chooses Cut-to-Pack vs in-house."""
    monkeypatch.setattr(
        production_db,
        "get_item_routing",
        lambda sku: ["Cutting", "Embroidery", "Handwork", "Stitching", "Finishing"],
    )
    monkeypatch.setattr(
        production_db,
        "get_component_routing",
        lambda sku: ["Cutting", "Embroidery", "Handwork", "Stitching", "Finishing"],
    )
    so_c2p = sales_db.create_order(
        {
            "production_mode": "cut_to_pack",
            "lines": [{"sku": "1001-XS", "qty": 50}],
        }
    )
    so_ih = sales_db.create_order(
        {
            "production_mode": "inhouse",
            "lines": [{"sku": "1001-XS", "qty": 50}],
        }
    )
    assert (
        production_db.get_next_process("1001-XS", "Cutting", so_number=so_c2p) == "Finishing"
    )
    assert (
        production_db.get_next_process("1001-XS", "Cutting", so_number=so_ih) == "Embroidery"
    )
    assert production_db.get_previous_process("1001-XS", "Finishing", so_number=so_c2p) == "Cutting"
    assert production_db.get_previous_process("1001-XS", "Finishing", so_number=so_ih) == "Stitching"


def test_cut_to_pack_does_not_appear_on_ready_to_stitch(iso, monkeypatch):
    monkeypatch.setattr(production_db, "get_item_routing", lambda sku: ["Cutting", "Stitching", "Finishing"])
    monkeypatch.setattr(production_db, "get_component_routing", lambda sku: ["Cutting", "Stitching", "Finishing"])
    so = sales_db.create_order(
        {"production_mode": "cut_to_pack", "lines": [{"sku": "1001-M", "qty": 40}]}
    )
    conn = production_db._connect()
    production_db._update_process_stock(conn, so, "1001-M", "Cutting", qty_in=40)
    conn.commit()
    conn.close()
    stitch = production_db.get_ready_to_process("Stitching")
    assert not any(r.get("sku") == "1001-M" and r.get("so_number") == so for r in stitch)
    finish = production_db.get_ready_to_process("Finishing")
    row = next(r for r in finish if r.get("sku") == "1001-M" and r.get("so_number") == so)
    assert int(row.get("available_qty") or 0) == 40


def test_ready_to_stitch_clears_after_jo_created(iso):
    """Issue to Stitching still shows Ready-To; creating the Stitching JO consumes it."""
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-R1", "1001-XS", "Stitching", qty_in=50)
    conn.commit()
    conn.close()
    before = production_db.get_ready_to_process("Stitching")
    row = next(r for r in before if r.get("sku") == "1001-XS")
    assert int(row.get("available_qty") or 0) == 50

    production_db.create_jo(
        {
            "so_number": "SO-R1",
            "so_source": "manual",
            "sku": "1001-XS",
            "process": "Stitching",
            "planned_qty": 50,
            "create_component_jos": False,
            "lines": [{"sku": "1001-XS", "style": "XS", "planned_qty": 50}],
        }
    )
    after = production_db.get_ready_to_process("Stitching")
    leftover = next((r for r in after if r.get("sku") == "1001-XS" and r.get("so_number") == "SO-R1"), None)
    assert leftover is None


def test_ready_to_stitch_remainder_after_partial_jo(iso):
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-R2", "1001-S", "Cutting", qty_in=80)
    conn.commit()
    conn.close()
    production_db.create_jo(
        {
            "so_number": "SO-R2",
            "so_source": "manual",
            "sku": "1001-S",
            "process": "Stitching",
            "planned_qty": 50,
            "create_component_jos": False,
            "lines": [{"sku": "1001-S", "planned_qty": 50}],
        }
    )
    after = production_db.get_ready_to_process("Stitching")
    row = next(r for r in after if r.get("sku") == "1001-S")
    assert int(row.get("available_qty") or 0) == 30
    assert int(row.get("already_planned") or 0) == 50


def test_incoming_ready_to_cut_nets_open_cutting_jo(iso):
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-C1", "1001-L", "Incoming", qty_in=40)
    conn.commit()
    conn.close()
    before = production_db.get_ready_to_process("Cutting")
    row = next(r for r in before if r.get("sku") == "1001-L" and r.get("so_number") == "SO-C1")
    assert int(row.get("available_qty") or 0) == 40

    production_db.create_jo(
        {
            "so_number": "SO-C1",
            "so_source": "manual",
            "sku": "1001-L",
            "process": "Cutting",
            "planned_qty": 40,
            "create_component_jos": False,
            "lines": [{"sku": "1001-L", "planned_qty": 40}],
        }
    )
    after = production_db.get_ready_to_process("Cutting")
    leftover = next((r for r in after if r.get("sku") == "1001-L" and r.get("so_number") == "SO-C1"), None)
    assert leftover is None


def test_validate_jo_creation_subtracts_open_planned(iso):
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-V1", "1001-M", "Cutting", qty_in=50)
    conn.commit()
    conn.close()
    production_db.create_jo(
        {
            "so_number": "SO-V1",
            "so_source": "manual",
            "sku": "1001-M",
            "process": "Stitching",
            "planned_qty": 50,
            "create_component_jos": False,
            "lines": [{"sku": "1001-M", "planned_qty": 50}],
        }
    )
    v = production_db.validate_jo_creation("Stitching", "SO-V1", "1001-M", 10)
    assert v["ok"] is False
    assert v["available"] == 0


def test_stich_alias_and_stitch_to_pack_path(iso, monkeypatch):
    from backend.services.so_production_path import normalize_production_mode

    assert normalize_production_mode("stich_to_pack") == "stitch_to_pack"
    assert normalize_production_mode("Cut-Pack") == "cut_to_pack"
    monkeypatch.setattr(
        production_db,
        "get_item_routing",
        lambda sku: ["Cutting", "Embroidery", "Stitching", "Finishing"],
    )
    monkeypatch.setattr(
        production_db,
        "get_component_routing",
        lambda sku: ["Cutting", "Embroidery", "Stitching", "Finishing"],
    )
    so = sales_db.create_order(
        {"production_mode": "stich_to_pack", "lines": [{"sku": "1001-XL", "qty": 30}]}
    )
    assert production_db.get_next_process("1001-XL", "Cutting", so_number=so) == "Stitching"
    assert production_db.get_next_process("1001-XL", "Stitching", so_number=so) == "Finishing"
    conn = production_db._connect()
    production_db._update_process_stock(conn, so, "1001-XL", "Cutting", qty_in=30)
    conn.commit()
    conn.close()
    stitch = production_db.get_ready_to_process("Stitching")
    assert any(r.get("sku") == "1001-XL" and r.get("so_number") == so for r in stitch)

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


def test_jo_cut_to_pack_overrides_inhouse_so_next_process(iso, monkeypatch):
    """Split path: JO production_mode=cut_to_pack must next to Finishing even if SO is inhouse."""
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
        {
            "production_mode": "inhouse",
            "lines": [{"sku": "1785YKRED-S", "qty": 48}],
        }
    )
    assert (
        production_db.get_next_process("1785YKRED-S", "Cutting", so_number=so) == "Embroidery"
    )
    assert (
        production_db.get_next_process(
            "1785YKRED-S",
            "Cutting",
            so_number=so,
            production_mode="cut_to_pack",
        )
        == "Finishing"
    )
    ok, _ = production_db.is_valid_routing_hop(
        "1785YKRED-S",
        "Cutting",
        "Finishing",
        so_number=so,
        production_mode="cut_to_pack",
    )
    assert ok
    bad, msg = production_db.is_valid_routing_hop(
        "1785YKRED-S",
        "Cutting",
        "Stitching",
        so_number=so,
        production_mode="cut_to_pack",
    )
    assert not bad
    assert "Stitching" in msg or "routing" in msg.lower()


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


def test_multi_size_finishing_jo_validates_each_sku_availability(iso, monkeypatch):
    """Combined planned qty must not be checked against only the header SKU's Ready qty."""
    monkeypatch.setattr(
        production_db,
        "get_item_routing",
        lambda sku: ["Cutting", "Finishing"],
    )
    monkeypatch.setattr(
        production_db,
        "get_component_routing",
        lambda sku: ["Cutting", "Finishing"],
    )
    monkeypatch.setattr(
        production_db,
        "get_previous_process",
        lambda sku, process, so_number=None, production_mode=None: "Cutting",
    )
    conn = production_db._connect()
    production_db._update_process_stock(conn, "01-2627", "1785YKRED-XXL", "Cutting", qty_in=119)
    production_db._update_process_stock(conn, "01-2627", "1785YKRED-XL", "Cutting", qty_in=116)
    production_db._update_process_stock(conn, "01-2627", "1785YKRED-L", "Cutting", qty_in=79)
    production_db._update_process_stock(conn, "01-2627", "1785YKRED-S", "Cutting", qty_in=48)
    production_db._update_process_stock(conn, "01-2627", "1785YKRED-5XL", "Cutting", qty_in=65)
    conn.commit()
    conn.close()

    # Header-only validation of the SUM would fail (427 > 119)
    bad = production_db.validate_jo_creation("Finishing", "01-2627", "1785YKRED-XXL", 427)
    assert bad["ok"] is False
    assert bad["available"] == 119

    lines = [
        {"sku": "1785YKRED-XXL", "planned_qty": 119},
        {"sku": "1785YKRED-XL", "planned_qty": 116},
        {"sku": "1785YKRED-L", "planned_qty": 79},
        {"sku": "1785YKRED-S", "planned_qty": 48},
        {"sku": "1785YKRED-5XL", "planned_qty": 65},
    ]
    for ln in lines:
        v = production_db.validate_jo_creation(
            "Finishing", "01-2627", ln["sku"], int(ln["planned_qty"])
        )
        assert v["ok"] is True, (ln, v)

    num = production_db.create_jo(
        {
            "so_number": "01-2627",
            "so_source": "manual",
            "sku": "1785YKRED-XXL",
            "process": "Finishing",
            "planned_qty": 427,
            "create_component_jos": False,
            "lines": lines,
        }
    )
    # Simulate router per-line gate then create
    assert isinstance(num, str)
    jo = production_db.get_jo_by_number(num)
    assert jo is not None
    assert len(jo["lines"]) == 5
    assert int(jo["planned_qty"]) == 427


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


def test_same_sku_split_across_production_paths(iso, monkeypatch):
    """SO+SKU can have multiple open JOs with different production_mode (qty-level path split)."""
    monkeypatch.setattr(production_db, "get_item_routing", lambda sku: ["Cutting", "Stitching", "Finishing"])
    monkeypatch.setattr(production_db, "get_component_routing", lambda sku: ["Cutting", "Stitching", "Finishing"])
    conn = production_db._connect()
    production_db._update_process_stock(conn, "SO-SPLIT", "1001-M", "Cutting", qty_in=1000)
    conn.commit()
    conn.close()

    production_db.create_jo(
        {
            "so_number": "SO-SPLIT",
            "so_source": "manual",
            "sku": "1001-M",
            "process": "Cutting",
            "production_mode": "inhouse",
            "planned_qty": 500,
            "create_component_jos": False,
            "lines": [{"sku": "1001-M", "planned_qty": 500}],
        }
    )
    production_db.create_jo(
        {
            "so_number": "SO-SPLIT",
            "so_source": "manual",
            "sku": "1001-M",
            "process": "Cutting",
            "production_mode": "cut_to_pack",
            "planned_qty": 500,
            "create_component_jos": False,
            "lines": [{"sku": "1001-M", "planned_qty": 500}],
        }
    )
    commit = production_db.get_path_commitment("SO-SPLIT", "1001-M", "Cutting")
    assert commit["total_planned"] == 1000
    assert commit["by_mode"]["inhouse"] == 500
    assert commit["by_mode"]["cut_to_pack"] == 500

    after = production_db.get_ready_to_process("Cutting")
    leftover = next((r for r in after if r.get("sku") == "1001-M" and r.get("so_number") == "SO-SPLIT"), None)
    assert leftover is None


def test_jo_production_mode_overrides_so_for_routing(iso, monkeypatch):
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
        {"production_mode": "inhouse", "lines": [{"sku": "1001-S", "qty": 20}]}
    )
    num = production_db.create_jo(
        {
            "so_number": so,
            "so_source": "system",
            "sku": "1001-S",
            "process": "Cutting",
            "production_mode": "cut_to_pack",
            "planned_qty": 20,
            "create_component_jos": False,
            "lines": [{"sku": "1001-S", "planned_qty": 20}],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    assert jo.get("production_mode") == "cut_to_pack"
    assert production_db.get_next_process(
        "1001-S", "Cutting", so_number=so, production_mode=jo.get("production_mode")
    ) == "Finishing"

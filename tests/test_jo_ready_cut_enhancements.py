"""JO planned qty edit, manual SO source, component ready-to-cut, fabric SKU filter."""
from __future__ import annotations

import os

import pytest

from backend.db import production_db, grey_db, sales_db
from backend.services.ready_to_cut_eligibility import expand_ready_to_cut_rows
from backend.services.fabric_sku_matching import sku_uses_fabric


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
    monkeypatch.setattr(grey_db, "_PRODUCTION_DB", prod)
    monkeypatch.setattr(sales_db, "_DB", sales)
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    yield


def test_update_planned_qty_writes_history(iso):
    num = production_db.create_jo(
        {
            "so_number": "SO-M1",
            "so_source": "manual",
            "sku": "STYLE-M",
            "process": "Cutting",
            "planned_qty": 10,
            "create_component_jos": False,
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    production_db.update_jo(jo["id"], {"planned_qty": 7, "changed_by": "test", "qty_change_remarks": "reduce"})
    jo2 = production_db.get_jo(jo["id"])
    assert int(jo2["planned_qty"]) == 7
    hist = production_db.list_jo_qty_history(jo["id"])
    assert hist and int(hist[0]["old_qty"]) == 10 and int(hist[0]["new_qty"]) == 7


def test_update_planned_qty_rejects_below_received(iso):
    num = production_db.create_jo(
        {
            "so_number": "SO-M2",
            "so_source": "manual",
            "sku": "STYLE-L",
            "process": "Cutting",
            "planned_qty": 10,
            "create_component_jos": False,
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    # Simulate received
    import sqlite3

    conn = sqlite3.connect(os.environ["PRODUCTION_DB_PATH"])
    conn.execute("UPDATE job_orders SET received_qty=6 WHERE id=?", (jo["id"],))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="below"):
        production_db.update_jo(jo["id"], {"planned_qty": 5})


def test_manual_so_multi_size_lines_on_single_jo(iso):
    """One JO can hold multiple size SKUs as separate lines."""
    num = production_db.create_jo(
        {
            "so_number": "WALKIN-MULTI",
            "so_source": "manual",
            "sku": "1294YKGREEN-S",
            "sku_name": "Green S",
            "process": "Cutting",
            "planned_qty": 25,
            "create_component_jos": False,
            "lines": [
                {
                    "so_number": "WALKIN-MULTI",
                    "sku": "1294YKGREEN-S",
                    "sku_name": "Green S",
                    "planned_qty": 10,
                },
                {
                    "so_number": "WALKIN-MULTI",
                    "sku": "1294YKGREEN-M",
                    "sku_name": "Green M",
                    "planned_qty": 15,
                },
            ],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    full = production_db.get_jo(jo["id"])
    skus = { (ln.get("sku") or "").upper() for ln in (full.get("lines") or []) }
    assert "1294YKGREEN-S" in skus
    assert "1294YKGREEN-M" in skus
    assert len(full.get("lines") or []) == 2
    assert sum(int(ln.get("planned_qty") or 0) for ln in full["lines"]) == 25


def test_multi_size_rejects_header_only_qty_edit(iso):
    num = production_db.create_jo(
        {
            "so_number": "WALKIN-QTY",
            "so_source": "manual",
            "sku": "1264YKGREEN-S",
            "process": "Cutting",
            "planned_qty": 50,
            "create_component_jos": False,
            "lines": [
                {"sku": "1264YKGREEN-S", "style": "S", "planned_qty": 20},
                {"sku": "1264YKGREEN-M", "style": "M", "planned_qty": 30},
            ],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    with pytest.raises(ValueError, match="multiple sizes"):
        production_db.update_jo(jo["id"], {"planned_qty": 55})


def test_edit_one_size_recalculates_jo_total(iso):
    num = production_db.create_jo(
        {
            "so_number": "WALKIN-QTY2",
            "so_source": "manual",
            "sku": "1264YKGREEN-S",
            "process": "Cutting",
            "planned_qty": 100,
            "create_component_jos": False,
            "lines": [
                {"sku": "1264YKGREEN-S", "style": "S", "planned_qty": 20},
                {"sku": "1264YKGREEN-M", "style": "M", "planned_qty": 30},
                {"sku": "1264YKGREEN-L", "style": "L", "planned_qty": 25},
                {"sku": "1264YKGREEN-XL", "style": "XL", "planned_qty": 25},
            ],
        }
    )
    jo = production_db.get_jo(next(j for j in production_db.list_jos() if j["jo_number"] == num)["id"])
    m_line = next(l for l in jo["lines"] if l["sku"] == "1264YKGREEN-M")
    production_db.update_jo(
        jo["id"],
        {"lines": [{"id": m_line["id"], "planned_qty": 35}], "qty_change_remarks": "M 30→35"},
    )
    jo2 = production_db.get_jo(jo["id"])
    assert int(jo2["planned_qty"]) == 105
    by_sku = {l["sku"]: int(l["planned_qty"]) for l in jo2["lines"]}
    assert by_sku["1264YKGREEN-S"] == 20
    assert by_sku["1264YKGREEN-M"] == 35
    hist = production_db.list_jo_qty_history(jo["id"])
    assert any("line:" in str(h.get("field")) for h in hist)


def test_size_qty_cannot_go_below_received(iso):
    num = production_db.create_jo(
        {
            "so_number": "WALKIN-QTY3",
            "so_source": "manual",
            "sku": "X-S",
            "process": "Cutting",
            "planned_qty": 20,
            "create_component_jos": False,
            "lines": [
                {"sku": "X-S", "planned_qty": 10},
                {"sku": "X-M", "planned_qty": 10},
            ],
        }
    )
    jo = production_db.get_jo(next(j for j in production_db.list_jos() if j["jo_number"] == num)["id"])
    line = jo["lines"][0]
    import sqlite3

    conn = sqlite3.connect(os.environ["PRODUCTION_DB_PATH"])
    conn.execute("UPDATE jo_lines SET received_qty=8 WHERE id=?", (line["id"],))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="below"):
        production_db.update_jo(jo["id"], {"lines": [{"id": line["id"], "planned_qty": 5}]})


def test_ready_component_not_blocked_by_sibling(iso, monkeypatch):
    """TOP ready when only TOP fabric reserved; PANT stays off list."""

    def _fake_bom(main):
        return {
            "style_key": main,
            "lines": [
                {
                    "component_code": "TOP",
                    "component_name": "Top",
                    "component_role": "SET_COMPONENT",
                    "qty_per_set": 1,
                    "materials": [{"material_code": "FAB-TOP", "quantity": 1.5, "unit": "MTR"}],
                },
                {
                    "component_code": "PANT",
                    "component_name": "Pant",
                    "component_role": "SET_COMPONENT",
                    "qty_per_set": 1,
                    "materials": [{"material_code": "FAB-PANT", "quantity": 1.2, "unit": "MTR"}],
                },
            ],
        }

    monkeypatch.setattr(
        "backend.services.component_bom.effective_set_bom_for_cutting",
        _fake_bom,
    )
    monkeypatch.setattr(
        "backend.services.component_bom.set_component_lines",
        lambda bom: bom["lines"] if bom else [],
    )
    rows = expand_ready_to_cut_rows(
        [
            {
                "so_number": "SO-1",
                "sku": "STYLE-M",
                "fabric_code": "FAB-TOP",
                "fabric_name": "Top cloth",
                "reserved_qty": 20,
            }
        ],
        jo_planned={},
        hide_if_jo=True,
    )
    assert len(rows) == 1
    assert rows[0]["component_code"] == "TOP"
    assert "PANT" not in (rows[0].get("component_code") or "")


def test_reserve_options_filters_by_fabric_bom(iso, monkeypatch):
    so_a = sales_db.create_order(
        {
            "buyer": "A",
            "status": "Confirmed",
            "lines": [{"sku": "STYLE-A", "sku_name": "A", "qty": 10}],
        }
    )
    so_b = sales_db.create_order(
        {
            "buyer": "B",
            "status": "Confirmed",
            "lines": [{"sku": "STYLE-B", "sku_name": "B", "qty": 10}],
        }
    )
    grey_db.insert_printed_fabric_unchecked("PX", 50, fabric_name="X", jwo_ref="JX", grn_ref="GX")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "PX", "fabric_name": "X", "jwo_ref": "JX", "passed_qty": 50, "qc_by": "QC"}
    )
    monkeypatch.setattr(
        "backend.services.fabric_sku_matching.skus_using_fabric",
        lambda fab: {"STYLE-A"} if fab.upper() == "PX" else set(),
    )
    monkeypatch.setattr(
        "backend.services.fabric_sku_matching.sku_uses_fabric",
        lambda sku, fab: sku == "STYLE-A" and fab.upper() == "PX",
    )
    opts = grey_db.printed_fabric_reserve_options(fabric_code="PX")
    assert opts.get("fabric_filter_active") is True
    skus = [ln["sku"] for o in opts["sales_orders"] for ln in o["lines"]]
    assert "STYLE-A" in skus
    assert "STYLE-B" not in skus
    sos = {o["so_number"] for o in opts["sales_orders"]}
    assert so_a in sos
    assert so_b not in sos


def test_reserve_rejects_unrelated_sku_when_mapped(iso, monkeypatch):
    grey_db.insert_printed_fabric_unchecked("P308", 100, fabric_name="Print A", jwo_ref="J1", grn_ref="G1")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "P308", "fabric_name": "Print A", "jwo_ref": "J1", "passed_qty": 100, "qc_by": "QC"}
    )
    monkeypatch.setattr(
        "backend.services.fabric_sku_matching.skus_using_fabric",
        lambda fab: {"STYLE-A"},
    )
    monkeypatch.setattr(
        "backend.services.fabric_sku_matching.sku_uses_fabric",
        lambda sku, fab: sku == "STYLE-A",
    )
    with pytest.raises(ValueError, match="does not use fabric"):
        grey_db.reserve_printed_fabric(
            {"fabric_code": "P308", "so_number": "SO-X", "sku": "OTHER", "qty": 5}
        )


def test_size_sku_inherits_parent_printed_fabric(iso, tmp_path, monkeypatch):
    """2020YKWHITE-S has no BOM; it still uses the parent style's printed fabric."""
    from backend.db import item_db
    from backend.services import jo_issue_notes
    from backend.services.fabric_sku_matching import item_master_skus_using_fabric, sku_uses_fabric

    path = str(tmp_path / "items.db")
    monkeypatch.setattr(item_db, "DB_PATH", path)
    monkeypatch.setattr(jo_issue_notes, "_ITEM_DB", path)
    item_db.init_db()
    conn = item_db._connect()
    fg = conn.execute("SELECT id FROM item_types WHERE code='FG'").fetchone()[0]
    sfg = conn.execute("SELECT id FROM item_types WHERE code='SFG'").fetchone()[0]
    conn.close()
    parent = item_db.create_item("2020YKWHITE", "White kurti", fg)
    item_db.create_item("2020YKWHITE-S", "White kurti S", fg, parent_id=parent, size_label="S")
    item_db.create_item("7104YKOFFWHITE-S-M", "Other", fg)
    fab = item_db.create_item("D2Slub1White", "Slub white", sfg, uom="MTR")
    bom_id = item_db.create_bom(parent, "Default", is_default=0)
    item_db.add_bom_line(
        bom_id, "D2Slub1White", component_type="SFG", quantity=1.98, unit="MTR", component_item_id=fab,
    )
    mapped = item_master_skus_using_fabric("D2Slub1White")
    assert "2020YKWHITE" in mapped
    assert "2020YKWHITE-S" in mapped
    assert "7104YKOFFWHITE-S-M" not in mapped
    assert sku_uses_fabric("2020YKWHITE-S", "D2Slub1White") is True
    assert sku_uses_fabric("7104YKOFFWHITE-S-M", "D2Slub1White") is False


def test_set_component_without_materials_stays_ready_to_cut(monkeypatch):
    def _fake(_main):
        return {
            "style_key": "2020YKWHITE",
            "lines": [{
                "component_code": "TOP",
                "component_name": "Top",
                "component_role": "SET_COMPONENT",
                "qty_per_set": 1,
                "materials": [],
            }],
        }

    monkeypatch.setattr("backend.services.component_bom.effective_set_bom_for_cutting", _fake)
    rows = expand_ready_to_cut_rows(
        [{
            "so_number": "SO-0009",
            "sku": "2020YKWHITE-S",
            "fabric_code": "D2Slub1White",
            "fabric_name": "Slub",
            "reserved_qty": 59.4,
        }],
        hide_if_jo=True,
    )
    assert len(rows) == 1
    assert rows[0]["sku"] == "2020YKWHITE-S"
    assert rows[0]["fabric_code"] == "D2Slub1White"


def test_cutting_jo_accepts_compound_and_numeric_style_skus(iso):
    cases = [
        ("SO-0009", "7104YKOFFWHITE-S-M"),
        ("SO-0010", "4011DRSBLUE-L"),
        ("SO-0010", "AK-237MULTI-4XL"),
    ]
    for so, sku in cases:
        num = production_db.create_jo({
            "so_number": so,
            "so_source": "manual",
            "sku": sku,
            "process": "Cutting",
            "planned_qty": 10,
            "fabric_code": "FAB1",
            "fabric_qty": 12.5,
            "lines": [{"sku": sku, "planned_qty": 10}],
            "create_component_jos": False,
        })
        assert isinstance(num, str) and num
        jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
        assert jo["sku"] == sku
        assert int(jo["planned_qty"]) == 10


def test_fractional_metre_payload_does_not_reject_jo():
    from backend.routers.production import JOIn

    body = JOIn.model_validate({
        "so_number": "SO-0010",
        "sku": "4011DRSBLUE-L",
        "process": "Cutting",
        "planned_qty": 14.4,
        "so_qty": 21.6,
        "lines": [{"sku": "AK-237MULTI-4XL", "planned_qty": 76.5}],
    })
    assert body.planned_qty == 14
    assert body.so_qty == 22
    assert body.lines[0].planned_qty == 76


def test_empty_material_set_bom_cutting_jo_then_hides_ready(iso, monkeypatch):
    def _fake(_main):
        return {
            "style_key": "2020YKWHITE",
            "lines": [{
                "component_code": "TOP",
                "component_name": "Top",
                "component_role": "SET_COMPONENT",
                "qty_per_set": 1,
                "creates_cutting_jo": True,
                "materials": [],
            }],
        }

    monkeypatch.setattr("backend.services.component_bom.effective_set_bom_for_cutting", _fake)
    monkeypatch.setattr("backend.services.fabric_sku_matching.skus_using_fabric", lambda _code: set())
    grey_db.insert_printed_fabric_unchecked("D2Slub1White", 100, fabric_name="Slub", jwo_ref="J", grn_ref="G")
    grey_db.do_printed_fabric_qc({
        "fabric_code": "D2Slub1White", "fabric_name": "Slub", "jwo_ref": "J", "passed_qty": 100, "qc_by": "QC",
    })
    grey_db.reserve_printed_fabric({
        "fabric_code": "D2Slub1White", "so_number": "SO-0009", "sku": "2020YKWHITE-S", "qty": 30,
    })
    ready = grey_db.list_printed_fabric_ready_to_cut()
    assert any(r["sku"] == "2020YKWHITE-S" for r in ready)
    nums = production_db.create_jo({
        "so_number": "SO-0009",
        "so_source": "manual",
        "sku": "2020YKWHITE-S",
        "process": "Cutting",
        "planned_qty": 30,
        "lines": [{"sku": "2020YKWHITE-S", "planned_qty": 30}],
    })
    created = nums if isinstance(nums, list) else [nums]
    assert created
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == created[0])
    assert str(jo["sku"]).endswith("-TOP")
    ready_after = grey_db.list_printed_fabric_ready_to_cut()
    assert all(r["sku"] != "2020YKWHITE-S" for r in ready_after)

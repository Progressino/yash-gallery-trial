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


def test_manual_so_source_persisted(iso):
    num = production_db.create_jo(
        {
            "so_number": "WALKIN-99",
            "so_source": "manual",
            "sku": "AB-S",
            "process": "Cutting",
            "planned_qty": 3,
            "create_component_jos": False,
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    assert jo.get("so_source") == "manual"
    assert jo.get("so_number") == "WALKIN-99"


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

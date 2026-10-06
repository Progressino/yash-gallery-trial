"""MRP → free checked Printed Fabric → allocate to SO/SKU → Ready to Cut."""
from __future__ import annotations

import pytest

from backend.db import grey_db, production_db, sales_db
from backend.services import fabric_allocation_engine as fae
from backend.services.fabric_allocation_engine import FabricAllocationError


@pytest.fixture(autouse=True)
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
    # No BOM mapping in the isolated item DB → every SKU may use the fabric.
    monkeypatch.setattr("backend.services.fabric_sku_matching.skus_using_fabric", lambda code: set())
    yield


def _seed_printed(code: str, qty: float) -> None:
    grey_db.insert_printed_fabric_unchecked(code, qty, fabric_name=code, jwo_ref=f"J-{code}", grn_ref=f"G-{code}")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": code, "fabric_name": code, "jwo_ref": f"J-{code}", "passed_qty": qty, "qc_by": "QC"}
    )


def _free(code: str) -> float:
    conn = grey_db._connect()
    r = conn.execute("SELECT available_qty, reserved_qty FROM printed_fabric_checked_stock WHERE fabric_code=?", (code,)).fetchone()
    conn.close()
    return float(r[0])


def _active(code: str) -> list[dict]:
    conn = grey_db._connect()
    rows = conn.execute(
        "SELECT so_number, sku, qty, stage FROM printed_fabric_reservations WHERE status='Active' AND fabric_code=? ORDER BY id",
        (code,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_bulk_allocation_reserves_and_reaches_ready_to_cut():
    _seed_printed("P752", 500)
    res = fae.allocate_printed_bulk(
        {
            "printed_code": "P752",
            "rows": [
                {"so_number": "SO-1", "fg_sku": "1742YKBLUE-M", "qty": 120},
                {"so_number": "SO-1", "fg_sku": "1742YKBLUE-L", "qty": 80},
            ],
            "user_name": "planner",
        }
    )
    assert res["ok"] and res["allocated_total"] == 200
    assert _free("P752") == 300
    rows = _active("P752")
    assert [(r["sku"], r["qty"], r["stage"]) for r in rows] == [
        ("1742YKBLUE-M", 120, "RESERVED"),
        ("1742YKBLUE-L", 80, "RESERVED"),
    ]
    hist = [h for h in fae.list_allocation_history() if h["event_type"] == "PF_ALLOCATE"]
    assert len(hist) == 2 and all(h["user_name"] == "planner" for h in hist)

    ready = production_db.get_ready_to_process("Cutting")
    got = {(r["so_number"], r["sku"], r["fabric_code"]) for r in ready}
    assert ("SO-1", "1742YKBLUE-M", "P752") in got
    assert ("SO-1", "1742YKBLUE-L", "P752") in got


def test_over_free_stock_moves_nothing():
    _seed_printed("P500", 100)
    with pytest.raises(FabricAllocationError, match="exceeds free"):
        fae.allocate_printed_bulk(
            {
                "printed_code": "P500",
                "rows": [
                    {"so_number": "SO-2", "fg_sku": "A-M", "qty": 70},
                    {"so_number": "SO-2", "fg_sku": "A-L", "qty": 40},
                ],
            }
        )
    assert _free("P500") == 100
    assert _active("P500") == []


def test_unchecked_fabric_rejected():
    with pytest.raises(FabricAllocationError, match="no checked printed stock"):
        fae.allocate_printed_bulk({"printed_code": "P999", "rows": [{"so_number": "SO", "fg_sku": "X", "qty": 1}]})


def test_top_up_merges_into_same_reservation():
    _seed_printed("P501", 300)
    fae.allocate_printed_bulk({"printed_code": "P501", "rows": [{"so_number": "SO-3", "fg_sku": "B-M", "qty": 50}]})
    fae.allocate_printed_bulk({"printed_code": "P501", "rows": [{"so_number": "SO-3", "fg_sku": "B-M", "qty": 25}]})
    rows = _active("P501")
    assert len(rows) == 1 and rows[0]["qty"] == 75
    assert _free("P501") == 225


def test_set_sku_can_hold_two_printed_fabrics():
    _seed_printed("PTOP1", 100)
    _seed_printed("PPANT1", 100)
    fae.allocate_printed_bulk({"printed_code": "PTOP1", "rows": [{"so_number": "SO-4", "fg_sku": "SET-M", "qty": 30}]})
    fae.allocate_printed_bulk({"printed_code": "PPANT1", "rows": [{"so_number": "SO-4", "fg_sku": "SET-M", "qty": 20}]})
    assert len(_active("PTOP1")) == 1 and len(_active("PPANT1")) == 1


def test_manual_reserve_keeps_single_reservation_rule():
    _seed_printed("P600", 100)
    grey_db.reserve_printed_fabric({"fabric_code": "P600", "so_number": "SO-5", "sku": "C-M", "qty": 10})
    with pytest.raises(ValueError, match="already reserved"):
        grey_db.reserve_printed_fabric({"fabric_code": "P600", "so_number": "SO-5", "sku": "C-M", "qty": 5})


def test_blocked_when_cutting_jo_exists(monkeypatch):
    _seed_printed("P700", 100)
    monkeypatch.setattr(grey_db, "_cutting_jo_planned_by_so_sku", lambda: {("SO-6", "D-M"): 10.0})
    with pytest.raises(FabricAllocationError, match="Cutting job order already exists"):
        fae.allocate_printed_bulk({"printed_code": "P700", "rows": [{"so_number": "SO-6", "fg_sku": "D-M", "qty": 5}]})
    assert _free("P700") == 100


def test_sku_not_using_fabric_rejected(monkeypatch):
    _seed_printed("P800", 100)
    monkeypatch.setattr("backend.services.fabric_sku_matching.skus_using_fabric", lambda code: {"OTHER-M"})
    monkeypatch.setattr("backend.services.fabric_sku_matching.sku_uses_fabric", lambda sku, code: sku == "OTHER-M")
    with pytest.raises(FabricAllocationError, match="does not use fabric"):
        fae.allocate_printed_bulk({"printed_code": "P800", "rows": [{"so_number": "SO-7", "fg_sku": "E-M", "qty": 5}]})


def test_mrp_annotation_shows_free_stock_and_allocation():
    _seed_printed("P752", 500)
    materials = {
        "P752": {
            "type": "SFG",
            "unit": "MTR",
            "breakdown": [
                {"so_no": "SO-1", "sku": "F-M", "qty_req": 100},
                {"so_no": "SO-1", "sku": "F-L", "qty_req": 60},
            ],
        }
    }
    fae.annotate_mrp_breakdown_with_allocations(materials)
    assert materials["P752"]["printed_free_qty"] == 500
    assert materials["P752"]["printed_in_checked_stock"] is True

    fae.allocate_printed_bulk({"printed_code": "P752", "rows": [{"so_number": "SO-1", "fg_sku": "F-M", "qty": 100}, {"so_number": "SO-1", "fg_sku": "F-L", "qty": 20}]})
    fae.annotate_mrp_breakdown_with_allocations(materials)
    mat = materials["P752"]
    assert mat["printed_free_qty"] == 380
    by_sku = {b["sku"]: b for b in mat["breakdown"]}
    assert by_sku["F-M"]["allocated_qty"] == 100 and by_sku["F-M"]["status"] == "Allocated"
    assert by_sku["F-L"]["allocated_qty"] == 20 and by_sku["F-L"]["status"] == "Partial"


def test_bulk_endpoint(client):
    _seed_printed("P900", 50)
    r = client.post(
        "/api/grey/planning/allocate-printed-bulk",
        json={"printed_code": "P900", "rows": [{"so_number": "SO-8", "fg_sku": "G-M", "qty": 20}], "user_name": "tester"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["allocated_total"] == 20
    bad = client.post(
        "/api/grey/planning/allocate-printed-bulk",
        json={"printed_code": "P900", "rows": [{"so_number": "SO-8", "fg_sku": "G-L", "qty": 31}]},
    )
    assert bad.status_code == 400 and "exceeds free" in bad.json()["detail"]

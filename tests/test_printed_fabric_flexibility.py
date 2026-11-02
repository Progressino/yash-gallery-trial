"""Style reservation and cutting-plan revision share the existing printed-fabric stock."""
from __future__ import annotations

import pytest

from backend.db import grey_db, sales_db
from backend.services.fabric_allocation_engine import FabricAllocationError, allocate_printed_bulk
from backend.services.printed_flexibility import (
    apply_cutting_plan,
    convert_advance_to_so,
    list_advance_reservations,
    release_advance,
    reserve_for_style,
    save_cutting_plan,
    transfer_advance,
)


@pytest.fixture(autouse=True)
def isolated_grey_sales(tmp_path, monkeypatch):
    grey_path = str(tmp_path / "grey.db")
    sales_path = str(tmp_path / "sales.db")
    monkeypatch.setenv("GREY_DB_PATH", grey_path)
    monkeypatch.setenv("SALES_DB_PATH", sales_path)
    monkeypatch.setattr(grey_db, "_DB", grey_path)
    monkeypatch.setattr(sales_db, "_DB", sales_path)
    grey_db.init_db()
    sales_db.init_db()
    yield


def _check(code: str, qty: float):
    grey_db.insert_printed_fabric_unchecked(code, qty, fabric_name=code, jwo_ref="J", grn_ref="G")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": code, "fabric_name": code, "jwo_ref": "J", "passed_qty": qty, "qc_by": "QC"}
    )


def _bom(sku: str, fabric: str) -> float:
    rates = {"KURTI-S": 1.5, "KURTI-M": 1.5, "KURTI-L": 1.5, "KURTI-XL": 1.8}
    return rates.get(sku.upper(), 0.0)


def test_advance_reserve_blocks_free_stock_from_other_styles():
    _check("PFLEX001", 1000)
    rid = reserve_for_style({
        "fabric_code": "PFLEX001", "style_code": "KURTI", "qty": 700, "reason": "Hold for next order",
    })
    assert rid
    vis = grey_db.printed_stock_visibility("PFLEX001")[0]
    assert vis["total_stock"] == 1000
    assert vis["advance_reserved"] == 700
    assert vis["so_allocated"] == 0
    assert vis["free_available"] == 300
    assert grey_db.list_printed_fabric_ready_to_cut() == []
    with pytest.raises(FabricAllocationError, match="exceeds free"):
        allocate_printed_bulk({
            "printed_code": "PFLEX001",
            "rows": [{"so_number": "SO-9", "sku": "OTHER-S", "qty": 301}],
        })
    allocate_printed_bulk({
        "printed_code": "PFLEX001",
        "rows": [{"so_number": "SO-9", "sku": "OTHER-S", "qty": 300}],
    })
    vis = grey_db.printed_stock_visibility("PFLEX001")[0]
    assert vis["free_available"] == 0
    assert vis["advance_reserved"] == 700
    assert vis["so_allocated"] == 300


def test_convert_moves_metres_without_a_second_deduction():
    _check("PFLEX002", 1000)
    rid = reserve_for_style({
        "fabric_code": "PFLEX002", "style_code": "KURTI", "qty": 700, "reason": "Hold",
    })
    before = grey_db.printed_stock_visibility("PFLEX002")[0]
    out = convert_advance_to_so(rid, [
        {"so_number": "SO-1", "sku": "KURTI-S", "qty": 200},
        {"so_number": "SO-1", "sku": "KURTI-M", "qty": 150},
    ], reason="SO raised")
    after = grey_db.printed_stock_visibility("PFLEX002")[0]
    assert after["free_available"] == before["free_available"]
    assert after["total_stock"] == before["total_stock"]
    assert after["advance_reserved"] == 350
    assert after["so_allocated"] == 350
    assert out["status"] == "Active"
    ready = grey_db.list_printed_fabric_ready_to_cut()
    skus = {r["sku"] for r in ready}
    assert skus == {"KURTI-S", "KURTI-M"}
    assert list_advance_reservations("PFLEX002")[0]["qty"] == 350


def test_release_and_transfer():
    _check("PFLEX003", 500)
    rid = reserve_for_style({
        "fabric_code": "PFLEX003", "style_code": "KURTI", "qty": 200, "reason": "Hold",
    })
    with pytest.raises(ValueError, match="reason"):
        release_advance(rid, 50, "")
    release_advance(rid, 50, "Order cancelled")
    vis = grey_db.printed_stock_visibility("PFLEX003")[0]
    assert vis["free_available"] == 350
    assert vis["advance_reserved"] == 150

    # Transfer keeps free stock still. A style outside the fabric mapping is refused.
    import backend.services.fabric_sku_matching as matching
    original = matching.sku_uses_fabric
    matching.sku_uses_fabric = lambda sku, fabric: str(sku).upper().startswith("KURTI") or str(sku).upper().startswith("SHIRT")
    try:
        with pytest.raises(ValueError, match="does not use"):
            transfer_advance(rid, "PANT", 40, "Wrong style")
        transfer_advance(rid, "SHIRT", 40, "Moved to shirt")
    finally:
        matching.sku_uses_fabric = original
    vis = grey_db.printed_stock_visibility("PFLEX003")[0]
    assert vis["free_available"] == 350
    assert vis["advance_reserved"] == 150
    styles = {r["style_code"]: r["qty"] for r in list_advance_reservations("PFLEX003")}
    assert styles["KURTI"] == 110
    assert styles["SHIRT"] == 40


def test_cutting_revision_uses_reserved_metres_then_free_stock(monkeypatch):
    monkeypatch.setattr(
        "backend.services.printed_flexibility.fabric_metres_per_piece",
        _bom,
    )
    _check("PFLEX004", 80)
    for sku, metres in (("KURTI-S", 15), ("KURTI-M", 15), ("KURTI-L", 15)):
        grey_db.reserve_printed_fabric({
            "fabric_code": "PFLEX004", "so_number": "SO-0009", "sku": sku, "qty": metres,
        })
    so = sales_db.create_order({
        "buyer": "Retail",
        "status": "Confirmed",
        "lines": [
            {"sku": "KURTI-S", "qty": 10},
            {"sku": "KURTI-M", "qty": 10},
            {"sku": "KURTI-L", "qty": 10},
        ],
    })
    lines_down = [
        {"sku": "KURTI-S", "so_qty": 10, "planned_qty": 7},
        {"sku": "KURTI-M", "so_qty": 10, "planned_qty": 8},
        {"sku": "KURTI-L", "so_qty": 10, "planned_qty": 10},
    ]
    with pytest.raises(ValueError, match="reason"):
        apply_cutting_plan("SO-0009", "PFLEX004", lines_down, "")
    smaller = apply_cutting_plan("SO-0009", "PFLEX004", lines_down, "Buyer reduced S and M", anchor_sku="KURTI-S")
    assert smaller["revised_metres"] == 37.5
    assert smaller["topped_up_metres"] == 0
    assert grey_db.printed_stock_visibility("PFLEX004")[0]["free_available"] == 35

    lines_up = [
        {"sku": "KURTI-S", "so_qty": 10, "planned_qty": 7},
        {"sku": "KURTI-M", "so_qty": 10, "planned_qty": 8},
        {"sku": "KURTI-L", "so_qty": 10, "planned_qty": 10},
        {"sku": "KURTI-XL", "so_qty": 0, "planned_qty": 8},
    ]
    bigger = apply_cutting_plan("SO-0009", "PFLEX004", lines_up, "Added XL from free stock", anchor_sku="KURTI-S")
    assert bigger["revised_metres"] == 51.9
    assert bigger["topped_up_metres"] == pytest.approx(6.9, abs=0.05)
    vis = grey_db.printed_stock_visibility("PFLEX004")[0]
    assert vis["free_available"] == pytest.approx(28.1, abs=0.05)
    save_cutting_plan(bigger, "JO-1")
    conn = grey_db._connect()
    row = conn.execute("SELECT revised_metres, reason FROM cutting_plan_revisions").fetchone()
    nlines = conn.execute("SELECT COUNT(*) c FROM cutting_plan_revision_lines").fetchone()["c"]
    conn.close()
    assert row["reason"] == "Added XL from free stock"
    assert nlines == 4
    order = next(o for o in sales_db.list_orders() if o["so_number"] == so)
    qty_by_sku = {ln["sku"]: ln["qty"] for ln in order["lines"]}
    assert qty_by_sku["KURTI-S"] == 10

    with pytest.raises(ValueError, match="not a size"):
        apply_cutting_plan("SO-0009", "PFLEX004", [
            {"sku": "PANT-S", "so_qty": 0, "planned_qty": 5},
        ], "Wrong style", anchor_sku="KURTI-S")

    with pytest.raises(ValueError, match="only"):
        apply_cutting_plan("SO-0009", "PFLEX004", [
            {"sku": "KURTI-XL", "so_qty": 0, "planned_qty": 100},
        ], "Too many", anchor_sku="KURTI-S")

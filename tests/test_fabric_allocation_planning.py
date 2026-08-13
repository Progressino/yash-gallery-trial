"""Grey Fabric Planning, Allocation & Reallocation — automated tests.

Covers PRD scenarios: allocate, partial, multi-SKU P-code, reallocation,
lock after cutting, shortage, audit trail, reporting accuracy.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from backend.db import grey_db, sales_db
from backend.services import fabric_allocation_engine as fae
from backend.services.fabric_allocation_engine import FabricAllocationError


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    grey_path = str(tmp_path / "grey.db")
    sales_path = str(tmp_path / "sales.db")
    monkeypatch.setenv("GREY_DB_PATH", grey_path)
    monkeypatch.setenv("SALES_DB_PATH", sales_path)
    monkeypatch.setattr(grey_db, "_DB", grey_path)
    monkeypatch.setattr(sales_db, "_DB", sales_path)
    grey_db.init_db()
    sales_db.init_db()
    yield


def _seed_grey(code: str, qty: float, name: str = "Slub 20x20") -> None:
    grey_db.save_fabric_check(
        {
            "fabric_code": code,
            "fabric_name": name,
            "checked_qty": qty,
            "passed_qty": qty,
            "rejected_qty": 0,
            "rework_qty": 0,
            "checked_by": "test",
        }
    )


def _seed_printed(code: str, qty: float, name: str = "P208 Print") -> None:
    grey_db.insert_printed_fabric_unchecked(
        code, qty, fabric_name=name, jwo_ref=f"J-{code}", grn_ref=f"G-{code}"
    )
    grey_db.do_printed_fabric_qc(
        {
            "fabric_code": code,
            "fabric_name": name,
            "jwo_ref": f"J-{code}",
            "passed_qty": qty,
            "qc_by": "QC",
        }
    )


def test_schema_has_allocation_tables():
    conn = grey_db._connect()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "grey_fabric_allocations" in tables
    assert "fabric_allocation_history" in tables
    cols = {r[1] for r in grey_db._connect().execute("PRAGMA table_info(printed_fabric_reservations)").fetchall()}
    assert "stage" in cols
    assert "cutting_issued_qty" in cols


# Scenario 1 — Normal allocation
def test_scenario_1_normal_grey_allocation():
    _seed_grey("SLUB20", 2000)
    r = fae.allocate_grey(
        {
            "grey_code": "SLUB20",
            "printed_code": "P208",
            "qty": 1000,
            "so_number": "SO-001",
            "fg_sku": "SKU1001",
            "user_name": "planner",
            "reason": "priority run",
        }
    )
    assert r["ok"] is True
    assert r["qty"] == 1000
    snap = fae.grey_stock_snapshot("SLUB20")[0]
    assert snap["grey_allocated_qty"] == 1000
    assert snap["available_qty"] == 1000  # free after move free→reserved
    hist = fae.list_allocation_history()
    assert any(h["event_type"] == fae.EVENT_GREY_ALLOC for h in hist)


# Scenario 2 — Partial allocation
def test_scenario_2_partial_grey_allocation():
    _seed_grey("SLUB20", 500)
    fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 200})
    fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 150})
    free = fae.grey_stock_snapshot("SLUB20")[0]["available_qty"]
    assert free == pytest.approx(150)
    allocs = fae.list_grey_allocations("Active")
    assert len(allocs) == 2
    assert sum(a["qty"] for a in allocs) == 350


def test_inline_multi_sku_grey_allocation_from_available_pool():
    """MRP screen: allocate available grey across SKUs; block when session total exceeds free."""
    _seed_grey("D7Denim1", 1000)
    rows = [
        ("SKU-A", 300),
        ("SKU-B", 250),
        ("SKU-C", 200),
        ("SKU-D", 150),
        ("SKU-E", 100),
    ]
    for sku, qty in rows:
        r = fae.allocate_grey(
            {
                "grey_code": "D7Denim1",
                "printed_code": "D7Denim1",
                "qty": qty,
                "so_number": "SO-0006",
                "fg_sku": sku,
                "reason": "MRP inline allocation",
            }
        )
        assert r["ok"] is True
    snap = fae.grey_stock_snapshot("D7Denim1")[0]
    assert snap["grey_allocated_qty"] == pytest.approx(1000)
    assert snap["available_qty"] == pytest.approx(0)
    with pytest.raises(FabricAllocationError, match="only"):
        fae.allocate_grey(
            {
                "grey_code": "D7Denim1",
                "printed_code": "D7Denim1",
                "qty": 1,
                "fg_sku": "SKU-A",
            }
        )


# Scenario 3 / 4 — Multiple SKUs same P-code + priority change
def test_scenario_3_4_multi_sku_and_reallocate():
    _seed_printed("P208", 1000)
    fae.allocate_printed(
        {
            "printed_code": "P208",
            "so_number": "SO-001",
            "fg_sku": "SKU1001",
            "qty": 1000,
            "user_name": "user1",
        }
    )
    # Priority change: move all to SKU1002
    out = fae.reallocate_printed(
        {
            "from_so": "SO-001",
            "from_sku": "SKU1001",
            "to_so": "SO-005",
            "to_sku": "SKU1002",
            "printed_code": "P208",
            "reason": "SKU1002 urgent",
            "user_name": "planner",
        }
    )
    assert out["ok"] is True
    assert out["to"]["sku"] == "SKU1002"
    active = fae.list_printed_allocations("Active")
    assert len(active) == 1
    assert active[0]["sku"] == "SKU1002"
    assert float(active[0]["qty"]) == 1000
    # Reporting uses current allocation
    rep = fae.fg_status_report("P208")
    skus = {r["fg_sku"]: r["report_status"] for r in rep}
    assert skus.get("SKU1002") == "Printed Available"
    assert "SKU1001" not in skus or skus.get("SKU1001") != "Printed Available"


# Scenario 5 — Reallocation before cutting succeeds
def test_scenario_5_realloc_before_cut_ok():
    _seed_printed("P208", 500)
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-A", "fg_sku": "SKU1", "qty": 300}
    )
    rid = fae.list_printed_allocations("Active")[0]["id"]
    r = fae.reallocate_printed(
        {
            "reservation_id": rid,
            "to_so": "SO-B",
            "to_sku": "SKU2",
            "qty": 100,
            "reason": "split demand",
        }
    )
    assert r["ok"]
    active = fae.list_printed_allocations("Active")
    by_sku = {a["sku"]: float(a["qty"]) for a in active}
    assert by_sku["SKU1"] == pytest.approx(200)
    assert by_sku["SKU2"] == pytest.approx(100)


# Scenario 6 — Reallocation after cutting fails
def test_scenario_6_realloc_after_cut_fails():
    _seed_printed("P208", 400)
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-X", "fg_sku": "SKU1", "qty": 400}
    )
    rid = fae.list_printed_allocations("Active")[0]["id"]
    fae.lock_printed_on_cutting_issue(
        jo_id=99, fabric_code="P208", so_number="SO-X", sku="SKU1", issued_qty=400, user_name="cutter"
    )
    with pytest.raises(FabricAllocationError, match="locked|Cutting"):
        fae.reallocate_printed(
            {
                "reservation_id": rid,
                "to_so": "SO-Y",
                "to_sku": "SKU2",
                "reason": "too late",
            }
        )
    hist = fae.list_allocation_history(printed_code="P208")
    assert any(h["event_type"] == fae.EVENT_PF_LOCK for h in hist)


# Scenario 7 — Partial printed receipt (stock then allocate partially)
def test_scenario_7_partial_printed_receipt_allocate():
    _seed_printed("P208", 500)  # QC passed 500 of a larger potential JO
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-1", "fg_sku": "SKU1", "qty": 200}
    )
    opts = grey_db.printed_fabric_reserve_options()
    fab = next(f for f in opts["fabrics"] if f["fabric_code"] == "P208")
    assert float(fab["available_qty"]) == pytest.approx(300)


# Scenario 8 — Multiple sales orders on same P-code
def test_scenario_8_multi_sales_orders():
    _seed_printed("P208", 600)
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-001", "fg_sku": "SKU1001", "qty": 250}
    )
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-005", "fg_sku": "SKU1002", "qty": 250}
    )
    active = fae.list_printed_allocations("Active")
    assert len(active) == 2
    assert {a["so_number"] for a in active} == {"SO-001", "SO-005"}


# Scenario 9 — Inventory shortage
def test_scenario_9_grey_shortage():
    _seed_grey("SLUB20", 100)
    with pytest.raises(FabricAllocationError, match="only"):
        fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 500})


def test_scenario_9_printed_shortage():
    _seed_printed("P208", 50)
    with pytest.raises(ValueError, match="only"):
        fae.allocate_printed(
            {"printed_code": "P208", "so_number": "SO-1", "fg_sku": "SKU1", "qty": 200}
        )


# Scenario 10 — Concurrent allocation
def test_scenario_10_concurrent_grey_alloc():
    _seed_grey("SLUB20", 1000)
    results: list = []
    errors: list = []

    def worker(q: float):
        try:
            results.append(
                fae.allocate_grey(
                    {"grey_code": "SLUB20", "printed_code": "P208", "qty": q, "user_name": "t"}
                )
            )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(700,))
    t2 = threading.Thread(target=worker, args=(700,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # At most one full success if both race; total allocated ≤ 1000
    total = sum(a["qty"] for a in fae.list_grey_allocations("Active"))
    assert total <= 1000 + 0.001
    assert len(results) + len(errors) == 2
    # With sqlite sequential commits, both may "succeed" if both read free before update —
    # engine free check is not transaction-locked globally; assert inventory not negative.
    snap = fae.grey_stock_snapshot("SLUB20")[0]
    assert float(snap["available_qty"]) >= -0.001


# Scenario 11 — Rollback / release
def test_scenario_11_release_grey_rollback():
    _seed_grey("SLUB20", 800)
    r = fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 400})
    fae.release_grey_allocation(r["allocation_id"], user_name="u", reason="cancel plan")
    snap = fae.grey_stock_snapshot("SLUB20")[0]
    assert snap["available_qty"] == pytest.approx(800)
    assert snap["grey_allocated_qty"] == 0
    assert fae.list_grey_allocations("Active") == []


# Scenario 12 / tree builds
def test_scenario_12_planning_tree_and_print_jo_status():
    _seed_grey("SLUB20", 1000)
    fae.allocate_grey(
        {
            "grey_code": "SLUB20",
            "printed_code": "P208",
            "qty": 500,
            "so_number": "SO-001",
            "fg_sku": "SKU1001",
        }
    )
    jo = fae.printing_jo_grey_status("P208")
    assert jo and float(jo[0]["grey_allocated"]) == 500
    assert jo[0]["ready_to_issue"] is True
    tree = fae.build_planning_tree()
    assert tree["ok"] is True
    assert "Grey" in tree["hierarchy"]


# Scenario 13 — Reporting accuracy after reallocation
def test_scenario_13_report_current_allocation_only():
    _seed_printed("P208", 1000)
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-001", "fg_sku": "SKU1001", "qty": 1000}
    )
    fae.reallocate_printed(
        {
            "from_so": "SO-001",
            "from_sku": "SKU1001",
            "to_so": "SO-005",
            "to_sku": "SKU1002",
            "reason": "priority",
        }
    )
    rep = {r["fg_sku"]: r for r in fae.fg_status_report("P208")}
    assert "SKU1002" in rep
    assert rep["SKU1002"]["report_status"] == "Printed Available"
    # Original intent SKU no longer active allocation
    assert "SKU1001" not in rep


# Scenario 14 — Audit history never deleted (append only)
def test_scenario_14_audit_trail():
    _seed_grey("SLUB20", 300)
    a = fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 100, "user_name": "a"})
    fae.release_grey_allocation(a["allocation_id"], user_name="b", reason="change")
    fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 50, "user_name": "c"})
    hist = fae.list_allocation_history(grey_code="SLUB20")
    types = [h["event_type"] for h in hist]
    assert fae.EVENT_GREY_ALLOC in types
    assert fae.EVENT_GREY_RELEASE in types
    # release does not delete history rows
    assert len(hist) >= 3


def test_release_printed_blocked_after_lock():
    _seed_printed("P208", 100)
    fae.allocate_printed(
        {"printed_code": "P208", "so_number": "SO-1", "fg_sku": "SKU1", "qty": 100}
    )
    rid = fae.list_printed_allocations("Active")[0]["id"]
    fae.lock_printed_on_cutting_issue(
        jo_id=1, fabric_code="P208", so_number="SO-1", sku="SKU1", issued_qty=100
    )
    with pytest.raises(FabricAllocationError, match="Cutting|locked"):
        fae.release_printed_allocation(rid, reason="nope")


def test_cannot_allocate_zero_or_missing():
    with pytest.raises(FabricAllocationError):
        fae.allocate_grey({"grey_code": "X", "printed_code": "P", "qty": 0})
    with pytest.raises(FabricAllocationError):
        fae.reallocate_printed({"to_so": "A", "to_sku": "B", "reason": ""})


def test_inventory_reconciliation_grey():
    _seed_grey("SLUB20", 1000)
    fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P208", "qty": 400})
    fae.allocate_grey({"grey_code": "SLUB20", "printed_code": "P209", "qty": 300})
    conn = grey_db._connect()
    row = conn.execute(
        "SELECT available_qty, reserved_qty, passed_qty FROM fabric_checked_stock WHERE fabric_code=?",
        ("SLUB20",),
    ).fetchone()
    conn.close()
    free = float(row["available_qty"])
    reserved = float(row["reserved_qty"])
    assert free == pytest.approx(300)
    assert reserved == pytest.approx(700)
    assert free + reserved == pytest.approx(float(row["passed_qty"]))

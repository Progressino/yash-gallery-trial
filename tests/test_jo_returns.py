"""JO returns (processed / unprocessed), material reconciliation and traceability."""
from __future__ import annotations

import pytest

from backend.db import grey_db, production_db, production_quality_db as pq, sales_db
from backend.services import jo_returns

ROUTE = ["Cutting", "Stitching", "Handwork", "Finishing"]


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
    monkeypatch.setattr(production_db, "get_item_routing", lambda sku: list(ROUTE))
    monkeypatch.setattr(production_db, "get_component_routing", lambda sku: list(ROUTE))
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    pq.init_quality_tables()
    yield


def _stock(so, sku, process, qty):
    conn = production_db._connect()
    production_db._update_process_stock(conn, so, sku, process, qty_in=qty)
    conn.commit()
    conn.close()


def _jo(so, sku, process, qty, vendor):
    num = production_db.create_jo(
        {
            "so_number": so,
            "so_source": "manual",
            "sku": sku,
            "process": process,
            "planned_qty": qty,
            "exec_type": "Outsource",
            "vendor_name": vendor,
            "create_component_jos": False,
            "lines": [{"sku": sku, "planned_qty": qty}],
        }
    )
    return production_db.get_jo_by_number(num)


def _ready(process, so, sku) -> int:
    row = next(
        (r for r in production_db.get_ready_to_process(process)
         if r.get("so_number") == so and r.get("sku") == sku),
        None,
    )
    return int(row["available_qty"]) if row else 0


def test_full_unprocessed_return_goes_back_to_ready_and_links_new_vendor(iso):
    _stock("SO-RT1", "1001-M", "Cutting", 100)
    jo_a = _jo("SO-RT1", "1001-M", "Stitching", 100, "Vendor A")
    assert _ready("Stitching", "SO-RT1", "1001-M") == 0

    res = jo_returns.record_return(
        jo_a["id"],
        {"lines": [{"jo_line_id": jo_a["lines"][0]["id"], "unprocessed_qty": 100}],
         "reason": "Vendor capacity"},
    )
    assert res["return_number"].startswith("RET-")
    assert res["unprocessed_qty"] == 100
    assert _ready("Stitching", "SO-RT1", "1001-M") == 100

    after = production_db.get_jo(jo_a["id"])
    assert after["status"] == "Closed"
    assert int(after["planned_qty"]) == 0
    assert int(after["unprocessed_return_qty"]) == 100
    assert int(after["lines"][0]["unprocessed_return_qty"]) == 100
    # Pieces-only JO with zero balance: nothing left for a person to reconcile.
    assert after["reconciliation_status"] == "Completed"

    jo_b = _jo("SO-RT1", "1001-M", "Stitching", 100, "Vendor B")
    assert _ready("Stitching", "SO-RT1", "1001-M") == 0

    hist = jo_returns.get_return_history(jo_a["id"])
    assert hist["summary"] == {
        "original_qty": 100,
        "processed_qty": 0,
        "unprocessed_returned_qty": 100,
        "reallocated_qty": 100,
        "pending_in_ready_qty": 0,
    }
    link = hist["outgoing"][0]
    assert link["source_vendor"] == "Vendor A"
    assert link["ready_at"]
    assert link["allocations"][0]["new_jo_number"] == jo_b["jo_number"]
    assert link["allocations"][0]["new_vendor"] == "Vendor B"

    incoming = jo_returns.get_return_history(jo_b["id"])["incoming"]
    assert incoming[0]["source_jo_number"] == jo_a["jo_number"]
    assert incoming[0]["return_number"] == res["return_number"]
    assert incoming[0]["qty"] == 100


def test_mixed_return_processed_moves_on_unprocessed_back_to_same_process(iso):
    _stock("SO-RT2", "1001-L", "Stitching", 100)
    hw = _jo("SO-RT2", "1001-L", "Handwork", 100, "HW Vendor")
    assert _ready("Handwork", "SO-RT2", "1001-L") == 0

    jo_returns.record_return(
        hw["id"],
        {"lines": [{"jo_line_id": hw["lines"][0]["id"], "processed_qty": 50, "unprocessed_qty": 50}]},
    )
    after = production_db.get_jo(hw["id"])
    assert int(after["received_qty"]) == 50
    assert int(after["planned_qty"]) == 50
    assert after["status"] == "Closed"
    # Processed pcs continue through the normal flow: issue forward from the Handwork JO.
    production_db.issue_pieces(
        hw["id"], {"issued_qty": 50, "jo_line_id": hw["lines"][0]["id"], "to_process": "Finishing"}
    )
    assert _ready("Handwork", "SO-RT2", "1001-L") == 50
    assert _ready("Finishing", "SO-RT2", "1001-L") == 50

    rows = jo_returns.get_reconciliation(hw["id"])["rows"]
    pieces = next(r for r in rows if r["material_kind"] == "PIECES")
    assert (pieces["issued_qty"], pieces["consumed_qty"], pieces["returned_qty"], pieces["balance_qty"]) == (
        100, 50, 50, 0
    )


def test_return_cannot_exceed_pending_with_vendor(iso):
    _stock("SO-RT3", "1001-S", "Cutting", 100)
    jo = _jo("SO-RT3", "1001-S", "Stitching", 100, "Vendor A")
    production_db.receive_pieces(jo["id"], {"received_qty": 70, "jo_line_id": jo["lines"][0]["id"]})
    with pytest.raises(ValueError, match="exceeds 30 pending"):
        jo_returns.record_return(
            jo["id"], {"lines": [{"jo_line_id": jo["lines"][0]["id"], "unprocessed_qty": 31}]}
        )
    with pytest.raises(ValueError, match="Enter processed"):
        jo_returns.record_return(jo["id"], {"lines": []})


def test_partial_return_keeps_reconciliation_pending_until_balance_settled(iso):
    _stock("SO-RT4", "1001-XL", "Cutting", 100)
    jo = _jo("SO-RT4", "1001-XL", "Stitching", 100, "Vendor A")
    line_id = jo["lines"][0]["id"]
    jo_returns.record_return(jo["id"], {"lines": [{"jo_line_id": line_id, "unprocessed_qty": 40}]})
    after = production_db.get_jo(jo["id"])
    assert after["status"] != "Closed"
    assert after["reconciliation_status"] == "Pending"
    assert _ready("Stitching", "SO-RT4", "1001-XL") == 40

    bill = pq.jo_billing_eligibility(jo["id"])
    assert bill["eligible_billing_qty"] == 0
    assert "Reconciliation pending" in bill["billing_status"]

    with pytest.raises(ValueError, match="balance with vendor remains"):
        jo_returns.save_reconciliation(jo["id"], {"rows": [], "complete": True})

    production_db.receive_pieces(jo["id"], {"received_qty": 55, "jo_line_id": line_id})
    rec = jo_returns.save_reconciliation(
        jo["id"],
        {"rows": [{"material_kind": "PIECES", "material_code": "1001-XL", "wastage_qty": 5,
                   "remarks": "Lost at vendor"}],
         "complete": True, "reconciled_by": "tester"},
    )
    assert rec["reconciliation_status"] == "Completed"
    final = production_db.get_jo(jo["id"])
    assert final["status"] == "Closed"
    assert final["reconciled_by"] == "tester"
    # Lost pieces stay committed — they must not reappear on Ready-To
    # (40 returned + 55 received still at Stitching until issued forward; never 100).
    assert _ready("Stitching", "SO-RT4", "1001-XL") == 95


def test_fabric_issued_requires_reconciliation_after_close(iso):
    _stock("SO-RT5", "1001-XS", "Cutting", 20)
    jo = _jo("SO-RT5", "1001-XS", "Stitching", 20, "Vendor A")
    production_db.issue_fabric(jo["id"], {"fabric_code": "FAB-1", "issued_qty": 10, "unit": "MTR"})
    production_db.receive_pieces(jo["id"], {"received_qty": 20, "jo_line_id": jo["lines"][0]["id"]})
    after = production_db.get_jo(jo["id"])
    assert after["status"] == "Closed"
    assert after["reconciliation_status"] == "Pending"

    fab = next(r for r in jo_returns.get_reconciliation(jo["id"])["rows"] if r["material_kind"] == "FABRIC")
    assert fab["issued_qty"] == 10
    assert not fab["settled"]

    with pytest.raises(ValueError, match="FAB-1"):
        jo_returns.save_reconciliation(
            jo["id"],
            {"rows": [{"material_kind": "FABRIC", "material_code": "FAB-1", "consumed_qty": 8}],
             "complete": True},
        )
    rec = jo_returns.save_reconciliation(
        jo["id"],
        {"rows": [{"material_kind": "FABRIC", "material_code": "FAB-1", "consumed_qty": 8,
                   "wastage_qty": 1}]},
    )
    assert rec["reconciliation_status"] == "Pending"
    fab = next(r for r in rec["rows"] if r["material_kind"] == "FABRIC")
    assert fab["balance_qty"] == 1

    production_db.update_jo(jo["id"], {})  # no-op edit must not disturb reconciliation
    with pytest.raises(ValueError, match="not issued"):
        jo_returns.save_reconciliation(
            jo["id"], {"rows": [{"material_kind": "ACCESSORY", "material_code": "BTN-X"}]}
        )
    rec = jo_returns.save_reconciliation(
        jo["id"],
        {"rows": [{"material_kind": "FABRIC", "material_code": "FAB-1", "consumed_qty": 8,
                   "wastage_qty": 2}],
         "complete": True},
    )
    assert rec["reconciliation_status"] == "Completed"


def test_accessories_from_issue_note_prefill_and_are_editable(iso, monkeypatch):
    _stock("SO-RT6", "1001-M", "Cutting", 10)
    jo = _jo("SO-RT6", "1001-M", "Stitching", 10, "Vendor A")
    note = {"lines": [
        {"material_code": "BTN-01", "material_name": "Button", "unit": "PCS",
         "bom_qty_per_unit": 6, "required_qty": 60, "issued_qty": 0},
        {"material_code": "THR-01", "material_name": "Thread", "unit": "CONE",
         "bom_qty_per_unit": 0.1, "required_qty": 1, "issued_qty": 0},
    ]}
    monkeypatch.setattr(
        "backend.services.jo_issue_notes.get_issue_note_by_jo_id", lambda joid: note
    )
    production_db.receive_pieces(jo["id"], {"received_qty": 10, "jo_line_id": jo["lines"][0]["id"]})
    assert production_db.get_jo(jo["id"])["reconciliation_status"] == "Pending"
    rows = {r["material_code"]: r for r in jo_returns.get_reconciliation(jo["id"])["rows"]}
    btn = rows["BTN-01"]
    assert btn["issued_qty"] == 60
    assert btn["expected_consumption"] == 60
    assert btn["consumed_qty"] == 60
    assert btn["settled"]
    assert not btn["confirmed"]

    rec = jo_returns.save_reconciliation(
        jo["id"],
        {"rows": [
            {"material_kind": "ACCESSORY", "material_code": "BTN-01", "issued_qty": 64,
             "consumed_qty": 60, "returned_qty": 3, "wastage_qty": 1},
            {"material_kind": "ACCESSORY", "material_code": "THR-01", "consumed_qty": 1},
        ], "complete": True},
    )
    assert rec["reconciliation_status"] == "Completed"
    btn = next(r for r in rec["rows"] if r["material_code"] == "BTN-01")
    assert (btn["issued_qty"], btn["returned_qty"], btn["balance_qty"]) == (64, 3, 0)


def test_cancelling_follow_up_jo_releases_returned_qty(iso):
    _stock("SO-RT7", "1001-S", "Cutting", 30)
    jo_a = _jo("SO-RT7", "1001-S", "Stitching", 30, "Vendor A")
    jo_returns.record_return(
        jo_a["id"], {"lines": [{"jo_line_id": jo_a["lines"][0]["id"], "unprocessed_qty": 30}]}
    )
    jo_b = _jo("SO-RT7", "1001-S", "Stitching", 30, "Vendor B")
    production_db.update_jo(jo_b["id"], {"status": "Cancelled"})
    hist = jo_returns.get_return_history(jo_a["id"])
    assert hist["summary"]["pending_in_ready_qty"] == 30
    assert hist["summary"]["reallocated_qty"] == 0
    assert _ready("Stitching", "SO-RT7", "1001-S") == 30
    jo_c = _jo("SO-RT7", "1001-S", "Stitching", 30, "Vendor C")
    alloc = jo_returns.get_return_history(jo_a["id"])["outgoing"][0]["allocations"]
    assert [a["new_vendor"] for a in alloc if not a["released"]] == ["Vendor C"]
    assert jo_c["id"]


def test_return_api_endpoints(iso, client):
    _stock("SO-RT8", "1001-M", "Cutting", 12)
    jo = _jo("SO-RT8", "1001-M", "Stitching", 12, "Vendor A")
    r = client.post(
        f"/api/production/orders/{jo['id']}/return",
        json={"lines": [{"jo_line_id": jo["lines"][0]["id"], "processed_qty": 2, "unprocessed_qty": 10}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["return_number"].startswith("RET-")
    bad = client.post(
        f"/api/production/orders/{jo['id']}/return",
        json={"lines": [{"jo_line_id": jo["lines"][0]["id"], "unprocessed_qty": 5}]},
    )
    assert bad.status_code == 400
    rec = client.get(f"/api/production/orders/{jo['id']}/reconciliation")
    assert rec.status_code == 200
    assert rec.json()["rows"][0]["material_kind"] == "PIECES"
    hist = client.get(f"/api/production/orders/{jo['id']}/return-history")
    assert hist.json()["summary"]["unprocessed_returned_qty"] == 10
    lst = client.get("/api/production/returns", params={"so_number": "SO-RT8"})
    assert lst.status_code == 200
    assert lst.json()[0]["links"][0]["unprocessed_qty"] == 10

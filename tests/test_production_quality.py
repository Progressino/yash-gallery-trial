"""Production QC + Rework WIP + Debit Notes + Billing eligibility."""
from __future__ import annotations

import pytest

from backend.db import production_db, production_quality_db as pq


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production_quality.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    pq.init_quality_tables()
    yield


def _seed_vendor_jo(process="Stitching", vendor="Vendor A", planned=100, received=100):
    num = production_db.create_jo(
        {
            "so_number": "SO-QC-1",
            "so_source": "manual",
            "sku": "SKU-QC-M",
            "process": process,
            "planned_qty": planned,
            "exec_type": "Outsource",
            "vendor_name": vendor,
            "vendor_rate": 25,
            "jo_date": "2026-09-01",
            "create_component_jos": False,
            "lines": [
                {
                    "sku": "SKU-QC-M",
                    "style": "M",
                    "planned_qty": planned,
                    "component_code": "TOP",
                    "parent_sku": "SKU-QC-M",
                }
            ],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    if received:
        production_db.receive_pieces(
            jo["id"],
            {
                "received_qty": received,
                "jo_line_id": jo["lines"][0]["id"],
                "receipt_date": "2026-09-02",
            },
        )
    return jo


def test_qc_creates_rework_and_holds_billing(iso):
    stitch = _seed_vendor_jo()
    fin_num = production_db.create_jo(
        {
            "so_number": "SO-QC-1",
            "so_source": "manual",
            "sku": "SKU-QC-M",
            "process": "Finishing",
            "planned_qty": 100,
            "exec_type": "Inhouse",
            "jo_date": "2026-09-03",
            "create_component_jos": False,
            "lines": [
                {
                    "sku": "SKU-QC-M",
                    "style": "M",
                    "planned_qty": 100,
                    "component_code": "TOP",
                    "parent_sku": "SKU-QC-M",
                }
            ],
        }
    )
    fin = next(j for j in production_db.list_jos() if j["jo_number"] == fin_num)

    report = pq.create_qc_report(
        {
            "found_at_process": "Finishing",
            "original_jo_id": fin["id"],
            "so_number": "SO-QC-1",
            "sku": "SKU-QC-M",
            "checked_qty": 100,
            "pass_qty": 90,
            "rework_qty": 10,
            "reject_qty": 0,
            "defects": [
                {
                    "defect_source_process": "Stitching",
                    "action": "Rework",
                    "qty": 10,
                    "reason": "Stitching defect",
                    "responsible_vendor": "Vendor A",
                    "responsible_jo_id": stitch["id"],
                    "rework_by_type": "SameVendor",
                    "rework_by_vendor": "Vendor A",
                    "debit_required": 0,
                }
            ],
        }
    )
    assert report["report_no"].startswith("QC-")
    assert report["pass_qty"] == 90
    assert len(report["defects"]) == 1
    assert len(report["created_reworks"]) == 1
    rw = report["created_reworks"][0]
    assert rw["planned_qty"] == 10
    assert rw["balance_qty"] == 10
    assert int(rw["chargeable"]) == 0
    assert rw["process"] == "Stitching"

    # Original stitching planned stays 100 — rework is separate WIP
    stitch2 = production_db.get_jo(stitch["id"])
    assert int(stitch2["planned_qty"]) == 100
    assert int(stitch2["received_qty"]) == 100

    wip = pq.rework_wip_summary(so_number="SO-QC-1")
    assert any(int(w["rework_pending"]) == 10 for w in wip)

    bill = pq.jo_billing_eligibility(stitch["id"])
    # QC linked via SO/SKU at Finishing (billing QC process)
    assert bill["qc_pass_qty"] >= 90
    assert bill["rework_pending_qty"] >= 10
    assert bill["eligible_billing_qty"] == 90
    assert "rework" in bill["billing_status"].lower() or bill["eligible_billing_qty"] == 90


def test_inhouse_rework_creates_debit(iso):
    stitch = _seed_vendor_jo()
    report = pq.create_qc_report(
        {
            "found_at_process": "Finishing",
            "original_jo_id": stitch["id"],
            "so_number": "SO-QC-1",
            "sku": "SKU-QC-M",
            "checked_qty": 100,
            "defects": [
                {
                    "defect_source_process": "Stitching",
                    "action": "Rework",
                    "qty": 10,
                    "reason": "Repair in-house",
                    "responsible_vendor": "Vendor A",
                    "responsible_jo_id": stitch["id"],
                    "rework_by_type": "Inhouse",
                    "debit_required": 1,
                }
            ],
        }
    )
    assert len(report["created_debits"]) == 1
    dn = report["created_debits"][0]
    assert dn["responsible_vendor"] == "Vendor A"
    assert float(dn["workmanship_amount"]) == 250.0  # 10 × 25
    assert int(dn["with_fabric"]) == 0


def test_final_reject_with_fabric_debit(iso):
    stitch = _seed_vendor_jo()
    report = pq.create_qc_report(
        {
            "found_at_process": "Finishing",
            "original_jo_id": stitch["id"],
            "so_number": "SO-QC-1",
            "sku": "SKU-QC-M",
            "checked_qty": 100,
            "defects": [
                {
                    "defect_source_process": "Stitching",
                    "action": "FinalReject",
                    "qty": 5,
                    "reason": "Unrepairable",
                    "responsible_vendor": "Vendor A",
                    "responsible_jo_id": stitch["id"],
                    "rework_by_type": "Inhouse",
                    "debit_required": 1,
                    "with_fabric": 1,
                }
            ],
        }
    )
    assert report["reject_qty"] == 5
    assert len(report["created_reworks"]) == 0  # FinalReject → no rework
    assert len(report["created_debits"]) == 1
    assert int(report["created_debits"][0]["with_fabric"]) == 1


def test_rework_partial_recovery(iso):
    stitch = _seed_vendor_jo()
    report = pq.create_qc_report(
        {
            "found_at_process": "Finishing",
            "original_jo_id": stitch["id"],
            "so_number": "SO-QC-1",
            "sku": "SKU-QC-M",
            "checked_qty": 100,
            "defects": [
                {
                    "defect_source_process": "Stitching",
                    "action": "Rework",
                    "qty": 10,
                    "responsible_vendor": "Vendor A",
                    "responsible_jo_id": stitch["id"],
                    "rework_by_type": "SameVendor",
                }
            ],
        }
    )
    rid = report["created_reworks"][0]["id"]
    out = pq.receive_rework(
        rid,
        {"received_qty": 10, "pass_qty": 8, "reject_qty": 2, "remarks": "partial recovery"},
    )
    assert out["pass_qty"] == 8
    assert out["reject_qty"] == 2
    assert out["balance_qty"] == 0
    assert out["status"] == "Completed"


def test_billing_hold_without_qc(iso):
    stitch = _seed_vendor_jo()
    bill = pq.jo_billing_eligibility(stitch["id"])
    assert bill["eligible_billing_qty"] == 0
    assert "QC pending" in bill["billing_status"]


def test_qc_billing_config(iso):
    assert pq.get_billing_qc_process("default") == "Finishing"
    pq.set_billing_qc_process("inhouse", "Initial")
    assert pq.get_billing_qc_process("inhouse") == "Initial"

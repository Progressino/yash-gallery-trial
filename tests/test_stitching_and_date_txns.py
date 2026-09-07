"""Stitching report + date-wise process transactions."""
from __future__ import annotations

from backend.db import production_db
from backend.services.cutting_reports import build_cutting_report
from backend.services.process_date_transactions import build_process_date_transactions


def _seed_stitching_jo(tmp_path, monkeypatch):
    db = str(tmp_path / "prod_stitch.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", db)
    monkeypatch.setattr(production_db, "_DB", db)
    production_db.init_db()
    num = production_db.create_jo(
        {
            "so_number": "SO-ST-1",
            "so_source": "manual",
            "sku": "SKU-1001-M",
            "process": "Stitching",
            "planned_qty": 100,
            "exec_type": "Outsource",
            "vendor_name": "Vendor A",
            "jo_date": "2026-09-01",
            "create_component_jos": False,
            "lines": [
                {
                    "sku": "SKU-1001-M",
                    "style": "M",
                    "planned_qty": 100,
                    "component_code": "TOP",
                    "parent_sku": "SKU-1001-M",
                }
            ],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    joid = jo["id"]
    lid = jo["lines"][0]["id"]
    production_db.receive_pieces(
        joid, {"receipt_date": "2026-09-02", "received_qty": 50, "jo_line_id": lid, "remarks": "partial 1"}
    )
    production_db.receive_pieces(
        joid, {"receipt_date": "2026-09-03", "received_qty": 20, "jo_line_id": lid, "remarks": "partial 2"}
    )
    production_db.receive_pieces(
        joid, {"receipt_date": "2026-09-04", "received_qty": 30, "jo_line_id": lid, "remarks": "partial 3"}
    )
    return joid


def test_stitching_report_includes_vendor(tmp_path, monkeypatch):
    _seed_stitching_jo(tmp_path, monkeypatch)
    rep = build_cutting_report(process="Stitching", group_by="vendor", export=True)
    assert rep["process"] == "Stitching"
    assert rep["kpis"]["planned_qty"] == 100
    assert rep["kpis"]["received_qty"] == 100
    assert any(r.get("vendor_name") == "Vendor A" for r in rep["rows"])
    assert "column_totals" in rep
    assert rep["column_totals"]["planned_qty"] == 100


def test_date_wise_receipts_are_separate_rows(tmp_path, monkeypatch):
    _seed_stitching_jo(tmp_path, monkeypatch)
    data = build_process_date_transactions(
        process="Stitching",
        txn_type="received",
        date_from="2026-09-01",
        date_to="2026-09-30",
        export=True,
    )
    recs = [r for r in data["rows"] if r["txn_type"] == "received"]
    assert len(recs) == 3
    assert sorted(int(r["qty"]) for r in recs) == [20, 30, 50]
    assert data["totals"]["received_qty"] == 100

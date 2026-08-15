"""Cutting report KPIs, aging, over/under, fabric saving."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.db import production_db
from backend.services.cutting_reports import build_cutting_report, _aging_bucket, _qty_status


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    yield


def test_aging_buckets():
    assert _aging_bucket(0) == "0-2"
    assert _aging_bucket(2) == "0-2"
    assert _aging_bucket(3) == "3-5"
    assert _aging_bucket(16) == "15+"
    assert _aging_bucket(None) == ""


def test_qty_status_over_under_pending():
    assert _qty_status(40, 42, "In Progress") == "over"
    assert _qty_status(40, 40, "Completed") == "exact"
    assert _qty_status(40, 35, "Completed") == "under"
    assert _qty_status(40, 35, "Created") == "pending"


def test_cutting_report_kpis_and_over_receipt(iso):
    num = production_db.create_jo(
        {
            "so_number": "SO-CUT-1",
            "so_source": "manual",
            "sku": "1001-M",
            "process": "Cutting",
            "planned_qty": 40,
            "fabric_code": "P-RED",
            "fabric_qty": 40,
            "component_code": "TOP",
            "main_sku": "1001-M",
            "sku_role": "COMPONENT",
            "create_component_jos": False,
            "jo_date": (date.today() - timedelta(days=4)).isoformat(),
            "lines": [{"sku": "1001-M", "style": "M", "planned_qty": 40, "component_code": "TOP", "parent_sku": "1001-M"}],
        }
    )
    jo = next(j for j in production_db.list_jos() if j["jo_number"] == num)
    production_db.receive_pieces(jo["id"], {"received_qty": 42, "jo_line_id": jo["lines"][0]["id"]})
    import sqlite3

    conn = sqlite3.connect(production_db._DB)
    conn.execute("UPDATE job_orders SET fabric_issued_qty=32, fabric_consumption=32 WHERE id=?", (jo["id"],))
    conn.commit()
    conn.close()

    out = build_cutting_report(export=True)
    assert out["ok"]
    assert out["kpis"]["planned_qty"] == 40
    assert out["kpis"]["received_qty"] == 42
    assert out["kpis"]["over_qty"] == 2
    row = out["rows"][0]
    assert row["qty_variance"] == 2
    assert row["balance_qty"] == -2
    assert row["status"] == "over"
    assert row["component"] == "TOP"
    assert row["bom_avg"] == 1.0
    assert row["actual_avg"] is not None
    assert row["fabric_saving"] is not None
    grouped = build_cutting_report(group_by="component", export=True)
    assert grouped["groups"][0]["group"] == "TOP"


def test_cutting_report_filters_and_zero_fabric(iso):
    production_db.create_jo(
        {
            "so_number": "SO-CUT-2",
            "sku": "2002-S",
            "process": "Cutting",
            "planned_qty": 10,
            "create_component_jos": False,
        }
    )
    out = build_cutting_report(so_number="SO-CUT-2", export=True)
    assert out["total"] == 1
    assert out["rows"][0]["actual_avg"] is None
    empty = build_cutting_report(so_number="NOPE", export=True)
    assert empty["total"] == 0
    assert empty["kpis"]["planned_qty"] == 0

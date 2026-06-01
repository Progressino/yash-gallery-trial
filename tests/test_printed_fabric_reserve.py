"""Printed fabric reserve — checked stock + SO options only."""
from __future__ import annotations

import sqlite3

import pytest

from backend.db import grey_db, sales_db


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


def test_reserve_options_only_checked_available():
    grey_db.insert_printed_fabric_unchecked("P308", 100, fabric_name="Print A", jwo_ref="J1", grn_ref="GRN-1")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "P308", "fabric_name": "Print A", "jwo_ref": "J1", "passed_qty": 100, "qc_by": "QC"}
    )
    grey_db.reserve_printed_fabric(
        {"fabric_code": "P308", "fabric_name": "Print A", "so_number": "SO-001", "sku": "SKU1", "qty": 40}
    )
    opts = grey_db.printed_fabric_reserve_options()
    codes = [f["fabric_code"] for f in opts["fabrics"]]
    assert "P308" in codes
    assert all(float(f.get("available_qty") or 0) > 0 for f in opts["fabrics"])


def test_reserve_rejects_unknown_fabric():
    sales_db.create_order(
        {
            "buyer": "Buyer",
            "status": "Confirmed",
            "lines": [{"sku": "STYLE-1", "sku_name": "Dress", "qty": 10}],
        }
    )
    with pytest.raises(ValueError, match="not in checked stock"):
        grey_db.reserve_printed_fabric(
            {"fabric_code": "NOPE", "so_number": "SO-001", "sku": "STYLE-1", "qty": 5}
        )


def test_reserve_options_include_open_so_lines():
    num = sales_db.create_order(
        {
            "buyer": "Retail",
            "status": "Confirmed",
            "lines": [{"sku": "1894GREEN", "sku_name": "Green Kurti", "qty": 50}],
        }
    )
    grey_db.insert_printed_fabric_unchecked("P999", 200, fabric_name="Test", jwo_ref="J2", grn_ref="GRN-2")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "P999", "fabric_name": "Test", "jwo_ref": "J2", "passed_qty": 200, "qc_by": "QC"}
    )
    opts = grey_db.printed_fabric_reserve_options()
    sos = [o["so_number"] for o in opts["sales_orders"]]
    assert num in sos
    so = next(o for o in opts["sales_orders"] if o["so_number"] == num)
    assert so["lines"][0]["sku"] == "1894GREEN"


def test_reserve_options_hide_reserved_sku(tmp_path, monkeypatch):
    so_num = sales_db.create_order(
        {
            "buyer": "Retail",
            "status": "Confirmed",
            "lines": [{"sku": "STYLE-A", "sku_name": "A", "qty": 10}],
        }
    )
    grey_db.insert_printed_fabric_unchecked("P100", 100, fabric_name="Fab", jwo_ref="J3", grn_ref="GRN-3")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "P100", "fabric_name": "Fab", "jwo_ref": "J3", "passed_qty": 100, "qc_by": "QC"}
    )
    grey_db.reserve_printed_fabric(
        {"fabric_code": "P100", "so_number": so_num, "sku": "STYLE-A", "qty": 20}
    )
    opts = grey_db.printed_fabric_reserve_options()
    sos = [o for o in opts["sales_orders"] if o["so_number"] == so_num]
    assert sos == []


def test_ready_to_cut_hidden_after_cutting_jo(tmp_path, monkeypatch):
    prod_path = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod_path)
    monkeypatch.setattr(grey_db, "_PRODUCTION_DB", prod_path)

    so_num = sales_db.create_order(
        {
            "buyer": "Retail",
            "status": "Confirmed",
            "lines": [{"sku": "STYLE-B", "sku_name": "B", "qty": 5}],
        }
    )
    grey_db.insert_printed_fabric_unchecked("P200", 50, fabric_name="Fab2", jwo_ref="J4", grn_ref="GRN-4")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "P200", "fabric_name": "Fab2", "jwo_ref": "J4", "passed_qty": 50, "qc_by": "QC"}
    )
    grey_db.reserve_printed_fabric(
        {"fabric_code": "P200", "so_number": so_num, "sku": "STYLE-B", "qty": 30}
    )
    assert len(grey_db.list_printed_fabric_ready_to_cut()) == 1

    conn = sqlite3.connect(prod_path)
    conn.execute(
        """CREATE TABLE job_orders (
            id INTEGER PRIMARY KEY, so_number TEXT, sku TEXT, process TEXT,
            status TEXT, planned_qty INTEGER)"""
    )
    conn.execute(
        "INSERT INTO job_orders(so_number, sku, process, status, planned_qty) VALUES (?,?,?,?,?)",
        (so_num, "STYLE-B", "Cutting", "Created", 5),
    )
    conn.commit()
    conn.close()

    assert grey_db.list_printed_fabric_ready_to_cut() == []

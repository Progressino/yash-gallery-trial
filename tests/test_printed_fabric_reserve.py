"""Printed fabric reserve — checked stock + SO options only."""
from __future__ import annotations

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

"""
Dry-run style integration tests for Finance DB: manual voucher + finance sales entry,
then Day Book, GSTR-3B, and Trial Balance visibility (isolated SQLite file).
"""

import json

import pytest

import backend.db.finance_db as fdb


@pytest.fixture
def fin_db(tmp_path, monkeypatch):
    path = tmp_path / "finance_dry_run.db"
    monkeypatch.setattr(fdb, "DB_PATH", str(path))
    fdb.init_db()
    return path


def test_dry_run_expense_and_sales_entry_reflect_in_reports(fin_db):
    """Accountant UAT simulation: post expense + sales line entry; verify summary queries."""
    vno = fdb.create_expense_voucher(
        {
            "voucher_type": "Expense",
            "voucher_date": "2026-05-15",
            "party_name": "Test Vendor Pvt Ltd",
            "party_gstin": "",
            "party_state": "Maharashtra",
            "bill_no": "BILL-DRY-1",
            "bill_date": "2026-05-15",
            "supply_type": "Intra",
            "narration": "Dry-run expense",
            "taxable_amount": 1000.0,
            "cgst_amount": 90.0,
            "sgst_amount": 90.0,
            "igst_amount": 0.0,
            "tds_section": "",
            "tds_rate": 0.0,
            "tds_amount": 0.0,
            "total_amount": 1180.0,
            "net_payable": 1180.0,
            "lines": [
                {
                    "expense_head": "Office Expenses",
                    "description": "Dry-run line",
                    "amount": 1000.0,
                    "cost_centre": "",
                    "is_debit": 1,
                }
            ],
        }
    )
    assert vno

    uid = fdb.create_finance_sales_upload(
        {
            "platform": "Amazon",
            "company_name": "Dry Run Co",
            "seller_gstin": "29AABCY3804E1ZF",
            "company_state": "Karnataka",
            "period": "2026-05",
            "filename": "dry.csv",
            "total_revenue": 500.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 500.0,
            "uploaded_by": "pytest",
            "upload_notes": "dry run",
        }
    )
    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-05",
                "voucher_date": "2026-05-15",
                "invoice_no": "INV-DRY-1",
                "order_id": "ORD-DRY-1",
                "party_name": "Buyer Dry",
                "party_gstin": "29AABCY3804E1ZF",
                "party_state": "Karnataka",
                "ship_to_state": "TN",
                "taxable_amount": 500.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 25.0,
                "total_amount": 525.0,
                "net_payable": 525.0,
                "narration": "Dry-run shipment",
                "source_filename": "dry.csv",
                "line_items": json.dumps(
                    [{"sku": "SKU-DRY", "quantity": 1, "invoice_amount": 500.0, "total_tax": 25.0, "ship_to_state": "TN"}]
                ),
            }
        ],
    )

    day = fdb.get_voucher_summary_by_date("2026-05-15")
    kinds = {(x["voucher_no"], x["voucher_type"]) for x in day}
    assert any(x["voucher_type"] == "Expense" for x in day)
    assert any(str(x["voucher_no"]).startswith("SUE-") for x in day)

    gstr = fdb.get_gstr3b_data("2026-05-01", "2026-05-31")
    assert gstr["outward"]["taxable"] >= 500.0
    br = gstr.get("breakdown") or []
    assert any("SUE-" in str(r.get("voucher_no", "")) for r in br)

    tb = fdb.get_trial_balance("2026-05-01", "2026-05-31")
    rows = {r["ledger"]: r for r in tb.get("rows", [])}
    assert "Finance Sales Upload A/c" in rows
    assert rows["Finance Sales Upload A/c"]["credit"] >= 525.0
    assert "Office Expenses" in rows

    detail = next(x for x in day if str(x["voucher_no"]).startswith("SUE-"))
    v = fdb.get_sales_entry_voucher(detail["id"])
    assert v is not None
    assert v["meta"]["invoice_no"] == "INV-DRY-1"
    assert v["meta"]["order_id"] == "ORD-DRY-1"

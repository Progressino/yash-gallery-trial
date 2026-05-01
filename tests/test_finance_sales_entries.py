"""Finance sales invoice-level entries (Day Book) — DB helpers."""

import json

import pytest

import backend.db.finance_db as fdb


@pytest.fixture
def fin_db(tmp_path, monkeypatch):
    path = tmp_path / "finance_test.db"
    monkeypatch.setattr(fdb, "DB_PATH", str(path))
    fdb.init_db()
    return path


def test_finance_sales_entries_daybook_by_voucher_date(fin_db):
    uid = fdb.create_finance_sales_upload(
        {
            "platform": "Amazon",
            "company_name": "TestCo",
            "seller_gstin": "29AABCY3804E1ZF",
            "company_state": "Karnataka",
            "period": "2026-04",
            "filename": "mtr.csv",
            "total_revenue": 100.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 100.0,
            "uploaded_by": "",
            "upload_notes": "",
        }
    )
    line_items = json.dumps(
        [{"sku": "SKU1", "quantity": 1, "invoice_amount": 100.0, "total_tax": 5.0, "ship_to_state": "TN"}]
    )
    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-04",
                "voucher_date": "2026-04-15",
                "invoice_no": "INV-1",
                "order_id": "O-99",
                "party_name": "Buyer A",
                "party_gstin": "29AABCY3804E1ZF",
                "party_state": "Karnataka",
                "ship_to_state": "TN",
                "taxable_amount": 100.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 5.0,
                "total_amount": 105.0,
                "net_payable": 105.0,
                "narration": "test",
                "source_filename": "mtr.csv",
                "line_items": line_items,
            }
        ],
    )
    day15 = fdb.get_voucher_summary_by_date("2026-04-15")
    nos = [v["voucher_no"] for v in day15]
    assert any(n.startswith("SUE-") for n in nos)
    assert not any(n.startswith("SU-") for n in nos)

    day30 = fdb.get_voucher_summary_by_date("2026-04-30")
    assert not any(v["voucher_no"].startswith("SUE-") for v in day30)

    sue_id = next(v["id"] for v in day15 if str(v["voucher_no"]).startswith("SUE-"))
    detail = fdb.get_sales_entry_voucher(sue_id)
    assert detail is not None
    assert detail["meta"]["invoice_no"] == "INV-1"
    assert detail["meta"]["order_id"] == "O-99"
    assert len(detail["meta"]["line_items"]) == 1


def test_delete_sales_upload_removes_entries(fin_db):
    uid = fdb.create_finance_sales_upload(
        {
            "platform": "Amazon",
            "period": "2026-04",
            "filename": "x.csv",
            "total_revenue": 10.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 10.0,
            "uploaded_by": "",
            "upload_notes": "",
        }
    )
    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-04",
                "voucher_date": "2026-04-01",
                "invoice_no": "A",
                "order_id": "1",
                "party_name": "P",
                "party_gstin": "",
                "party_state": "",
                "ship_to_state": "",
                "taxable_amount": 10.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": 10.0,
                "net_payable": 10.0,
                "narration": "",
                "source_filename": "",
                "line_items": "[]",
            }
        ],
    )
    assert fdb.delete_finance_sales_upload(uid)
    day = fdb.get_voucher_summary_by_date("2026-04-01")
    assert not any(v["voucher_no"].startswith("SUE-") for v in day)


def test_chart_of_accounts_has_groups(fin_db):
    tree = fdb.get_chart_of_accounts()
    assert "groups" in tree
    assert isinstance(tree["groups"], list)
    assert len(tree["groups"]) >= 1


def test_trial_balance_returns_structure(fin_db):
    tb = fdb.get_trial_balance("2026-01-01", "2026-12-31")
    assert "rows" in tb and isinstance(tb["rows"], list)
    assert "balanced" in tb and "total_debit" in tb and "total_credit" in tb

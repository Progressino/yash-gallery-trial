"""HTTP tests for /api/finance/* on an isolated finance DB."""


def test_finance_pin_required(client):
    r = client.get("/api/finance/pin-required")
    assert r.status_code == 200
    body = r.json()
    assert "required" in body
    assert isinstance(body["required"], bool)


def test_finance_verify_pin_when_unset(client, monkeypatch):
    monkeypatch.setattr("backend.routers.finance._FINANCE_PIN", "")
    r = client.post("/api/finance/verify-pin", json={"pin": "anything"})
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_finance_verify_pin_rejects_wrong(client, monkeypatch):
    monkeypatch.setattr("backend.routers.finance._FINANCE_PIN", "secret99")
    r = client.post("/api/finance/verify-pin", json={"pin": "wrong"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is False


def test_finance_verify_pin_accepts_match(client, monkeypatch):
    monkeypatch.setattr("backend.routers.finance._FINANCE_PIN", "secret99")
    r = client.post("/api/finance/verify-pin", json={"pin": "secret99"})
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_finance_expense_crud(finance_isolated_db, client):
    r = client.post(
        "/api/finance/expenses",
        json={
            "date": "2026-03-10",
            "category": "Other",
            "description": "api test",
            "amount": 250.0,
            "gst_amount": 45.0,
        },
    )
    assert r.status_code == 200
    new_id = r.json()["id"]

    r2 = client.get("/api/finance/expenses?start_date=2026-03-01&end_date=2026-03-31")
    assert r2.status_code == 200
    rows = r2.json()
    assert any(x["id"] == new_id for x in rows)

    r3 = client.delete(f"/api/finance/expenses/{new_id}")
    assert r3.status_code == 200


def test_finance_voucher_and_daybook(finance_isolated_db, client):
    payload = {
        "voucher_date": "2026-03-15",
        "voucher_type": "Expense",
        "party_name": "API Vendor",
        "party_state": "Karnataka",
        "taxable_amount": 1000.0,
        "cgst_amount": 90.0,
        "sgst_amount": 90.0,
        "igst_amount": 0.0,
        "total_amount": 1180.0,
        "net_payable": 1180.0,
        "lines": [
            {
                "expense_head": "Office Expenses",
                "description": "api voucher",
                "amount": 1000.0,
                "is_debit": 1,
            }
        ],
    }
    r = client.post("/api/finance/vouchers", json=payload)
    assert r.status_code == 200
    vno = r.json()["voucher_no"]
    assert vno

    r2 = client.get("/api/finance/daybook?date=2026-03-15")
    assert r2.status_code == 200
    day = r2.json()
    assert any(str(x.get("voucher_no")) == vno for x in day)


def test_finance_chart_of_accounts(finance_isolated_db, client):
    r = client.get("/api/finance/chart-of-accounts")
    assert r.status_code == 200
    tree = r.json()
    assert "groups" in tree
    assert len(tree["groups"]) >= 1


def test_finance_trial_balance(finance_isolated_db, client):
    r = client.get("/api/finance/trial-balance?start_date=2026-01-01&end_date=2026-12-31")
    assert r.status_code == 200
    tb = r.json()
    assert "rows" in tb and "total_debit" in tb and "total_credit" in tb
    assert "balanced" in tb


def test_finance_tally_pl_roundtrip(finance_isolated_db, client):
    r = client.post(
        "/api/finance/tally-pl",
        json={
            "fy": "2026-27",
            "opening_stock": 1.0,
            "purchases": 2.0,
            "direct_expenses": 3.0,
            "indirect_expenses": 4.0,
            "sales": 100.0,
            "closing_stock": 5.0,
            "indirect_incomes": 0.0,
            "notes": "pytest",
        },
    )
    assert r.status_code == 200

    r2 = client.get("/api/finance/tally-pl")
    assert r2.status_code == 200
    rows = r2.json()
    fy_row = next((x for x in rows if x.get("fy") == "2026-27"), None)
    assert fy_row is not None
    assert fy_row.get("net_profit") is not None

    r3 = client.delete("/api/finance/tally-pl/2026-27")
    assert r3.status_code == 200


def test_finance_pl_finance_lock_empty_uploads(finance_isolated_db, client, session_for_client):
    """P&amp;L with finance_lock should respond without error when uploads DB is empty."""
    _sid, _sess = session_for_client
    r = client.get(
        "/api/finance/pl?revenue_source=finance_lock&start_date=2026-01-01&end_date=2026-01-31"
    )
    assert r.status_code == 200
    body = r.json()
    assert "net_profit" in body
    assert body.get("revenue_source") == "finance_lock"


def test_finance_sales_invoices_from_upload_entries(finance_isolated_db, client):
    # Create upload + one invoice-level entry through existing API.
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-04",
            "filename": "x.csv",
            "total_revenue": 100.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 100.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    upload_id = int(up.json().get("id") or 0)
    assert upload_id > 0
    # No finance_sales_entries for JSON-only upload → list should still show one upload-summary row (SUP-*).
    r = client.get("/api/finance/sales-invoices?start_date=2026-04-01&end_date=2026-04-30")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    sup = [x for x in rows if str(x.get("voucher_no") or "").startswith("SUP-")]
    assert len(sup) >= 1
    assert any(int(x.get("id") or 0) >= 10_000_000 for x in sup)
    vid = 10_000_000 + upload_id
    rv = client.get(f"/api/finance/vouchers/{vid}")
    assert rv.status_code == 200
    assert rv.json().get("voucher_type") == "Sales Upload"

    import backend.db.finance_db as fdb

    fdb.create_finance_sales_entries(
        upload_id,
        [
            {
                "platform": "Amazon",
                "period": "2026-04",
                "voucher_date": "2026-04-10",
                "invoice_no": "INV-X",
                "order_id": "ORD-1",
                "party_name": "Buyer",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "",
                "taxable_amount": 50.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": 50.0,
                "net_payable": 50.0,
                "narration": "",
                "source_filename": "x.csv",
                "line_items": "[]",
            }
        ],
    )
    r2 = client.get("/api/finance/sales-invoices?start_date=2026-04-01&end_date=2026-04-30")
    assert r2.status_code == 200
    rows2 = r2.json()
    assert any(x.get("voucher_no") == f"SUP-{upload_id}" for x in rows2)
    sue_rows = [x for x in rows2 if str(x.get("voucher_no") or "").startswith("SUE-")]
    assert len(sue_rows) >= 1
    assert sue_rows[0].get("sales_upload_id") == upload_id
    assert sue_rows[0].get("row_kind") == "entry"

    r_all = client.get("/api/finance/sales-invoices")
    assert r_all.status_code == 200
    assert len(r_all.json()) >= len(rows2)


def test_finance_sales_invoices_exclude_upload_summaries_param(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-07",
            "filename": "suphide.csv",
            "total_revenue": 50.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 50.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    upload_id = int(up.json().get("id") or 0)
    r_all = client.get("/api/finance/sales-invoices?start_date=2026-07-01&end_date=2026-07-31")
    assert r_all.status_code == 200
    assert any(x.get("voucher_no") == f"SUP-{upload_id}" for x in r_all.json())
    r_no = client.get(
        "/api/finance/sales-invoices?start_date=2026-07-01&end_date=2026-07-31&include_upload_summaries=false"
    )
    assert r_no.status_code == 200
    assert not any(str(x.get("voucher_no") or "").startswith("SUP-") for x in r_no.json())


def test_finance_inventory_movements_api(finance_isolated_db, client):
    import json

    import backend.db.finance_db as fdb

    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-08",
            "filename": "inv.csv",
            "total_revenue": 200.0,
            "total_orders": 2,
            "total_returns": 0.0,
            "net_revenue": 200.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    uid = int(up.json()["id"])
    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-08",
                "voucher_date": "2026-08-10",
                "invoice_no": "INV-M1",
                "order_id": "O1",
                "party_name": "Buyer",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "KA",
                "taxable_amount": 200.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": 200.0,
                "net_payable": 200.0,
                "narration": "",
                "source_filename": "inv.csv",
                "line_items": json.dumps([{"sku": "SKU-MOVE-1", "Quantity": 3, "product_name": "Widget"}]),
            },
            {
                "platform": "Amazon",
                "period": "2026-08",
                "voucher_date": "2026-08-11",
                "invoice_no": "CN-M1",
                "order_id": "O1-R",
                "party_name": "Buyer",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "KA",
                "taxable_amount": -50.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": -50.0,
                "net_payable": -50.0,
                "narration": "Refund",
                "source_filename": "inv.csv",
                "line_items": json.dumps([{"sku": "SKU-MOVE-1", "Quantity": 1, "product_name": "Widget"}]),
            },
        ],
    )
    r = client.get("/api/finance/inventory-movements?start_date=2026-08-01&end_date=2026-08-31")
    assert r.status_code == 200
    body = r.json()
    hit = next((x for x in body if x.get("sku") == "SKU-MOVE-1"), None)
    assert hit is not None
    assert hit["qty_out"] == 3.0
    assert hit["qty_in"] == 1.0
    assert hit["net_qty"] == 2.0


def test_finance_sales_invoices_document_kind_sales_vs_credit_memo(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-05",
            "filename": "mix.csv",
            "total_revenue": 150.0,
            "total_orders": 2,
            "total_returns": 30.0,
            "net_revenue": 120.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    upload_id = int(up.json().get("id") or 0)
    assert upload_id > 0

    import backend.db.finance_db as fdb

    fdb.create_finance_sales_entries(
        upload_id,
        [
            {
                "platform": "Amazon",
                "period": "2026-05",
                "voucher_date": "2026-05-12",
                "invoice_no": "INV-OK",
                "order_id": "ORD-A",
                "party_name": "Buyer A",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "",
                "taxable_amount": 80.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": 80.0,
                "net_payable": 80.0,
                "narration": "",
                "source_filename": "mix.csv",
                "line_items": "[]",
            },
            {
                "platform": "Amazon",
                "period": "2026-05",
                "voucher_date": "2026-05-12",
                "invoice_no": "CN-1",
                "order_id": "ORD-R",
                "party_name": "Buyer B",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "",
                "taxable_amount": -30.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": -30.0,
                "net_payable": -30.0,
                "narration": "refund adjustment",
                "source_filename": "mix.csv",
                "line_items": "[]",
            },
        ],
    )

    r_sales = client.get("/api/finance/sales-invoices?document_kind=sales&start_date=2026-05-01&end_date=2026-05-31")
    assert r_sales.status_code == 200
    sales_rows = r_sales.json()
    sue_sales = [x for x in sales_rows if str(x.get("voucher_no") or "").startswith("SUE-")]
    assert len(sue_sales) >= 1
    assert all(x.get("document_subtype") != "sales_credit_memo" for x in sue_sales)
    assert any(str(x.get("voucher_no") or "").startswith("SUP-") for x in sales_rows)

    r_cred = client.get("/api/finance/sales-invoices?document_kind=credit_memo&start_date=2026-05-01&end_date=2026-05-31")
    assert r_cred.status_code == 200
    cred_rows = r_cred.json()
    sue_cred = [x for x in cred_rows if str(x.get("voucher_no") or "").startswith("SUE-")]
    assert len(sue_cred) >= 1
    assert all(x.get("document_subtype") == "sales_credit_memo" for x in sue_cred)
    assert not any(str(x.get("voucher_no") or "").startswith("SUP-") for x in cred_rows)


def test_finance_customer_ledger_entries_bc_shape(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-06",
            "filename": "ledger.csv",
            "total_revenue": 50.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 50.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    uid = int(up.json().get("id") or 0)
    import backend.db.finance_db as fdb

    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-06",
                "voucher_date": "2026-06-01",
                "invoice_no": "BLR8-22106",
                "order_id": "403-8195605-4690736",
                "party_name": "MAJESTY",
                "party_gstin": "29AABCY3804E1ZF",
                "party_state": "KA",
                "ship_to_state": "KA",
                "taxable_amount": 637.14,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 31.857,
                "total_amount": 669.0,
                "net_payable": 669.0,
                "narration": "Invoice SI-40797",
                "source_filename": "ledger.csv",
                "line_items": "[]",
            },
        ],
    )
    r = client.get("/api/finance/customer-ledger-entries?start_date=2026-06-01&end_date=2026-06-30")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 1
    row = next(x for x in rows if x.get("document_no") == "BLR8-22106")
    assert row.get("document_type") == "Invoice"
    assert row.get("customer_name") == "MAJESTY"
    assert row.get("location_code") == "BLR8"
    assert row.get("external_document_no") == "403-8195605-4690736"
    assert row.get("gst_jurisdiction_type") == "Interstate"
    assert abs(float(row.get("gst_amount") or 0) - 31.857) < 0.02


def test_customer_ledger_uses_bill_date_patch_for_document_date(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-06",
            "filename": "ledger-date.csv",
            "total_revenue": 100.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 100.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    uid = int(up.json().get("id") or 0)
    import backend.db.finance_db as fdb

    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-06",
                "voucher_date": "2026-06-10",
                "invoice_no": "INV-DATE-1",
                "order_id": "ORD-DATE-1",
                "party_name": "Buyer D",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "KA",
                "taxable_amount": 100.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 5.0,
                "total_amount": 105.0,
                "net_payable": 105.0,
                "narration": "",
                "source_filename": "ledger-date.csv",
                "line_items": "[]",
            }
        ],
    )
    sales = client.get("/api/finance/sales-invoices?start_date=2026-06-01&end_date=2026-06-30").json()
    sue = next(x for x in sales if x.get("invoice_no") == "INV-DATE-1")
    vid = int(sue["id"])
    p = client.patch(f"/api/finance/sales-invoices/{vid}", json={"bill_date": "2026-06-07"})
    assert p.status_code == 200

    r = client.get("/api/finance/customer-ledger-entries?start_date=2026-06-01&end_date=2026-06-30")
    assert r.status_code == 200
    row = next(x for x in r.json() if x.get("document_no") == "INV-DATE-1")
    assert row.get("document_date") == "2026-06-07"
    assert row.get("due_date") == "2026-06-07"


def test_customer_ledger_prefers_line_item_invoice_date_when_bill_date_missing(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-07",
            "filename": "ledger-line-date.csv",
            "total_revenue": 100.0,
            "total_orders": 1,
            "total_returns": 0.0,
            "net_revenue": 100.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    uid = int(up.json().get("id") or 0)
    import backend.db.finance_db as fdb

    fdb.create_finance_sales_entries(
        uid,
        [
            {
                "platform": "Amazon",
                "period": "2026-07",
                "voucher_date": "2026-07-10",
                "invoice_no": "INV-LINE-DATE-1",
                "order_id": "ORD-LINE-DATE-1",
                "party_name": "Buyer E",
                "party_gstin": "",
                "party_state": "KA",
                "ship_to_state": "KA",
                "taxable_amount": 100.0,
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 5.0,
                "total_amount": 105.0,
                "net_payable": 105.0,
                "narration": "",
                "source_filename": "ledger-line-date.csv",
                "line_items": '[{"invoice_date":"09/07/2026"}]',
            }
        ],
    )

    r = client.get("/api/finance/customer-ledger-entries?start_date=2026-07-01&end_date=2026-07-31")
    assert r.status_code == 200
    row = next(x for x in r.json() if x.get("document_no") == "INV-LINE-DATE-1")
    assert row.get("document_date") == "2026-07-09"
    assert row.get("due_date") == "2026-07-09"


def test_finance_sales_invoice_patch_upload_summary(finance_isolated_db, client):
    up = client.post(
        "/api/finance/sales-uploads",
        json={
            "platform": "Amazon",
            "period": "2026-04",
            "filename": "mtr.csv",
            "total_revenue": 200.0,
            "total_orders": 2,
            "total_returns": 0.0,
            "net_revenue": 200.0,
            "uploaded_by": "pytest",
            "upload_notes": "",
        },
    )
    assert up.status_code == 200
    uid = int(up.json().get("id") or 0)
    vid = 10_000_000 + uid
    p = client.patch(
        f"/api/finance/sales-invoices/{vid}",
        json={"invoice_no": "EXT-INV-99", "ship_to_state": "Maharashtra", "net_payable": 199.5},
    )
    assert p.status_code == 200
    assert p.json().get("ok") is True
    lst = client.get("/api/finance/sales-invoices?start_date=2026-04-01&end_date=2026-04-30").json()
    row = next(x for x in lst if int(x.get("id") or 0) == vid)
    assert row.get("invoice_no") == "EXT-INV-99"
    assert row.get("ship_to_state") == "Maharashtra"
    assert float(row.get("net_payable") or 0) == 199.5
    rv = client.get(f"/api/finance/vouchers/{vid}")
    assert rv.status_code == 200
    j = rv.json()
    assert j.get("bill_no") == "EXT-INV-99"
    assert float(j.get("net_payable") or 0) == 199.5

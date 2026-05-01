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
    # DB helper endpoint persists entries via upload-file in real flow; here we only verify list endpoint shape.
    # Existing synthetic coverage already verifies SUE persistence in finance_db tests.
    r = client.get("/api/finance/sales-invoices?start_date=2026-04-01&end_date=2026-04-30")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

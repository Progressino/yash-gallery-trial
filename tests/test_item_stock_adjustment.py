"""Item Master — manual stock adjustment (admin-only) and tracking ledger."""

import pytest


@pytest.fixture()
def item_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "items_stock_test.db")
    monkeypatch.setenv("ITEM_DB_PATH", db_path)
    from backend.db import item_db

    monkeypatch.setattr(item_db, "DB_PATH", db_path)
    item_db.init_db()
    return item_db


def _create_rm(item_db, code: str = "ELASTIC-15", stock: float = 0.0) -> int:
    types = item_db.list_item_types()
    rm_type = next(t for t in types if t["code"] == "RM")
    item_id = item_db.create_item(code, f"Test {code}", rm_type["id"], uom="MTR")
    if stock:
        item_db.update_item_stock(item_id, stock)
    return item_id


def test_adjust_item_stock_in_and_out(item_db):
    item_id = _create_rm(item_db)
    out_in = item_db.adjust_item_stock(
        item_id, 50, "IN", entry_date="2026-01-01", reason="Opening / existing stock"
    )
    assert out_in["stock_after"] == 50.0
    assert out_in["direction"] == "IN"

    out_out = item_db.adjust_item_stock(
        item_id, 10, "OUT", entry_date="2026-01-02", reason="Physical count correction"
    )
    assert out_out["stock_after"] == 40.0

    row = item_db.get_item(item_id)
    assert float(row["stock"]) == 40.0


def test_adjust_item_stock_rejects_insufficient(item_db):
    item_id = _create_rm(item_db, stock=5.0)
    with pytest.raises(ValueError, match="Insufficient stock"):
        item_db.adjust_item_stock(item_id, 10, "OUT", entry_date="2026-01-01", reason="Test")


def test_list_stock_adjustments_and_book_stock(item_db):
    item_id = _create_rm(item_db, code="BTN-NAVY")
    item_db.adjust_item_stock(item_id, 25, "IN", entry_date="2026-01-05", reason="Opening stock")
    rows = item_db.list_stock_adjustments("BTN-NAVY")
    assert len(rows) == 1
    assert rows[0]["direction"] == "IN"
    assert float(rows[0]["qty"]) == 25.0
    assert item_db.book_stock_for_code("BTN-NAVY") == 25.0


def test_admin_can_adjust_via_api(client, item_db, monkeypatch):
    monkeypatch.setattr("backend.db.item_db.DB_PATH", item_db.DB_PATH)

    item_id = _create_rm(item_db, code="API-STOCK-1")
    r = client.post(
        f"/api/items/{item_id}/stock/adjust",
        json={
            "qty": 100,
            "direction": "IN",
            "entry_date": "2026-06-01",
            "reason": "Opening / existing stock",
            "reference_no": "OPEN-001",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["stock_after"] == 100.0

    tr = client.get("/api/items/API-STOCK-1/tracking")
    assert tr.status_code == 200, tr.text
    ledger = tr.json()
    assert ledger["current_stock"] == 100.0
    assert ledger["book_stock"] == 100.0
    types = {t["txn_type"] for t in ledger["transactions"]}
    assert "Stock Adjustment" in types


def test_non_admin_cannot_adjust_stock(client, item_db, monkeypatch):
    monkeypatch.setattr("backend.db.item_db.DB_PATH", item_db.DB_PATH)

    def _decode(token: str | None):
        if token == "user-token":
            return {"sub": "clerk", "role": "Clerk", "permissions": []}
        if token == "test-token":
            return {"sub": "tester", "role": "Admin", "permissions": []}
        return None

    monkeypatch.setattr("backend.main.decode_token", _decode)
    item_id = _create_rm(item_db, code="API-STOCK-2")
    client.cookies.set("auth_token", "user-token")
    r = client.post(
        f"/api/items/{item_id}/stock/adjust",
        json={"qty": 5, "direction": "IN", "reason": "Should fail"},
    )
    assert r.status_code == 403, r.text

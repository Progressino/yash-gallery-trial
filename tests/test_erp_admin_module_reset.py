import sqlite3


def _sales_counts(db_path: str) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    try:
        so = conn.execute("SELECT COUNT(*) FROM sales_orders").fetchone()[0]
        ln = conn.execute("SELECT COUNT(*) FROM so_lines").fetchone()[0]
        return int(so), int(ln)
    finally:
        conn.close()


def test_admin_can_reset_sales_orders_module(client, tmp_path, monkeypatch):
    from backend.db import sales_db

    db_path = str(tmp_path / "sales_reset_test.db")
    monkeypatch.setattr(sales_db, "_DB", db_path)
    sales_db.init_db()
    sales_db.create_order(
        {
            "so_date": "2026-06-01",
            "buyer": "Test Buyer",
            "lines": [{"sku": "SKU-1", "sku_name": "Sample", "qty": 10}],
        }
    )
    before = _sales_counts(db_path)
    assert before[0] == 1 and before[1] == 1

    r = client.post("/api/erp-admin/reset-module-data", json={"module": "sales_orders"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["module"] == "sales_orders"

    after = _sales_counts(db_path)
    assert after == (0, 0)


def test_non_admin_cannot_reset_module_data(client, monkeypatch):
    def _decode(token: str | None):
        if token == "user-token":
            return {"sub": "tester2", "role": "Employee", "permissions": []}
        if token == "test-token":
            return {"sub": "tester", "role": "Admin", "permissions": []}
        return None

    monkeypatch.setattr("backend.main.decode_token", _decode)
    client.cookies.set("auth_token", "user-token")
    r = client.post("/api/erp-admin/reset-module-data", json={"module": "sales_orders"})
    assert r.status_code == 403, r.text

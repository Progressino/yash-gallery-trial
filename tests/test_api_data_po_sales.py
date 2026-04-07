"""HTTP tests: /api/data/* , /api/po/* , /api/sales/* (requires auth middleware bypass)."""

import pandas as pd


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_sales_summary_empty_session(client, session_for_client):
    from backend.session import wipe_app_session

    _, sess = session_for_client
    wipe_app_session(sess)
    r = client.get("/api/data/sales-summary?months=0")
    assert r.status_code == 200
    j = r.json()
    assert j["total_units"] == 0
    assert j["total_returns"] == 0


def test_platform_summary_with_mtr(client, session_for_client):
    _, sess = session_for_client
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-04-01"]),
            "SKU": ["Z1"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [7.0],
        }
    )
    r = client.get("/api/data/platform-summary")
    assert r.status_code == 200
    plat = r.json()
    assert isinstance(plat, list)
    amz = next(p for p in plat if p["platform"] == "Amazon")
    assert amz["total_units"] == 7


def test_po_calculate_needs_inventory(client, session_for_client):
    _, sess = session_for_client
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["API-SKU"],
            "TxnDate": pd.to_datetime(["2025-12-01"]),
            "Transaction Type": ["Shipment"],
            "Quantity": [5],
            "Units_Effective": [5],
            "Source": ["Meesho"],
        }
    )
    sess.inventory_df_variant = pd.DataFrame()
    r = client.post(
        "/api/po/calculate",
        json={"period_days": 30, "lead_time": 7, "target_days": 60, "safety_pct": 0},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is False


def test_po_calculate_ok(client, session_for_client):
    _, sess = session_for_client
    days = pd.date_range("2025-12-01", periods=20, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["API-SKU"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [3] * len(days),
            "Units_Effective": [3] * len(days),
            "Source": ["Meesho"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["API-SKU"], "Total_Inventory": [100]}
    )
    r = client.post(
        "/api/po/calculate",
        json={"period_days": 30, "lead_time": 7, "target_days": 60, "safety_pct": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("rows")
    assert "PO_Qty" in body["columns"]


def test_po_quarterly_loaded(client, session_for_client):
    _, sess = session_for_client
    q_days = pd.date_range("2025-04-01", periods=10, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["Q-SKU"] * len(q_days),
            "TxnDate": q_days,
            "Transaction Type": ["Shipment"] * len(q_days),
            "Quantity": [1] * len(q_days),
            "Units_Effective": [1] * len(q_days),
            "Source": ["Flipkart"] * len(q_days),
        }
    )
    sess._quarterly_cache.clear()
    r = client.get("/api/po/quarterly?n_quarters=4")
    assert r.status_code == 200
    j = r.json()
    assert j.get("loaded") is True
    assert j.get("rows")


def test_sales_demands_list(client):
    r = client.get("/api/sales/demands")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_sales_export_csv(client, session_for_client):
    import pandas as pd

    _, sess = session_for_client
    from backend.session import wipe_app_session

    wipe_app_session(sess)
    sess.pause_auto_data_restore = True
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2025-06-01", "2025-06-02"]),
            "Sku": ["X1", "X1"],
            "Transaction Type": ["Shipment", "Refund"],
            "Quantity": [3, 1],
            "Units_Effective": [3, -1],
            "Source": ["Amazon", "Amazon"],
            "OrderId": ["", ""],
        }
    )
    r = client.get("/api/data/sales-export?months=0&start_date=2025-06-01&end_date=2025-06-30")
    assert r.status_code == 200
    assert "text/csv" in (r.headers.get("content-type") or "")
    body = r.text
    assert "TxnDate" in body and "Amazon" in body and "Shipment" in body

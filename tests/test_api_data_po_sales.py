"""HTTP tests: /api/data/* , /api/po/* , /api/sales/* (requires auth middleware bypass)."""

import time
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
    kick = r.json()
    assert kick.get("ok") is True
    assert kick.get("status") == "running"
    body = kick
    for _ in range(60):
        st = client.get("/api/po/calculate/status")
        assert st.status_code == 200
        body = st.json()
        if body.get("status") == "done":
            break
        if body.get("status") == "error":
            raise AssertionError(body.get("message") or "PO calculate failed")
        time.sleep(0.5)
    assert body.get("ok") is True
    res = client.get(
        "/api/po/calculate/result",
        params={"offset": 0, "limit": 500, "compact": 0},
    )
    assert res.status_code == 200
    full = res.json()
    assert full.get("ok") is True
    rows = full.get("rows") or []
    if not rows and full.get("rows_matrix") and full.get("columns"):
        rows = [
            dict(zip(full["columns"], r))
            for r in full["rows_matrix"]
        ]
    assert rows
    assert "PO_Qty" in full["columns"]
    assert full.get("has_more") is False


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
    assert "OMS_Sku" in body


def test_sales_export_recover_meesho_sku_from_meesho_df(client, session_for_client):
    import io

    import pandas as pd

    _, sess = session_for_client
    from backend.session import wipe_app_session

    wipe_app_session(sess)
    sess.pause_auto_data_restore = True
    oid = "171474035005_1"
    day = "2025-12-24"
    sess.meesho_df = pd.DataFrame(
        {
            "Date": pd.to_datetime([day]),
            "TxnType": ["Shipment"],
            "Quantity": [1.0],
            "SKU": ["1592YKBLUE-5XL"],
            "OrderId": [oid],
        }
    )
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime([f"{day} 00:00:00"]),
            "Sku": ["MEESHO_TOTAL"],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Units_Effective": [1],
            "Source": ["Meesho"],
            "OrderId": [oid],
        }
    )
    r = client.get("/api/data/sales-export?months=0")
    assert r.status_code == 200
    out = pd.read_csv(io.StringIO(r.text))
    assert out["Sku"].iloc[0] == "1592YKBLUE-5XL"


def test_sales_export_blanks_oms_for_note_like_sku(client, session_for_client):
    import io

    import pandas as pd

    _, sess = session_for_client
    from backend.session import wipe_app_session

    wipe_app_session(sess)
    sess.pause_auto_data_restore = True
    sess.meesho_df = pd.DataFrame()
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2025-06-01"]),
            "Sku": ["SIZE CHANGE"],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Units_Effective": [1],
            "Source": ["Meesho"],
            "OrderId": ["x1"],
        }
    )
    r = client.get("/api/data/sales-export?months=0")
    assert r.status_code == 200
    out = pd.read_csv(io.StringIO(r.text))
    assert str(out["OMS_Sku"].iloc[0]) in ("", "nan") or pd.isna(out["OMS_Sku"].iloc[0])


def test_po_raise_ledger_import_csv(client, session_for_client):
    import io

    from backend.session import wipe_app_session

    _, sess = session_for_client
    wipe_app_session(sess)
    csv_content = "OMS_SKU,PO_Qty\nSKU-A,10\nSKU-B,5\n"
    files = {"file": ("po_recommendation.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    data = {"raised_date": "2026-05-13", "group_by_parent": "false", "replace_day": "true"}
    r = client.post("/api/po/raise-ledger/import-csv", files=files, data=data)
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("imported_skus") == 2
    assert j.get("total_units") == 15
    assert j.get("raised_date") == "2026-05-13"
    df = sess.po_raise_ledger_df
    assert df is not None and not df.empty


def test_po_raise_ledger_import_accepts_raise_export_final_po_qty_header(client, session_for_client):
    """Raise PO modal CSV uses Final_PO_Qty — re-import must recognise it."""
    import io

    from backend.session import wipe_app_session

    _, sess = session_for_client
    wipe_app_session(sess)
    csv_content = "OMS_SKU,Priority,Final_PO_Qty\nSKU-A,OK,12\n"
    files = {"file": ("raise_po_2026-05-14.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    data = {"raised_date": "2026-05-14", "group_by_parent": "false", "replace_day": "true"}
    r = client.post("/api/po/raise-ledger/import-csv", files=files, data=data)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("ok") is True
    assert j.get("imported_skus") == 1
    assert j.get("total_units") == 12


def test_po_raise_ledger_import_accepts_ledger_dump_raised_qty(client, session_for_client):
    import io

    from backend.session import wipe_app_session

    _, sess = session_for_client
    wipe_app_session(sess)
    csv_content = "OMS_SKU,Raised_Qty,Raised_Date\nSKU-X,7,2026-05-01\n"
    files = {"file": ("ledger_dump.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    data = {"raised_date": "2026-05-14", "group_by_parent": "false", "replace_day": "true"}
    r = client.post("/api/po/raise-ledger/import-csv", files=files, data=data)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("ok") is True
    assert j.get("total_units") == 7

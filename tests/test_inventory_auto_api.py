"""HTTP smoke test for /api/upload/inventory-auto — must not 500."""

from __future__ import annotations

import pandas as pd


def test_inventory_auto_accepts_oms_csv(client, session_for_client):
    _sid, sess = session_for_client
    sess.sku_mapping = {"TEST-SKU": "TEST-SKU"}

    csv = "Item SkuCode,Inventory\nTEST-SKU,42\n"
    r = client.post(
        "/api/upload/inventory-auto",
        files=[("files", ("oms_inventory.csv", csv.encode("utf-8"), "text/csv"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True or body.get("ingest_async") is True


def test_tier3_columns_only_metrics_include_order_ids():
    from backend.services.daily_store import _PLATFORM_METRICS_COLUMNS

    for plat in ("myntra", "meesho", "flipkart", "snapdeal"):
        cols = _PLATFORM_METRICS_COLUMNS[plat]
        assert "OrderId" in cols, plat
    assert "Order_Id" in _PLATFORM_METRICS_COLUMNS["amazon"]


def test_build_sales_from_tier3_frames_without_orderid_does_not_crash():
    from backend.routers.data import _build_sales_from_tier3_frames
    from backend.session import AppSession

    sess = AppSession()
    sess.sku_mapping = {"SKU1": "SKU1"}
    myntra = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "OMS_SKU": ["SKU1"],
            "TxnType": ["Shipment"],
            "Quantity": [5],
            "State": [""],
        }
    )
    built = _build_sales_from_tier3_frames(sess, {"myntra": myntra})
    assert not built.empty
    assert int(built["Quantity"].sum()) == 5

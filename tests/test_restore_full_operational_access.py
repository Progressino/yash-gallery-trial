"""After POST /data/restore-full, core datasets and intelligence APIs must be usable."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.session import wipe_app_session

# Platforms the business cares about (Snapdeal optional).
_REQUIRED_COVERAGE = (
    "sku_mapping",
    "mtr",
    "myntra",
    "meesho",
    "flipkart",
    "sales",
    "inventory",
)


@pytest.fixture
def full_operational_warm(monkeypatch):
    """Simulate a healthy VPS warm cache (bulk history + inventory, no Snapdeal)."""
    import backend.main as main

    days = pd.date_range("2025-06-01", periods=40, freq="D")
    n = len(days)
    sku = "REST-SKU"

    months = days.to_period("M").astype(str)

    def _order_plat() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": days,
                "OMS_SKU": [sku] * n,
                "OrderId": [f"ord-{i}" for i in range(n)],
                "TxnType": ["Shipment"] * n,
                "Quantity": [2.0] * n,
                "Month": months,
                "State": ["MH"] * n,
            }
        )

    mtr = pd.DataFrame(
        {
            "Date": days,
            "SKU": [sku] * n,
            "Transaction_Type": ["Shipment"] * n,
            "Quantity": [3.0] * n,
            "Order_Id": [f"mtr-{i}" for i in range(n)],
            "Invoice_Number": [""] * n,
        }
    )

    myntra_df = _order_plat()
    meesho_df = _order_plat()
    flipkart_df = _order_plat()
    from backend.services.sales import build_sales_df

    sales_df = build_sales_df(
        mtr_df=mtr,
        myntra_df=myntra_df,
        meesho_df=meesho_df,
        flipkart_df=flipkart_df,
        snapdeal_df=pd.DataFrame(),
        sku_mapping={sku: sku, "ALT": sku},
    )

    main._warm_cache = {
        "sku_mapping": {sku: sku, "ALT": sku},
        "mtr_df": mtr,
        "myntra_df": myntra_df,
        "meesho_df": meesho_df,
        "flipkart_df": flipkart_df,
        "snapdeal_df": pd.DataFrame(),
        "sales_df": sales_df,
        "inventory_df_variant": pd.DataFrame(
            {"OMS_SKU": [sku], "Total_Inventory": [120], "Channel": ["All"]}
        ),
        "inventory_df_parent": pd.DataFrame(),
    }
    main._warm_cache_generation = 99
    yield
    main._warm_cache = {}
    main._warm_cache_generation = 0


def _assert_required_coverage(body: dict) -> None:
    missing = [k for k in _REQUIRED_COVERAGE if not body.get(k)]
    assert not missing, f"coverage missing: {missing}; body keys={list(body.keys())}"
    assert body["mtr_rows"] > 0
    assert body["sales_rows"] > 0
    assert body["myntra_rows"] > 0
    assert body["meesho_rows"] > 0
    assert body["flipkart_rows"] > 0


def test_restore_full_then_data_apis_accessible(client, session_for_client, full_operational_warm):
    _, sess = session_for_client
    wipe_app_session(sess)
    sess.pause_auto_data_restore = True
    assert not sess.mtr_df.empty is False or sess.mtr_df.empty

    r = client.post("/api/data/restore-full")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True, body.get("message")
    _assert_required_coverage(body)

    cov = client.get("/api/data/coverage")
    assert cov.status_code == 200
    _assert_required_coverage(cov.json())

    sales = client.get("/api/data/sales-summary", params={"months": "12"})
    assert sales.status_code == 200
    sj = sales.json()
    assert sj["total_units"] > 0

    plat = client.get("/api/data/platform-summary")
    assert plat.status_code == 200
    names = {p["platform"] for p in plat.json()}
    assert {"Amazon", "Myntra", "Meesho", "Flipkart"}.issubset(names)

    by_src = client.get("/api/data/sales-by-source")
    assert by_src.status_code == 200
    assert len(by_src.json()) >= 4

    inv = client.get("/api/data/inventory", params={"limit": 10})
    assert inv.status_code == 200
    inv_body = inv.json()
    assert inv_body.get("loaded") is True
    assert int(inv_body.get("total_rows") or 0) >= 1

    for path in (
        "/api/data/mtr-analytics",
        "/api/data/myntra-analytics",
        "/api/data/meesho-analytics",
        "/api/data/flipkart-analytics",
    ):
        ar = client.get(path, params={"months": "3"})
        assert ar.status_code == 200, f"{path}: {ar.text[:200]}"

    dsr = client.get("/api/data/daily-dsr", params={"days": "7"})
    assert dsr.status_code == 200

    top = client.get("/api/data/top-skus", params={"limit": 5, "months": "3"})
    assert top.status_code == 200

"""POST /api/po/calculate must return immediately while work runs in background."""

import time


def test_po_calculate_post_returns_before_job_finishes(client, monkeypatch, session_for_client):
    import pandas as pd

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

    def _slow_execute(*_a, **_kw):
        time.sleep(3)
        return {
            "ok": True,
            "total_rows": 1,
            "columns": ["OMS_SKU", "PO_Qty"],
            "rows": [],
            "sales_through": "2025-12-20",
            "planning_date": "2025-12-20",
        }

    monkeypatch.setattr(
        "backend.services.po_calculate_run.execute_po_calculate",
        _slow_execute,
    )

    t0 = time.monotonic()
    r = client.post(
        "/api/po/calculate",
        json={"period_days": 30, "lead_time": 7, "target_days": 60, "safety_pct": 0},
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, f"POST blocked {elapsed:.1f}s — should return immediately"
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("status") == "running"

    for _ in range(40):
        st = client.get("/api/po/calculate/status")
        assert st.status_code == 200
        if st.json().get("status") == "done":
            break
        time.sleep(0.25)
    assert st.json().get("status") == "done"

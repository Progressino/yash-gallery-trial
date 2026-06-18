"""PO Engine 2 defaults: app-only target-cover (180d post-PO, no lead gate)."""

import time

import pandas as pd

PO2_DEFAULT_BODY = {
    "period_days": 30,
    "lead_time": 60,
    "target_days": 180,
    "grace_days": 0,
    "demand_basis": "Sold",
    "group_by_parent": False,
    "safety_pct": 0,
    "enforce_two_size_minimum": True,
    "enforce_lead_time_release_gate": False,
    "use_shared_cache": False,
}


def _seed_minimal_po_session(sess):
    days = pd.date_range("2025-12-01", periods=30, freq="D")
    frames = []
    for sku in ("STYLE-M", "STYLE-L"):
        frames.append(
            pd.DataFrame(
                {
                    "Sku": [sku] * len(days),
                    "TxnDate": days,
                    "Transaction Type": ["Shipment"] * len(days),
                    "Quantity": [2] * len(days),
                    "Units_Effective": [2] * len(days),
                    "Source": ["Meesho"] * len(days),
                }
            )
        )
    sess.sales_df = pd.concat(frames, ignore_index=True)
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["STYLE-M", "STYLE-L"],
            "Total_Inventory": [5, 5],
        }
    )


def _wait_po_done(client, kick):
    job_id = kick.get("job_id")
    if job_id:
        from tests.conftest import wait_po_job_done

        return wait_po_job_done(client, job_id)
    body = kick
    if body.get("status") == "done":
        return body
    for _ in range(90):
        st = client.get("/api/po/calculate/status")
        assert st.status_code == 200
        body = st.json()
        if body.get("status") == "done":
            return body
        if body.get("status") == "error":
            raise AssertionError(body.get("message") or "PO calculate failed")
        time.sleep(0.5)
    raise AssertionError("PO calculate timed out")


def test_po2_defaults_calculate_ok(client, session_for_client):
    _, sess = session_for_client
    _seed_minimal_po_session(sess)
    r = client.post("/api/po/calculate", json=PO2_DEFAULT_BODY)
    assert r.status_code == 200
    kick = r.json()
    assert kick.get("ok") is True
    body = _wait_po_done(client, kick)
    job_id = kick.get("job_id") or body.get("job_id")
    result_path = f"/api/po/calculate/result/{job_id}" if job_id else "/api/po/calculate/result"
    res = client.get(
        result_path,
        params={"offset": 0, "limit": 500, "compact": 0},
    )
    assert res.status_code == 200
    full = res.json()
    assert full.get("ok") is True
    assert "PO_Qty" in (full.get("columns") or [])
    rows = full.get("rows") or []
    if not rows and full.get("rows_matrix") and full.get("columns"):
        rows = [dict(zip(full["columns"], row)) for row in full["rows_matrix"]]
    style_rows = [row for row in rows if str(row.get("OMS_SKU", "")).startswith("STYLE-")]
    assert len(style_rows) >= 2


def test_po2_defaults_no_lead_gate(client, session_for_client):
    """Lead-time release gate must stay off for PO Engine 2."""
    _, sess = session_for_client
    _seed_minimal_po_session(sess)
    body = {**PO2_DEFAULT_BODY, "enforce_lead_time_release_gate": False}
    r = client.post("/api/po/calculate", json=body)
    assert r.status_code == 200
    assert r.json().get("ok") is True

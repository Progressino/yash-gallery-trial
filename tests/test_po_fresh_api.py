"""PO Engine (Fresh) API contract — same app-only defaults as the fresh UI."""

import time

import pandas as pd

PO_FRESH_BODY = {
    "period_days": 30,
    "lead_time": 60,
    "target_days": 180,
    "grace_days": 0,
    "demand_basis": "Sold",
    "group_by_parent": False,
    "safety_pct": 0,
    "enforce_two_size_minimum": True,
    "enforce_lead_time_release_gate": True,
    "use_ly_fallback": True,
    "use_seasonality": True,
    "use_shared_cache": False,
}


def _seed_two_size_parent(sess):
    days = pd.date_range("2025-12-01", periods=30, freq="D")
    frames = []
    for sku in ("FRESH-M", "FRESH-L"):
        frames.append(
            pd.DataFrame(
                {
                    "Sku": [sku] * len(days),
                    "TxnDate": days,
                    "Transaction Type": ["Shipment"] * len(days),
                    "Quantity": [10] * len(days),
                    "Units_Effective": [10] * len(days),
                    "Source": ["Meesho"] * len(days),
                }
            )
        )
    sess.sales_df = pd.concat(frames, ignore_index=True)
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["FRESH-M", "FRESH-L"], "Total_Inventory": [0, 0]}
    )


def _seed_1023ykpblue_high_cover(sess):
    """Style with projected cover well above 60d lead — must not receive PO."""
    parent = "1023YKPBLUE"
    sizes = ["6XL", "M", "XS", "L", "XL"]
    skus = [f"{parent}-{sz}" for sz in sizes]
    days = pd.date_range("2025-12-01", periods=30, freq="D")
    frames = []
    for sku in skus:
        frames.append(
            pd.DataFrame(
                {
                    "Sku": [sku] * len(days),
                    "TxnDate": days,
                    "Transaction Type": ["Shipment"] * len(days),
                    "Quantity": [1] * len(days),
                    "Units_Effective": [1] * len(days),
                    "Source": ["Amazon"] * len(days),
                }
            )
        )
    sess.sales_df = pd.concat(frames, ignore_index=True)
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": skus,
            "Total_Inventory": [120, 110, 100, 150, 140],
        }
    )
    sess.existing_po_df = pd.DataFrame(
        {
            "OMS_SKU": skus,
            "PO_Pipeline_Total": [50, 65, 75, 80, 70],
        }
    )
    sess.sku_status_lead_df = pd.DataFrame(
        {
            "OMS_SKU": skus,
            "SKU_Sheet_Status": ["High selling"] * len(skus),
            "Lead_Time_From_Sheet": [60.0] * len(skus),
            "SKU_Sheet_Closed": [False] * len(skus),
        }
    )


def _rows_from_result(full: dict) -> list[dict]:
    rows = full.get("rows") or []
    if not rows and full.get("rows_matrix") and full.get("columns"):
        rows = [dict(zip(full["columns"], row)) for row in full["rows_matrix"]]
    return rows


def _wait_done(client, kick):
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
            raise AssertionError(body.get("message") or "PO failed")
        time.sleep(0.5)
    raise AssertionError("PO calculate timed out")


def test_po_fresh_calculate_ok(client, session_for_client, monkeypatch):
    _, sess = session_for_client
    sess.existing_po_df = pd.DataFrame()
    monkeypatch.setattr(
        "backend.services.existing_po.ensure_existing_po_hydrated",
        lambda sess: False,
    )
    _seed_two_size_parent(sess)
    r = client.post("/api/po/calculate", json=PO_FRESH_BODY)
    assert r.status_code == 200
    kick = r.json()
    assert kick.get("ok") is True
    _wait_done(client, kick)
    res = client.get(
        "/api/po/calculate/result",
        params={"offset": 0, "limit": 500, "compact": 0},
    )
    full = res.json()
    assert full.get("ok") is True
    rows = _rows_from_result(full)
    fresh = [r for r in rows if str(r.get("OMS_SKU", "")).startswith("FRESH-")]
    assert len(fresh) >= 2
    cols = full.get("columns") or []
    assert "PO_Pipeline_Total" in cols
    assert "Gross_PO_Qty" in cols
    assert full.get("po_merge_version") == 36


def test_po_fresh_lead_gate_blocks_high_cover_even_if_api_sends_false(
    client, session_for_client, monkeypatch
):
    """Regression: API default used to pass gate=False and ignore body default True."""
    _, sess = session_for_client
    monkeypatch.setattr(
        "backend.services.existing_po.ensure_existing_po_hydrated",
        lambda sess: False,
    )
    _seed_1023ykpblue_high_cover(sess)
    body = {
        **PO_FRESH_BODY,
        "enforce_lead_time_release_gate": False,
        "use_shared_cache": False,
    }
    r = client.post("/api/po/calculate", json=body)
    assert r.status_code == 200
    kick = r.json()
    assert kick.get("ok") is True
    _wait_done(client, kick)
    res = client.get(
        "/api/po/calculate/result",
        params={"offset": 0, "limit": 500, "compact": 0},
    )
    full = res.json()
    assert full.get("ok") is True
    rows = _rows_from_result(full)
    style = [r for r in rows if str(r.get("OMS_SKU", "")).startswith("1023YKPBLUE")]
    assert len(style) >= 5
    for row in style:
        proj = float(row.get("Projected_Running_Days") or 0)
        if proj > 60:
            assert int(row.get("PO_Qty") or 0) == 0, row.get("OMS_SKU")
    assert sum(int(r.get("PO_Qty") or 0) for r in style) == 0

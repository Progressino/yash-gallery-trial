"""Shared PO calculate cache — cross-session reuse on the same server."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from backend.services import po_shared_cache as psc
from backend.services.po_result_spill import has_spill, spill_row_count


@pytest.fixture(autouse=True)
def isolated_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PO_SHARED_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("PO_RESULT_SPILL_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("PO_SHARED_CACHE_ENABLED", "1")
    yield


def _minimal_session():
    from backend.session import AppSession

    sess = AppSession()
    days = pd.date_range("2025-12-01", periods=10, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU-A"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [2] * len(days),
            "Units_Effective": [2] * len(days),
            "Source": ["Meesho"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "Total_Inventory": [50]}
    )
    return sess


def test_cache_key_changes_when_params_differ():
    sess = _minimal_session()
    body = {"planning_date": "2025-12-20", "period_days": 30, "lead_time": 7}
    k1, _ = psc.build_cache_key(sess, body)
    body2 = {**body, "period_days": 45}
    k2, _ = psc.build_cache_key(sess, body2)
    assert k1 != k2


def test_save_lookup_and_apply_to_session():
    sess = _minimal_session()
    body = {
        "planning_date": "2025-12-20",
        "period_days": 30,
        "lead_time": 7,
        "target_days": 60,
        "group_by_parent": False,
    }
    po_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "PO_Qty": [10, 5],
        }
    )
    result = {
        "ok": True,
        "sales_through": "2025-12-10",
        "planning_date": "2025-12-20",
        "raise_ledger_rows": 0,
    }
    key = psc.save_shared_cache(sess, body, po_df, result)
    assert key
    assert psc.lookup_shared_cache(sess, body) is not None

    sess2 = _minimal_session()
    out = psc.apply_shared_cache_to_session(sess2, "session-b", body)
    assert out is not None
    assert out.get("from_shared_cache") is True
    assert out.get("status") == "done"
    assert has_spill("session-b")
    assert spill_row_count("session-b") == 2


def test_cache_key_changes_when_existing_po_changes():
    sess = _minimal_session()
    body = {"planning_date": "2025-12-20", "period_days": 30}
    k1, _ = psc.build_cache_key(sess, body)
    sess.existing_po_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "PO_Pipeline_Total": [10, 5],
            "Pending_Cutting": [3, 0],
            "Balance_to_Dispatch": [7, 5],
        }
    )
    sess.existing_po_uploaded_at = "2026-06-04T12:00:00Z"
    sess.existing_po_filename = "Po 4-Jun-26.xlsx"
    k2, _ = psc.build_cache_key(sess, body)
    assert k1 != k2


def test_apply_miss_when_existing_po_changes():
    sess = _minimal_session()
    body = {"planning_date": "2025-12-20", "period_days": 30}
    po_df = pd.DataFrame({"OMS_SKU": ["SKU-A"], "PO_Qty": [1], "Pending_Cutting": [99]})
    psc.save_shared_cache(sess, body, po_df, {"ok": True})

    sess2 = _minimal_session()
    sess2.existing_po_df = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "PO_Pipeline_Total": [5], "Pending_Cutting": [2]}
    )
    sess2.existing_po_uploaded_at = "2026-06-05T10:00:00Z"
    assert psc.apply_shared_cache_to_session(sess2, "session-ep", body) is None


def test_apply_miss_when_inventory_changes():
    sess = _minimal_session()
    body = {"planning_date": "2025-12-20", "period_days": 30}
    po_df = pd.DataFrame({"OMS_SKU": ["SKU-A"], "PO_Qty": [1]})
    psc.save_shared_cache(sess, body, po_df, {"ok": True})

    sess2 = _minimal_session()
    sess2.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU-A", "SKU-C"], "Total_Inventory": [50, 20]}
    )
    assert psc.apply_shared_cache_to_session(sess2, "session-c", body) is None


def test_po_calculate_post_uses_shared_cache(client, monkeypatch, session_for_client):
    """Second session should get immediate done from shared cache without running engine."""
    import time

    _, sess = session_for_client
    days = pd.date_range("2025-12-01", periods=10, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["CACHE-SKU"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [1] * len(days),
            "Units_Effective": [1] * len(days),
            "Source": ["Meesho"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["CACHE-SKU"], "Total_Inventory": [99]}
    )
    body = {
        "planning_date": "2025-12-20",
        "period_days": 30,
        "lead_time": 7,
        "target_days": 60,
        "use_shared_cache": True,
    }
    po_df = pd.DataFrame({"OMS_SKU": ["CACHE-SKU"], "PO_Qty": [7]})
    psc.save_shared_cache(
        sess,
        body,
        po_df,
        {"ok": True, "sales_through": "2025-12-10", "planning_date": "2025-12-20"},
    )

    ran = {"n": 0}

    def _should_not_run(*_a, **_kw):
        ran["n"] += 1
        time.sleep(5)
        return {"ok": True, "total_rows": 0, "columns": []}

    monkeypatch.setattr(
        "backend.services.po_calculate_run.execute_po_calculate",
        _should_not_run,
    )

    from backend.session import store

    sid2, sess2 = store.get_or_create(None)
    sess2.sales_df = sess.sales_df.copy()
    sess2.inventory_df_variant = sess.inventory_df_variant.copy()
    store._sessions[sid2] = sess2

    c2 = client.__class__(client.app)
    c2.cookies.set("auth_token", "test-token")
    c2.cookies.set("session_id", sid2)

    t0 = time.monotonic()
    r = c2.post("/api/po/calculate", json=body)
    elapsed = time.monotonic() - t0

    assert r.status_code == 200
    data = r.json()
    assert data.get("from_shared_cache") is True
    assert data.get("status") == "done"
    assert elapsed < 2.0
    assert ran["n"] == 0

    st = c2.get("/api/po/calculate/result", params={"offset": 0, "limit": 50, "compact": 1})
    assert st.status_code == 200
    page = st.json()
    assert page.get("total") == 1


def test_shared_cache_info_endpoint(client, session_for_client):
    _, sess = session_for_client
    body = {"planning_date": "2025-12-21", "period_days": 30}
    psc.save_shared_cache(
        sess,
        body,
        pd.DataFrame({"OMS_SKU": ["X"], "PO_Qty": [1]}),
        {"ok": True},
    )
    r = client.get(
        "/api/po/calculate/shared-cache",
        params={"planning_date": "2025-12-21", "period_days": 30},
    )
    assert r.status_code == 200
    assert r.json().get("available") is True
    assert r.json().get("row_count") == 1

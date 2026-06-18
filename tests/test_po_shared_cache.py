"""Shared PO calculate cache — cross-session reuse on the same server."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from backend.services import po_shared_cache as psc
from backend.services.po_result_spill import has_spill, spill_row_count

# Use today + 1 day as the planning date so _meta_is_fresh always passes.
# (The cache considers planning dates older than yesterday stale.)
_PLAN_DATE = (date.today() + timedelta(days=1)).isoformat()
_PLAN_DATE_ALT = (date.today() + timedelta(days=2)).isoformat()


@pytest.fixture(autouse=True)
def isolated_shared_cache(tmp_path, monkeypatch):
    import backend.main as main_mod

    warm = tmp_path / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PO_SHARED_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("PO_RESULT_SPILL_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("PO_SHARED_CACHE_ENABLED", "1")
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    main_mod.clear_warm_cache()
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


def test_po_fresh_warmup_profile_matches_ui_defaults():
    from backend.scripts.run_po_calculate_production import _PROFILES

    fresh = next(p for p in _PROFILES if p["label"] == "po_fresh_default")
    assert fresh["period_days"] == 30
    assert fresh["lead_time"] == 60
    assert fresh["target_days"] == 180
    assert fresh["use_seasonality"] is True
    assert fresh["use_ly_fallback"] is True
    assert fresh["enforce_lead_time_release_gate"] is True
    assert fresh["raise_ledger_lookback_days"] == 45


def test_cache_key_changes_when_return_overlay_changes():
    sess = _minimal_session()
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
    k1, _ = psc.build_cache_key(sess, body)
    sess.po_return_overlay_df = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "Return_Overlay_Units": [3]}
    )
    sess.return_overlay_as_of = "2026-06-01"
    k2, _ = psc.build_cache_key(sess, body)
    assert k1 != k2


def test_cache_key_changes_when_params_differ():
    sess = _minimal_session()
    body = {"planning_date": _PLAN_DATE, "period_days": 30, "lead_time": 7}
    k1, _ = psc.build_cache_key(sess, body)
    body2 = {**body, "period_days": 45}
    k2, _ = psc.build_cache_key(sess, body2)
    assert k1 != k2


def test_save_lookup_and_apply_to_session():
    sess = _minimal_session()
    body = {
        "planning_date": _PLAN_DATE,
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
        "planning_date": _PLAN_DATE,
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
    assert out.get("po_merge_version") == psc.PO_MERGE_LOGIC_VERSION
    assert has_spill("session-b")
    assert spill_row_count("session-b") == 2


def test_cache_key_changes_when_merge_version_changes():
    sess = _minimal_session()
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
    k1, fp1 = psc.build_cache_key(sess, body)
    assert fp1.get("po_merge_version") == psc.PO_MERGE_LOGIC_VERSION
    assert "git_sha" in fp1


def test_po_merge_result_is_stale():
    assert psc.po_merge_result_is_stale(None) is False
    assert psc.po_merge_result_is_stale({}) is False
    assert psc.po_merge_result_is_stale({"po_merge_version": None}) is True
    assert psc.po_merge_result_is_stale({"po_merge_version": 1}) is True
    assert psc.po_merge_result_is_stale({"po_merge_version": psc.PO_MERGE_LOGIC_VERSION}) is False


def test_apply_miss_when_merge_version_in_meta_differs():
    sess = _minimal_session()
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
    po_df = pd.DataFrame({"OMS_SKU": ["SKU-A"], "PO_Qty": [1], "Pending_Cutting": [99]})
    key = psc.save_shared_cache(sess, body, po_df, {"ok": True})
    assert key
    meta_path = psc._meta_path(key)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fingerprint"]["po_merge_version"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert psc.apply_shared_cache_to_session(sess, "session-old-merge", body) is None


def test_cache_key_changes_when_existing_po_changes():
    sess = _minimal_session()
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
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
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
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
    body = {"planning_date": _PLAN_DATE, "period_days": 30}
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

    from backend.routers.po import PORequest

    monkeypatch.setattr(
        "backend.services.po_session_hydrate.ensure_po_return_overlay_from_server",
        lambda _sess: False,
    )
    monkeypatch.setattr(
        "backend.services.po_session_hydrate.hydrate_po_session_for_calculate",
        lambda _sess: {},
    )
    monkeypatch.setattr(
        "backend.services.existing_po.read_existing_po_disk_meta",
        lambda: None,
    )
    monkeypatch.setattr(
        "backend.services.po_shared_cache._shared_cache_stale_vs_disk",
        lambda _meta: False,
    )

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
    # Use model_dump() so the fingerprint has the same Pydantic defaults as the API.
    body = PORequest(
        planning_date=_PLAN_DATE, period_days=30, lead_time=7, target_days=60, use_shared_cache=True
    ).model_dump()
    po_df = pd.DataFrame({"OMS_SKU": ["CACHE-SKU"], "PO_Qty": [7]})
    psc.save_shared_cache(
        sess,
        body,
        po_df,
        {"ok": True, "sales_through": "2025-12-10", "planning_date": _PLAN_DATE},
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
    sess2.daily_inventory_history_df = getattr(
        sess, "daily_inventory_history_df", pd.DataFrame()
    ).copy()
    sess2.sku_status_lead_df = getattr(sess, "sku_status_lead_df", pd.DataFrame()).copy()
    sess2.sku_mapping = dict(getattr(sess, "sku_mapping", None) or {})
    sess2.po_return_overlay_df = getattr(sess, "po_return_overlay_df", pd.DataFrame()).copy()
    sess2.return_overlay_as_of = getattr(sess, "return_overlay_as_of", "") or ""
    sess2.existing_po_df = getattr(sess, "existing_po_df", pd.DataFrame()).copy()
    sess2.existing_po_generation = int(getattr(sess, "existing_po_generation", 0) or 0)
    sess2.existing_po_uploaded_at = getattr(sess, "existing_po_uploaded_at", "") or ""
    sess2.existing_po_filename = getattr(sess, "existing_po_filename", "") or ""
    sess2.inventory_snapshot_uploaded_at = (
        getattr(sess, "inventory_snapshot_uploaded_at", "") or ""
    )
    sess2.inventory_snapshot_date = getattr(sess, "inventory_snapshot_date", "") or ""
    store._sessions[sid2] = sess2

    c2 = client.__class__(client.app)
    c2.cookies.set("auth_token", "test-token")
    c2.cookies.set("session_id", sid2)

    t0 = time.monotonic()
    r = c2.post("/api/po/calculate", json=body)
    elapsed = time.monotonic() - t0

    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    job_id = data.get("job_id")
    assert job_id
    assert elapsed < 2.0
    from tests.conftest import wait_po_job_done

    st = wait_po_job_done(c2, job_id, max_sec=15)
    assert st.get("status") == "done"
    assert ran["n"] == 0

    res = c2.get(f"/api/po/calculate/result/{job_id}", params={"offset": 0, "limit": 50, "compact": 1})
    assert res.status_code == 200
    page = res.json()
    assert page.get("total") == 1


def test_shared_cache_used_when_session_has_existing_po(client, monkeypatch, session_for_client):
    """Existing PO in session must not block shared cache (fingerprint already tracks the sheet)."""
    import time

    from backend.routers.po import PORequest

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
    sess.existing_po_df = pd.DataFrame(
        {
            "OMS_SKU": ["CACHE-SKU"],
            "PO_Pipeline_Total": [12],
            "Balance_to_Dispatch": [12],
        }
    )
    sess.existing_po_generation = 3
    sess.existing_po_uploaded_at = "2026-06-06T10:00:00"
    sess.po_calculate_existing_po_generation = 3

    # Use model_dump() so the fingerprint has the same Pydantic defaults as the API.
    body = PORequest(
        planning_date=_PLAN_DATE, period_days=30, lead_time=7, target_days=60, use_shared_cache=True
    ).model_dump()
    from backend.services.po_session_hydrate import hydrate_po_session_for_calculate

    hydrate_po_session_for_calculate(sess)
    psc.save_shared_cache(
        sess,
        body,
        pd.DataFrame({"OMS_SKU": ["CACHE-SKU"], "PO_Qty": [9]}),
        {"ok": True, "sales_through": "2025-12-10", "planning_date": _PLAN_DATE},
    )

    ran = {"n": 0}

    def _should_not_run(*_a, **_kw):
        ran["n"] += 1
        return {"ok": True, "total_rows": 0, "columns": []}

    monkeypatch.setattr(
        "backend.services.po_calculate_run.execute_po_calculate",
        _should_not_run,
    )

    r = client.post("/api/po/calculate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data.get("job_id")
    from tests.conftest import wait_po_job_done

    st = wait_po_job_done(client, data["job_id"], max_sec=15)
    assert st.get("from_shared_cache") is True
    assert ran["n"] == 0


def test_shared_cache_info_endpoint(client, session_for_client):
    _, sess = session_for_client
    # Use the same body shape as the GET /shared-cache endpoint so fingerprints match.
    body = {
        "planning_date": _PLAN_DATE_ALT,
        "period_days": 30,
        "lead_time": 30,
        "target_days": 135,
        "demand_basis": "Sold",
        "group_by_parent": False,
        "raise_ledger_lookback_days": 14,
    }
    psc.save_shared_cache(
        sess,
        body,
        pd.DataFrame({"OMS_SKU": ["X"], "PO_Qty": [1]}),
        {"ok": True},
    )
    r = client.get(
        "/api/po/calculate/shared-cache",
        params={"planning_date": _PLAN_DATE_ALT, "period_days": 30},
    )
    assert r.status_code == 200
    assert r.json().get("available") is True
    assert r.json().get("row_count") == 1

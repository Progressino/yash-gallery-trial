"""Light coverage must not block on Tier-3 restore lock."""

import time
from unittest.mock import patch

import pandas as pd

from backend.session import AppSession
from backend.routers.data import (
    get_coverage,
    _hydrate_queued,
    _build_coverage_response,
    _coverage_sales_ready,
    _unified_sales_build_queued,
)
from backend.routers.upload import clear_stale_background_jobs


def test_clear_stuck_daily_ingest_orphan_started():
    sess = AppSession()
    sess.daily_auto_ingest_status = "running"
    sess.daily_auto_ingest_started = 0.0
    clear_stale_background_jobs(sess)
    assert sess.daily_auto_ingest_status != "running"


def test_clear_stuck_sales_rebuild_orphan_started():
    from backend.routers.upload import _clear_stuck_sales_rebuild

    sess = AppSession()
    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_started = 0.0
    _clear_stuck_sales_rebuild(sess, force=False)
    assert sess.sales_rebuild_status == "idle"


def test_light_coverage_skips_heavy_restore(client, auth_token, monkeypatch):
    """GET /data/coverage?light=1 must not call Tier-3 restore on the request thread."""
    called = {"restore": False, "rebuild": False}

    def _track_restore(*_a, **_k):
        called["restore"] = True

    def _track_rebuild(*_a, **_k):
        called["rebuild"] = True

    monkeypatch.setattr("backend.routers.data._restore_daily_if_needed", _track_restore)
    monkeypatch.setattr("backend.routers.data._ensure_sales_rebuilt", _track_rebuild)
    monkeypatch.setattr("backend.routers.data._maybe_queue_light_session_hydrate", lambda *_a, **_k: None)
    monkeypatch.setattr("backend.routers.data._maybe_queue_unified_sales_build", lambda *_a, **_k: None)

    t0 = time.perf_counter()
    r = client.get("/api/data/coverage", params={"light": "1"})
    elapsed = time.perf_counter() - t0

    assert r.status_code == 200, r.text
    assert elapsed < 5.0
    assert called["restore"] is False
    _hydrate_queued.clear()


def test_light_coverage_applies_warm_cache_when_session_empty(client, auth_token, monkeypatch):
    """Hard refresh: first light coverage should copy warm cache into session without Tier-3."""
    import pandas as pd

    import backend.main as main_mod
    from backend.session import store

    main_mod._warm_cache = {
        "sku_mapping": {"SKU-A": "SKU-A"},
        "mtr_df": pd.DataFrame({"Sku": ["SKU-A"], "Quantity": [1], "TxnDate": ["2026-01-01"]}),
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
        "sales_df": pd.DataFrame({"Sku": ["SKU-A"], "Quantity": [1]}),
        "inventory_df_variant": pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [5]}),
        "inventory_df_parent": pd.DataFrame(),
    }
    main_mod._warm_cache_generation = 3
    main_mod._warm_cache_ready.set()

    monkeypatch.setattr("backend.routers.data._maybe_queue_light_session_hydrate", lambda *_a, **_k: None)

    r = client.get("/api/data/coverage", params={"light": "1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("sku_mapping") is True
    assert body.get("mtr") is True
    assert body.get("sales") is True
    assert body.get("inventory") is True

    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    assert not sess.mtr_df.empty
    assert not sess.sales_df.empty
    main_mod.clear_warm_cache()


def test_po_session_only_disk_load_skips_platform_frames(monkeypatch, tmp_path):
    """Local Mac mode: load sales + PO inputs without 1M+ row platform history."""
    import json

    import pandas as pd

    import backend.main as main_mod

    warm = tmp_path / "warm"
    warm.mkdir()
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    monkeypatch.setenv("WARM_CACHE_PO_SESSION_ONLY", "1")

    pd.DataFrame({"Sku": ["A"], "Quantity": [1]}).to_parquet(warm / "sales_df.parquet", index=False)
    pd.DataFrame({"Sku": ["A"] * 100}).to_parquet(warm / "mtr_df.parquet", index=False)
    with open(warm / "sku_mapping.json", "w") as f:
        json.dump({"A": "A"}, f)
    manifest = {
        "saved_at": "2026-06-17T12:00:00+05:30",
        "keys": ["sales_df", "mtr_df", "sku_mapping"],
    }
    with open(warm / "_manifest.json", "w") as f:
        json.dump(manifest, f)

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    assert ok
    assert "sales_df" in data
    assert "mtr_df" not in data


def test_coverage_sales_ready_when_platforms_loaded_without_sales_df():
    """Warm-cache Phase 2: platform history without unified sales_df should count as sales loaded."""
    sess = AppSession()
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    sess.mtr_df = pd.DataFrame({"Sku": ["SKU-A"], "Quantity": [1]})
    assert sess.sales_df.empty
    assert _coverage_sales_ready(sess) is True
    cov = _build_coverage_response(sess)
    assert cov.sales is True


def test_light_coverage_marks_sales_ready_and_queues_build(client, auth_token, monkeypatch):
    """Platforms-only session: coverage reports sales=True and queues background unified build."""
    import backend.main as main_mod
    from backend.session import store

    main_mod._warm_cache = {
        "sku_mapping": {"SKU-A": "SKU-A"},
        "mtr_df": pd.DataFrame({"SKU": ["SKU-A"], "Quantity": [3], "Date": pd.to_datetime(["2026-01-15"]), "Transaction_Type": ["Shipment"]}),
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
        "inventory_df_variant": pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [5]}),
        "inventory_df_parent": pd.DataFrame(),
    }
    main_mod._warm_cache_generation = 5
    main_mod._warm_cache_ready.set()
    monkeypatch.setattr("backend.routers.data._maybe_queue_light_session_hydrate", lambda *_a, **_k: None)

    queued = {"n": 0}

    def _track_queue(sess, sid):
        queued["n"] += 1

    monkeypatch.setattr("backend.routers.data._maybe_queue_unified_sales_build", _track_queue)

    r = client.get("/api/data/coverage", params={"light": "1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("sales") is True

    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    assert not sess.mtr_df.empty
    assert queued["n"] == 1
    main_mod.clear_warm_cache()
    _unified_sales_build_queued.clear()


def test_full_coverage_rebuilds_sales_when_platforms_only(client, auth_token, monkeypatch):
    """Non-light coverage rebuilds unified sales when platform history is in session."""
    import pandas as pd

    import backend.main as main_mod
    from backend.session import store

    main_mod._warm_cache = {
        "sku_mapping": {"SKU-A": "SKU-A"},
        "mtr_df": pd.DataFrame({"SKU": ["SKU-A"], "Quantity": [3], "Date": pd.to_datetime(["2026-01-15"]), "Transaction_Type": ["Shipment"]}),
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
        "inventory_df_variant": pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [5]}),
        "inventory_df_parent": pd.DataFrame(),
    }
    main_mod._warm_cache_generation = 4
    main_mod._warm_cache_ready.set()
    monkeypatch.setattr("backend.routers.data._maybe_queue_light_session_hydrate", lambda *_a, **_k: None)
    monkeypatch.setattr("backend.routers.data._restore_daily_if_needed", lambda *_a, **_k: None)

    def _fake_rebuild(sess):
        sess.sales_df = pd.DataFrame(
            {
                "Sku": ["SKU-A"],
                "Quantity": [3],
                "TxnDate": pd.to_datetime(["2026-01-15"]),
                "Transaction Type": ["Shipment"],
            }
        )

    monkeypatch.setattr("backend.routers.data._ensure_sales_rebuilt", _fake_rebuild)

    r = client.get("/api/data/coverage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("sales") is True
    assert int(body.get("sales_rows") or 0) > 0

    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    assert not sess.sales_df.empty
    main_mod.clear_warm_cache()

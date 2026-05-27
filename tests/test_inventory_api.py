"""Inventory tab API — pagination and snapshot metadata."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.services.inventory import (
    apply_inventory_session_meta,
    ensure_inventory_snapshot_metadata,
    inventory_rows_for_api,
    inventory_session_meta_bundle,
)


@pytest.fixture
def inv_sess():
    from backend.session import AppSession

    s = AppSession()
    s.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": [f"SKU-{i}" for i in range(120)],
            "OMS_Inventory": [10] * 120,
            "Total_Inventory": [10] * 120,
        }
    )
    s.inventory_debug = {
        "oms": "120 SKUs",
        "amz_disclaimer": {"latest_report_date": "2026-05-21"},
    }
    ensure_inventory_snapshot_metadata(s)
    return s


def test_inventory_rows_paginated():
    df = pd.DataFrame({"OMS_SKU": ["A", "B", "C"], "Total_Inventory": [1, 2, 3]})
    rows, total = inventory_rows_for_api(df, offset=1, limit=1)
    assert total == 3
    assert len(rows) == 1
    assert rows[0]["OMS_SKU"] == "B"


def test_ensure_snapshot_metadata_from_debug(inv_sess):
    inv_sess.inventory_snapshot_date = ""
    inv_sess.inventory_snapshot_date_label = ""
    ensure_inventory_snapshot_metadata(inv_sess)
    assert inv_sess.inventory_snapshot_date == "2026-05-21"
    assert "21 May" in inv_sess.inventory_snapshot_date_label


def test_inventory_api_endpoint(client, inv_sess, monkeypatch):
    from backend.routers import data as data_router

    def _fake_sess(_request):
        return inv_sess

    monkeypatch.setattr(data_router, "_sess", _fake_sess)
    monkeypatch.setattr(data_router, "_restore_inventory_from_warm", lambda _s: None)

    r = client.get("/api/data/inventory", params={"offset": 0, "limit": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loaded"] is True
    assert body["total_rows"] == 120
    assert len(body["rows"]) == 50
    assert body["snapshot_date_label"]


def test_inventory_api_cache_totals(inv_sess):
    from backend.services.inventory import inventory_column_totals, refresh_inventory_api_cache

    refresh_inventory_api_cache(inv_sess)
    assert inv_sess.inventory_api_totals.get("Total_Inventory") == 1200
    assert inv_sess.inventory_api_totals == inventory_column_totals(inv_sess.inventory_df_variant)


def test_get_inventory_running_blocks_stale_warm(client, inv_sess, monkeypatch):
    from backend.routers import data as data_router

    inv_sess.inventory_upload_status = "running"
    inv_sess.inventory_upload_message = "Parsing RAR…"
    inv_sess.inventory_upload_progress = 42

    def _fake_sess(_request):
        return inv_sess

    monkeypatch.setattr(data_router, "_sess", _fake_sess)
    monkeypatch.setattr(data_router, "_restore_inventory_from_warm", lambda _s: None)

    r = client.get("/api/data/inventory")
    assert r.status_code == 200
    body = r.json()
    assert body.get("upload_in_progress") is True
    assert body.get("loaded") is False
    assert body.get("inventory_upload_progress") == 42


def test_sync_inventory_prefers_newer_warm(inv_sess):
    from backend.services.inventory import (
        apply_inventory_snapshot_metadata,
        inventory_snapshot_upload_epoch,
        sync_inventory_snapshot_from_warm,
    )
    import backend.main as main

    inv_sess.inventory_snapshot_uploaded_at = "2026-05-20T08:00:00Z"
    old_total = int(inv_sess.inventory_df_variant["OMS_Inventory"].sum())

    newer = inv_sess.inventory_df_variant.copy()
    newer["OMS_Inventory"] = newer["OMS_Inventory"] + 100
    main._warm_cache = {
        "inventory_df_variant": newer,
        "inventory_df_parent": newer,
        "inventory_session_meta": {
            "inventory_debug": {"oms": "120 SKUs"},
            "inventory_snapshot_uploaded_at": "2026-05-26T12:00:00Z",
            "inventory_snapshot_date": "2026-05-26",
            "inventory_snapshot_date_label": "26 May 2026",
            "inventory_snapshot_date_sources": [],
        },
    }
    sync_inventory_snapshot_from_warm(inv_sess)
    assert int(inv_sess.inventory_df_variant["OMS_Inventory"].sum()) == old_total + 12000
    assert inventory_snapshot_upload_epoch(inv_sess.inventory_snapshot_uploaded_at) > inventory_snapshot_upload_epoch(
        "2026-05-20T08:00:00Z"
    )


def test_restore_backup_after_missing_oms(inv_sess):
    """Backup/restore utility still works; but upload finalize now prefers
    marketplace-based snapshot when Total_Inventory > 0."""
    from backend.services.inventory import (
        backup_inventory_before_upload,
        refresh_inventory_api_cache,
    )
    from backend.routers.upload import _inventory_apply_parse_result

    # Pre-upload snapshot (backup source)
    inv_sess.inventory_df_variant = inv_sess.inventory_df_variant.copy()
    old_df = inv_sess.inventory_df_variant.copy()
    inv_sess.inventory_debug = {"oms": "120 SKUs"}
    refresh_inventory_api_cache(inv_sess)
    backup_inventory_before_upload(inv_sess)

    # Simulate a new parse where OMS is missing in debug, but marketplace stock exists
    # (Total_Inventory > 0). This should NOT restore the old snapshot.
    df_new = inv_sess.inventory_df_variant.iloc[:10].copy()
    df_new["OMS_Inventory"] = 0
    df_new["Total_Inventory"] = 5
    df_new["Marketplace_Total"] = 5  # optional column (used by UI totals only)
    debug_missing_oms = {"oms": "0 SKUs"}

    payload = _inventory_apply_parse_result(
        inv_sess,
        df_variant=df_new,
        df_parent=df_new,
        debug=debug_missing_oms,
        file_parts=[("OMS_Inventory_25-05-2026.rar", b"x")],
        detected=["Amazon (RAR)"],
        warnings=[],
    )
    assert payload["ok"] is True
    assert len(inv_sess.inventory_df_variant) == 10

    # Now simulate a parse that yields zero Total_Inventory => we should restore backup.
    inv_sess.inventory_df_variant = old_df.copy()
    inv_sess.inventory_debug = {"oms": "120 SKUs"}
    backup_inventory_before_upload(inv_sess)
    df_zero = inv_sess.inventory_df_variant.iloc[:3].copy()
    df_zero["OMS_Inventory"] = 0
    df_zero["Total_Inventory"] = 0
    payload2 = _inventory_apply_parse_result(
        inv_sess,
        df_variant=df_zero,
        df_parent=df_zero,
        debug=debug_missing_oms,
        file_parts=[("OMS_Inventory_25-05-2026.rar", b"x")],
        detected=["Amazon (RAR)"],
        warnings=[],
    )
    assert payload2["ok"] is False
    assert len(inv_sess.inventory_df_variant) == 120


def test_inventory_session_meta_roundtrip(inv_sess):
    meta = inventory_session_meta_bundle(inv_sess)
    from backend.session import AppSession

    empty = AppSession()
    apply_inventory_session_meta(empty, meta)
    assert empty.inventory_snapshot_date_label == inv_sess.inventory_snapshot_date_label

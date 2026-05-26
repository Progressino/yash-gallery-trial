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


def test_restore_backup_after_missing_oms(inv_sess):
    from backend.services.inventory import (
        backup_inventory_before_upload,
        refresh_inventory_api_cache,
        restore_inventory_upload_backup,
    )

    inv_sess.inventory_df_variant = inv_sess.inventory_df_variant.copy()
    inv_sess.inventory_debug = {"oms": "120 SKUs"}
    refresh_inventory_api_cache(inv_sess)
    backup_inventory_before_upload(inv_sess)

    inv_sess.inventory_df_variant = inv_sess.inventory_df_variant.iloc[:10].copy()
    inv_sess.inventory_debug = {"oms": "0 SKUs"}
    assert restore_inventory_upload_backup(inv_sess) is True
    assert len(inv_sess.inventory_df_variant) == 120


def test_inventory_session_meta_roundtrip(inv_sess):
    meta = inventory_session_meta_bundle(inv_sess)
    from backend.session import AppSession

    empty = AppSession()
    apply_inventory_session_meta(empty, meta)
    assert empty.inventory_snapshot_date_label == inv_sess.inventory_snapshot_date_label

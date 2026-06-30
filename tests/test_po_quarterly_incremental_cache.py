"""Incremental quarterly cache — stale detection, merge, PO fast path."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.services import po_quarterly_cache as qc
from backend.services.po_quarterly_warmup import (
    QUARTERLY_CACHE_SCHEMA,
    attach_quarterly_columns_to_po_df,
    expected_quarter_columns,
    get_quarterly_payload_for_po,
    quarterly_cache_key,
)


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "_DISK_CACHE_DIR", str(tmp_path))
    with qc._lock:
        qc._payloads.clear()
    yield tmp_path
    with qc._lock:
        qc._payloads.clear()


def test_quarterly_cache_schema_v16():
    assert quarterly_cache_key(False, 8)[0] == QUARTERLY_CACHE_SCHEMA == 16


def test_store_stamps_tier3_metadata(tmp_cache_dir, monkeypatch):
    monkeypatch.setattr(
        "backend.services.po_quarterly_cache._current_tier3_token",
        lambda: {"amazon": "1:100:2026-06-01"},
    )
    key = quarterly_cache_key(False, 8)
    qc.store_shared_quarterly(
        key, {"loaded": True, "columns": ["OMS_SKU"], "rows": [{"OMS_SKU": "A1"}]}
    )
    payload = qc.get_shared_quarterly(key)
    assert payload["tier3_sync_token"] == {"amazon": "1:100:2026-06-01"}
    assert payload.get("built_at")


def test_quarterly_is_stale_when_token_changes(tmp_cache_dir, monkeypatch):
    monkeypatch.setattr(
        "backend.services.po_quarterly_cache._current_tier3_token",
        lambda: {"amazon": "1:100:2026-06-01"},
    )
    key = quarterly_cache_key(False, 8)
    qc.store_shared_quarterly(
        key, {"loaded": True, "columns": ["OMS_SKU"], "rows": [{"OMS_SKU": "A1"}]}
    )
    payload = qc.get_shared_quarterly(key)
    assert qc.quarterly_is_stale(payload) is False

    monkeypatch.setattr(
        "backend.services.po_quarterly_cache._current_tier3_token",
        lambda: {"amazon": "2:200:2026-06-02"},
    )
    assert qc.quarterly_is_stale(payload) is True


def test_merge_incremental_patches_recent_quarters_only(tmp_cache_dir):
    recent = expected_quarter_columns(8)[-2:]
    older = expected_quarter_columns(8)[:-2]
    existing_row = {"OMS_SKU": "SKU1", "Units_90d": 1, "ADS": 0.1}
    for c in older:
        existing_row[c] = 100
    for c in recent:
        existing_row[c] = 10

    inc_row = {"OMS_SKU": "SKU1", "Units_90d": 50, "ADS": 2.5}
    for c in recent:
        inc_row[c] = 99

    merged = qc.merge_incremental_quarterly_payload(
        {"loaded": True, "columns": list(existing_row.keys()), "rows": [existing_row]},
        {"loaded": True, "columns": list(inc_row.keys()), "rows": [inc_row]},
        recent_quarter_cols=recent,
    )
    out = merged["rows"][0]
    for c in older:
        assert out[c] == 100
    for c in recent:
        assert out[c] == 99
    assert out["Units_90d"] == 50
    assert out["ADS"] == 2.5


def test_attach_quarterly_uses_shared_cache_without_sync_build(
    tmp_cache_dir, monkeypatch
):
    from backend.session import AppSession

    monkeypatch.setattr(
        "backend.services.po_quarterly_warmup.try_build_quarterly_payload_sync",
        lambda *a, **k: pytest.fail("PO calculate must not block on quarterly rebuild"),
    )
    key = quarterly_cache_key(False, 8)
    cols = ["OMS_SKU"] + expected_quarter_columns(2)[-1:]
    qc.store_shared_quarterly(
        key,
        {
            "loaded": True,
            "columns": cols,
            "rows": [{"OMS_SKU": "SKU1", cols[-1]: 42}],
        },
    )
    sess = AppSession()
    po_df = pd.DataFrame({"OMS_SKU": ["SKU1"], "Qty": [1]})
    out = attach_quarterly_columns_to_po_df(po_df, sess, n_quarters=8)
    assert int(out[cols[-1]].iloc[0]) == 42


def test_get_quarterly_payload_for_po_schedules_full_build_on_miss(monkeypatch):
    from backend.session import AppSession

    scheduled: list[bool] = []

    def _sched(key, sess, *, force_full=False):
        scheduled.append(force_full)
        return True

    monkeypatch.setattr(qc, "get_shared_quarterly", lambda _k: None)
    monkeypatch.setattr(qc, "schedule_quarterly_refresh_if_stale", _sched)
    sess = AppSession()
    payload = get_quarterly_payload_for_po(sess)
    assert payload.get("loaded") is False
    assert scheduled == [True]

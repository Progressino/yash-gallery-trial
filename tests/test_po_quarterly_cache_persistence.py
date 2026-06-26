"""Shared PO quarterly cache must survive a backend restart via disk persistence."""

import importlib

import pytest

from backend.services import po_quarterly_cache as qc
from backend.services.po_quarterly_warmup import quarterly_cache_key


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "_DISK_CACHE_DIR", str(tmp_path))
    with qc._lock:
        qc._payloads.clear()
    yield tmp_path
    with qc._lock:
        qc._payloads.clear()


def test_disk_path_uses_schema_group_and_n_quarters(tmp_cache_dir):
    key = quarterly_cache_key(False, 8)
    path = qc._disk_path(key)
    assert path.endswith("quarterly_v9_0_8.json")


def test_store_persists_to_disk_and_survives_memory_clear(tmp_cache_dir):
    key = quarterly_cache_key(False, 8)
    payload = {"loaded": True, "columns": ["SKU", "Q1"], "rows": [{"SKU": "A1", "Q1": 10}]}

    qc.store_shared_quarterly(key, payload)
    assert qc.get_shared_quarterly(key) is not None

    # Simulate a restart: in-memory cache wiped, disk survives (Docker volume).
    with qc._lock:
        qc._payloads.clear()
    restored = qc.get_shared_quarterly(key)
    assert restored is not None
    assert restored["rows"][0]["SKU"] == "A1"


def test_load_from_disk_returns_none_when_absent(tmp_cache_dir):
    key = quarterly_cache_key(True, 8)
    assert qc.load_shared_quarterly_from_disk(key) is None


def test_load_from_disk_ignores_unloaded_payload(tmp_cache_dir):
    key = quarterly_cache_key(False, 8)
    qc.save_shared_quarterly_to_disk(key, {"loaded": False, "rows": []})
    assert qc.load_shared_quarterly_from_disk(key) is None


def test_invalidate_clears_memory_and_disk(tmp_cache_dir):
    key = quarterly_cache_key(False, 8)
    payload = {"loaded": True, "columns": ["SKU"], "rows": [{"SKU": "A1"}]}
    qc.store_shared_quarterly(key, payload)
    assert qc.get_shared_quarterly(key) is not None
    assert qc.load_shared_quarterly_from_disk(key) is not None

    qc.invalidate_shared_quarterly()

    assert qc.get_shared_quarterly(key) is None
    assert qc.load_shared_quarterly_from_disk(key) is None


def test_json_default_handles_numpy_scalars(tmp_cache_dir):
    import numpy as np

    key = quarterly_cache_key(True, 8)
    payload = {
        "loaded": True,
        "columns": ["SKU", "Q1"],
        "rows": [{"SKU": "A1", "Q1": np.int64(42)}],
    }
    qc.store_shared_quarterly(key, payload)
    restored = qc.load_shared_quarterly_from_disk(key)
    assert restored["rows"][0]["Q1"] == 42

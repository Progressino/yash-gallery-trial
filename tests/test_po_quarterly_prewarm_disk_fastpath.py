"""Shared quarterly prewarm should reuse a persisted disk payload instead of
paying for a 30-180s Tier-3 streaming rebuild on every restart."""

import threading
import time

from backend.services import po_quarterly_cache as qc
from backend.services import po_quarterly_warmup as qw


def _join_prewarm_threads(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        threads = [t for t in threading.enumerate() if t.name == "po-qtr-prewarm"]
        if not threads:
            time.sleep(0.05)
            continue
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.time()))
        return
    raise AssertionError("po-qtr-prewarm thread never started")


def test_prewarm_uses_persisted_disk_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "_DISK_CACHE_DIR", str(tmp_path))
    with qc._lock:
        qc._payloads.clear()

    key = qw.quarterly_cache_key(False, 8)
    payload = {"loaded": True, "columns": ["SKU", "Q1"], "rows": [{"SKU": "A1", "Q1": 5}]}
    qc.save_shared_quarterly_to_disk(key, payload)

    # Guard: if the fast path didn't short-circuit, this would hang waiting on
    # the upload-memory lock / sleep for ~30-150s.
    def _boom():
        raise AssertionError("upload_memory_lock_held should not be checked on disk-cache hit")

    monkeypatch.setattr(
        "backend.concurrency.upload_memory_lock_held", _boom, raising=False
    )
    monkeypatch.setattr(qw, "po_quarterly_demand_bases", lambda: ("Sold",))

    qw.schedule_shared_quarterly_prewarm()
    _join_prewarm_threads()

    loaded = qc.get_shared_quarterly(key)
    assert loaded and loaded.get("loaded")
    assert loaded.get("rows") == [{"SKU": "A1", "Q1": 5, "Oct-Dec 2024": 0, "Jan-Mar 2025": 0, "Apr-Jun 2025": 0, "Jul-Sep 2025": 0, "Oct-Dec 2025": 0, "Jan-Mar 2026": 0, "Apr-Jun 2026": 0, "Jul-Sep 2026": 0}]

    with qc._lock:
        qc._payloads.clear()

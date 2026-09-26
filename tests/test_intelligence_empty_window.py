"""Windows past the latest upload must report zero, fast — never all-time totals."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from backend.services import intelligence_guard as guard


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-08-01", "2026-08-15", "2026-09-05"],
            "OMS_SKU": ["A", "B", "A"],
            "TxnType": ["Shipment", "Shipment", "Shipment"],
            "Quantity": [100, 200, 300],
        }
    )


def test_platform_metrics_window_without_rows_is_zero_not_all_time():
    from backend.services.sales import _compute_platform_metrics

    out = _compute_platform_metrics(
        _history_frame(), "Myntra", "OMS_SKU", "TxnType",
        start_date="2026-09-20", end_date="2026-09-26",
    )
    assert out["loaded"] is True
    assert out["total_units"] == 0
    assert out["net_units"] == 0


def test_platform_metrics_window_with_rows_counts_only_window():
    from backend.services.sales import _compute_platform_metrics

    out = _compute_platform_metrics(
        _history_frame(), "Myntra", "OMS_SKU", "TxnType",
        start_date="2026-09-01", end_date="2026-09-26",
    )
    assert out["total_units"] == 300


def test_window_after_latest_data(monkeypatch):
    monkeypatch.setattr(guard, "latest_sales_date", lambda: "2026-09-06")
    assert guard.window_after_latest_data("2026-09-20", "2026-09-26") == "2026-09-06"
    # One day of IST slack: a window starting the day after the latest is not "empty".
    assert guard.window_after_latest_data("2026-09-07", "2026-09-13") is None
    assert guard.window_after_latest_data("2026-08-27", "2026-09-26") is None
    monkeypatch.setattr(guard, "latest_sales_date", lambda: None)
    assert guard.window_after_latest_data("2026-09-20", "2026-09-26") is None


def test_empty_window_payload_is_final_and_zero():
    p = guard.empty_window_payload("2026-09-20", "2026-09-26", "2026-09-06")
    assert p["status"] == "ready"
    assert p["data_completeness"] == "full"
    assert p["empty_window"] is True
    assert p["data_max_date"] == "2026-09-06"
    assert p["sales_summary"]["total_units"] == 0
    assert all(pl["total_units"] == 0 and pl["loaded"] for pl in p["platform_summary"])
    assert "2026-09-06" in p["message"]


def test_single_flight_runs_once_for_concurrent_callers():
    calls = []
    started = threading.Event()

    def build():
        calls.append(1)
        started.set()
        time.sleep(0.2)
        return {"units": 7}

    results = []

    def worker():
        results.append(guard.single_flight(("t", "same"), build))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    threads[0].start()
    started.wait(1.0)
    for t in threads[1:]:
        t.start()
    for t in threads:
        t.join(2.0)
    assert len(calls) == 1
    assert results == [{"units": 7}] * 4
    # Waiters get their own dict so mutating one response can't corrupt another.
    assert len({id(r) for r in results}) == 4


def test_background_build_yields_to_waiting_request():
    order: list[str] = []
    fg_holding = threading.Event()
    release_fg = threading.Event()

    def fg_build():
        fg_holding.set()
        release_fg.wait(1.0)
        order.append("fg1")
        return {}

    t_fg1 = threading.Thread(target=lambda: guard.single_flight(("t", "fg1"), fg_build))
    t_fg1.start()
    fg_holding.wait(1.0)

    t_bg = threading.Thread(target=lambda: guard.gated(lambda: order.append("bg")))
    t_bg.start()
    time.sleep(0.05)
    t_fg2 = threading.Thread(
        target=lambda: guard.single_flight(("t", "fg2"), lambda: order.append("fg2") or {})
    )
    t_fg2.start()
    time.sleep(0.05)
    release_fg.set()
    for t in (t_fg1, t_bg, t_fg2):
        t.join(3.0)
    assert order == ["fg1", "fg2", "bg"]


def test_gate_is_reentrant_for_nested_builds():
    assert guard.gated(lambda: guard.gated(lambda: 5)) == 5
    assert guard.single_flight(("t", "nest"), lambda: guard.gated(lambda: {"ok": 1})) == {"ok": 1}


def test_standard_windows_match_ui_presets():
    from backend.services.intelligence_artifacts import IST, standard_intelligence_windows

    today = datetime.now(IST).date()
    starts = [s for s, _ in standard_intelligence_windows()]
    assert starts[0] == (today - timedelta(days=6)).isoformat()
    assert starts[1] == (today - timedelta(days=30)).isoformat()


def test_artifact_build_skips_window_past_latest_data(monkeypatch):
    from backend.services import intelligence_artifacts as ia

    monkeypatch.setattr(guard, "latest_sales_date", lambda: "2026-09-06")

    def boom(*a, **k):
        raise AssertionError("must not build an empty window")

    monkeypatch.setattr(ia, "_build_hot_payload", boom)
    monkeypatch.setattr(ia, "_build_deep_payload", boom)
    assert ia.build_and_store_artifact(None, "2026-09-20", "2026-09-26", ia.KIND_HOT) is None


def test_artifact_without_current_schema_is_ignored(tmp_path, monkeypatch):
    import json

    from backend.services import intelligence_artifacts as ia

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    s, e = "2026-09-20", "2026-09-26"
    with ia._MEM_LOCK:
        ia._MEM.pop((s, e, ia.KIND_HOT), None)
    legacy = {
        "version": ia.intelligence_version_for_window(s, e),
        "kind": ia.KIND_HOT,
        "start_date": s,
        "end_date": e,
        "payload": {"sales_summary": {"total_units": 896122}, "platform_summary": [{"platform": "Amazon"}]},
    }
    with open(ia._artifact_json_path(s, e, ia.KIND_HOT), "w", encoding="utf-8") as f:
        json.dump(legacy, f)
    payload, _meta = ia.load_artifact(s, e, ia.KIND_HOT)
    assert payload is None


def test_legacy_day_parquet_is_discarded(tmp_path, monkeypatch):
    from backend.services import intelligence_artifact_store as store

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(store, "_maybe_fetch_cdn_file", lambda *a, **k: False)
    day = "2026-09-20"
    path = store.day_parquet_path(day)
    pd.DataFrame(
        [{"section": "day_platform", "date": day, "platform": "Amazon", "loaded": True,
          "total_units": 973418, "total_returns": 0, "net_units": 973418, "return_rate": 0.0}]
    ).to_parquet(path, index=False)
    assert store.read_day_parquet(day) is None

    payload = {
        "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 12}],
        "sales_summary": {"total_units": 12},
    }
    assert store.write_day_parquet(day, payload)
    out = store.read_day_parquet(day)
    assert out and out["platform_summary"][0]["total_units"] == 12
    assert "schema" not in out["sales_summary"]

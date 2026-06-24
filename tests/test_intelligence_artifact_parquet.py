"""Parquet intelligence artifacts — roundtrip and per-day drill-down."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


@pytest.fixture
def artifact_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "intelligence", "daily")
        os.makedirs(root, exist_ok=True)
        monkeypatch.setenv("WARM_CACHE_DIR", os.path.join(tmp))
        yield tmp


def test_deep_parquet_roundtrip(artifact_env):
    from backend.services.intelligence_artifact_store import (
        read_deep_parquet,
        write_deep_parquet,
    )

    payload = {
        "source": "test",
        "sales_summary": {"total_units": 100, "net_units": 90, "return_rate": 10.0},
        "platform_summary": [
            {
                "platform": "Amazon",
                "loaded": True,
                "total_units": 60,
                "total_returns": 5,
                "net_units": 55,
                "return_rate": 8.3,
                "top_sku": "SKU1",
                "trend_direction": "up",
                "trend_direction_net": "up",
                "daily": [{"date": "2026-06-01", "units": 30, "returns": 2, "net_units": 28}],
                "monthly": [],
                "by_state": [],
            }
        ],
        "top_skus": [{"sku": "SKU1", "units": 30, "platform": "Amazon"}],
    }
    assert write_deep_parquet("2026-06-01", "2026-06-24", payload)
    loaded = read_deep_parquet("2026-06-01", "2026-06-24")
    assert loaded is not None
    assert loaded["sales_summary"]["total_units"] == 100
    assert len(loaded["platform_summary"]) == 1
    assert loaded["platform_summary"][0]["platform"] == "Amazon"
    assert loaded["platform_summary"][0]["daily"][0]["units"] == 30
    assert loaded["top_skus"][0]["sku"] == "SKU1"


def test_day_parquet_roundtrip(artifact_env):
    from backend.services.intelligence_artifact_store import read_day_parquet, write_day_parquet

    payload = {
        "platform_summary": [
            {
                "platform": "Myntra",
                "loaded": True,
                "total_units": 12,
                "total_returns": 1,
                "net_units": 11,
                "return_rate": 8.3,
            }
        ],
        "sales_summary": {"total_units": 12, "net_units": 11},
    }
    assert write_day_parquet("2026-06-20", payload)
    loaded = read_day_parquet("2026-06-20")
    assert loaded is not None
    assert loaded["date"] == "2026-06-20"
    assert loaded["platform_summary"][0]["platform"] == "Myntra"


def test_save_deep_uses_parquet_storage(artifact_env, monkeypatch):
    from backend.services import intelligence_artifacts as ia

    monkeypatch.setattr(
        ia,
        "intelligence_version_for_window",
        lambda *a, **k: "ver-test",
    )
    payload = {
        "source": "tier3",
        "data_completeness": "full",
        "sales_summary": {"total_units": 1},
        "platform_summary": [{"platform": "Amazon", "loaded": True, "total_units": 1, "daily": []}],
        "top_skus": [],
    }
    ia.save_artifact("2026-06-01", "2026-06-07", ia.KIND_DEEP, payload)
    path = os.path.join(artifact_env, "intelligence", "daily", "intelligence_bundle_2026-06-01_2026-06-07_deep.json")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("storage") == "parquet"
    loaded, m = ia.load_artifact("2026-06-01", "2026-06-07", ia.KIND_DEEP, allow_stale=False)
    assert loaded is not None
    assert m.get("source") in ("disk", "parquet", "memory")

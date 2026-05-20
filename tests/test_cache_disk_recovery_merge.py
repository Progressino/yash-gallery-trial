"""GitHub load + on-disk warm-cache recovery merge (cache router)."""
import pandas as pd
import pytest

from backend.routers.cache import _merge_disk_warm_cache_into_loaded


def test_merge_disk_recovers_rows_missing_from_github(monkeypatch):
    small = pd.DataFrame(
        {
            "OrderId": ["a1", "a2"],
            "LineKey": ["L1", "L2"],
            "OMS_SKU": ["S1", "S2"],
            "Date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [1, 1],
            "RawStatus": ["x", "y"],
        }
    )
    big = pd.DataFrame(
        {
            "OrderId": [f"o{i}" for i in range(80)],
            "LineKey": [f"DL{i}" for i in range(80)],
            "OMS_SKU": [f"SK{i % 4}" for i in range(80)],
            "Date": [pd.Timestamp("2024-06-01")] * 80,
            "TxnType": ["Shipment"] * 80,
            "Quantity": [1] * 80,
            "RawStatus": [f"r{i}" for i in range(80)],
        }
    )

    def fake_disk(ignore_age=True):
        return True, {"mtr_df": big, "sku_mapping": {"A": 1, "B": 2, "C": 3}}

    monkeypatch.setattr("backend.routers.cache._disk_warm_load", fake_disk)

    loaded = {"mtr_df": small, "sku_mapping": {"A": 9}}
    out, note = _merge_disk_warm_cache_into_loaded(loaded)

    assert len(out["mtr_df"]) == 82
    assert out["sku_mapping"]["A"] == 9
    assert out["sku_mapping"]["B"] == 2
    assert "mtr_df" in note


def test_merge_disk_noop_when_disk_empty(monkeypatch):
    monkeypatch.setattr("backend.routers.cache._disk_warm_load", lambda ignore_age=True: (False, {}))
    loaded = {"mtr_df": pd.DataFrame({"x": [1]})}
    out, note = _merge_disk_warm_cache_into_loaded(loaded)
    assert len(out["mtr_df"]) == 1
    assert note == ""

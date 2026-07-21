"""Automated data-health suite basics."""
import json

import pandas as pd

from backend.services import data_health as dh


def test_inventory_component_sum_check_flags_mismatch():
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B"],
            "OMS_Inventory": [10, 20],
            "Amazon_Inventory": [5, 0],
            "Total_Inventory": [15, 30],  # B is 10 too high
        }
    )
    c = dh._inventory_component_sum_check(inv)
    assert c["ok"] is False
    assert c["data"]["delta"] == 10

    inv["Total_Inventory"] = [15, 20]
    c2 = dh._inventory_component_sum_check(inv)
    assert c2["ok"] is True


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    result = {
        "generated_at": "2026-07-21T00:00:00+00:00",
        "ok": False,
        "fail_count": 1,
        "warn_count": 0,
        "checks": [
            {
                "id": "x",
                "area": "sales",
                "title": "t",
                "ok": False,
                "severity": "fail",
                "detail": "d",
                "data": {},
            }
        ],
    }
    dh._cache_path().write_text(json.dumps(result))
    out = dh.get_cached_data_health()
    assert out is not None
    assert out["ok"] is False
    assert out["fail_count"] == 1
    # Stale cache is ignored
    assert dh.get_cached_data_health(max_age_sec=-1) is None


def test_inventory_amazon_ledger_check_uses_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    meta = {
        "inventory_debug": {"amz_disclaimer": {"sellable_non_znne_units": 100}},
    }
    (tmp_path / "inventory_session_meta.json").write_text(json.dumps(meta))
    inv = pd.DataFrame({"OMS_SKU": ["A"], "Amazon_Inventory": [90]})
    c = dh._inventory_amazon_ledger_check(inv)
    assert c["ok"] is False
    assert c["data"]["delta"] == -10

    inv["Amazon_Inventory"] = [100]
    assert dh._inventory_amazon_ledger_check(inv)["ok"] is True

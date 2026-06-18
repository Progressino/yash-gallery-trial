"""PostgreSQL shared operational snapshot + Tier-3 dual-write."""

import pandas as pd
import pytest

from backend.db import forecast_ops_pg as ops
from backend.session import AppSession
from backend.services.inventory import clear_inventory_snapshot


@pytest.fixture(autouse=True)
def _enable_ops_pg(monkeypatch):
    monkeypatch.setattr(ops, "_table_ready", True)
    monkeypatch.setenv("FORECAST_OPS_PG", "1")
    monkeypatch.setenv("FORECAST_SESSION_DATABASE_URL", "postgresql://test:test@localhost/test")


def test_warm_cache_bundle_roundtrip():
    cache = {
        "sku_mapping": {"A": "A"},
        "mtr_df": pd.DataFrame({"Sku": ["A"], "Quantity": [1]}),
        "inventory_df_variant": pd.DataFrame({"OMS_SKU": ["A"], "Total_Inventory": [5]}),
    }
    blob, manifest = ops.warm_cache_dict_to_bundle(cache)
    assert blob
    assert "mtr_df" in manifest.get("keys", [])
    restored = ops.warm_cache_dict_from_bundle(blob)
    assert restored is not None
    assert restored["sku_mapping"] == {"A": "A"}
    assert len(restored["mtr_df"]) == 1


def test_snapshot_rejects_empty_cache():
    assert ops._snapshot_has_operational_data({}) is False
    assert ops._snapshot_has_operational_data({"snapdeal_df": pd.DataFrame()}) is False


def test_clear_inventory_does_not_affect_snapshot_keys():
    sess = AppSession()
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["A"], "Total_Inventory": [1]})
    clear_inventory_snapshot(sess)
    assert sess.inventory_df_variant.empty


def test_pg_save_daily_file_calls_insert(monkeypatch):
    calls: list[str] = []

    class FakeConn:
        def execute(self, sql, params=None):
            calls.append(" ".join(str(sql).split()))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ops, "_require_conn", lambda: FakeConn())
    ops.pg_save_daily_file("amazon", "t.csv", "2026-06-18", "2026-06-18", "2026-06-18", b"PAR1", 1)
    assert any("INSERT INTO forecast_daily_uploads" in c for c in calls)

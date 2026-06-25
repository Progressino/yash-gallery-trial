"""PO sidecars and existing PO must load from loose disk parquets when manifest omits them."""
import json
from pathlib import Path

import pandas as pd
import pytest


def test_loose_parquets_include_daily_inventory_history(tmp_path, monkeypatch):
    from backend import main as m

    monkeypatch.setattr(m, "_DISK_CACHE_DIR", str(tmp_path))
    hist = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-A"],
            "Date": pd.to_datetime(["2026-06-24", "2026-06-25"]),
            "Qty": [10, 12],
        }
    )
    hist.to_parquet(tmp_path / "daily_inventory_history_df.parquet", index=False)
    loose = m._warm_cache_loose_parquets_from_dir(tmp_path)
    assert "daily_inventory_history_df" in loose
    assert len(loose["daily_inventory_history_df"]) == 2


def test_top_up_po_sidecars_from_loose_disk(monkeypatch, tmp_path):
    from backend import main as m

    monkeypatch.setattr(m, "_DISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    m._warm_cache = {"inventory_df_variant": pd.DataFrame({"OMS_SKU": ["X"]})}
    hist = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-B"] * 3,
            "Date": pd.to_datetime(["2026-06-20", "2026-06-21", "2026-06-22"]),
            "Qty": [1, 2, 3],
        }
    )
    hist.to_parquet(tmp_path / "daily_inventory_history_df.parquet", index=False)
    ep = pd.DataFrame({"OMS_SKU": ["SKU-B"], "PO_Pipeline_Total": [50]})
    ep.to_parquet(tmp_path / "existing_po_df.parquet", index=False)
    (tmp_path / "existing_po_meta.json").write_text(
        json.dumps({"existing_po_generation": 2, "existing_po_rows": 1}),
        encoding="utf-8",
    )

    m._top_up_po_sidecars_from_loose_disk()
    assert len(m._warm_cache["daily_inventory_history_df"]) == 3
    assert len(m._warm_cache["existing_po_df"]) == 1


def test_ensure_po_sidecars_prefers_newer_inventory_max_date(monkeypatch, tmp_path):
    from backend.main import _DISK_CACHE_DIR
    from backend.services.po_session_hydrate import ensure_po_sidecars_hydrated
    from backend.session import AppSession

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    stale = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"] * 100,
            "Date": pd.to_datetime(["2026-05-28", "2026-05-29", "2026-05-30"] * 34)[:100],
            "Qty": [5] * 100,
        }
    )
    fresh = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-A"],
            "Date": pd.to_datetime(["2026-06-24", "2026-06-25"]),
            "Qty": [8, 9],
        }
    )
    fresh.to_parquet(tmp_path / "daily_inventory_history_df.parquet", index=False)

    import backend.main as m

    m._warm_cache = {"daily_inventory_history_df": stale}
    sess = AppSession()
    sess.daily_inventory_history_df = stale.copy()
    stats = ensure_po_sidecars_hydrated(sess)
    assert stats["daily_inventory_history_df"] == 2
    mx = pd.to_datetime(sess.daily_inventory_history_df["Date"]).max()
    assert str(mx.date()) == "2026-06-25"

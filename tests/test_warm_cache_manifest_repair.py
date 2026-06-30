"""Partial manifest saves must not hide bulk platform parquets on disk."""
import json

import pandas as pd
import pytest

import backend.main as main


@pytest.fixture
def tmp_warm_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_DISK_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_repair_manifest_adds_loose_parquets(tmp_warm_dir):
    (tmp_warm_dir / "myntra_df.parquet").write_bytes(b"")
    pd.DataFrame({"OMS_SKU": ["A"], "Date": ["2026-06-01"]}).to_parquet(
        tmp_warm_dir / "myntra_df.parquet", index=False
    )
    pd.DataFrame({"OMS_SKU": ["B"], "Date": ["2026-06-01"]}).to_parquet(
        tmp_warm_dir / "mtr_df.parquet", index=False
    )
    manifest = {
        "saved_at": "2026-06-30T10:00:00+05:30",
        "keys": ["sales_df", "sku_mapping"],
    }
    (tmp_warm_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    main._repair_disk_manifest_from_loose_parquets()

    repaired = json.loads((tmp_warm_dir / "_manifest.json").read_text(encoding="utf-8"))
    keys = set(repaired.get("keys") or [])
    assert "myntra_df" in keys
    assert "mtr_df" in keys
    assert "sales_df" in keys


def test_merge_loose_disk_restores_platform_frames(tmp_warm_dir):
    pd.DataFrame({"OMS_SKU": ["X"], "Date": ["2026-06-01"]}).to_parquet(
        tmp_warm_dir / "meesho_df.parquet", index=False
    )
    cache = {"sales_df": pd.DataFrame({"OMS_SKU": ["S"], "TxnDate": ["2026-06-01"]})}
    merged = main._merge_loose_disk_into_cache_dict(cache)
    assert "meesho_df" in merged
    assert len(merged["meesho_df"]) == 1

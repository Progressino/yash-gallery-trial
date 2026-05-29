"""Fast restore: local cache first, selective GitHub downloads."""
import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.services import github_cache as gh


def _fake_manifest(row_counts: dict) -> dict:
    return {"saved_at": "2026-05-29T12:00:00", "saved_at_display": "May 29", "row_counts": row_counts}


def test_load_cache_from_drive_only_keys_limits_fetch(monkeypatch):
    manifest = _fake_manifest({"mtr_df": 1000})
    assets = {
        "_manifest.json": (1, "http://manifest"),
        "mtr_df.parquet": (2, "http://mtr"),
        "sales_df.parquet": (3, "http://sales"),
    }
    fetched: list[str] = []

    monkeypatch.setattr(gh, "_get_gh_release", lambda: (99, assets, None))

    def fake_fetch(url, *, manifest_saved_at, filename):
        fetched.append(filename)
        if filename.endswith(".json"):
            return json.dumps(manifest).encode()
        return b"PARQUET"

    monkeypatch.setattr(gh, "_fetch_asset_bytes", fake_fetch)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda buf, engine: pd.DataFrame({"x": [1, 2, 3]}),
    )

    ok, msg, loaded = gh.load_cache_from_drive(only_keys=frozenset({"mtr_df"}))
    assert ok
    assert "mtr_df" in loaded
    assert "sales_df" not in loaded
    assert "mtr_df.parquet" in fetched
    assert "sales_df.parquet" not in fetched


def test_keys_needing_github_download_skips_when_local_full():
    loaded = {"mtr_df": pd.DataFrame({"a": range(950_000)})}
    manifest = _fake_manifest({"mtr_df": 1_000_000})
    need = gh._keys_needing_github_download(loaded, manifest)
    assert "mtr_df" not in need


def test_keys_needing_github_download_includes_gap():
    loaded = {"mtr_df": pd.DataFrame({"a": range(50_000)})}
    manifest = _fake_manifest({"mtr_df": 1_000_000})
    need = gh._keys_needing_github_download(loaded, manifest)
    assert "mtr_df" in need


def test_load_history_for_restore_skips_network_when_disk_sufficient(monkeypatch):
    big_mtr = pd.DataFrame({"a": range(1_000_000)})
    disk = {"mtr_df": big_mtr, "sku_mapping": {"X": "X"}}
    manifest = _fake_manifest({"mtr_df": 1_000_000, "sku_mapping": 1})

    import backend.main as main

    monkeypatch.setattr(main, "_warm_cache", {})
    monkeypatch.setattr(main, "_load_warm_cache_from_disk", lambda ignore_age=False: (True, disk))
    monkeypatch.setattr(main, "warm_cache_disk_recovery_dict", lambda: {})
    monkeypatch.setattr(gh, "get_cache_manifest", lambda: manifest)

    github_called = {"n": 0}

    def no_github(*a, **k):
        github_called["n"] += 1
        return False, "should not run", {}

    monkeypatch.setattr(gh, "load_cache_from_drive", no_github)

    ok, msg, loaded, used_github = gh.load_history_for_restore()
    assert ok
    assert used_github is False
    assert github_called["n"] == 0
    assert len(loaded["mtr_df"]) == 1_000_000
    assert "no GitHub download" in msg


def test_union_history_dicts_keeps_largest_frame():
    small = pd.DataFrame({"a": [1]})
    big = pd.DataFrame({"a": range(100)})
    out = gh._union_history_dicts({"mtr_df": small}, {"mtr_df": big})
    assert len(out["mtr_df"]) == 100

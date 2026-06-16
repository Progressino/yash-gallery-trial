"""Phase-0 disk cache must self-heal from an all-empty/corrupt snapshot.

Reproduces the production "0/8 Data loaded" incident: a partial session published
an empty platform snapshot to the warm cache + /data/warm_cache disk parquets.
Because production runs WARM_CACHE_MIN_MTR_ROWS=0 (so small accounts fast-path),
the old corrupt-cache guard was disabled and every restart reloaded the empty disk
cache, skipping the GitHub rebuild → every session saw "0/8 Data loaded" forever.

The always-on hard floor (_DISK_CACHE_HARD_MIN_ROWS) must reject an all-empty disk
cache regardless of the configurable mtr guard.
"""
import pandas as pd
import pytest

import backend.main as main


def _empty_disk_cache() -> dict:
    return {
        "mtr_df": pd.DataFrame(),
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
        # A partial publish may still carry sidecar/meta keys — these must NOT
        # count as "platform data" for the corruption check.
        "sku_status_lead_df": pd.DataFrame({"SKU": ["A1"], "Lead time": [14]}),
    }


def _full_disk_cache() -> dict:
    return {
        "mtr_df": pd.DataFrame({"OMS_SKU": ["A1"] * 10, "Date": pd.NaT}),
        "myntra_df": pd.DataFrame(),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
    }


def test_warm_total_platform_rows_ignores_non_platform_keys():
    assert main._warm_total_platform_rows(_empty_disk_cache()) == 0
    assert main._warm_total_platform_rows(_full_disk_cache()) == 10
    assert main._warm_total_platform_rows({}) == 0
    assert main._warm_total_platform_rows(None) == 0


def test_all_empty_disk_cache_is_corrupt_even_with_mtr_guard_disabled(monkeypatch):
    # Production config: configurable mtr guard disabled so small accounts fast-path.
    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 0)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 1)

    reason = main._disk_cache_corruption_reason(_empty_disk_cache())
    assert reason is not None
    assert "platform" in reason.lower()


def test_full_disk_cache_is_not_flagged_corrupt(monkeypatch):
    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 0)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 1)

    assert main._disk_cache_corruption_reason(_full_disk_cache()) is None


def test_small_account_passes_when_mtr_guard_disabled(monkeypatch):
    """A legitimately small account (a handful of rows in one platform) must NOT
    be treated as corrupt when the mtr guard is disabled — only the all-empty
    case trips the hard floor."""
    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 0)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 1)

    small = {
        "mtr_df": pd.DataFrame(),
        "myntra_df": pd.DataFrame({"OMS_SKU": ["X"], "Date": pd.NaT}),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
    }
    assert main._disk_cache_corruption_reason(small) is None


def test_mtr_guard_still_fires_when_enabled(monkeypatch):
    """When the configurable mtr guard is enabled, a too-small mtr_df is corrupt
    even if other platforms carry rows (the historical race-condition case)."""
    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 500_000)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 1)

    cache = {
        "mtr_df": pd.DataFrame({"OMS_SKU": ["A1"] * 100}),
        "myntra_df": pd.DataFrame({"OMS_SKU": ["B"] * 100}),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
    }
    reason = main._disk_cache_corruption_reason(cache)
    assert reason is not None
    assert "mtr_df" in reason


def test_empty_dict_is_not_corrupt():
    """No disk cache at all is a distinct 'absent' case, not 'corrupt'."""
    assert main._disk_cache_corruption_reason({}) is None
    assert main._disk_cache_corruption_reason(None) is None


def test_hard_floor_can_be_disabled(monkeypatch):
    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 0)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 0)
    # With both guards off, even an all-empty cache passes (operator opt-out).
    assert main._disk_cache_corruption_reason(_empty_disk_cache()) is None


def test_partial_disk_cache_stale_vs_github_forces_rebuild(monkeypatch):
    from backend.services import github_cache as gh

    monkeypatch.setattr(main, "_DISK_CACHE_MIN_MTR_ROWS", 0)
    monkeypatch.setattr(main, "_DISK_CACHE_HARD_MIN_ROWS", 1)
    monkeypatch.setattr(
        gh,
        "github_manifest_row_counts",
        lambda: {"mtr_df": 1_191_815},
    )
    partial = {
        "mtr_df": pd.DataFrame({"OMS_SKU": ["A1"] * 190_555}),
        "myntra_df": pd.DataFrame({"OMS_SKU": ["B"] * 80_000}),
        "meesho_df": pd.DataFrame(),
        "flipkart_df": pd.DataFrame(),
        "snapdeal_df": pd.DataFrame(),
    }
    reason = main._disk_cache_corruption_reason(partial)
    assert reason is not None
    assert "GitHub" in reason

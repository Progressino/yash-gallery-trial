"""Restore-full must load complete Tier-3 SQLite history, not a 12-month window."""
import pandas as pd
import pytest

from backend.session import AppSession, wipe_app_session
from tests.test_daily_auto_merge_preserves_bulk import _tier3_line_df


def test_restore_full_mode_loads_unbounded_tier3(monkeypatch):
    from backend.routers import data as data_router

    calls: list[dict] = []

    def fake_load(platform, months=None, dedup=False, max_files=None):
        calls.append(
            {"platform": platform, "months": months, "max_files": max_files, "dedup": dedup}
        )
        if platform == "myntra":
            return _tier3_line_df("myntra", 120)
        return pd.DataFrame()

    monkeypatch.setattr("backend.services.daily_store.load_platform_data", fake_load)
    monkeypatch.setattr("backend.services.daily_store.get_summary", lambda: {})

    sess = AppSession()
    wipe_app_session(sess)
    sess.daily_restored = False

    data_router._restore_daily_if_needed(sess, force=True, restore_full_mode=True)

    myntra_calls = [c for c in calls if c["platform"] == "myntra"]
    assert myntra_calls
    assert myntra_calls[0]["months"] is None
    assert myntra_calls[0]["max_files"] is None
    assert len(sess.myntra_df) == 120


def test_tier3_topup_when_session_row_count_far_below_sqlite_total():
    from backend.routers.data import _tier3_session_needs_topup

    sess = AppSession()
    sess.myntra_df = _tier3_line_df("myntra", 84)

    summary = {
        "myntra": {
            "min_date": "2023-12-07",
            "max_date": "2026-06-15",
            "total_rows": 195549,
            "file_count": 113,
        }
    }

    import backend.services.daily_store as ds

    orig = ds.get_summary
    ds.get_summary = lambda: summary
    try:
        assert _tier3_session_needs_topup(sess) is True
    finally:
        ds.get_summary = orig

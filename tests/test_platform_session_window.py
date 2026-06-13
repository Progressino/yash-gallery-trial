"""Session platform date window (2-year default for sales-build copies only)."""
import datetime

import pandas as pd

from backend.services.platform_session_window import (
    AUTO_RESTORE_MONTHS_DEFAULT,
    SESSION_PLATFORM_MAX_DAYS,
    platform_frames_trimmed_for_sales_build,
    session_platform_shorter_than_tier3,
    trim_platform_df,
    trimmed_copy_for_sales_build,
    trim_session_platform_frames,
)


def test_default_window_is_two_years():
    assert SESSION_PLATFORM_MAX_DAYS == 730
    assert AUTO_RESTORE_MONTHS_DEFAULT == 12


def test_trim_platform_df_drops_old_rows():
    today = datetime.date.today()
    old = (today - datetime.timedelta(days=800)).isoformat()
    recent = (today - datetime.timedelta(days=30)).isoformat()
    df = pd.DataFrame({"Date": [old, recent], "Qty": [1, 2]})
    out = trim_platform_df(df, max_days=730)
    assert len(out) == 1
    assert out.iloc[0]["Qty"] == 2


def test_trimmed_copy_for_sales_build_does_not_mutate_session_frame():
    today = datetime.date.today()
    old = (today - datetime.timedelta(days=800)).isoformat()
    recent = (today - datetime.timedelta(days=30)).isoformat()
    original = pd.DataFrame({"Date": [old, recent], "Qty": [1, 2]})
    copy = trimmed_copy_for_sales_build(original, max_days=730)
    assert len(copy) == 1
    assert len(original) == 2


def test_platform_frames_trimmed_for_sales_build():
    class Sess:
        mtr_df = pd.DataFrame({"TxnDate": ["2020-01-01"], "x": [1]})
        myntra_df = pd.DataFrame()
        meesho_df = pd.DataFrame()
        flipkart_df = pd.DataFrame()
        snapdeal_df = pd.DataFrame()

    frames = platform_frames_trimmed_for_sales_build(Sess(), max_days=730)
    assert len(Sess.mtr_df) == 1
    assert frames["mtr_df"].empty


def test_trim_session_platform_frames():
    class Sess:
        mtr_df = pd.DataFrame()
        myntra_df = pd.DataFrame()
        meesho_df = pd.DataFrame()
        flipkart_df = pd.DataFrame()
        snapdeal_df = pd.DataFrame()

    today = datetime.date.today()
    old = (today - datetime.timedelta(days=900)).isoformat()
    recent = (today - datetime.timedelta(days=10)).isoformat()
    Sess.mtr_df = pd.DataFrame({"TxnDate": [old, recent], "x": [1, 2]})
    assert trim_session_platform_frames(Sess, max_days=730)
    assert len(Sess.mtr_df) == 1


def test_session_platform_shorter_than_tier3_detects_older_sqlite(monkeypatch):
    class Sess:
        myntra_df = pd.DataFrame(
            {"Date": pd.to_datetime(["2025-01-15", "2025-06-01"]), "Quantity": [1, 2]}
        )

    monkeypatch.setattr(
        "backend.services.platform_session_window.get_summary",
        lambda: {
            "myntra": {
                "min_date": "2024-06-13",
                "max_date": "2025-06-01",
                "file_count": 3,
                "total_rows": 100,
            }
        },
        raising=False,
    )
    # Patch via daily_store import inside the function
    import backend.services.daily_store as ds

    monkeypatch.setattr(
        ds,
        "get_summary",
        lambda: {
            "myntra": {
                "min_date": "2024-06-13",
                "max_date": "2025-06-01",
                "file_count": 3,
                "total_rows": 100,
            }
        },
    )
    assert session_platform_shorter_than_tier3(Sess()) is True

"""Session platform date window (2-year default)."""
import datetime

import pandas as pd

from backend.services.platform_session_window import (
    AUTO_RESTORE_MONTHS_DEFAULT,
    SESSION_PLATFORM_MAX_DAYS,
    trim_platform_df,
    trim_session_platform_frames,
)


def test_default_window_is_two_years():
    assert SESSION_PLATFORM_MAX_DAYS == 730
    assert AUTO_RESTORE_MONTHS_DEFAULT == 24


def test_trim_platform_df_drops_old_rows():
    today = datetime.date.today()
    old = (today - datetime.timedelta(days=800)).isoformat()
    recent = (today - datetime.timedelta(days=30)).isoformat()
    df = pd.DataFrame({"Date": [old, recent], "Qty": [1, 2]})
    out = trim_platform_df(df, max_days=730)
    assert len(out) == 1
    assert out.iloc[0]["Qty"] == 2


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

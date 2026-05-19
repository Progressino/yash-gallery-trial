"""Daily inventory history must be trimmed to recent days at upload time to prevent OOM."""

import pandas as pd
import pytest
from datetime import date, timedelta


def _make_history(n_skus: int, n_days: int) -> pd.DataFrame:
    """Build a synthetic tall inventory history DataFrame."""
    today = pd.Timestamp.now().normalize()
    dates = [today - pd.Timedelta(days=i) for i in range(n_days)]
    rows = []
    for sku_i in range(n_skus):
        sku = f"SKU-{sku_i:04d}"
        for d in dates:
            rows.append({"OMS_SKU": sku, "Date": d, "Qty": 10})
    return pd.DataFrame(rows)


def test_trim_keeps_only_recent_days():
    from backend.services.daily_inventory_upload_run import _trim_history_to_recent

    df = _make_history(n_skus=10, n_days=1000)
    assert len(df) == 10_000

    trimmed, note = _trim_history_to_recent(df, max_days=400)
    # Exactly 10 SKUs × 401 day-buckets (cutoff <= date <= max, inclusive)
    assert len(trimmed) <= 10 * 401
    assert len(trimmed) >= 10 * 399   # allow ±1 day for timestamp rounding
    assert "400" in note
    assert len(note) > 0


def test_trim_noop_when_already_short():
    from backend.services.daily_inventory_upload_run import _trim_history_to_recent

    df = _make_history(n_skus=5, n_days=30)
    trimmed, note = _trim_history_to_recent(df, max_days=400)
    assert len(trimmed) == len(df)
    assert note == ""


def test_trim_noop_when_disabled():
    from backend.services.daily_inventory_upload_run import _trim_history_to_recent

    df = _make_history(n_skus=5, n_days=1000)
    trimmed, note = _trim_history_to_recent(df, max_days=0)
    assert len(trimmed) == len(df)
    assert note == ""


def test_execute_applies_trim(monkeypatch):
    """execute_daily_inventory_upload must trim the stored dataframe to _MAX_HISTORY_DAYS."""
    import backend.services.daily_inventory_upload_run as mod
    import backend.services.daily_inventory_history as dih_mod
    from backend.session import AppSession

    monkeypatch.setattr(mod, "_MAX_HISTORY_DAYS", 60)

    # Build a fake 5 SKUs × 500 days history (analogous to 9500 SKU × 3000 day problem).
    big_df = _make_history(n_skus=5, n_days=500)

    def _fake_parse(fp, filename, sku_mapping=None):
        return big_df

    # The function does a local import from .daily_inventory_history — patch that module.
    monkeypatch.setattr(dih_mod, "parse_daily_inventory_history_upload", _fake_parse)

    sess = AppSession()
    sess.sku_mapping = {}
    result = mod.execute_daily_inventory_upload(sess, b"dummy", "test.xlsx")

    assert result["ok"] is True
    # After trimming to 60 days: 5 SKUs × ~61 days ≈ 305 rows (not 2500)
    assert result["rows"] <= 5 * 62
    assert result["rows"] >= 5 * 59
    assert sess.daily_inventory_history_df is not None
    assert len(sess.daily_inventory_history_df) == result["rows"]
    assert "Trimmed" in result["message"] or "trimmed" in result["message"].lower()

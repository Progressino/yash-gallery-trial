"""Daily-auto must merge parsed slices without reloading full Tier-3 SQLite per file."""

import pandas as pd
import pytest

from backend.session import AppSession


def test_daily_auto_merge_slice_does_not_call_load_platform_data(monkeypatch):
    """Regression: per-file touch must not reload all Tier-3 blobs from SQLite."""
    from backend.routers import upload as upload_router

    calls: list[str] = []

    def _boom_load(platform, **kwargs):
        calls.append(platform)
        raise AssertionError(f"load_platform_data must not run during ingest: {platform}")

    monkeypatch.setattr(
        "backend.services.daily_store.load_platform_data",
        _boom_load,
    )
    monkeypatch.setattr(
        "backend.routers.upload.save_daily_file",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "backend.routers.upload._detect_platform",
        lambda _fn, _raw: "myntra",
    )
    monkeypatch.setattr(
        "backend.routers.upload._rebuild_sales_sync",
        lambda _s: (True, "ok"),
    )

    df = pd.DataFrame(
        {
            "sub order id": ["1"],
            "sku id": ["S1"],
            "order_created_date": ["2024-06-01"],
            "product_mrp": [100],
        }
    )

    def _fake_parse(raw, fname, sku_mapping):
        return df, "OK"

    monkeypatch.setattr(
        "backend.services.myntra._parse_myntra_csv",
        _fake_parse,
    )
    monkeypatch.setattr(
        "backend.routers.upload.apply_dsr_segment_from_upload_filename",
        lambda d, *a: d,
    )

    sess = AppSession()
    sess.mtr_df = pd.DataFrame({"OrderId": [f"b{i}" for i in range(50)]})
    payload = upload_router._process_daily_auto_sync(
        sess,
        [("myntra.csv", b"x")],
        rebuild_sales=False,
    )
    assert payload["ok"] is True
    assert calls == []
    assert len(sess.myntra_df) >= 1

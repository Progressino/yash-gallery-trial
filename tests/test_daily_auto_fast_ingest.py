"""Fast daily-auto: SQLite during ingest, bounded rebuild — not 1M-row session merges."""

import time
from pathlib import Path

import pandas as pd
import pytest

from backend.session import AppSession


def test_fast_ingest_skips_session_merge(monkeypatch):
    from backend.routers import upload as upload_router

    monkeypatch.setenv("DAILY_AUTO_FAST_INGEST", "1")
    merges: list[str] = []

    def _spy_merge(sess, platform, df_slice, **kwargs):
        merges.append(platform)

    monkeypatch.setattr(upload_router, "_merge_slice_into_session", _spy_merge)
    monkeypatch.setattr(
        upload_router,
        "save_daily_file",
        lambda *a, **k: ("2026-05-23", 10, None),
    )
    monkeypatch.setattr(
        upload_router,
        "_detect_platform",
        lambda _fn, _raw: "meesho_csv",
    )
    monkeypatch.setattr(
        "backend.services.meesho.parse_meesho_csv",
        lambda _raw: (
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-05-23"]),
                    "OMS_SKU": ["A"],
                    "TxnType": ["Shipment"],
                    "Quantity": [1],
                    "OrderId": ["1"],
                    "LineKey": [""],
                }
            ),
            "OK",
        ),
    )
    monkeypatch.setattr(
        upload_router,
        "apply_dsr_segment_from_upload_filename",
        lambda d, *a: d,
    )

    sess = AppSession()
    sess.meesho_df = pd.DataFrame({"Date": [], "OMS_SKU": [], "TxnType": [], "Quantity": [], "OrderId": [], "LineKey": []})
    payload = upload_router._process_daily_auto_sync(
        sess,
        [("Orders_test.csv", b"x")],
        rebuild_sales=False,
    )
    assert payload["ok"] is True
    assert merges == []
    assert "meesho" in sess._daily_auto_platforms_touched


@pytest.mark.skipif(
    not Path("/Users/samraisinghani/Downloads/Sales 23-24-5-26.rar").is_file(),
    reason="Sales 23-24-5-26.rar not on disk",
)
def test_fast_ingest_sales_rar_completes_under_two_minutes(monkeypatch, tmp_path):
    """Operator bundle (~12 files) should not spend 15+ min merging into session RAM."""
    import os

    from backend.routers import upload as upload_router
    from backend.services import daily_store

    monkeypatch.setenv("DAILY_AUTO_FAST_INGEST", "1")
    monkeypatch.setattr(daily_store, "_DB_PATH", tmp_path / "daily_sales_fast.db")

    raw = Path("/Users/samraisinghani/Downloads/Sales 23-24-5-26.rar").read_bytes()
    from backend.routers.upload import _extract_rar_files

    file_parts = _extract_rar_files(raw)
    sess = AppSession()
    sess.sku_mapping = {}

    t0 = time.monotonic()
    payload = upload_router._process_daily_auto_sync(sess, file_parts, rebuild_sales=False)
    ingest_sec = time.monotonic() - t0

    assert payload["ok"] is True
    assert ingest_sec < 120, f"ingest took {ingest_sec:.0f}s"
    assert len(sess._daily_auto_platforms_touched) >= 4

    assert "amazon" in sess._daily_auto_parsed_buffers

    t1 = time.monotonic()
    ok, msg = upload_router._rebuild_sales_sync(
        sess,
        refresh_sqlite=False,
        platforms_touched=sess._daily_auto_platforms_touched,
    )
    rebuild_sec = time.monotonic() - t1

    assert ok is True
    assert "updated" in msg.lower() or "rebuilt" in msg.lower()
    assert rebuild_sec < 60, f"incremental rebuild took {rebuild_sec:.0f}s"
    assert len(sess.sales_df) > 3000, msg
    assert not sess._daily_auto_parsed_buffers

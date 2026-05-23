"""Daily upload file outcomes and async build-sales."""

from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.routers import upload as upload_router
from backend.services.daily_store import save_daily_file


def test_save_daily_file_returns_block_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("DAILY_DB_PATH", str(tmp_path / "daily.db"))
    df = pd.DataFrame({"Date": ["2099-01-15"], "OrderId": ["1"], "SKU": ["X"], "Quantity": [1]})
    _fd, rows, block = save_daily_file("amazon", "2026-05-22.csv", df)
    assert rows == 0
    assert block


def test_save_daily_file_tracked_records_skip():
    detected: list[str] = []
    warnings: list[str] = []
    file_results: list[dict] = []
    ok = upload_router._save_daily_file_tracked(
        "amazon",
        "empty.csv",
        pd.DataFrame(),
        detected=detected,
        warnings=warnings,
        file_results=file_results,
        detected_label="Amazon (empty.csv)",
    )
    assert ok is False
    assert not detected
    assert file_results[0]["status"] == "skipped"
    assert "No data extracted" in file_results[0]["reason"]


def test_build_sales_returns_pending(client, auth_token, monkeypatch):
    submitted: list[tuple] = []

    def _fake_submit(fn, *args, **kwargs):
        submitted.append((fn, args, kwargs))

    monkeypatch.setattr(upload_router.DAILY_UPLOAD_EXECUTOR, "submit", _fake_submit)

    r = client.post("/api/upload/build-sales")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body.get("sales_rebuild") == "pending"
    assert len(submitted) == 1
    assert submitted[0][0] is upload_router._run_sales_rebuild_worker


def test_store_ingest_result_includes_file_results():
    sess = MagicMock()
    upload_router._store_daily_auto_ingest_result(
        sess,
        {
            "ok": True,
            "message": "ok",
            "detected_platforms": ["Amazon (a.csv)"],
            "warnings": ["b.csv: unknown"],
            "processed_files": 2,
            "detected_files": 1,
            "unknown_files": 1,
            "expanded_files": 2,
            "saved_files": 1,
            "file_results": [
                {"filename": "a.csv", "status": "saved", "platform": "amazon", "rows": 10},
                {"filename": "b.csv", "status": "skipped", "reason": "unknown"},
            ],
        },
    )
    res = sess.daily_auto_ingest_result
    assert res["saved_files"] == 1
    assert len(res["file_results"]) == 2

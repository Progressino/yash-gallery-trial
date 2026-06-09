"""Returns RAR import returns immediately; net sales rebuild runs in background."""

import time
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd


def test_returns_import_accepts_rar_and_queues_followup(client, auth_token, monkeypatch):
    queued: list[str] = []

    def _fake_followup(session_id: str):
        queued.append(session_id)

    monkeypatch.setattr(
        "backend.routers.upload._run_returns_import_followup",
        _fake_followup,
    )

    overlay = pd.DataFrame({"OMS_SKU": ["SKU-A"], "Return_Units": [2]})
    monkeypatch.setattr(
        "backend.services.po_return_import.parse_return_upload_bytes",
        lambda *a, **k: (overlay, None),
    )
    monkeypatch.setattr(
        "backend.services.po_return_import.apply_return_overlay_import",
        lambda sess, df, **k: {
            "ok": True,
            "message": "Return sheet: 1 SKU(s), 2 return units.",
            "skus": 1,
            "total_units": 2,
        },
    )
    monkeypatch.setattr(
        "backend.routers.po._sync_po_sidecars_to_durable_storage",
        lambda *a, **k: None,
    )

    # DAILY_UPLOAD_EXECUTOR has max_workers=1. In the full test suite, previous
    # tests can leave queued work that saturates the executor for > 5s.  Patch
    # submit() to run the callable synchronously in this thread instead.
    import concurrent.futures
    from backend.concurrency import DAILY_UPLOAD_EXECUTOR

    class _SyncFuture:
        def result(self, timeout=None):
            return None

    def _sync_submit(fn, *args, **kwargs):
        fn(*args, **kwargs)
        return _SyncFuture()

    monkeypatch.setattr(DAILY_UPLOAD_EXECUTOR, "submit", _sync_submit)

    r = client.post(
        "/api/po/returns/import-file",
        files=[("file", ("Return Data.rar", BytesIO(b"Rar!" + b"\x1a\x07\x00" + b"x"), "application/x-rar"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body.get("sales_rebuild") == "pending"
    assert body.get("sales_rebuilt") is False
    assert "background" in (body.get("message") or "").lower()
    assert len(queued) == 1


def test_skip_meesho_lost_member():
    from backend.services.po_return_import import _skip_return_archive_member

    assert _skip_return_archive_member(
        "Return Data/completed_lost_____Meesho Lost.csv"
    )
    assert not _skip_return_archive_member(
        "Return Data/completed_delivered_Meesho Return.csv"
    )

"""Tier-3 daily-auto returns before sales rebuild completes."""

from io import BytesIO


def test_daily_auto_returns_pending_sales_rebuild(client, auth_token, monkeypatch):
    def _fake_process(sess, file_parts, *, rebuild_sales=True):
        assert rebuild_sales is False
        return {
            "ok": True,
            "message": "Loaded 1 file(s): Myntra.",
            "detected_platforms": ["Myntra (x.csv)"],
            "warnings": [],
            "processed_files": 1,
            "detected_files": 1,
            "unknown_files": 0,
            "sales_rebuild": "pending",
        }

    rebuild_called = []

    def _fake_rebuild(session_id: str):
        rebuild_called.append(session_id)

    monkeypatch.setattr(
        "backend.routers.upload._process_daily_auto_sync",
        _fake_process,
    )
    monkeypatch.setattr(
        "backend.routers.upload._run_daily_auto_sales_rebuild",
        _fake_rebuild,
    )

    r = client.post(
        "/api/upload/daily-auto",
        files=[("files", ("orders.csv", BytesIO(b"a,b\n1,2"), "text/csv"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["sales_rebuild"] == "pending"
    assert len(rebuild_called) == 1

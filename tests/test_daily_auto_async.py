"""Tier-3 daily-auto returns immediately; ingest and sales rebuild run in background."""

from io import BytesIO


def test_daily_auto_accepts_upload_and_queues_background_ingest(client, auth_token, monkeypatch):
    pipeline_called: list[tuple[str, list]] = []

    def _fake_pipeline(session_id: str, file_parts: list[tuple[str, bytes]]):
        pipeline_called.append((session_id, file_parts))

    monkeypatch.setattr(
        "backend.routers.upload._run_daily_auto_ingest_pipeline",
        _fake_pipeline,
    )

    r = client.post(
        "/api/upload/daily-auto",
        files=[("files", ("orders.csv", BytesIO(b"a,b\n1,2"), "text/csv"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body.get("ingest_async") is True
    assert body["sales_rebuild"] == "pending"
    assert len(pipeline_called) == 1
    assert pipeline_called[0][1][0][0] == "orders.csv"

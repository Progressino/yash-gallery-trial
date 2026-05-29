"""Async restore-full must return immediately and set session_restore_status=running."""


def test_restore_full_async_returns_immediately(client, session_for_client):
    _, sess = session_for_client
    r = client.post("/api/data/restore-full")
    assert r.status_code == 200
    body = r.json()
    assert body.get("restore_async") is True
    assert sess.session_restore_status in ("running", "done")

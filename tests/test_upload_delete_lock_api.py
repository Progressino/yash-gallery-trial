"""HTTP enforcement of owner-only upload delete."""

import pytest


@pytest.mark.parametrize(
    "method,path",
    [
        ("DELETE", "/api/upload/clear/mtr"),
        ("DELETE", "/api/data/daily-uploads/999999"),
        ("POST", "/api/cache/reset-all"),
    ],
)
def test_non_owner_cannot_delete_upload_data(client, monkeypatch, method, path):
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner_only")
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    if method == "DELETE":
        r = client.delete(path)
    else:
        r = client.post(path, json={})
    assert r.status_code == 403


def test_owner_username_may_call_clear_when_listed(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "tester")
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")

    def _decode(token: str | None):
        if token == "test-token":
            return {"sub": "tester", "role": "Clerk", "permissions": []}
        return None

    monkeypatch.setattr("backend.main.decode_token", _decode)
    r = client.delete("/api/upload/clear/mtr")
    assert r.status_code == 200
    assert r.json().get("ok") is True

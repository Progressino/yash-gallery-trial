"""Super Admin can bypass OTP when SUPER_ADMIN_OTP_BYPASS=1."""
import os
import tempfile

import bcrypt
import pytest


@pytest.fixture
def bypass_env(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("USERS_DB_PATH", path)
    monkeypatch.setenv("OTP_REQUIRED", "1")
    monkeypatch.setenv("SUPER_ADMIN_OTP_BYPASS", "1")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("AUTH_USERNAME", "admin")
    pw_hash = bcrypt.hashpw(b"secret123", bcrypt.gensalt()).decode()
    monkeypatch.setenv("AUTH_PASSWORD_HASH", pw_hash)
    monkeypatch.setenv("SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("SUPER_ADMIN_PHONE", "9876543210")

    from backend.db import users_db

    users_db.init_db()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def test_super_admin_login_skips_otp_with_bypass(bypass_env):
    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "secret123"},
        headers={"X-Device-Id": "brand-new-device-never-seen"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("role") == "Super Admin"
    assert not data.get("otp_required")
    assert client.cookies.get("auth_token")

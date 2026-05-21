"""OTP login and trusted device tests."""
import os
import tempfile

import bcrypt
import pytest


@pytest.fixture
def otp_env(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("USERS_DB_PATH", path)
    monkeypatch.setenv("OTP_REQUIRED", "1")
    monkeypatch.setenv("SMS_OTP_DEV", "1")
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


def test_login_requires_otp_on_new_device(otp_env):
    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("otp_required") is True
    assert data.get("challenge_id")
    assert "auth_token" not in client.cookies


def test_otp_verify_issues_token(otp_env, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
    challenge_id = r.json()["challenge_id"]
    otp_code = None
    for rec in caplog.records:
        if "SMS_OTP_DEV" in rec.message and "is " in rec.message:
            otp_code = rec.message.rsplit(" is ", 1)[-1].strip()
            break
    assert otp_code and len(otp_code) == 6

    v = client.post(
        "/api/auth/otp/verify",
        json={"challenge_id": challenge_id, "code": otp_code, "trust_device": True},
        headers={"X-Device-Id": "test-device-1"},
    )
    assert v.status_code == 200
    assert v.json().get("ok") is True
    assert v.json().get("role") == "Super Admin"
    assert client.cookies.get("auth_token")
    assert client.cookies.get("device_trust")


def test_trusted_device_skips_otp(otp_env, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    from starlette.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    headers = {"X-Device-Id": "trusted-browser-xyz"}
    r1 = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "secret123"},
        headers=headers,
    )
    challenge_id = r1.json()["challenge_id"]
    otp_code = None
    for rec in caplog.records:
        if "SMS_OTP_DEV" in rec.message and " is " in rec.message:
            otp_code = rec.message.rsplit(" is ", 1)[-1].strip()
    client.post(
        "/api/auth/otp/verify",
        json={"challenge_id": challenge_id, "code": otp_code, "trust_device": True},
        headers=headers,
    )
    # Logout clears session but keeps device_trust — same browser should skip OTP.
    client.post("/api/auth/logout", headers=headers)
    r2 = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "secret123"},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
    assert r2.json().get("role") == "Super Admin"
    assert not r2.json().get("otp_required")


def test_normalize_india_phone():
    from backend.services.login_otp import normalize_india_phone

    assert normalize_india_phone("9876543210") == "919876543210"
    assert normalize_india_phone("+91 98765 43210") == "919876543210"
    assert normalize_india_phone("09876543210") == "919876543210"
    assert normalize_india_phone("invalid") is None

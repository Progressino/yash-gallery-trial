"""Upload policy: lock historical data for non-admin roles."""
import os

import pytest

from backend.services.upload_policy import (
    check_upload_api_access,
    may_upload_historical,
    upload_policy_for_role,
)


def test_clerk_blocked_from_bulk_upload_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_upload_historical("Clerk") is False
    err = check_upload_api_access("Clerk", "POST", "/api/upload/mtr")
    assert err is not None
    assert "locked" in err.lower() or "historical" in err.lower()


def test_clerk_may_daily_auto_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access("Clerk", "POST", "/api/upload/daily-auto") is None
    assert check_upload_api_access("Clerk", "POST", "/api/po/daily-inventory-history") is None


def test_admin_may_bulk_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_upload_historical("Admin") is True
    assert check_upload_api_access("Admin", "POST", "/api/upload/mtr") is None


def test_clerk_blocked_clear_and_reset(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access("Executive", "DELETE", "/api/upload/clear/mtr") is not None
    assert check_upload_api_access("Viewer", "POST", "/api/cache/reset-all") is not None


def test_upload_policy_payload():
    pol = upload_policy_for_role("Clerk")
    assert pol["may_upload_daily"] is True
    assert pol["historical_upload_locked"] is True

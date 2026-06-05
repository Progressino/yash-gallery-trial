"""Upload policy: lock historical data for non-admin roles; owner-only delete."""
import os

import pytest

from backend.services.upload_policy import (
    _delete_allowed_usernames,
    check_upload_api_access,
    may_delete_upload_data,
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
    assert check_upload_api_access("Clerk", "POST", "/api/upload/inventory-auto") is None
    assert check_upload_api_access("Clerk", "POST", "/api/po/returns/import-file") is None
    err = check_upload_api_access("Clerk", "POST", "/api/po/daily-inventory-history")
    assert err is not None


def test_admin_may_bulk_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_upload_historical("Admin") is True
    assert check_upload_api_access("Admin", "POST", "/api/upload/mtr") is None


def test_non_owner_blocked_from_all_delete_paths(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner_only")
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/upload/clear/mtr", username="admin"
    ) is not None
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/data/daily-uploads/99", username="admin"
    ) is not None
    assert check_upload_api_access(
        "Super Admin", "POST", "/api/cache/reset-all", username="other"
    ) is not None
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/po/returns/overlay", username="clerk1"
    ) is not None


def test_owner_may_delete_when_listed(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner_only,sam")
    assert may_delete_upload_data("sam")
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/data/daily-uploads/1", username="sam"
    ) is None
    assert check_upload_api_access(
        "Manager", "DELETE", "/api/upload/clear/mtr", username="owner_only"
    ) is None


def test_manager_cannot_delete_even_when_historical_upload_allowed(monkeypatch):
    """Regression: may_upload_historical must not bypass owner-only delete."""
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner")
    assert may_upload_historical("Manager") is True
    assert check_upload_api_access(
        "Manager", "DELETE", "/api/upload/clear/mtr", username="manager"
    ) is not None


def test_upload_policy_payload_denies_delete_for_non_owner(monkeypatch):
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner")
    pol = upload_policy_for_role("Admin", "not_owner")
    assert pol["may_upload_daily"] is True
    assert pol["may_delete_daily_upload"] is False
    assert pol["may_reset_all"] is False
    assert pol["may_clear_platform"] is False
    assert pol["upload_delete_locked"] is True
    pol_owner = upload_policy_for_role("Clerk", "owner")
    assert pol_owner["may_delete_daily_upload"] is True


def test_default_delete_allowlist_uses_super_admin_env(monkeypatch):
    monkeypatch.delenv("UPLOAD_DELETE_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SUPER_ADMIN_USERNAME", "yash_admin")
    monkeypatch.setenv("AUTH_USERNAME", "legacy")
    assert _delete_allowed_usernames() == frozenset({"yash_admin", "legacy"})
    assert may_delete_upload_data("YASH_ADMIN")


def test_clerk_blocked_clear_and_reset(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner")
    assert check_upload_api_access("Executive", "DELETE", "/api/upload/clear/mtr") is not None
    assert check_upload_api_access("Viewer", "POST", "/api/cache/reset-all") is not None


def test_upload_policy_payload_clerk():
    pol = upload_policy_for_role("Clerk", "clerk_user")
    assert pol["may_upload_daily"] is True
    assert pol["historical_upload_locked"] is True
    assert pol["may_upload_po_baseline"] is False


def test_manager_blocked_sku_mapping_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access("Manager", "POST", "/api/upload/mtr") is None
    assert check_upload_api_access("Manager", "POST", "/api/upload/sku-mapping") is not None


def test_admin_may_sku_mapping_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access("Admin", "POST", "/api/upload/sku-mapping") is None


def test_owner_may_clear_returns_overlay(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    monkeypatch.setenv("UPLOAD_DELETE_ALLOWED_USERS", "owner")
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/po/returns/overlay", username="owner"
    ) is None
    assert check_upload_api_access(
        "Manager", "DELETE", "/api/po/returns/overlay", username="mgr"
    ) is not None

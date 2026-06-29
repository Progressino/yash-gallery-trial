"""Upload policy: lock historical data for non-admin roles; Super Admin-only delete."""
import pytest

from backend.services.upload_policy import (
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
    assert check_upload_api_access("Clerk", "POST", "/api/upload/chunk/init") is None
    assert check_upload_api_access("Clerk", "POST", "/api/upload/chunk/complete") is None
    assert check_upload_api_access("Clerk", "POST", "/api/po/returns/import-file") is None
    err = check_upload_api_access("Clerk", "POST", "/api/po/daily-inventory-history")
    assert err is not None


def test_admin_may_bulk_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_upload_historical("Admin") is True
    assert check_upload_api_access("Admin", "POST", "/api/upload/mtr") is None


def test_non_super_admin_blocked_from_all_delete_paths(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/upload/clear/mtr", username="admin"
    ) is not None
    assert check_upload_api_access(
        "Admin", "DELETE", "/api/data/daily-uploads/99", username="admin"
    ) is not None
    assert check_upload_api_access(
        "Manager", "POST", "/api/cache/reset-all", username="mgr"
    ) is not None
    assert check_upload_api_access(
        "Executive", "DELETE", "/api/po/returns/overlay", username="exec"
    ) is not None


def test_super_admin_may_delete(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_delete_upload_data("Super Admin", "admin")
    assert check_upload_api_access(
        "Super Admin", "DELETE", "/api/data/daily-uploads/1", username="admin"
    ) is None
    assert check_upload_api_access(
        "Super Admin", "DELETE", "/api/upload/clear/mtr", username="admin"
    ) is None
    assert check_upload_api_access(
        "Super Admin", "POST", "/api/cache/reset-all", username="admin"
    ) is None


def test_manager_cannot_delete_even_when_historical_upload_allowed(monkeypatch):
    """Regression: may_upload_historical must not grant delete."""
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert may_upload_historical("Manager") is True
    assert may_delete_upload_data("Manager", "manager") is False
    assert check_upload_api_access(
        "Manager", "DELETE", "/api/upload/clear/mtr", username="manager"
    ) is not None


def test_upload_policy_payload_denies_delete_for_non_super_admin(monkeypatch):
    pol = upload_policy_for_role("Admin", "admin")
    assert pol["may_upload_daily"] is True
    assert pol["may_delete_daily_upload"] is False
    assert pol["may_reset_all"] is False
    assert pol["may_clear_platform"] is False
    assert pol["upload_delete_locked"] is True
    pol_sa = upload_policy_for_role("Super Admin", "admin")
    assert pol_sa["may_delete_daily_upload"] is True
    assert pol_sa["may_reset_all"] is True
    assert pol_sa["may_clear_platform"] is True
    assert pol_sa["upload_delete_locked"] is False


def test_clerk_blocked_clear_and_reset(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
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


def test_super_admin_may_clear_returns_overlay(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    assert check_upload_api_access(
        "Super Admin", "DELETE", "/api/po/returns/overlay", username="admin"
    ) is None
    assert check_upload_api_access(
        "Manager", "DELETE", "/api/po/returns/overlay", username="mgr"
    ) is not None


def test_irfan_may_upload_existing_po_when_locked(monkeypatch):
    monkeypatch.setenv("UPLOAD_HISTORICAL_LOCKED", "1")
    pol = upload_policy_for_role("Manager", "irfan")
    assert pol["may_upload_po_baseline"] is True
    assert check_upload_api_access(
        "Manager", "POST", "/api/upload/existing-po", username="irfan"
    ) is None
    assert check_upload_api_access(
        "Manager", "POST", "/api/upload/sku-mapping", username="irfan"
    ) is None
    assert check_upload_api_access(
        "Manager", "POST", "/api/po/sku-status-lead", username="irfan"
    ) is None
    assert check_upload_api_access(
        "Manager", "POST", "/api/upload/existing-po", username="other_mgr"
    ) is not None

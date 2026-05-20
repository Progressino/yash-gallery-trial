"""Who may change historical vs daily-only uploads (org-wide historical lock)."""
from __future__ import annotations

import os

# Roles allowed to replace bulk history, clear platforms, or reset shared cache.
_HISTORICAL_UPLOAD_ROLES = frozenset({"Admin", "Manager"})
_RESET_DATA_ROLES = frozenset({"Admin"})

# POST paths any authenticated non-karigar user may call when historical data is locked.
_DAILY_UPLOAD_PREFIXES = (
    "/api/upload/daily-auto",
    "/api/upload/daily",
    "/api/upload/existing-po",
    "/api/upload/build-sales",
    "/api/po/daily-inventory-history",
)

# Always allowed (read shared cache into session; does not overwrite org history).
_CACHE_LOAD_PREFIXES = (
    "/api/cache/load",
    "/api/cache/status",
)

# Mutating cache/history — Admin only when locked.
_RESTRICTED_CACHE_PREFIXES = (
    "/api/cache/save",
    "/api/cache/reload-fresh",
    "/api/cache/reset-all",
)

_RESTRICTED_CACHE_EXACT = frozenset({"/api/cache"})

# Bulk Tier-1 / Tier-2 upload POST paths (blocked when locked for non-admin roles).
_HISTORICAL_UPLOAD_PREFIXES = (
    "/api/upload/sku-mapping",
    "/api/upload/mtr",
    "/api/upload/myntra",
    "/api/upload/meesho",
    "/api/upload/flipkart",
    "/api/upload/snapdeal",
    "/api/upload/inventory-auto",
    "/api/upload/inventory",
    "/api/upload/amazon-b2c",
    "/api/upload/amazon-b2b",
    "/api/upload/existing-po",
    "/api/upload/cogs",
)


def historical_upload_locked() -> bool:
    return os.environ.get("UPLOAD_HISTORICAL_LOCKED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def may_upload_historical(role: str) -> bool:
    if not historical_upload_locked():
        return True
    return role in _HISTORICAL_UPLOAD_ROLES


def may_reset_shared_data(role: str) -> bool:
    if not historical_upload_locked():
        return role in _HISTORICAL_UPLOAD_ROLES
    return role in _RESET_DATA_ROLES


def may_clear_platform_data(role: str) -> bool:
    return may_upload_historical(role)


def may_delete_daily_upload_file(role: str) -> bool:
    """Deleting a saved Tier-3 file removes history — treat like reset."""
    return may_reset_shared_data(role)


def upload_policy_for_role(role: str) -> dict:
    hist = may_upload_historical(role)
    locked = historical_upload_locked()
    return {
        "historical_upload_locked": locked,
        "may_upload_historical": hist,
        "may_upload_daily": True,
        "may_clear_platform": may_clear_platform_data(role),
        "may_reset_all": may_reset_shared_data(role),
        "may_save_shared_cache": hist,
        "may_reload_shared_cache": hist,
        "may_delete_daily_upload": may_delete_daily_upload_file(role),
    }


def check_upload_api_access(role: str, method: str, path: str) -> str | None:
    """
    Return an error message if this role may not call ``path`` with ``method``.
    None means allowed.
    """
    if not historical_upload_locked():
        return None
    if may_upload_historical(role):
        return None

    m = method.upper()
    p = path.rstrip("/") or path

    if m == "GET" or m == "HEAD" or m == "OPTIONS":
        return None

    if any(p.startswith(prefix) for prefix in _DAILY_UPLOAD_PREFIXES):
        return None

    if m == "POST" and any(p.startswith(prefix) for prefix in _CACHE_LOAD_PREFIXES):
        return None

    if p.startswith("/api/upload/clear/") and m == "DELETE":
        return "Historical data is locked. Platform clear is not available for your role."

    if p.startswith("/api/data/daily-uploads/") and m == "DELETE":
        return "Historical data is locked. Removing saved daily files is not available for your role."

    if m == "DELETE" and p in _RESTRICTED_CACHE_EXACT:
        return "Historical data is locked. Session wipe is not available for your role."

    if m == "POST" and (
        p in _RESTRICTED_CACHE_EXACT
        or any(p.startswith(prefix) for prefix in _RESTRICTED_CACHE_PREFIXES)
    ):
        return "Historical data is locked. Shared cache changes require Admin."

    if m == "POST" and any(p.startswith(prefix) for prefix in _HISTORICAL_UPLOAD_PREFIXES):
        return (
            "Historical bulk uploads are locked. Use Tier 3 daily file drop "
            "(or daily inventory on PO Engine) — contact Admin to replace base history."
        )

    return None

"""Who may change historical vs daily-only uploads (org-wide historical lock)."""
from __future__ import annotations

import os

# Roles allowed to replace bulk history (upload POST — not delete).
_HISTORICAL_UPLOAD_ROLES = frozenset({"Super Admin", "Admin", "Manager"})
_RESET_DATA_ROLES = frozenset({"Super Admin", "Admin"})

# POST paths any authenticated non-karigar user may call when historical data is locked.
_DAILY_UPLOAD_PREFIXES = (
    "/api/upload/daily-auto",
    "/api/upload/daily",
    "/api/upload/build-sales",
    "/api/upload/inventory-auto",
    "/api/po/returns/import-file",
)

# Always allowed (read shared cache into session; does not overwrite org history).
_CACHE_LOAD_PREFIXES = (
    "/api/cache/load",
    "/api/cache/hydrate-warm",
    "/api/cache/status",
)

# Mutating cache/history — Admin only when locked (except delete — owner username only).
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
    "/api/upload/inventory",
    "/api/upload/amazon-b2c",
    "/api/upload/amazon-b2b",
    "/api/upload/existing-po",
    "/api/upload/cogs",
)

# PO baseline sheets (wide daily inventory history, SKU status/lead) — Admin only when locked.
_PO_ADMIN_BASELINE_PREFIXES = (
    "/api/po/daily-inventory-history",
    "/api/po/sku-status-lead",
    "/api/po/manual-intransit-sheet",
)

_DELETE_DENIED_MSG = (
    "Uploaded data is locked. Only the designated owner account may delete "
    "saved uploads, clear platforms, or reset shared cache."
)


def _delete_allowed_usernames() -> frozenset[str]:
    """Usernames allowed to delete uploaded / shared ERP sales data (case-insensitive)."""
    raw = (os.environ.get("UPLOAD_DELETE_ALLOWED_USERS") or "").strip()
    if raw:
        return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
    defaults: list[str] = []
    for key in ("SUPER_ADMIN_USERNAME", "AUTH_USERNAME"):
        v = (os.environ.get(key) or "").strip()
        if v:
            defaults.append(v.lower())
    if not defaults:
        defaults.append("admin")
    return frozenset(defaults)


def may_delete_upload_data(username: str | None) -> bool:
    """True only for usernames listed in UPLOAD_DELETE_ALLOWED_USERS (or super-admin env default)."""
    u = (username or "").strip().lower()
    return bool(u) and u in _delete_allowed_usernames()


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


def may_reset_shared_data(role: str, username: str | None = None) -> bool:
    """Session wipe, reset-all, Tier-3 purge — owner username only."""
    _ = role
    return may_delete_upload_data(username)


def may_clear_platform_data(role: str, username: str | None = None) -> bool:
    _ = role
    return may_delete_upload_data(username)


def may_delete_daily_upload_file(role: str, username: str | None = None) -> bool:
    """Deleting a saved Tier-3 file removes history — owner only."""
    _ = role
    return may_delete_upload_data(username)


def may_admin_po_session_edits(role: str) -> bool:
    """PO raise-ledger edits — Admin/Super Admin (not upload delete)."""
    return role in _RESET_DATA_ROLES


def may_upload_po_baseline(role: str) -> bool:
    """Wide PO inventory history + SKU status/lead — Admin only when org lock is on."""
    if not historical_upload_locked():
        return role in _HISTORICAL_UPLOAD_ROLES or role in _RESET_DATA_ROLES
    return role in _RESET_DATA_ROLES


def upload_policy_for_role(role: str, username: str | None = None) -> dict:
    locked = historical_upload_locked()
    can_delete = may_delete_upload_data(username)
    hist = may_upload_historical(role)
    return {
        "historical_upload_locked": locked,
        "may_upload_historical": hist,
        "may_upload_daily": True,
        "may_clear_platform": can_delete,
        "may_reset_all": can_delete,
        "may_save_shared_cache": hist,
        "may_reload_shared_cache": hist,
        "may_delete_daily_upload": can_delete,
        "may_upload_po_baseline": may_upload_po_baseline(role),
        "upload_delete_locked": not can_delete,
    }


def _path_deletes_upload_data(method: str, path: str) -> bool:
    m = method.upper()
    p = path.rstrip("/") or path
    if m == "DELETE" and p.startswith("/api/upload/clear/"):
        return True
    if m == "DELETE" and p.startswith("/api/data/daily-uploads/"):
        return True
    if m == "DELETE" and p == "/api/cache":
        return True
    if m == "POST" and p == "/api/cache/reset-all":
        return True
    if m == "DELETE" and p == "/api/po/returns/overlay":
        return True
    if m == "DELETE" and p.startswith("/api/po/daily-inventory-history"):
        return True
    return False


def check_upload_api_access(
    role: str,
    method: str,
    path: str,
    *,
    username: str | None = None,
) -> str | None:
    """
    Return an error message if this role may not call ``path`` with ``method``.
    None means allowed.
    """
    m = method.upper()
    p = path.rstrip("/") or path

    if _path_deletes_upload_data(m, p):
        if not may_delete_upload_data(username):
            return _DELETE_DENIED_MSG
        return None

    if not historical_upload_locked():
        return None

    if m == "GET" or m == "HEAD" or m == "OPTIONS":
        return None

    if any(p.startswith(prefix) for prefix in _DAILY_UPLOAD_PREFIXES):
        return None

    if m == "POST" and any(p.startswith(prefix) for prefix in _CACHE_LOAD_PREFIXES):
        return None

    # Admin-only while lock is on: PO baselines, SKU master map, existing PO sheet.
    if role not in _RESET_DATA_ROLES:
        if m == "POST" and any(p.startswith(prefix) for prefix in _PO_ADMIN_BASELINE_PREFIXES):
            return (
                "PO baseline uploads are Admin-only while historical data is locked. "
                "Use the Upload → Daily uploads tab for daily sales, snapshot inventory, and returns."
            )
        if m == "DELETE" and p.startswith("/api/po/daily-inventory-history"):
            return _DELETE_DENIED_MSG
        if m == "POST" and p.startswith("/api/upload/sku-mapping"):
            return "SKU mapping is Admin-only while historical data is locked."
        if m == "POST" and p.startswith("/api/upload/existing-po"):
            return "Existing PO baseline upload is Admin-only while historical data is locked."
        if p == "/api/po/returns/overlay" and m == "DELETE":
            return _DELETE_DENIED_MSG

    if may_upload_historical(role):
        return None

    if m == "POST" and (
        p in _RESTRICTED_CACHE_EXACT
        or any(p.startswith(prefix) for prefix in _RESTRICTED_CACHE_PREFIXES)
    ):
        return "Historical data is locked. Shared cache changes require Admin."

    if m == "POST" and any(p.startswith(prefix) for prefix in _HISTORICAL_UPLOAD_PREFIXES):
        return (
            "Historical bulk uploads are locked. Use the Daily uploads tab for daily sales, "
            "snapshot inventory, and returns — contact Admin to replace base history."
        )

    return None

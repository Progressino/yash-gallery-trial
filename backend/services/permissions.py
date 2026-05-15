"""Role-based API access for ERP users."""
from __future__ import annotations

# Roles with full ERP + analytics access (existing env admin maps to Admin).
FULL_ACCESS_ROLES = frozenset({"Admin", "Manager", "Executive", "Clerk", "Viewer"})

KARIGAR_ROLE = "Karigar"

# Stitching endpoints a karigar may call (production entry only).
_KARIGAR_STITCHING_PREFIXES = (
    "/api/stitching/status",
    "/api/stitching/sheets/karigar_master",
    "/api/stitching/sheets/style_master",
    "/api/stitching/sheets/challan_master",
    "/api/stitching/production-entry",
)

# Admin-only ERP user management.
_ERP_ADMIN_ROLES = frozenset({"Admin", "Manager"})


def permissions_for_role(role_name: str) -> list[str]:
    if role_name == KARIGAR_ROLE:
        return ["stitching.production_entry"]
    return ["*"]


def karigar_may_access_api(path: str, method: str) -> bool:
    """Return True if a Karigar role may call this API path."""
    if path.startswith("/api/auth/"):
        return True
    if path == "/api/health":
        return True
    if not path.startswith("/api/stitching/"):
        return False
    return any(path.startswith(prefix) for prefix in _KARIGAR_STITCHING_PREFIXES)


def may_access_erp_admin(role_name: str) -> bool:
    return role_name in _ERP_ADMIN_ROLES

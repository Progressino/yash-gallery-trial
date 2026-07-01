"""Role-based API access for ERP users."""
from __future__ import annotations

# Roles with full ERP + analytics access (existing env admin maps to Admin).
FULL_ACCESS_ROLES = frozenset({"Super Admin", "Admin", "Manager", "Executive", "Clerk", "Viewer"})

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
_ERP_ADMIN_ROLES = frozenset({"Super Admin", "Admin", "Manager", "Sir"})


def permissions_for_role(role_name: str) -> list[str]:
    if role_name == KARIGAR_ROLE:
        return ["stitching.production_entry"]
    from .upload_policy import upload_policy_for_role

    pol = upload_policy_for_role(role_name)
    perms = ["upload.daily", "data.read"]
    if pol.get("may_upload_historical"):
        perms.append("upload.historical")
    if pol.get("may_reset_all"):
        perms.append("data.reset")
    if pol.get("may_clear_platform"):
        perms.append("upload.clear")
    # Backward compat: Admin/Manager retain wildcard for ERP modules not yet permission-gated.
    if role_name in _ERP_ADMIN_ROLES:
        perms.append("*")
    elif role_name in FULL_ACCESS_ROLES:
        perms.extend(["*"])
    return perms


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


_DEPARTMENT_ADMIN_USERS = frozenset({"harsh"})


def may_manage_erp_departments(role_name: str, username: str | None = None) -> bool:
    """Add ERP user departments (Admin → Users dropdown). Managers/Admins + named ops users."""
    if may_access_erp_admin(role_name):
        return True
    return (username or "").strip().lower() in _DEPARTMENT_ADMIN_USERS


def may_delete_stitching_attendance(role_name: str | None, username: str | None = None) -> bool:
    """Managers and Himanshu may delete karigar attendance rows."""
    if (role_name or "").strip() == "Manager":
        return True
    if str(username or "").strip().lower() == "himanshu":
        return True
    return False

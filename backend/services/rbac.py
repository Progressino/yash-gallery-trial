"""Role-based access control — ERP module visibility and HRM data scope."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Target HRM RBAC roles (also seeded in users_db).
ROLE_SUPER_ADMIN = "Super Admin"
ROLE_ADMIN = "Admin"
ROLE_SIR = "Sir"
ROLE_HOD = "HOD"
ROLE_EMPLOYEE = "Employee"
ROLE_KARIGAR = "Karigar"

# Legacy roles that retain full ERP access.
LEGACY_FULL_ERP_ROLES = frozenset({"Manager", "Executive", "Clerk", "Viewer"})

FULL_ERP_ROLES = frozenset({ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_SIR}) | LEGACY_FULL_ERP_ROLES
HRM_ONLY_ROLES = frozenset({ROLE_HOD, ROLE_EMPLOYEE})

# Sidebar / route module keys (must match frontend).
ALL_MODULES = (
    "intelligence",
    "upload",
    "amazon",
    "myntra",
    "meesho",
    "flipkart",
    "snapdeal",
    "forecast",
    "finance",
    "sku_deepdive",
    "sales",
    "items",
    "purchase",
    "tna",
    "production",
    "stitching",
    "grey",
    "hrm",
    "inventory",
    "po",
    "admin",
    "marketplace",
)

HRM_ONLY_MODULES = ("hrm",)

# Route prefix → module key (first match wins).
ROUTE_MODULE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/", "intelligence"),
    ("/upload", "upload"),
    ("/mtr", "amazon"),
    ("/myntra", "myntra"),
    ("/meesho", "meesho"),
    ("/flipkart", "flipkart"),
    ("/snapdeal", "snapdeal"),
    ("/forecast", "forecast"),
    ("/finance", "finance"),
    ("/sku-deepdive", "sku_deepdive"),
    ("/sales", "sales"),
    ("/items", "items"),
    ("/purchase", "purchase"),
    ("/tna", "tna"),
    ("/production-entry", "stitching"),
    ("/production", "production"),
    ("/stitching-costing", "stitching"),
    ("/grey", "grey"),
    ("/hrm", "hrm"),
    ("/inventory", "inventory"),
    ("/po", "po"),
    ("/admin", "admin"),
    ("/marketplace-connections", "marketplace"),
)


@dataclass(frozen=True)
class HrmScope:
    """Resolved HRM visibility for the current user."""

    level: str  # all | department | self
    role: str
    user_id: int | None = None
    employee_id: int | None = None
    department_id: int | None = None
    reporting_hod_user_id: int | None = None

    @property
    def can_manage_org(self) -> bool:
        return self.role in (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_SIR) or self.role in LEGACY_FULL_ERP_ROLES

    @property
    def is_hod(self) -> bool:
        return self.role == ROLE_HOD

    @property
    def is_employee(self) -> bool:
        return self.role == ROLE_EMPLOYEE


def _parse_module_access(raw: str | None) -> list[str] | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in s.split(",") if p.strip()]


def default_modules_for_role(role_name: str) -> list[str]:
    if role_name == ROLE_KARIGAR:
        return ["stitching"]
    if role_name in FULL_ERP_ROLES:
        return list(ALL_MODULES)
    if role_name in HRM_ONLY_ROLES:
        return list(HRM_ONLY_MODULES)
    return list(ALL_MODULES)


def resolve_module_access(role_name: str, module_access_raw: str | None = None) -> list[str]:
    override = _parse_module_access(module_access_raw)
    if override:
        valid = [m for m in override if m in ALL_MODULES]
        return valid if valid else default_modules_for_role(role_name)
    return default_modules_for_role(role_name)


def may_access_module(role_name: str, module_key: str, module_access_raw: str | None = None) -> bool:
    return module_key in resolve_module_access(role_name, module_access_raw)


def route_to_module(path: str) -> str | None:
    if path == "/":
        return "intelligence"
    for prefix, mod in ROUTE_MODULE_PREFIXES:
        if prefix != "/" and path.startswith(prefix):
            return mod
    return None


def build_hrm_scope(profile: dict[str, Any] | None, *, role: str | None = None) -> HrmScope:
    """Build HRM data scope from erp_users profile row."""
    role_name = (profile or {}).get("role_name") or role or ROLE_ADMIN
    emp_id = profile.get("employee_id") if profile else None
    dept_id = profile.get("hrm_department_id") if profile else None
    user_id = profile.get("id") if profile else None
    hod_uid = profile.get("reporting_hod_user_id") if profile else None

    try:
        emp_id = int(emp_id) if emp_id is not None else None
    except (TypeError, ValueError):
        emp_id = None
    try:
        dept_id = int(dept_id) if dept_id is not None else None
    except (TypeError, ValueError):
        dept_id = None
    try:
        hod_uid = int(hod_uid) if hod_uid is not None else None
    except (TypeError, ValueError):
        hod_uid = None

    if role_name in (ROLE_SUPER_ADMIN, ROLE_ADMIN, ROLE_SIR) or role_name in LEGACY_FULL_ERP_ROLES:
        return HrmScope(level="all", role=role_name, user_id=user_id, employee_id=emp_id, department_id=dept_id, reporting_hod_user_id=hod_uid)
    if role_name == ROLE_HOD:
        return HrmScope(level="department", role=role_name, user_id=user_id, employee_id=emp_id, department_id=dept_id, reporting_hod_user_id=hod_uid)
    if role_name == ROLE_EMPLOYEE:
        return HrmScope(level="self", role=role_name, user_id=user_id, employee_id=emp_id, department_id=dept_id, reporting_hod_user_id=hod_uid)
    # Unknown roles: treat as full access (backward compatible).
    return HrmScope(level="all", role=role_name, user_id=user_id, employee_id=emp_id, department_id=dept_id, reporting_hod_user_id=hod_uid)


def hrm_scope_filters(
    scope: HrmScope,
    *,
    department_id: int | None = None,
    employee_id: int | None = None,
) -> tuple[int | None, int | None]:
    """Apply scope to client-requested filters; returns enforced (department_id, employee_id)."""
    if scope.level == "all":
        return department_id, employee_id
    if scope.level == "department":
        dept = scope.department_id
        if dept is None:
            return -1, employee_id  # no dept configured → empty results
        if department_id is not None and int(department_id) != int(dept):
            return -1, None
        if employee_id is not None:
            return dept, employee_id
        return dept, None
    # self
    if scope.employee_id is None:
        return scope.department_id, -1
    return scope.department_id, scope.employee_id


def assert_department_in_scope(scope: HrmScope, department_id: int) -> None:
    from fastapi import HTTPException

    if scope.level == "all":
        return
    if scope.level == "department":
        if scope.department_id is None or int(department_id) != int(scope.department_id):
            raise HTTPException(403, "Access denied for this department")
        return
    raise HTTPException(403, "Access denied")


def assert_employee_in_scope(scope: HrmScope, employee_id: int, *, conn=None) -> None:
    from fastapi import HTTPException

    if scope.level == "all":
        return
    if scope.level == "self":
        if scope.employee_id is None or int(employee_id) != int(scope.employee_id):
            raise HTTPException(403, "Access denied for this employee")
        return
    # department — verify employee belongs to HOD department
    if scope.department_id is None:
        raise HTTPException(403, "HOD department not configured on your account")
    if conn is None:
        from ..db.hrm_db import employee_department_id

        dept = employee_department_id(employee_id)
    else:
        row = conn.execute("SELECT department_id FROM employees WHERE id=?", (employee_id,)).fetchone()
        dept = row["department_id"] if row else None
    if dept is None or int(dept) != int(scope.department_id):
        raise HTTPException(403, "Access denied for this employee")


def assert_hrm_write_org(scope: HrmScope) -> None:
    from fastapi import HTTPException

    if not scope.can_manage_org:
        raise HTTPException(403, "Only Admin or management can manage departments and employees")


def assert_responsibility_in_scope(scope: HrmScope, responsibility_id: int) -> int:
    """Return employee_id for the responsibility; raise 403/404 if out of scope."""
    from fastapi import HTTPException
    from ..db.hrm_db import get_responsibility_owner

    owner = get_responsibility_owner(responsibility_id)
    if owner is None:
        raise HTTPException(404, "Responsibility not found")
    assert_employee_in_scope(scope, owner)
    return owner

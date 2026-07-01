"""Admin Module router — Users, Roles, Activity Log"""
import sqlite3
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from ..db.users_db import (
    list_roles, create_role,
    list_users, create_user, update_user, deactivate_user,
    list_activity, get_admin_stats, log_activity,
    list_erp_departments, create_erp_department,
)
from ..services.permissions import may_manage_erp_departments
from ..services.upload_policy import may_admin_po_session_edits

router = APIRouter()

class RoleIn(BaseModel):
    role_name: str
    description: Optional[str] = ''

class UserIn(BaseModel):
    username: str
    email: Optional[str] = ''
    password: Optional[str] = 'changeme123'
    full_name: Optional[str] = ''
    role_id: Optional[int] = None
    department: Optional[str] = ''
    phone: Optional[str] = None
    karigar_id: Optional[str] = ''
    employee_id: Optional[int] = None
    hrm_department_id: Optional[int] = None
    reporting_hod_user_id: Optional[int] = None
    module_access: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    role_id: Optional[int] = None
    department: Optional[str] = None
    active: Optional[int] = None
    password: Optional[str] = None
    phone: Optional[str] = None
    karigar_id: Optional[str] = None
    employee_id: Optional[int] = None
    hrm_department_id: Optional[int] = None
    reporting_hod_user_id: Optional[int] = None
    module_access: Optional[str] = None

class ActivityIn(BaseModel):
    username: str
    action: str
    document_type: Optional[str] = ''
    document_no: Optional[str] = ''
    details: Optional[str] = ''


class DepartmentIn(BaseModel):
    name: str


class ModuleDataResetIn(BaseModel):
    module: str


_ALLOWED_RESET_MODULES = {
    "sales_orders",
    "item_master",
    "purchase",
    "tna",
    "production",
    "grey_fabric",
}


def _request_role(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("role") or "Admin")


def _request_username(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("sub") or "")


@contextmanager
def _sqlite_conn(path: str):
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _clear_tables(path: str, table_names: list[str]) -> int:
    deleted = 0
    with _sqlite_conn(path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        for t in table_names:
            try:
                cur = conn.execute(f"DELETE FROM {t}")
                if cur.rowcount and cur.rowcount > 0:
                    deleted += int(cur.rowcount)
            except sqlite3.OperationalError:
                # Missing table on older DB versions — skip silently.
                continue
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except Exception:
            pass
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
    return deleted


def _reset_sales_orders_data() -> int:
    from ..db import sales_db

    return _clear_tables(sales_db._DB, ["so_lines", "sales_orders", "demand_lines", "demands"])


def _reset_item_master_data() -> int:
    from ..db import item_db

    # Keep static setup tables (item_types, size_groups, routing_steps).
    return _clear_tables(
        item_db.DB_PATH,
        ["item_buyer_packaging", "bom_lines", "bom_headers", "item_routing", "items", "merchants", "buyers"],
    )


def _reset_purchase_data() -> int:
    from ..db import purchase_db

    return _clear_tables(
        purchase_db._DB,
        [
            "po_lines",
            "po_headers",
            "pr_lines",
            "pr_headers",
            "jwo_lines",
            "jwo_headers",
            "grn_lines",
            "grn_headers",
            "material_issue_notes",
            "min_lines",
            "gate_pass_lines",
            "gate_passes",
            "suppliers",
            "processors",
        ],
    )


def _reset_tna_data() -> int:
    from ..db import tna_db

    return _clear_tables(tna_db._DB, ["tna_lines", "tna_list"])


def _reset_production_data() -> int:
    from ..db import production_db

    return _clear_tables(
        production_db._DB,
        [
            "jo_cost_entries",
            "jo_piece_receipts",
            "jo_piece_issues",
            "jo_fabric_returns",
            "jo_fabric_issues",
            "jo_lines",
            "job_orders",
            "process_stock",
            "soft_reservations",
            "mrp_soft_reservations",
            "mrp_last_run",
            "mrp_material_commitments",
        ],
    )


def _reset_grey_fabric_data() -> int:
    from ..db import grey_db

    return _clear_tables(
        grey_db._DB,
        [
            "grey_ledger",
            "grey_tracker",
            "hard_reservations",
            "grey_mrp_requirement",
            "grey_printer_issue",
            "grey_conversion",
            "grey_qc_event",
            "grey_return_vendor",
            "fabric_check_log",
            "fabric_checked_stock",
            "grey_unchecked_stock",
            "printed_fabric_stock",
            "printed_fabric_checked_stock",
            "printed_fabric_reservations",
        ],
    )


def _run_module_reset(module: str) -> int:
    if module == "sales_orders":
        return _reset_sales_orders_data()
    if module == "item_master":
        return _reset_item_master_data()
    if module == "purchase":
        return _reset_purchase_data()
    if module == "tna":
        return _reset_tna_data()
    if module == "production":
        return _reset_production_data()
    if module == "grey_fabric":
        return _reset_grey_fabric_data()
    raise ValueError(f"Unsupported module: {module}")

@router.get("/stats")
def get_stats():
    return get_admin_stats()

# ── Roles ─────────────────────────────────────────────────────────────────────
@router.get("/roles")
def get_roles():
    return list_roles()

@router.post("/roles")
def post_role(body: RoleIn):
    create_role(body.role_name, body.description or '')
    return {"ok": True}


# ── Departments (user assignment dropdown) ────────────────────────────────────
@router.get("/departments")
def get_departments():
    return list_erp_departments(active_only=True)


@router.post("/departments")
def post_department(request: Request, body: DepartmentIn):
    role = _request_role(request)
    username = _request_username(request)
    if not may_manage_erp_departments(role, username):
        raise HTTPException(status_code=403, detail="Not allowed to add departments")
    try:
        row = create_erp_department(body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        log_activity(
            username,
            "create",
            "erp_department",
            row.get("name") or body.name,
            f"Added department: {row.get('name') or body.name}",
        )
    except Exception:
        pass
    return {"ok": True, **row}


# ── Users ─────────────────────────────────────────────────────────────────────
@router.get("/users")
def get_users(include_inactive: bool = False):
    """By default only active users. Set ``include_inactive=true`` to audit deactivated accounts."""
    return list_users(active_only=not include_inactive)


@router.post("/users")
def post_user(body: UserIn):
    try:
        create_user(body.model_dump())
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sqlite3.IntegrityError as e:
        raw = str(e).lower()
        if "username" in raw:
            detail = (
                "That username is already in the database (including deactivated accounts). "
                "Enable ‘Show inactive users’, reactivate that row, or pick another username."
            )
        elif "email" in raw:
            detail = "That email is already registered. Use a different address or leave email blank."
        else:
            detail = f"Could not create user: {e}"
        raise HTTPException(status_code=409, detail=detail) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@router.patch("/users/{uid}")
def patch_user(uid: int, body: UserUpdate):
    update_user(uid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}

@router.delete("/users/{uid}")
def delete_user(uid: int):
    deactivate_user(uid)
    return {"ok": True}

# ── Activity Log ──────────────────────────────────────────────────────────────
@router.get("/activity")
def get_activity(limit: int = 100):
    return list_activity(limit)

@router.post("/activity")
def post_activity(body: ActivityIn):
    log_activity(body.username, body.action, body.document_type or '',
                 body.document_no or '', body.details or '')
    return {"ok": True}


@router.post("/reset-module-data")
def reset_module_data(request: Request, body: ModuleDataResetIn):
    role = _request_role(request)
    from ..services.upload_policy import may_delete_upload_data

    if not may_delete_upload_data(role):
        raise HTTPException(status_code=403, detail="Only Super Admin can remove ERP module data.")

    module = (body.module or "").strip().lower()
    if module not in _ALLOWED_RESET_MODULES:
        raise HTTPException(status_code=400, detail="Invalid module key.")

    try:
        deleted_rows = _run_module_reset(module)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}") from e

    try:
        actor = str((getattr(request.state, "auth", None) or {}).get("sub") or "admin")
        log_activity(
            actor,
            "delete",
            "module_data_reset",
            module,
            f"Admin reset module data: {module}, rows_deleted={deleted_rows}",
        )
    except Exception:
        pass

    return {
        "ok": True,
        "module": module,
        "rows_deleted": int(deleted_rows),
        "message": f"{module} test data removed.",
    }

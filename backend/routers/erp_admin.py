"""Admin Module router — Users, Roles, Activity Log"""
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..db.users_db import (
    list_roles, create_role,
    list_users, create_user, update_user, deactivate_user,
    list_activity, get_admin_stats, log_activity
)

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

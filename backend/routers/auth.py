"""
Authentication router.

POST /api/auth/login        → password; new devices require OTP
POST /api/auth/otp/resend   → resend OTP SMS
POST /api/auth/otp/verify   → verify OTP + optional trust device
POST /api/auth/logout       → clear cookies
GET  /api/auth/me           → current user profile + role
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, Response, HTTPException
from jose import jwt, JWTError
from pydantic import BaseModel, Field

from ..db.users_db import verify_erp_user, get_user_auth_profile
from ..services.permissions import permissions_for_role, KARIGAR_ROLE
from ..services.upload_policy import upload_policy_for_role
from ..services.rbac import build_hrm_scope, resolve_module_access, ROLE_SUPER_ADMIN
from ..services.login_otp import (
    otp_required_globally,
    super_admin_otp_bypass_enabled,
    is_super_admin_user,
    normalize_india_phone,
    mask_phone,
    start_login_challenge,
    resend_challenge,
    verify_otp_code,
)
from ..services.device_trust import is_device_trusted, set_trusted_device_cookie

router = APIRouter()


def _cookie_secure(request: Request) -> bool:
    xf = request.headers.get("x-forwarded-proto")
    if xf:
        return xf.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


_SECRET = os.environ.get("JWT_SECRET", "change-me-set-jwt-secret-in-env")
_ALGO = "HS256"
_TTL_H = int(os.environ.get("JWT_TTL_HOURS", "24"))


def create_token(
    username: str,
    *,
    role: str = "Admin",
    user_id: int | None = None,
    full_name: str = "",
    karigar_id: str = "",
) -> str:
    perms = permissions_for_role(role)
    exp = datetime.now(tz=timezone.utc) + timedelta(hours=_TTL_H)
    payload = {
        "sub": username,
        "role": role,
        "user_id": user_id,
        "full_name": full_name,
        "karigar_id": karigar_id or "",
        "permissions": perms,
        "exp": exp,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGO])
    except JWTError:
        return None


def verify_token(token: str) -> str | None:
    payload = decode_token(token)
    return payload.get("sub") if payload else None


def _set_auth_cookie(request: Request, response: Response, token: str) -> None:
    sec = _cookie_secure(request)
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=sec,
        max_age=_TTL_H * 3600,
    )


def _reset_session_cookie(request: Request, response: Response) -> None:
    """Drop stale session_id so login is not blocked restoring a huge PostgreSQL blob."""
    sec = _cookie_secure(request)
    response.delete_cookie("session_id", path="/", secure=sec, httponly=True, samesite="lax")


def _login_payload(user: dict) -> dict:
    role = user.get("role_name") or user.get("role") or "Clerk"
    username = user.get("username") or ""
    modules = resolve_module_access(role, user.get("module_access"))
    hrm_scope = build_hrm_scope(user, role=role)
    redirect = "/production-entry" if role == KARIGAR_ROLE else "/"
    if role not in (KARIGAR_ROLE,) and modules == ["hrm"]:
        redirect = "/hrm"
    pol = upload_policy_for_role(role)
    return {
        "ok": True,
        "username": username,
        "role": role,
        "full_name": user.get("full_name") or "",
        "karigar_id": user.get("karigar_id") or "",
        "employee_id": user.get("employee_id"),
        "hrm_department_id": user.get("hrm_department_id"),
        "user_id": user.get("id"),
        "modules": modules,
        "hrm_scope": {
            "level": hrm_scope.level,
            "department_id": hrm_scope.department_id,
            "employee_id": hrm_scope.employee_id,
            "can_manage_org": hrm_scope.can_manage_org,
        },
        "permissions": permissions_for_role(role),
        **pol,
        "redirect": redirect,
    }


def _complete_login(request: Request, response: Response, user: dict) -> dict:
    role = user.get("role_name") or user.get("role") or "Clerk"
    username = user.get("username") or ""
    token = create_token(
        username,
        role=role,
        user_id=user.get("id"),
        full_name=user.get("full_name") or "",
        karigar_id=str(user.get("karigar_id") or ""),
    )
    _set_auth_cookie(request, response, token)
    _reset_session_cookie(request, response)
    return _login_payload(user)


def _resolve_phone(user: dict | None, username: str) -> str | None:
    if user:
        p = normalize_india_phone(user.get("phone") or "")
        if p:
            return p
    if username == (os.environ.get("SUPER_ADMIN_USERNAME") or os.environ.get("AUTH_USERNAME") or "admin").strip():
        return normalize_india_phone(os.environ.get("SUPER_ADMIN_PHONE", ""))
    return None


def _needs_otp(request: Request, user: dict, phone: str | None) -> bool:
    if not otp_required_globally():
        return False
    if super_admin_otp_bypass_enabled() and is_super_admin_user(
        user, user.get("username") or ""
    ):
        return False
    if not phone:
        return False
    if is_device_trusted(
        request,
        user_id=user.get("id"),
        username=user.get("username") or "",
    ):
        return False
    return True


def _authenticate_password(username: str, password: str) -> dict | None:
    user = verify_erp_user(username, password)
    if user:
        return user

    expected_user = os.environ.get("AUTH_USERNAME", "")
    expected_hash = os.environ.get("AUTH_PASSWORD_HASH", "").encode()
    super_name = (os.environ.get("SUPER_ADMIN_USERNAME") or expected_user or "admin").strip()
    if expected_user and expected_hash and username in (expected_user, super_name):
        if bcrypt.checkpw(password.encode(), expected_hash):
            profile = get_user_auth_profile(super_name) or get_user_auth_profile(expected_user)
            if profile:
                return profile
            return {
                "username": super_name,
                "role_name": ROLE_SUPER_ADMIN,
                "full_name": "Super Administrator",
                "karigar_id": "",
            }
    return None


class LoginRequest(BaseModel):
    username: str
    password: str


class OtpResendRequest(BaseModel):
    challenge_id: str


class OtpVerifyRequest(BaseModel):
    challenge_id: str
    code: str = Field(min_length=4, max_length=8)
    trust_device: bool = True


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    """Sync handler — fast SQLite+bcrypt; must not queue behind upload thread pool."""
    username = body.username.strip()
    password = body.password

    user = _authenticate_password(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    phone = _resolve_phone(user, username)
    if _needs_otp(request, user, phone):
        try:
            challenge_id, masked = start_login_challenge(
                user_id=user.get("id"),
                username=user.get("username") or username,
                phone=phone,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "ok": False,
            "otp_required": True,
            "challenge_id": challenge_id,
            "masked_phone": masked,
            "message": f"OTP sent to {masked}. Required on new devices.",
        }

    return _complete_login(request, response, user)


@router.post("/otp/resend")
def otp_resend(body: OtpResendRequest):
    try:
        masked = resend_challenge(body.challenge_id.strip())
        return {"ok": True, "masked_phone": masked}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/otp/verify")
def otp_verify(body: OtpVerifyRequest, request: Request, response: Response):
    try:
        row = verify_otp_code(body.challenge_id.strip(), body.code.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    username = row.get("username") or ""
    profile = get_user_auth_profile(username)
    if profile:
        user = profile
    else:
        user = {
            "username": username,
            "role_name": ROLE_SUPER_ADMIN,
            "full_name": "Super Administrator",
            "karigar_id": "",
        }

    if body.trust_device:
        set_trusted_device_cookie(
            request,
            response,
            user_id=user.get("id"),
            username=username,
        )

    return _complete_login(request, response, user)


@router.post("/logout")
def logout(request: Request, response: Response):
    sid = request.cookies.get("session_id")
    if sid:
        try:
            from ..db.forecast_session_pg import delete_session_bundle, pg_session_persist_enabled

            if pg_session_persist_enabled():
                delete_session_bundle(sid)
        except Exception:
            pass
        from ..session import store as _store

        _store.delete(sid)
    sec = _cookie_secure(request)
    response.delete_cookie("auth_token", path="/", secure=sec, httponly=True, samesite="lax")
    response.delete_cookie("session_id", path="/", secure=sec, httponly=True, samesite="lax")
    # Keep device_trust cookie so OTP is not required again on this browser after logout.
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

    username = payload.get("sub")
    role = payload.get("role", "Admin")
    profile = get_user_auth_profile(username) if username else None

    if profile:
        role = profile.get("role_name") or role
        pol = upload_policy_for_role(role)
        hrm_scope = build_hrm_scope(profile, role=role)
        modules = resolve_module_access(role, profile.get("module_access"))
        return {
            "username": username,
            "role": role,
            "full_name": profile.get("full_name") or payload.get("full_name", ""),
            "karigar_id": profile.get("karigar_id") or "",
            "user_id": profile.get("id"),
            "department": profile.get("department") or "",
            "employee_id": profile.get("employee_id"),
            "hrm_department_id": profile.get("hrm_department_id"),
            "reporting_hod_user_id": profile.get("reporting_hod_user_id"),
            "module_access": profile.get("module_access") or "",
            "modules": modules,
            "hrm_scope": {
                "level": hrm_scope.level,
                "department_id": hrm_scope.department_id,
                "employee_id": hrm_scope.employee_id,
                "can_manage_org": hrm_scope.can_manage_org,
            },
            "permissions": permissions_for_role(role),
            "is_karigar": role == KARIGAR_ROLE,
            **pol,
        }

    pol = upload_policy_for_role(role)
    return {
        "username": username,
        "role": role,
        "full_name": payload.get("full_name", ""),
        "karigar_id": payload.get("karigar_id", ""),
        "user_id": payload.get("user_id"),
        "department": "",
        "permissions": payload.get("permissions") or permissions_for_role(role),
        "is_karigar": role == KARIGAR_ROLE,
        **pol,
    }

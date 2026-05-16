"""
Authentication router.

POST /api/auth/login   → ERP users (users.db) or legacy env admin
POST /api/auth/logout  → clear cookie
GET  /api/auth/me      → current user profile + role
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, Response, HTTPException
from jose import jwt, JWTError
from pydantic import BaseModel

from ..db.users_db import verify_erp_user, get_user_auth_profile
from ..services.permissions import permissions_for_role, KARIGAR_ROLE

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


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response):
    from starlette.concurrency import run_in_threadpool

    username = body.username.strip()
    password = body.password

    # bcrypt + SQLite are sync; keep the event loop free during busy uploads.
    user = await run_in_threadpool(verify_erp_user, username, password)
    if user:
        role = user.get("role_name") or "Clerk"
        token = create_token(
            username,
            role=role,
            user_id=user.get("id"),
            full_name=user.get("full_name") or "",
            karigar_id=str(user.get("karigar_id") or ""),
        )
        _set_auth_cookie(request, response, token)
        _reset_session_cookie(request, response)
        return {
            "ok": True,
            "username": username,
            "role": role,
            "full_name": user.get("full_name") or "",
            "karigar_id": user.get("karigar_id") or "",
            "permissions": permissions_for_role(role),
            "redirect": "/production-entry" if role == KARIGAR_ROLE else "/",
        }

    expected_user = os.environ.get("AUTH_USERNAME", "")
    expected_hash = os.environ.get("AUTH_PASSWORD_HASH", "").encode()
    if expected_user and expected_hash:

        def _legacy_admin_ok() -> bool:
            return username == expected_user and bcrypt.checkpw(password.encode(), expected_hash)

        password_ok = await run_in_threadpool(_legacy_admin_ok)
        if password_ok:
            token = create_token(username, role="Admin", full_name="Administrator")
            _set_auth_cookie(request, response, token)
            _reset_session_cookie(request, response)
            return {
                "ok": True,
                "username": username,
                "role": "Admin",
                "full_name": "Administrator",
                "karigar_id": "",
                "permissions": permissions_for_role("Admin"),
                "redirect": "/",
            }

    raise HTTPException(status_code=401, detail="Invalid username or password")


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
        return {
            "username": username,
            "role": role,
            "full_name": profile.get("full_name") or payload.get("full_name", ""),
            "karigar_id": profile.get("karigar_id") or "",
            "user_id": profile.get("id"),
            "department": profile.get("department") or "",
            "permissions": permissions_for_role(role),
            "is_karigar": role == KARIGAR_ROLE,
        }

    return {
        "username": username,
        "role": role,
        "full_name": payload.get("full_name", ""),
        "karigar_id": payload.get("karigar_id", ""),
        "user_id": payload.get("user_id"),
        "department": "",
        "permissions": payload.get("permissions") or permissions_for_role(role),
        "is_karigar": role == KARIGAR_ROLE,
    }

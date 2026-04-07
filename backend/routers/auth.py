"""
Authentication router.

POST /api/auth/login   → verify credentials, set httpOnly JWT cookie
POST /api/auth/logout  → clear cookie
GET  /api/auth/me      → return current username or 401
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, Response, HTTPException
from jose import jwt, JWTError
from pydantic import BaseModel

router = APIRouter()

_SECRET = os.environ.get("JWT_SECRET", "change-me-set-jwt-secret-in-env")
_ALGO   = "HS256"
_TTL_H  = 24


# ── Token helpers (also used by auth middleware in main.py) ───────────────────

def create_token(username: str) -> str:
    exp = datetime.now(tz=timezone.utc) + timedelta(hours=_TTL_H)
    return jwt.encode({"sub": username, "exp": exp}, _SECRET, algorithm=_ALGO)


def verify_token(token: str) -> str | None:
    """Return username if valid, None otherwise."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGO])
        return payload.get("sub")
    except JWTError:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginRequest, response: Response):
    expected_user = os.environ.get("AUTH_USERNAME", "")
    expected_hash = os.environ.get("AUTH_PASSWORD_HASH", "").encode()

    if not expected_user or not expected_hash:
        raise HTTPException(status_code=500, detail="Auth not configured on server")

    password_ok = bcrypt.checkpw(body.password.encode(), expected_hash)
    if body.username != expected_user or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(body.username)
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=_TTL_H * 3600,
    )
    return {"ok": True, "username": body.username}


@router.post("/logout")
def logout(request: Request, response: Response):
    # Destroy the server-side session so next login starts with clean state
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
    response.delete_cookie("auth_token")
    response.delete_cookie("session_id")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return {"username": username}

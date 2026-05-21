"""Trusted device cookie — skip OTP on known browsers."""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError

from ..db.users_db import register_trusted_device, trusted_device_exists

DEVICE_COOKIE = "device_trust"
_TRUST_DAYS = int(os.environ.get("DEVICE_TRUST_DAYS", "90"))
_ALGO = "HS256"


def _secret() -> str:
    return os.environ.get("JWT_SECRET", "change-me-set-jwt-secret-in-env") + ":device"


def device_fingerprint(request) -> str:
    ua = (request.headers.get("user-agent") or "")[:512]
    client_id = (request.headers.get("x-device-id") or "")[:64]
    raw = f"{ua}|{client_id}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cookie_secure(request) -> bool:
    xf = request.headers.get("x-forwarded-proto")
    if xf:
        return xf.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def is_device_trusted(request, *, user_id: int | None, username: str) -> bool:
    token = request.cookies.get(DEVICE_COOKIE)
    if not token:
        return False
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALGO])
    except JWTError:
        return False
    if payload.get("sub") != username:
        return False
    uid = payload.get("uid")
    if user_id is not None and uid is not None and int(uid) != int(user_id):
        return False
    fp = payload.get("fp")
    if not fp or fp != device_fingerprint(request):
        return False
    device_id = payload.get("did")
    if not device_id:
        return False
    return trusted_device_exists(device_id=device_id, username=username)


def set_trusted_device_cookie(request, response, *, user_id: int | None, username: str) -> None:
    fp = device_fingerprint(request)
    device_id = secrets.token_urlsafe(18)
    register_trusted_device(
        device_id=device_id,
        user_id=user_id,
        username=username,
        fingerprint=fp,
        user_agent=(request.headers.get("user-agent") or "")[:512],
        trust_days=_TRUST_DAYS,
    )
    exp = datetime.now(tz=timezone.utc) + timedelta(days=_TRUST_DAYS)
    token = jwt.encode(
        {
            "sub": username,
            "uid": user_id,
            "fp": fp,
            "did": device_id,
            "exp": exp,
        },
        _secret(),
        algorithm=_ALGO,
    )
    sec = _cookie_secure(request)
    response.set_cookie(
        key=DEVICE_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=sec,
        max_age=_TRUST_DAYS * 86400,
        path="/",
    )


def clear_trusted_device_cookie(request, response) -> None:
    sec = _cookie_secure(request)
    response.delete_cookie(DEVICE_COOKIE, path="/", secure=sec, httponly=True, samesite="lax")

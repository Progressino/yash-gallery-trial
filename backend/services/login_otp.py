"""India mobile OTP — send, verify, rate limits. MSG91 or dev console."""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import secrets
from datetime import datetime, timedelta, timezone

import requests

from ..db.users_db import create_otp_challenge, get_otp_challenge, increment_otp_attempts, mark_otp_verified

_log = logging.getLogger(__name__)

OTP_TTL_MIN = int(os.environ.get("OTP_TTL_MINUTES", "10"))
OTP_MAX_ATTEMPTS = int(os.environ.get("OTP_MAX_ATTEMPTS", "5"))
OTP_RESEND_SEC = int(os.environ.get("OTP_RESEND_SECONDS", "60"))


def otp_required_globally() -> bool:
    return os.environ.get("OTP_REQUIRED", "1").strip().lower() not in ("0", "false", "no", "off")


def normalize_india_phone(raw: str) -> str | None:
    """Return E.164-style digits only: 91 + 10-digit mobile, or None."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw).strip())
    if len(digits) == 10 and digits[0] in "6789":
        return "91" + digits
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return digits
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return "91" + digits[1:]
    return None


def mask_phone(phone: str) -> str:
    p = normalize_india_phone(phone) or phone
    if len(p) >= 4:
        return f"******{p[-4:]}"
    return "******"


def _otp_pepper() -> bytes:
    return (os.environ.get("JWT_SECRET", "otp-pepper") + ":otp").encode()


def hash_otp(code: str) -> str:
    return hashlib.sha256(_otp_pepper() + code.strip().encode()).hexdigest()


def generate_otp_code() -> str:
    return f"{random.SystemRandom().randint(0, 999999):06d}"


def start_login_challenge(*, user_id: int | None, username: str, phone: str) -> tuple[str, str]:
    """Create DB challenge and send SMS. Returns (challenge_id, masked_phone)."""
    code = generate_otp_code()
    challenge_id = secrets.token_urlsafe(24)
    expires = (datetime.now(tz=timezone.utc) + timedelta(minutes=OTP_TTL_MIN)).isoformat()
    create_otp_challenge(
        challenge_id=challenge_id,
        user_id=user_id,
        username=username,
        phone=phone,
        code_hash=hash_otp(code),
        expires_at=expires,
    )
    send_otp_sms(phone, code)
    return challenge_id, mask_phone(phone)


def resend_challenge(challenge_id: str) -> str:
    row = get_otp_challenge(challenge_id)
    if not row:
        raise ValueError("Invalid or expired verification session")
    if row.get("verified"):
        raise ValueError("Already verified")
    exp = row.get("expires_at") or ""
    try:
        exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    except Exception:
        exp_dt = datetime.now(tz=timezone.utc)
    if exp_dt < datetime.now(tz=timezone.utc):
        raise ValueError("Verification session expired — sign in again")
    created = row.get("created_at") or ""
    try:
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except Exception:
        created_dt = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    if (datetime.now(tz=timezone.utc) - created_dt).total_seconds() < OTP_RESEND_SEC:
        raise ValueError(f"Wait {OTP_RESEND_SEC} seconds before requesting another OTP")
    code = generate_otp_code()
    expires = (datetime.now(tz=timezone.utc) + timedelta(minutes=OTP_TTL_MIN)).isoformat()
    create_otp_challenge(
        challenge_id=challenge_id,
        user_id=row.get("user_id"),
        username=row.get("username") or "",
        phone=row.get("phone") or "",
        code_hash=hash_otp(code),
        expires_at=expires,
    )
    send_otp_sms(row["phone"], code)
    return mask_phone(row["phone"])


def verify_otp_code(challenge_id: str, code: str) -> dict:
    row = get_otp_challenge(challenge_id)
    if not row:
        raise ValueError("Invalid or expired verification session")
    if row.get("verified"):
        raise ValueError("Already verified")
    attempts = int(row.get("attempts") or 0)
    if attempts >= OTP_MAX_ATTEMPTS:
        raise ValueError("Too many attempts — sign in again")
    exp = row.get("expires_at") or ""
    try:
        exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    except Exception:
        exp_dt = datetime.now(tz=timezone.utc) - timedelta(seconds=1)
    if exp_dt < datetime.now(tz=timezone.utc):
        raise ValueError("OTP expired — sign in again")
    if hash_otp(code) != row.get("code_hash"):
        increment_otp_attempts(challenge_id)
        left = max(0, OTP_MAX_ATTEMPTS - attempts - 1)
        raise ValueError(f"Incorrect OTP ({left} attempts left)")
    mark_otp_verified(challenge_id)
    return row


def send_otp_sms(phone: str, code: str) -> None:
    normalized = normalize_india_phone(phone)
    if not normalized:
        raise ValueError("Invalid Indian mobile number")

    if os.environ.get("SMS_OTP_DEV", "").strip().lower() in ("1", "true", "yes"):
        _log.warning("SMS_OTP_DEV: OTP for %s is %s", mask_phone(normalized), code)
        return

    auth_key = os.environ.get("MSG91_AUTH_KEY", "").strip()
    if not auth_key:
        _log.error("MSG91_AUTH_KEY not set — cannot send OTP to %s", mask_phone(normalized))
        raise RuntimeError("SMS is not configured on the server. Contact your administrator.")

    sender = os.environ.get("MSG91_SENDER_ID", "PROGIN").strip() or "PROGIN"
    template_id = os.environ.get("MSG91_TEMPLATE_ID", "").strip()
    mobile = normalized[2:] if normalized.startswith("91") else normalized

    # MSG91 v5 flow API (India DLT template)
    if template_id:
        url = "https://control.msg91.com/api/v5/flow"
        payload = {
            "template_id": template_id,
            "short_url": "0",
            "recipients": [{"mobiles": f"91{mobile}", "otp": code, "var": code}],
        }
        headers = {"authkey": auth_key, "Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=15)
    else:
        # Legacy OTP API (requires MSG91 OTP widget / route)
        url = "https://control.msg91.com/api/v5/otp"
        payload = {"template_id": "default", "mobile": f"91{mobile}", "otp": code}
        headers = {"authkey": auth_key, "Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=15)

    if r.status_code >= 400:
        _log.error("MSG91 send failed: %s %s", r.status_code, r.text[:500])
        raise RuntimeError("Could not send OTP SMS. Try again in a minute.")

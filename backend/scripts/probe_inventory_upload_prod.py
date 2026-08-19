#!/usr/bin/env python3
"""Prod probe: upload daily inventory RAR fixtures and verify snapshot dates."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("PROBE_BASE_URL", "http://127.0.0.1:8000")
FIXTURE_DIR = Path(os.environ.get("INV_FIXTURE_DIR", "ops/test-fixtures/inventory"))
EXPECTED_DATES = ("2026-08-01", "2026-08-02", "2026-08-03")
POLL_SEC = 1800


def _env_auth() -> tuple[str, str]:
    user = os.environ.get("AUTH_USERNAME", "").strip()
    pw = os.environ.get("AUTH_PASSWORD", "").strip()
    if not user or not pw:
        env_path = Path("/root/app/.env")
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "AUTH_USERNAME":
                    user = user or v
                elif k == "AUTH_PASSWORD":
                    pw = pw or v
    if not user or not pw:
        raise SystemExit("AUTH_USERNAME/PASSWORD missing")
    return user, pw


def _wait_backend(session: requests.Session) -> str:
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            h = session.get(f"{BASE}/api/health", timeout=20)
            if h.ok:
                body = h.json()
                if body.get("status") == "ok":
                    sha = str(body.get("git_sha") or "")
                    print("HEALTH_OK", sha, "warm", body.get("warm_cache"))
                    return sha
        except Exception as e:
            print("health_wait", e)
        time.sleep(10)
    raise SystemExit("backend not healthy after 10 min")


def _login(session: requests.Session) -> None:
    user, pw = _env_auth()
    for attempt in range(1, 8):
        try:
            r = session.post(
                f"{BASE}/api/auth/login",
                json={"username": user, "password": pw},
                headers={"X-Device-Id": "gha-inv-upload-probe"},
                timeout=60,
            )
            print("LOGIN", r.status_code, (r.text or "")[:160])
            r.raise_for_status()
            body = r.json()
            if body.get("requires_otp"):
                raise SystemExit("login requires OTP")
            return
        except Exception as e:
            print("login_retry", attempt, e)
            time.sleep(5)
    raise SystemExit("login failed")


def _clear_stuck(session: requests.Session) -> None:
    r = session.post(f"{BASE}/api/upload/inventory-auto/reset-stuck", timeout=30)
    print("RESET_STUCK", r.status_code, (r.text or "")[:120])


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int = 12,
    backoff: float = 5.0,
    **kwargs,
) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = session.request(method, url, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_err = e
            print("request_retry", method, url, attempt, e)
            time.sleep(backoff * attempt)
    raise SystemExit(f"request failed after {retries} retries: {last_err}")


def _coverage(session: requests.Session) -> dict:
    r = _request_with_retry(
        session,
        "GET",
        f"{BASE}/api/data/coverage",
        params={"light": "1"},
        timeout=60,
    )
    return r.json()


def _wait_inventory_done(session: requests.Session) -> dict:
    start = time.time()
    saw_running = False
    while time.time() - start < POLL_SEC:
        try:
            cov = _coverage(session)
        except SystemExit as e:
            print("coverage_retry", e)
            time.sleep(10)
            continue
        st = str(cov.get("inventory_upload_status") or "idle")
        msg = str(cov.get("inventory_upload_message") or "")
        pct = int(cov.get("inventory_upload_progress") or 0)
        print("INV_STATUS", st, pct, msg[:120])
        if st == "running":
            saw_running = True
        elif st == "error":
            raise SystemExit(f"inventory upload error: {msg}")
        elif st == "done":
            return cov
        elif saw_running and st == "idle":
            return cov
        time.sleep(5)
    raise SystemExit("inventory upload poll timeout")


def _upload_rar(session: requests.Session, path: Path) -> None:
    print("UPLOAD", path.name, path.stat().st_size)
    with path.open("rb") as fh:
        r = session.post(
            f"{BASE}/api/upload/inventory-auto",
            files=[("files", (path.name, fh, "application/x-rar-compressed"))],
            timeout=600,
        )
    print("UPLOAD_RESP", r.status_code, (r.text or "")[:240])
    r.raise_for_status()
    body = r.json()
    if not body.get("ok"):
        raise SystemExit(f"upload rejected: {body.get('message')}")
    if body.get("ingest_async"):
        cov = _wait_inventory_done(session)
        print(
            "UPLOAD_DONE",
            path.name,
            "rows",
            cov.get("inventory_upload_rows"),
            "snapshot",
            cov.get("inventory_snapshot_date"),
        )
        return
    print("UPLOAD_SYNC_OK", body.get("message", ""))


def _matrix_has_date(session: requests.Session, day: str) -> bool:
    r = _request_with_retry(
        session,
        "GET",
        f"{BASE}/api/po/daily-inventory-history/matrix",
        params={"days": 60, "limit": 3, "channel": "oms"},
        timeout=120,
    )
    body = r.json() or {}
    uploaded = {str(x)[:10] for x in (body.get("uploaded_dates") or [])}
    gaps = {str(x)[:10] for x in (body.get("gap_dates") or [])}
    ok = day in uploaded
    print("MATRIX_DAY", day, "uploaded", ok, "in_gaps", day in gaps)
    return ok


def main() -> int:
    if os.environ.get("POST_DEPLOY_COOLDOWN", "").strip() == "1":
        print("POST_DEPLOY_COOLDOWN 120s")
        time.sleep(120)

    session = requests.Session()
    sha = _wait_backend(session)
    _login(session)
    _clear_stuck(session)

    cov = _coverage(session)
    if not cov.get("sku_mapping"):
        raise SystemExit("SKU mapping not loaded — cannot upload inventory")

    fixtures = sorted(FIXTURE_DIR.glob("Inventory*.rar"))
    if not fixtures:
        raise SystemExit(f"No RAR fixtures in {FIXTURE_DIR}")

    for path in fixtures:
        _clear_stuck(session)
        # Skip if this snapshot date is already present (safe to re-run probe).
        day_hint = path.stem.split("-Aug-26")[0].split()[-1]
        if day_hint.isdigit():
            maybe = f"2026-08-{int(day_hint):02d}"
            if _matrix_has_date(session, maybe):
                print("SKIP_ALREADY_UPLOADED", path.name, maybe)
                continue
        _upload_rar(session, path)
        print("COOLDOWN 30s before next upload")
        time.sleep(30)

    missing = [d for d in EXPECTED_DATES if not _matrix_has_date(session, d)]
    if missing:
        raise SystemExit(f"Inv. History missing uploaded dates after probe: {missing}")

    print("INV_UPLOAD_PROBE_OK", json.dumps({"sha": sha, "dates": EXPECTED_DATES}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

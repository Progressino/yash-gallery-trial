#!/usr/bin/env python3
"""Read-only smoke of upload *API* shapes on production.

Do NOT upload synthetic Meesho rows into prod. An earlier version of this script
wrote SKU ``SMOKE-TEST-SKU`` into Tier-3 / sales_df and polluted Sales History.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import requests

BASE = "https://app.progressino.com"
USER = "admin"
PASSWORD = "ProgressinoAdmin2026!"


def login(sess: requests.Session) -> None:
    r = sess.post(
        f"{BASE}/api/auth/login",
        json={"username": USER, "password": PASSWORD},
        headers={"X-Device-Id": "upload-smoke-prod"},
        timeout=60,
    )
    r.raise_for_status()
    if r.json().get("otp_required"):
        raise SystemExit("OTP required for smoke test user")


def coverage(sess: requests.Session) -> dict:
    r = sess.get(f"{BASE}/api/data/coverage", params={"light": "1"}, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    sess = requests.Session()
    print("Login…")
    login(sess)

    cov = coverage(sess)
    print(
        "Coverage:",
        "daily_ingest=",
        cov.get("daily_auto_ingest_status"),
        "inv_upload=",
        cov.get("inventory_upload_status"),
        "daily_inv=",
        cov.get("daily_inventory_upload_status"),
        "sales_rows=",
        cov.get("sales_rows"),
    )

    # Chunk init only — policy + session, no fake rows written to warm cache.
    fname = f"Meesho Orders {(date.today() - timedelta(days=1)).strftime('%d-%b-%y')}.csv"
    r = sess.post(
        f"{BASE}/api/upload/chunk/init",
        json={
            "target": "daily-auto",
            "files": [{"name": fname, "size": 128}],
        },
        timeout=60,
    )
    print(f"Chunk init: status={r.status_code}")
    if r.status_code == 403:
        print(r.text)
        return 2
    chunk = r.json()
    if not chunk.get("ok"):
        print(json.dumps(chunk, indent=2))
        return 2

    print("OK — read-only smoke passed (no SMOKE-TEST-SKU written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

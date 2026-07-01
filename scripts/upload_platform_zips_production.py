#!/usr/bin/env python3
"""Upload Tier-1 Myntra/Flipkart ZIP archives to production and wait for completion."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

BASE = "https://app.progressino.com"
USER = "admin"
PASSWORD = "ProgressinoAdmin2026!"

UPLOADS = [
    ("myntra", Path("/Users/samraisinghani/Downloads/Myntra PPMP Jan-2024 to Jan-2026.zip")),
    ("myntra", Path("/Users/samraisinghani/Downloads/Myntra Sjit Jan-2024 to Jan-2026.zip")),
    ("flipkart", Path("/Users/samraisinghani/Downloads/Flipkart Akiko Jan-2024 To Dec-2025.zip")),
    ("flipkart", Path("/Users/samraisinghani/Downloads/Flipkart AG Jan-2024 To Dec-2025.zip")),
]


def login(sess: requests.Session) -> None:
    r = sess.post(
        f"{BASE}/api/auth/login",
        json={"username": USER, "password": PASSWORD},
        headers={"X-Device-Id": "platform-zip-audit"},
        timeout=120,
    )
    r.raise_for_status()
    if r.json().get("otp_required"):
        raise SystemExit("OTP required — cannot upload unattended")


def coverage(sess: requests.Session) -> dict:
    r = sess.get(f"{BASE}/api/data/coverage", params={"light": "1"}, timeout=120)
    r.raise_for_status()
    return r.json()


def wait_bulk(sess: requests.Session, label: str, timeout_sec: int = 1800) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        cov = coverage(sess)
        status = cov.get("tier1_bulk_status") or "idle"
        msg = cov.get("tier1_bulk_message") or ""
        print(f"  [{label}] tier1_bulk_status={status} {msg[:120]}")
        if status in ("done", "idle"):
            return cov
        if status == "error":
            raise RuntimeError(f"Tier-1 upload failed: {msg}")
        time.sleep(8)
    raise TimeoutError(f"Timed out waiting for {label}")


def upload_zip(sess: requests.Session, platform: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    url = f"{BASE}/api/upload/{platform}"
    print(f"Uploading {platform}: {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)…")
    with path.open("rb") as fh:
        r = sess.post(
            url,
            files={"file": (path.name, fh, "application/zip")},
            timeout=600,
        )
    r.raise_for_status()
    body = r.json()
    print("  accept:", body.get("message") or body)
    if not body.get("ok", True):
        raise RuntimeError(body.get("message") or "upload rejected")
    wait_bulk(sess, path.name)


def sync_tier3(sess: requests.Session) -> None:
    print("Syncing Tier-3 → warm cache…")
    r = sess.post(f"{BASE}/api/cache/sync-tier3", timeout=120)
    r.raise_for_status()
    print(" ", r.json().get("message") or r.json())
    # background worker — poll coverage until tier1 idle and row counts stable
    time.sleep(15)
    for _ in range(60):
        cov = coverage(sess)
        if cov.get("tier1_bulk_status") not in ("running",):
            print(
                "  myntra_rows=", cov.get("myntra_rows"),
                "flipkart_rows=", cov.get("flipkart_rows"),
            )
            return
        time.sleep(10)


def main() -> int:
    sess = requests.Session()
    login(sess)
    cov = coverage(sess)
    print(
        "Before:",
        "myntra_rows=", cov.get("myntra_rows"),
        "flipkart_rows=", cov.get("flipkart_rows"),
    )
    for platform, path in UPLOADS:
        upload_zip(sess, platform, path)
    sync_tier3(sess)
    cov = coverage(sess)
    print(
        "After:",
        "myntra_rows=", cov.get("myntra_rows"),
        "flipkart_rows=", cov.get("flipkart_rows"),
        "myntra=", cov.get("myntra"),
        "flipkart=", cov.get("flipkart"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

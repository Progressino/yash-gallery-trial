#!/usr/bin/env python3
"""Regression: Intelligence dashboard fast path (summary must complete under concurrent load)."""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = (os.environ.get("REGRESSION_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
START = os.environ.get("REGRESSION_START", "2026-05-25")
END = os.environ.get("REGRESSION_END", "2026-06-24")
SUMMARY_MAX_SEC = float(os.environ.get("REGRESSION_SUMMARY_MAX_SEC", "45"))
WALL_MAX_SEC = float(os.environ.get("REGRESSION_WALL_MAX_SEC", "60"))


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    dot = ROOT / ".env"
    if dot.is_file():
        for line in dot.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def login(session: requests.Session, env: dict[str, str]) -> None:
    user = (env.get("AUTH_USERNAME") or env.get("SUPER_ADMIN_USERNAME") or "admin").strip()
    pw = (env.get("AUTH_PASSWORD") or env.get("SUPER_ADMIN_PASSWORD") or "").strip()
    if not pw:
        raise SystemExit("AUTH_PASSWORD missing in .env")
    r = session.post(
        f"{BASE}/api/auth/login",
        json={"username": user, "password": pw},
        headers={"X-Device-Id": "dashboard-regression"},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("otp_required"):
        raise SystemExit("OTP required — use SUPER_ADMIN_OTP_BYPASS locally or trusted device")
    if not body.get("ok") and not body.get("username"):
        raise SystemExit(f"Login failed: {body}")


def main() -> int:
    env = _load_env()
    s = requests.Session()
    login(s, env)
    print(f"base={BASE} window={START}..{END}")

    def summary():
        t0 = time.time()
        r = s.get(
            f"{BASE}/api/data/dashboard/summary",
            params={"start_date": START, "end_date": END},
            timeout=int(SUMMARY_MAX_SEC + 30),
        )
        r.raise_for_status()
        j = r.json()
        units = int((j.get("sales_summary") or {}).get("total_units") or 0)
        plats = len(j.get("platform_summary") or [])
        return time.time() - t0, units, plats, j.get("source")

    def parity():
        r = s.get(f"{BASE}/api/data/parity", params={"planning_date": END}, timeout=30)
        r.raise_for_status()
        return r.json().get("tier3_sync_mismatch")

    def coverage():
        r = s.get(f"{BASE}/api/data/coverage", params={"light": "1"}, timeout=90)
        r.raise_for_status()
        return r.status_code

    def readiness():
        r = s.get(f"{BASE}/api/data/intelligence/readiness", timeout=60)
        r.raise_for_status()
        return r.json().get("dashboard_ready")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            "summary": pool.submit(summary),
            "parity": pool.submit(parity),
            "coverage": pool.submit(coverage),
            "readiness": pool.submit(readiness),
        }
        results: dict[str, object] = {}
        for name, fut in futs.items():
            try:
                results[name] = fut.result(timeout=SUMMARY_MAX_SEC + 30)
            except Exception as exc:
                print(f"FAIL {name}: {exc}")
                return 1

    wall = time.time() - t0
    dt, units, plats, source = results["summary"]  # type: ignore[misc]
    print(f"summary: {dt:.2f}s units={units:,} plats={plats} source={source}")
    print(f"parity tier3_sync_mismatch={results['parity']}")
    print(f"wall={wall:.2f}s")

    if dt > SUMMARY_MAX_SEC:
        print(f"FAIL: summary slower than {SUMMARY_MAX_SEC}s")
        return 1
    if wall > WALL_MAX_SEC:
        print(f"FAIL: concurrent wall slower than {WALL_MAX_SEC}s")
        return 1
    if units <= 0 or plats < 1:
        print("FAIL: summary returned no displayable units")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

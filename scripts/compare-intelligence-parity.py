#!/usr/bin/env python3
"""Compare Intelligence 30D gross units: local vs production (requires credentials)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
cfg = dotenv_values(ROOT / ".env")

LOCAL = os.environ.get("LOCAL_API", "http://127.0.0.1:8000")
PROD = os.environ.get("PROD_API", "https://app.progressino.com")
START = os.environ.get("COMPARE_START", "2026-05-19")
END = os.environ.get("COMPARE_END", "2026-06-18")
TOLERANCE = int(os.environ.get("COMPARE_TOLERANCE_UNITS", "50"))


def login(base: str, session: requests.Session) -> None:
    user = (cfg.get("AUTH_USERNAME") or "admin").strip()
    password = (cfg.get("AUTH_PASSWORD") or "").strip()
    if not password:
        raise SystemExit("AUTH_PASSWORD missing in .env")
    r = session.post(f"{base}/api/auth/login", json={"username": user, "password": password}, timeout=60)
    r.raise_for_status()
    body = r.json()
    if not body.get("ok"):
        raise SystemExit(f"Login failed at {base}: {body}")


def gross_units(base: str, session: requests.Session) -> dict:
    parity = session.get(
        f"{base}/api/data/parity",
        params={"planning_date": END},
        timeout=60,
    ).json()
    bundle = session.get(
        f"{base}/api/data/intelligence-bundle",
        params={"start_date": START, "end_date": END, "limit": 5, "basis": "gross"},
        timeout=180,
    ).json()
    ss = bundle.get("sales_summary") or {}
    return {
        "gross": int(ss.get("total_units") or 0),
        "net": int(ss.get("net_units") or 0),
        "returns": int(ss.get("total_returns") or 0),
        "tier3_auto_pull": bundle.get("tier3_auto_pull"),
        "tier3_files": int(parity.get("tier3_file_count") or 0),
        "warnings": parity.get("warnings") or [],
    }


def main() -> int:
    local_s = requests.Session()
    prod_s = requests.Session()
    login(LOCAL, local_s)
    login(PROD, prod_s)

    local = gross_units(LOCAL, local_s)
    prod = gross_units(PROD, prod_s)

    print(f"Window: {START} → {END}")
    print(f"LOCAL  gross={local['gross']:,}  tier3_files={local['tier3_files']}  tier3_pull={local['tier3_auto_pull']}")
    if local["warnings"]:
        for w in local["warnings"]:
            print(f"  warn: {w}")
    print(f"PROD   gross={prod['gross']:,}  tier3_files={prod['tier3_files']}  tier3_pull={prod['tier3_auto_pull']}")
    if prod["warnings"]:
        for w in prod["warnings"]:
            print(f"  warn: {w}")

    delta = abs(local["gross"] - prod["gross"])
    print(f"Delta: {delta:,} units (tolerance {TOLERANCE:,})")

    if local["tier3_files"] == 0:
        print("FAIL: local Tier-3 DB empty — run scripts/sync-tier3-from-vps.sh")
        return 1
    if delta > TOLERANCE:
        print("FAIL: gross units mismatch")
        return 1
    print("PASS: local matches production within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())

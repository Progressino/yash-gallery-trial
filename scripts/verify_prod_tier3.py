#!/usr/bin/env python3
"""Smoke-check prod (or local) Tier-3 + intelligence for recent daily uploads."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

BASE = (os.environ.get("VERIFY_BASE_URL") or "https://app.progressino.com").rstrip("/")
START = os.environ.get("VERIFY_START_DATE", "2026-06-17")
END = os.environ.get("VERIFY_END_DATE", "2026-06-18")


def main() -> int:
    r = requests.get(f"{BASE}/api/health", timeout=30)
    r.raise_for_status()
    health = r.json()
    print(f"health: git_sha={health.get('git_sha')} warm_cache={health.get('warm_cache')}")

    from backend.services.daily_store import get_summary, platforms_with_uploads_in_range
    from backend.session import AppSession
    from backend.routers.data import _tier3_direct_has_units

    summary = get_summary() or {}
    files = sum(int((summary.get(p) or {}).get("file_count") or 0) for p in summary)
    plats = platforms_with_uploads_in_range(START, END)
    print(f"tier3: {files} files, window {START}..{END} platforms={plats}")
    max_dates = [str((summary.get(p) or {}).get("max_date") or "")[:10] for p in summary]
    max_dates = [d for d in max_dates if len(d) == 10]
    tier3_through = max(max_dates) if max_dates else ""
    print(f"  tier3 max_date: {tier3_through}")
    if plats and tier3_through < START:
        print(f"FAIL: tier3 platforms overlap window but max_date {tier3_through} < {START}")
        return 1

    sess = AppSession()
    sess.sku_mapping = {}
    # Single-day spot check (fast); range totals can include dispatch-dated rows on prior day.
    spot = START if START == END else "2026-06-17"
    t3 = _tier3_direct_has_units(spot, spot, sess, 10, "gross")
    units = int(t3[0].get("total_units") or 0) if t3 else 0
    print(f"  intelligence-tier3 spot {spot}: {units} gross units")
    if units <= 0 and plats:
        print(f"FAIL: tier3 metadata has platforms but 0 units for spot day {spot}")
        return 1

    user = os.environ.get("AUTH_USERNAME", "").strip()
    pw = os.environ.get("AUTH_PASSWORD", "").strip()
    if user and pw:
        s = requests.Session()
        lr = s.post(
            f"{BASE}/api/auth/login",
            json={"username": user, "password": pw},
            headers={"X-Device-Id": "verify-prod-tier3"},
            timeout=60,
        )
        lr.raise_for_status()
        if lr.json().get("otp_required"):
            print("WARN: OTP required — skipping authenticated intelligence check")
        else:
            ir = s.get(
                f"{BASE}/api/data/intelligence-bundle",
                params={
                    "start_date": START,
                    "end_date": END,
                    "limit": 5,
                    "basis": "gross",
                    "include_extras": "0",
                },
                timeout=120,
            )
            ir.raise_for_status()
            units = int((ir.json().get("sales_summary") or {}).get("total_units") or 0)
            print(f"  live intelligence-bundle {START}..{END}: {units} gross units")
            if units <= 0 and plats:
                print("FAIL: live intelligence returned 0 with tier3 uploads present")
                return 1
            try:
                pr = s.get(f"{BASE}/api/data/parity", params={"planning_date": END}, timeout=30)
                if pr.ok:
                    parity = pr.json()
                    print(
                        f"  parity: sales_through={parity.get('sales_through')} "
                        f"mismatch={parity.get('tier3_sync_mismatch')} ok={parity.get('ok')}"
                    )
                else:
                    print(f"  parity: HTTP {pr.status_code} (non-fatal)")
            except requests.RequestException as exc:
                print(f"  parity: skipped ({exc.__class__.__name__})")
    else:
        print("Set AUTH_USERNAME/AUTH_PASSWORD to verify live session intelligence + parity")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

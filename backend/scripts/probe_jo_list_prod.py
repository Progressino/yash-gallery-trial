#!/usr/bin/env python3
"""Prod probe: time GET /api/production/orders (Cutting, light + full)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("PROBE_BASE_URL", "http://127.0.0.1:8000")


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


def main() -> int:
    s = requests.Session()
    h = s.get(f"{BASE}/api/health", timeout=30)
    h.raise_for_status()
    health = h.json()
    print("HEALTH", json.dumps({k: health.get(k) for k in ("status", "git_sha", "warm_cache")}))

    user, pw = _env_auth()
    r = s.post(
        f"{BASE}/api/auth/login",
        json={"username": user, "password": pw},
        headers={"X-Device-Id": "gha-jo-list-probe"},
        timeout=60,
    )
    print("LOGIN", r.status_code)
    r.raise_for_status()

    results = {}
    for label, path in (
        ("cutting_light", "/api/production/orders?process=Cutting&light=1&limit=150"),
        ("cutting_full", "/api/production/orders?process=Cutting&limit=150"),
        ("stats", "/api/production/stats"),
        ("ready_cutting", "/api/production/ready-to-process/Cutting"),
    ):
        t0 = time.time()
        try:
            resp = s.get(f"{BASE}{path}", timeout=180)
            elapsed = time.time() - t0
            n = len(resp.json()) if resp.ok and isinstance(resp.json(), list) else None
            results[label] = {
                "status": resp.status_code,
                "seconds": round(elapsed, 3),
                "bytes": len(resp.content),
                "count": n,
            }
            print(
                "TIMING",
                label,
                f"{elapsed:.3f}s",
                "status",
                resp.status_code,
                "bytes",
                len(resp.content),
                "count",
                n,
            )
            resp.raise_for_status()
        except Exception as e:
            elapsed = time.time() - t0
            results[label] = {"error": str(e), "seconds": round(elapsed, 3)}
            print("TIMING_FAIL", label, f"{elapsed:.3f}s", e)
            raise SystemExit(f"{label} failed: {e}") from e

    light_s = float(results["cutting_light"]["seconds"])
    if light_s > 20:
        raise SystemExit(f"Cutting light list too slow: {light_s:.1f}s (target <20s for first page)")
    print("JO_LIST_PROBE_OK", json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

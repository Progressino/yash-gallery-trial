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


def _wait_warm_cache(session: requests.Session, *, timeout_s: float = 600.0) -> dict:
    """Deploy can report health ok while warm_cache is still false for several minutes."""
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        try:
            h = session.get(f"{BASE}/api/health", timeout=30)
            h.raise_for_status()
            last = h.json() or {}
            print(
                "HEALTH_WAIT",
                json.dumps(
                    {
                        k: last.get(k)
                        for k in ("status", "git_sha", "warm_cache", "upload_memory_lock_held")
                    }
                ),
            )
            if last.get("status") == "ok" and last.get("warm_cache"):
                return last
        except Exception as e:
            print("HEALTH_WAIT_ERR", e)
        time.sleep(10)
    raise SystemExit(
        f"warm_cache still false after {int(timeout_s)}s — "
        f"last={json.dumps({k: last.get(k) for k in ('status', 'git_sha', 'warm_cache')})}"
    )


def main() -> int:
    s = requests.Session()
    health = _wait_warm_cache(s)
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

    # Session attach: empty cookie session must leave 0/8 within seconds via hydrate-warm.
    # After deploy, concurrent probes / reconcile can make the first attach slow — retry.
    hydrate_s = 0.0
    c0: dict = {}
    c1: dict = {}
    hyd_msg = ""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            cov0 = s.get(f"{BASE}/api/data/coverage?light=1", timeout=180)
            cov0.raise_for_status()
            c0 = cov0.json() or {}
            hyd = s.post(f"{BASE}/api/cache/hydrate-warm", timeout=180)
            hyd.raise_for_status()
            hyd_msg = str((hyd.json() or {}).get("message") or "")
            cov1 = s.get(f"{BASE}/api/data/coverage?light=1", timeout=180)
            cov1.raise_for_status()
            c1 = cov1.json() or {}
            hydrate_s = time.time() - t0
            sales_ok = bool(c1.get("sales")) or int(c1.get("sales_rows") or 0) > 0
            sku_ok = bool(c1.get("sku_mapping"))
            print(
                "TIMING hydrate",
                attempt,
                f"{hydrate_s:.3f}s",
                "sales",
                sales_ok,
                "sku",
                sku_ok,
                hyd_msg,
            )
            if sales_ok or sku_ok:
                break
            last_err = SystemExit("hydrate-warm left session with no sales and no sku_mapping")
        except Exception as e:
            last_err = e
            hydrate_s = time.time() - t0
            print("TIMING_FAIL hydrate", attempt, f"{hydrate_s:.3f}s", e)
            time.sleep(5 * attempt)
    else:
        raise SystemExit(f"hydrate-warm failed after retries: {last_err}")

    sales_ok = bool(c1.get("sales")) or int(c1.get("sales_rows") or 0) > 0
    sku_ok = bool(c1.get("sku_mapping"))
    results["hydrate"] = {
        "seconds": round(hydrate_s, 3),
        "before_sales": bool(c0.get("sales")),
        "after_sales": sales_ok,
        "after_sku": sku_ok,
        "message": hyd_msg,
    }
    if not sales_ok and not sku_ok:
        raise SystemExit("hydrate-warm left session with no sales and no sku_mapping (0/8 stuck)")
    # Attached shared warm cache is typically <5s; allow headroom under post-deploy load.
    if hydrate_s > 120:
        raise SystemExit(f"hydrate-warm too slow: {hydrate_s:.1f}s (target <120s)")

    print("JO_LIST_PROBE_OK", json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

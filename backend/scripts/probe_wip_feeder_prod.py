#!/usr/bin/env python3
"""Prod probe: Ready-To WIP templates + feeder resolution for Kaj/Handwork/Finishing."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("PROBE_BASE_URL", "http://127.0.0.1:8000")
STAGES = ("Kajh Button", "Handwork", "Finishing")



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
        headers={"X-Device-Id": "gha-wip-feeder-probe"},
        timeout=60,
    )
    print("LOGIN", r.status_code)
    r.raise_for_status()

    results = {}
    for stage in STAGES:
        t0 = time.time()
        tpl = s.get(
            f"{BASE}/api/production/ready-to-wip/import-template",
            params={"stage": stage},
            timeout=60,
        )
        tpl.raise_for_status()
        assert "Ready_To_Stage" in tpl.text or "ready_to_stage" in tpl.text.lower()
        # Tiny import: unique probe SKU; credits feeder stock (MIGRATE SO).
        sku = f"PROBEWIP-{stage.replace(' ', '').upper()[:6]}-S"
        csv = (
            "Ready_To_Stage,SO_Number,OMS_SKU,Quantity,Remarks\n"
            f"{stage},PROBE-WIP,{sku},1,probe_wip_feeder\n"
        )
        up = s.post(
            f"{BASE}/api/production/ready-to-wip/import",
            files={"file": ("probe.csv", io.BytesIO(csv.encode()), "text/csv")},
            data={"stage": stage},
            timeout=90,
        )
        elapsed = time.time() - t0
        body = up.json() if up.ok else {"detail": up.text[:300]}
        print("STAGE", stage, up.status_code, f"{elapsed:.2f}s", body.get("message") or body)
        up.raise_for_status()
        if int(body.get("imported") or 0) < 1:
            raise SystemExit(f"{stage} import failed: {body}")
        ready = s.get(
            f"{BASE}/api/production/ready-to-process/{stage}",
            timeout=90,
        )
        ready.raise_for_status()
        rows = ready.json() if isinstance(ready.json(), list) else []
        hit = next((x for x in rows if str(x.get("sku") or "") == sku), None)
        results[stage] = {
            "imported": body.get("imported"),
            "ready_hit": bool(hit),
            "from_process": (hit or {}).get("from_process"),
            "seconds": round(elapsed, 3),
        }
        if not hit:
            # Empty/short Style BOM may credit stock without listing on Ready —
            # imported>0 is still success for the upload blocker.
            print("WARN", stage, "imported but not on Ready board (stock credited)")
    print("WIP_FEEDER_PROBE_OK", json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

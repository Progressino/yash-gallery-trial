#!/usr/bin/env python3
"""Prod probe: PATCH responsibility with full edit fields and verify persistence."""
from __future__ import annotations

import json
import os
import sys
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
    print("HEALTH", h.json().get("git_sha"), "warm", h.json().get("warm_cache"))

    user, pw = _env_auth()
    r = s.post(
        f"{BASE}/api/auth/login",
        json={"username": user, "password": pw},
        headers={"X-Device-Id": "gha-hrm-edit-probe"},
        timeout=60,
    )
    print("LOGIN", r.status_code)
    r.raise_for_status()

    # Frontend bundle must expose edit labels (cache-bust via asset map).
    idx = s.get(f"{BASE.replace(':8000','')}/" if False else "https://app.progressino.com/", timeout=30)
    # Prefer loopback frontend if available; else public.
    for base_ui in ("http://127.0.0.1", "https://app.progressino.com"):
        try:
            idx = s.get(base_ui + "/", timeout=20)
            if idx.ok and "index-" in idx.text:
                break
        except Exception:
            continue
    import re

    m = re.search(r"assets/(index-[A-Za-z0-9_-]+\.js)", idx.text or "")
    if not m:
        print("WARN: could not find index asset")
    else:
        index_js = s.get(f"https://app.progressino.com/assets/{m.group(1)}", timeout=60).text
        hm = re.search(r'HRM-[A-Za-z0-9_-]+\.js', index_js)
        if not hm:
            raise SystemExit("HRM chunk not found in index")
        hrm_js = s.get(f"https://app.progressino.com/assets/{hm.group(0)}", timeout=60).text
        for needle in (
            "Linked To (supervisor / approver)",
            "Mandatory",
            "Anchor month",
            "Day of month",
            "Select a weekday for Weekly",
        ):
            if needle not in hrm_js:
                raise SystemExit(f"HRM bundle missing edit UI string: {needle}")
            print("BUNDLE_OK", needle)

    resps = s.get(f"{BASE}/api/hrm/responsibilities", timeout=60)
    print("RESPS", resps.status_code)
    resps.raise_for_status()
    rows = resps.json() if isinstance(resps.json(), list) else (resps.json() or {}).get("rows") or []
    if not rows:
        raise SystemExit("No responsibilities to edit-test")
    row = rows[0]
    rid = int(row["id"])
    print("TARGET", rid, row.get("title"), "freq", row.get("frequency"))

    original = {
        "title": row.get("title"),
        "description": row.get("description") or "",
        "frequency": row.get("frequency") or "Daily",
        "category": row.get("category") or "General",
        "employee_id": row.get("employee_id"),
        "linked_to_employee_id": row.get("linked_to_employee_id"),
        "priority": row.get("priority") or "Medium",
        "mandatory": bool(row.get("mandatory")),
        "schedule_weekday": row.get("schedule_weekday") or "",
        "schedule_month_day": int(row.get("schedule_month_day") or 0),
        "schedule_month": int(row.get("schedule_month") or 0),
        "time_period": row.get("time_period") or "",
    }

    emps = s.get(f"{BASE}/api/hrm/employees", timeout=30)
    emps.raise_for_status()
    emp_list = emps.json() if isinstance(emps.json(), list) else []
    link_id = None
    for e in emp_list:
        if int(e["id"]) != int(original["employee_id"] or 0):
            link_id = int(e["id"])
            break

    probe_payload = {
        **original,
        "frequency": "Weekly",
        "schedule_weekday": "Wednesday",
        "schedule_month_day": 0,
        "schedule_month": 0,
        "mandatory": True,
        "priority": "High",
        "time_period": "Morning",
        "linked_to_employee_id": link_id,
    }
    patch = s.patch(f"{BASE}/api/hrm/responsibilities/{rid}", json=probe_payload, timeout=60)
    print("PATCH", patch.status_code, (patch.text or "")[:200])
    patch.raise_for_status()

    again = s.get(f"{BASE}/api/hrm/responsibilities", timeout=60)
    again.raise_for_status()
    rows2 = again.json() if isinstance(again.json(), list) else (again.json() or {}).get("rows") or []
    updated = next(r for r in rows2 if int(r["id"]) == rid)
    checks = {
        "frequency": updated.get("frequency") == "Weekly",
        "schedule_weekday": (updated.get("schedule_weekday") or "") == "Wednesday",
        "mandatory": bool(updated.get("mandatory")) is True,
        "priority": (updated.get("priority") or "") == "High",
        "time_period": (updated.get("time_period") or "") == "Morning",
        "linked_to": (
            link_id is None
            or int(updated.get("linked_to_employee_id") or 0) == int(link_id)
        ),
    }
    print("VERIFY", json.dumps(checks), "linked", updated.get("linked_to_employee_id"), updated.get("linked_to_employee_name"))
    if not all(checks.values()):
        # restore then fail
        s.patch(f"{BASE}/api/hrm/responsibilities/{rid}", json=original, timeout=60)
        raise SystemExit(f"Edit fields did not persist: {checks}")

    # Restore original so we don't leave probe mutations
    restore = s.patch(f"{BASE}/api/hrm/responsibilities/{rid}", json=original, timeout=60)
    print("RESTORE", restore.status_code)
    restore.raise_for_status()
    print("HRM_EDIT_PROBE_OK", json.dumps({"id": rid, "checks": checks}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

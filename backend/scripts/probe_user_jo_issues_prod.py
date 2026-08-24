#!/usr/bin/env python3
"""Prod probe: JO search beyond page 150, Cut-to-Pack visibility, multi-size Finishing create."""
from __future__ import annotations

import sys

import requests


def _env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in open("/root/app/.env", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = _env()
    user, pw = env.get("AUTH_USERNAME"), env.get("AUTH_PASSWORD")
    if not user or not pw:
        print("missing AUTH_USERNAME/PASSWORD")
        return 2
    base = "http://127.0.0.1:8000"
    s = requests.Session()
    r = s.post(
        f"{base}/api/auth/login",
        json={"username": user, "password": pw},
        headers={"X-Device-Id": "probe-user-jo-issues"},
        timeout=60,
    )
    r.raise_for_status()
    print("LOGIN ok", r.json().get("username"))
    print("HEALTH", s.get(f"{base}/api/health", timeout=20).json().get("git_sha"))

    j2751 = s.get(
        f"{base}/api/production/orders",
        params={"process": "Cutting", "light": 1, "limit": 50, "q": "PJO-2751"},
        timeout=60,
    )
    print(
        "search_PJO-2751",
        [
            (x.get("jo_number"), x.get("production_mode"), x.get("sku"))
            for x in (j2751.json() if j2751.ok else [])
        ],
    )

    page150 = s.get(
        f"{base}/api/production/orders",
        params={"process": "Cutting", "light": 1, "limit": 150},
        timeout=90,
    ).json()
    deep = s.get(
        f"{base}/api/production/orders",
        params={"process": "Cutting", "light": 1, "limit": 1, "offset": 400},
        timeout=90,
    ).json()
    if deep:
        target = deep[0]
        qsku = (target.get("sku") or "")[:12]
        found = s.get(
            f"{base}/api/production/orders",
            params={"process": "Cutting", "light": 1, "limit": 5000, "q": qsku},
            timeout=90,
        ).json()
        print(
            "deep_sku_search",
            qsku,
            "jo",
            target.get("jo_number"),
            "in_first_150",
            target["jo_number"] in {x["jo_number"] for x in page150},
            "matches",
            len(found),
            "has_target",
            any(x["jo_number"] == target["jo_number"] for x in found),
        )

    ready = s.get(
        f"{base}/api/production/ready-to-process/Finishing",
        params={"q": "1785YKRED"},
        timeout=90,
    )
    rows = [r0 for r0 in (ready.json() if ready.ok else []) if str(r0.get("so_number")) == "01-2627"]
    print("ready_fin_01-2627", len(rows))
    for r0 in rows[:10]:
        print(
            " ",
            r0.get("sku"),
            r0.get("available_qty") or r0.get("qty"),
            "already",
            r0.get("already_planned"),
        )

    so, sku = "01-2627", "1785YKRED-XXL"
    v_sum = s.get(
        f"{base}/api/production/orders/validate",
        params={"process": "Finishing", "so_number": so, "sku": sku, "planned_qty": 427},
        timeout=30,
    )
    print("validate_sum_427", v_sum.json() if v_sum.ok else v_sum.text[:200])
    lines = []
    for sk, pq in [
        ("1785YKRED-XXL", 119),
        ("1785YKRED-XL", 116),
        ("1785YKRED-L", 79),
        ("1785YKRED-S", 48),
        ("1785YKRED-5XL", 65),
    ]:
        v = s.get(
            f"{base}/api/production/orders/validate",
            params={"process": "Finishing", "so_number": so, "sku": sk, "planned_qty": pq},
            timeout=30,
        )
        body = v.json() if v.ok else {}
        print(f"validate_{sk}", body)
        lines.append({"sku": sk, "planned_qty": pq})

    bad = s.post(
        f"{base}/api/production/orders",
        json={
            "so_number": so,
            "sku": sku,
            "process": "Finishing",
            "planned_qty": 427,
            "exec_type": "Outsource",
            "vendor_name": "PROBE-DO-NOT-USE",
            "lines": [],
        },
        timeout=60,
    )
    print("post_empty_lines", bad.status_code, (bad.text or "")[:220])

    good = s.post(
        f"{base}/api/production/orders",
        json={
            "so_number": so,
            "sku": sku,
            "process": "Finishing",
            "planned_qty": 427,
            "exec_type": "Outsource",
            "vendor_name": "PROBE-MULTI-SIZE",
            "lines": lines,
            "remarks": "AUTO-PROBE-DELETE-ME",
        },
        timeout=60,
    )
    print("post_with_lines", good.status_code, (good.text or "")[:400])
    if good.status_code == 200:
        jid = good.json().get("id")
        jn = good.json().get("jo_number")
        print("CREATED", jn, jid)
        if jid:
            c = s.patch(
                f"{base}/api/production/orders/{jid}",
                json={"status": "Cancelled"},
                timeout=30,
            )
            print("cancel", c.status_code, (c.text or "")[:160])
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

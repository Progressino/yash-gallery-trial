#!/usr/bin/env python3
"""Smoke-test daily + inventory upload paths on production (server-side)."""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
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
        "Before:",
        "daily_ingest=",
        cov.get("daily_auto_ingest_status"),
        "inv_upload=",
        cov.get("inventory_upload_status"),
        "daily_inv=",
        cov.get("daily_inventory_upload_status"),
    )

    # 1) Tiny daily CSV (direct, not chunked)
    qty = int(time.time()) % 1000 + 50
    sale_day = (date.today() - timedelta(days=1)).strftime("%d-%b-%Y")
    csv_body = (
        "Reason for Credit Entry,Sub Order No,Order Date,SKU,Quantity\n"
        f"DELIVERED,SMOKE-{qty},{sale_day},SMOKE-TEST-SKU,{qty}\n"
    ).encode("utf-8")
    fname = f"Meesho Orders {(date.today() - timedelta(days=1)).strftime('%d-%b-%y')}.csv"
    t0 = time.time()
    r = sess.post(
        f"{BASE}/api/upload/daily-auto",
        files={"files": (fname, csv_body, "text/csv")},
        timeout=120,
    )
    daily_elapsed = round(time.time() - t0, 2)
    daily = r.json()
    print(f"Daily-auto: status={r.status_code} ok={daily.get('ok')} elapsed={daily_elapsed}s")
    if not daily.get("ok"):
        print(json.dumps(daily, indent=2))
        return 1

    # Wait for ingest (max 3 min)
    for _ in range(60):
        cov = coverage(sess)
        ingest = cov.get("daily_auto_ingest_status") or "idle"
        rebuild = cov.get("sales_rebuild") or "idle"
        if ingest not in ("running",) and rebuild not in ("running",):
            break
        time.sleep(3)
    print(
        "After daily:",
        "ingest=",
        cov.get("daily_auto_ingest_status"),
        "rebuild=",
        cov.get("sales_rebuild"),
        "sales_rows=",
        cov.get("sales_rows"),
    )

    # 2) Chunk init for daily (policy + session)
    r = sess.post(
        f"{BASE}/api/upload/chunk/init",
        json={
            "target": "daily-auto",
            "files": [{"name": fname, "size": len(csv_body)}],
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

    # 3) Tiny wide inventory history CSV (admin PO baseline)
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26
SMOKE-SKU-A,10,10,5,8,12
"""
    t0 = time.time()
    r = sess.post(
        f"{BASE}/api/po/daily-inventory-history",
        files={"file": ("smoke-inventory-matrix.csv", csv.encode(), "text/csv")},
        timeout=120,
    )
    inv_hist = r.json()
    print(f"Daily-inv-history POST: status={r.status_code} ok={inv_hist.get('ok')} elapsed={round(time.time()-t0,2)}s")
    if not inv_hist.get("ok"):
        print(json.dumps(inv_hist, indent=2))
        return 3

    for _ in range(90):
        st = sess.get(f"{BASE}/api/po/daily-inventory-history/upload-status", timeout=30).json()
        status = st.get("status") or "idle"
        if status == "done":
            print(f"Daily-inv-history done: rows={st.get('rows')} skus={st.get('skus')} elapsed={round(time.time()-t0,2)}s")
            break
        if status == "error":
            print("Daily-inv-history error:", st.get("message"))
            return 3
        time.sleep(2)
    else:
        print("Daily-inv-history timed out")
        return 3

    print("OK — daily upload + chunk policy + inventory history smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

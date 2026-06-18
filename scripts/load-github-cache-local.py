#!/usr/bin/env python3
"""Local helper: login + POST /api/cache/load and print coverage."""
from __future__ import annotations

import sys
import time

import requests
from dotenv import dotenv_values

BASE = "http://127.0.0.1:8000"
cfg = dotenv_values(".env")
user = (cfg.get("AUTH_USERNAME") or "admin").strip()
password = (cfg.get("AUTH_PASSWORD") or "").strip()
if not password:
    print("AUTH_PASSWORD missing in .env", file=sys.stderr)
    sys.exit(1)

s = requests.Session()
r = s.post(f"{BASE}/api/auth/login", json={"username": user, "password": password}, timeout=60)
r.raise_for_status()
body = r.json()
if body.get("otp_required"):
    print("OTP required — set SUPER_ADMIN_OTP_BYPASS=1 and restart backend", file=sys.stderr)
    sys.exit(1)
if not body.get("ok"):
    print("Login failed:", body, file=sys.stderr)
    sys.exit(1)
print("Logged in as", body.get("username"), flush=True)

print("Reloading from GitHub (full download, may take 3–10 minutes)…", flush=True)
r = s.post(f"{BASE}/api/cache/reload-fresh", timeout=900)
print("reload-fresh:", r.status_code, r.text[:800], flush=True)

for i in range(120):
    c = s.get(f"{BASE}/api/data/coverage", params={"light": "1"}, timeout=180).json()
    loaded = sum(
        1
        for k in ("sku_mapping", "mtr", "sales", "inventory", "myntra", "meesho", "flipkart")
        if c.get(k)
    )
    inv = c.get("inventory")
    print(f"  [{i+1}] coverage {loaded}/7 inventory={inv} sales_rows={c.get('sales_rows')}")
    if inv and loaded >= 7:
        print("OK — inventory loaded, 7/7 ready for PO")
        sys.exit(0)
    time.sleep(3)

print("Timed out — check Upload page or backend logs", file=sys.stderr)
sys.exit(1)

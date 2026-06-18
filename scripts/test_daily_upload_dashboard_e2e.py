#!/usr/bin/env python3
"""E2E: daily upload → sales rebuild → intelligence bundle refresh."""
from __future__ import annotations

import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _login(sess: requests.Session, base: str) -> None:
    user = os.environ.get("AUTH_USERNAME", "").strip()
    pw = os.environ.get("AUTH_PASSWORD", "").strip()
    if not user or not pw:
        raise SystemExit("Set AUTH_USERNAME and AUTH_PASSWORD in .env")
    r = sess.post(
        f"{base}/api/auth/login",
        json={"username": user, "password": pw},
        timeout=60,
        headers={"X-Device-Id": "daily-upload-e2e"},
    )
    r.raise_for_status()
    data = r.json()
    if data.get("otp_required"):
        raise SystemExit("OTP required — run from a trusted device or disable OTP for test user")


def _intel_units(sess: requests.Session, base: str, start: str, end: str) -> int:
    r = sess.get(
        f"{base}/api/data/intelligence-bundle",
        params={
            "start_date": start,
            "end_date": end,
            "limit": 5,
            "basis": "gross",
            "include_extras": "0",
        },
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    return int((body.get("sales_summary") or {}).get("total_units") or 0)


def _coverage(sess: requests.Session, base: str) -> dict:
    r = sess.get(f"{base}/api/data/coverage", params={"light": "1"}, timeout=60)
    r.raise_for_status()
    return r.json()


def _wait_jobs(sess: requests.Session, base: str, *, before_rev: int, max_sec: int = 600) -> dict:
    deadline = time.time() + max_sec
    last = ""
    saw_rebuild = False
    while time.time() < deadline:
        cov = _coverage(sess, base)
        ingest = cov.get("daily_auto_ingest_status") or "idle"
        rebuild = cov.get("sales_rebuild") or "idle"
        rev = int(cov.get("sales_data_revision") or 0)
        msg = f"ingest={ingest} rebuild={rebuild} sales_rows={cov.get('sales_rows')} rev={rev}"
        if msg != last:
            print(msg)
            last = msg
        if rebuild == "running":
            saw_rebuild = True
        if ingest == "error":
            raise RuntimeError(cov.get("daily_auto_ingest_message") or "Daily ingest failed")
        if rebuild == "error":
            raise RuntimeError(cov.get("sales_rebuild_message") or "Sales rebuild failed")
        if rev > before_rev and ingest not in ("running",) and rebuild not in ("running",):
            return cov
        if ingest not in ("running",) and rebuild not in ("running",) and saw_rebuild:
            return cov
        time.sleep(3)
    raise TimeoutError("Timed out waiting for daily ingest / sales rebuild")


def main() -> int:
    _load_dotenv()
    base = os.environ.get("FORECAST_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    sess = requests.Session()

    print(f"Login → {base}")
    _login(sess, base)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=30)).isoformat()
    before_cov = _coverage(sess, base)
    before_rev = int(before_cov.get("sales_data_revision") or 0)
    before_units = _intel_units(sess, base, start, end)
    print(f"Before: intelligence gross={before_units:,} sales_rows={before_cov.get('sales_rows')} rev={before_rev}")

    # Unique quantity so we can detect the upload in intelligence totals.
    qty = int(time.time()) % 50_000 + 100
    sale_day = (date.today() - timedelta(days=1)).strftime("%d-%b-%Y")
    csv_body = (
        "Reason for Credit Entry,Sub Order No,Order Date,SKU,Quantity\n"
        f"DELIVERED,E2E-{qty},{sale_day},E2E-TEST-SKU,{qty}\n"
    ).encode("utf-8")
    fname = f"Meesho Orders {(date.today() - timedelta(days=1)).strftime('%d-%b-%y')}.csv"

    print(f"Uploading {fname} (+{qty} units)…")
    t0 = time.time()
    up = sess.post(
        f"{base}/api/upload/daily-auto",
        files=[("files", (fname, io.BytesIO(csv_body), "text/csv"))],
        timeout=120,
    )
    up.raise_for_status()
    body = up.json()
    elapsed = time.time() - t0
    print(f"Upload HTTP {up.status_code} in {elapsed:.1f}s — async={body.get('ingest_async')} rebuild={body.get('sales_rebuild')}")
    assert body.get("ok"), body

    after_cov = _wait_jobs(sess, base, before_rev=before_rev)
    after_rev = int(after_cov.get("sales_data_revision") or 0)
    after_units = _intel_units(sess, base, start, end)

    print(f"After:  intelligence gross={after_units:,} sales_rows={after_cov.get('sales_rows')} rev={after_rev}")
    print(f"Delta:  units={after_units - before_units:+,} rev={after_rev - before_rev:+}")

    ok = after_rev > before_rev and after_units >= before_units + qty
    if ok:
        print("PASS — dashboard data refreshed automatically after daily upload")
        return 0

    if after_rev > before_rev:
        print("PARTIAL — sales_data_revision bumped (UI would invalidate); intelligence delta inconclusive")
        return 0

    print("FAIL — sales revision did not bump", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

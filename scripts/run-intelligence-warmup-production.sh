#!/usr/bin/env bash
# Post-deploy: warm the Intelligence dashboard (UI 7D / 30D presets) via the live API.
# Avoid docker-exec'ing a second Python that reloads /data/warm_cache while uvicorn
# already holds it — that double-loads and OOM-kills the 7GB VPS (exit 137).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="${INTELLIGENCE_WARMUP_BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_SEC="${INTELLIGENCE_WARMUP_TIMEOUT_SEC:-600}"

echo "==> Intelligence warmup via API (${BASE})"

for i in $(seq 1 60); do
  if curl -sf --max-time 5 "${BASE}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if ! curl -sf --max-time 5 "${BASE}/api/health" >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping Intelligence warmup"
  exit 0
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

export INTELLIGENCE_WARMUP_BASE_URL="$BASE"
export INTELLIGENCE_WARMUP_TIMEOUT_SEC="$TIMEOUT_SEC"

set +e
python3 <<'PY'
import os, sys, time
from datetime import datetime, timedelta, timezone

import requests

base = os.environ["INTELLIGENCE_WARMUP_BASE_URL"].rstrip("/")
deadline = time.time() + int(os.environ.get("INTELLIGENCE_WARMUP_TIMEOUT_SEC") or 600)
user = os.environ.get("AUTH_USERNAME") or ""
pw = os.environ.get("AUTH_PASSWORD") or ""
if not user or not pw:
    print("WARN: AUTH_USERNAME/PASSWORD missing — skip Intelligence warmup")
    sys.exit(0)

s = requests.Session()
for _ in range(90):
    try:
        h = s.get(f"{base}/api/health", timeout=15)
        if h.ok and h.json().get("warm_cache"):
            break
    except Exception as e:
        print("health wait", e, flush=True)
    time.sleep(2)

r = s.post(
    f"{base}/api/auth/login",
    json={"username": user, "password": pw},
    headers={"X-Device-Id": "gha-intelligence-warmup"},
    timeout=60,
)
print("login", r.status_code, flush=True)
r.raise_for_status()

# Same start offsets as the Dashboard presets (IST): 7D = today-6, 30D = today-30.
today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
ok = True
for label, days in (("7D", 6), ("30D", 30)):
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()
    bundle = {"start_date": start, "end_date": end, "limit": 10, "basis": "gross", "include_extras": 0}
    for path, params in (
        ("/api/data/dashboard/summary", {"start_date": start, "end_date": end, "limit": 10}),
        ("/api/data/intelligence-bundle", {**bundle, "mode": "fast"}),
        ("/api/data/intelligence-bundle", {**bundle, "mode": "full"}),
    ):
        left = deadline - time.time()
        if left <= 5:
            print("WARN: warmup timeout reached", flush=True)
            sys.exit(1)
        t0 = time.time()
        try:
            resp = s.get(f"{base}{path}", params=params, timeout=min(left, 300))
            body = resp.json()
        except Exception as e:
            print(f"WARN {label} {path}: {e}", flush=True)
            ok = False
            continue
        units = (body.get("sales_summary") or {}).get("total_units")
        print(f"{label} {path} {params.get('mode', '')}: http={resp.status_code} {time.time()-t0:.1f}s units={units} "
              f"empty_window={body.get('empty_window', False)}", flush=True)
        ok = ok and resp.ok
sys.exit(0 if ok else 1)
PY
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "OK: Intelligence warmup finished"
else
  echo "WARN: Intelligence warmup exited ${rc} — dashboard will build on first request"
fi
exit 0

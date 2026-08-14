#!/usr/bin/env bash
# Post-deploy: pre-compute shared PO cache via the live API (one process / warm cache).
# Avoid docker-exec'ing a second Python that reloads /data/warm_cache while uvicorn
# already holds it — that double-loads and OOM-kills the 7GB VPS (exit 137).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMEOUT_SEC="${PO_WARMUP_TIMEOUT_SEC:-90}"
BASE="${PO_WARMUP_BASE_URL:-http://127.0.0.1:8000}"

echo "==> PO shared-cache warmup via API (${BASE})"

echo "==> Waiting for backend health…"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 "${BASE}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -sf --max-time 5 "${BASE}/api/health" >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping PO warmup"
  exit 0
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

export PO_WARMUP_BASE_URL="$BASE"
export PO_WARMUP_TIMEOUT_SEC="$TIMEOUT_SEC"

python3 <<'PY'
import json, os, sys, time
import requests

base = os.environ.get("PO_WARMUP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
timeout_total = int(os.environ.get("PO_WARMUP_TIMEOUT_SEC") or 2400)
user = os.environ.get("AUTH_USERNAME") or ""
pw = os.environ.get("AUTH_PASSWORD") or ""
if not user or not pw:
    print("WARN: AUTH_USERNAME/PASSWORD missing — skip PO warmup")
    sys.exit(0)

# UI-matching fingerprints (order: most used first). One full calc fills shared cache
# for that fingerprint; later API Calculate PO returns in seconds.
PROFILES = (
    {
        "label": "po_fresh_default",
        "period_days": 30,
        "lead_time": 60,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "use_ly_fallback": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 45,
        "inventory_history_channel": "oms",
        "use_oms_inventory_only": False,
    },
    {
        "label": "ui_default_30",
        "period_days": 30,
        "lead_time": 45,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": False,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 14,
        "inventory_history_channel": "oms",
        "use_oms_inventory_only": False,
    },
)

s = requests.Session()
# Wait until warm_cache is true (deploy hydrate may still be running).
for i in range(90):
    try:
        h = s.get(f"{base}/api/health", timeout=15)
        if h.ok and h.json().get("warm_cache"):
            print("health warm_cache=true", flush=True)
            break
    except Exception as e:
        print("health wait", e, flush=True)
    time.sleep(2)
else:
    print("WARN: warm_cache never became true — continuing anyway", flush=True)

r = s.post(
    f"{base}/api/auth/login",
    json={"username": user, "password": pw},
    headers={"X-Device-Id": "gha-po-shared-warmup"},
    timeout=60,
)
print("login", r.status_code, flush=True)
r.raise_for_status()

hy = s.post(f"{base}/api/cache/hydrate-warm", timeout=600)
print("hydrate", hy.status_code, (hy.text or "")[:200], flush=True)
hy.raise_for_status()

deadline = time.time() + timeout_total
saved = []

for prof in PROFILES:
    if time.time() >= deadline:
        print("WARN: overall timeout before all profiles", flush=True)
        break
    label = prof["label"]
    body = {k: v for k, v in prof.items() if k != "label"}
    body["planning_date"] = time.strftime("%Y-%m-%d")
    body["raise_view_date"] = body["planning_date"]
    # First build forces a real calc so the shared parquet is created.
    body["use_shared_cache"] = False
    print("POST calculate", label, flush=True)
    post = s.post(f"{base}/api/po/calculate", json=body, timeout=120)
    print("POST", label, post.status_code, (post.text or "")[:240], flush=True)
    if post.status_code >= 400:
        print(f"WARN: profile {label} post failed — skip rest", flush=True)
        break
    data = post.json()
    job_id = data.get("job_id")
    done = False
    while time.time() < deadline:
        path = f"/api/po/calculate/status/{job_id}" if job_id else "/api/po/calculate/status"
        try:
            st = s.get(f"{base}{path}", timeout=45)
        except Exception as e:
            print("status transient", e, flush=True)
            time.sleep(5)
            continue
        if st.status_code >= 500:
            time.sleep(5)
            continue
        payload = st.json() if st.ok else {}
        status = payload.get("status")
        print(f"  {label}: {status} p={payload.get('progress')} {payload.get('message','')[:80]}", flush=True)
        if status == "done":
            # Confirm a page of results so spill + session are warm.
            res = s.get(
                f"{base}/api/po/calculate/result" + (f"/{job_id}" if job_id else ""),
                params={"offset": 0, "limit": 3, "compact": 1},
                timeout=90,
            )
            print("  result", res.status_code, (res.text or "")[:180], flush=True)
            if res.ok and res.json().get("ok"):
                saved.append({"label": label, "total": res.json().get("total"), "job_id": job_id})
                done = True
            break
        if status == "error":
            print("WARN: calc error", payload, flush=True)
            break
        time.sleep(4)
    if not done:
        print(f"WARN: profile {label} did not finish", flush=True)
        break

print(json.dumps({"ok": bool(saved), "profiles_saved": saved}, indent=2), flush=True)
sys.exit(0 if saved else 1)
PY
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "OK: PO shared-cache warmup finished"
else
  echo "WARN: PO warmup exited ${rc} — operators can still Calculate PO manually"
  exit 0
fi

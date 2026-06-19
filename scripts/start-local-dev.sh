#!/usr/bin/env bash
# Start local backend + frontend for PO Fresh testing.
# Do NOT `source .env` in shell — bcrypt hash contains $ and breaks login.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/.local-data/warm_cache" "$ROOT/.local-data/github_cache"

if ! [[ -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

# PostgreSQL — same architecture as production (normalized ops tables + shared frames).
bash "$ROOT/scripts/ensure-local-postgres.sh"

export LOCAL_DEV=1
export WARM_CACHE_DIR="$ROOT/.local-data/warm_cache"
export GITHUB_BLOB_CACHE_DIR="$ROOT/.local-data/github_cache"
export DAILY_SALES_DB="$ROOT/daily_sales.db"
export WARM_CACHE_PO_SESSION_ONLY=1
export SUPER_ADMIN_OTP_BYPASS=1
export FORECAST_SESSION_DATABASE_URL="${FORECAST_SESSION_DATABASE_URL:-postgresql://forecast:forecast@127.0.0.1:5433/forecast}"
export FORECAST_OPS_NORMALIZED="${FORECAST_OPS_NORMALIZED:-1}"
export SESSION_SHARED_FRAMES="${SESSION_SHARED_FRAMES:-1}"
export DAILY_SALES_BACKEND="${DAILY_SALES_BACKEND:-postgres}"
# Perf logging only — does not change PO math.
export PERF_PO_STAGES=1
export DB_SLOW_QUERY_LOG=0

echo "Starting backend on http://127.0.0.1:8000"
echo "  WARM_CACHE_DIR=$WARM_CACHE_DIR"
echo "  FORECAST_SESSION_DATABASE_URL=$FORECAST_SESSION_DATABASE_URL"
echo "  LOCAL_DEV=1 (sync warm-cache hydrate on login)"
"$ROOT/.venv/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACK_PID=$!

cleanup() {
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 3
if curl -sf http://127.0.0.1:8000/api/health >/dev/null; then
  echo "Backend health OK"
  curl -s http://127.0.0.1:8000/api/health | "$ROOT/.venv/bin/python" -m json.tool 2>/dev/null | head -20 || true
else
  echo "WARN: backend not responding yet on :8000"
fi

echo ""
echo "Starting frontend on http://127.0.0.1:5173 …"
echo "Open: http://127.0.0.1:5173/  (Dashboard)  http://127.0.0.1:5173/po-fresh"
cd "$ROOT/frontend"
exec npm run dev -- --host 127.0.0.1 --port 5173

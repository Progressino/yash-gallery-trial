#!/usr/bin/env bash
# Start local backend + frontend for PO Fresh testing.
# Do NOT `source .env` in shell — bcrypt hash contains $ and breaks login.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/.local-data/warm_cache"

if ! [[ -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

export WARM_CACHE_DIR="$ROOT/.local-data/warm_cache"
export GITHUB_BLOB_CACHE_DIR="$ROOT/.local-data/github_cache"
export SUPER_ADMIN_OTP_BYPASS=1

echo "Starting backend on http://127.0.0.1:8000 (WARM_CACHE_DIR=$WARM_CACHE_DIR)…"
"$ROOT/.venv/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACK_PID=$!

cleanup() {
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
echo "Starting frontend on http://127.0.0.1:5173 …"
echo "Open: http://127.0.0.1:5173/po-fresh"
cd "$ROOT/frontend"
exec npm run dev -- --host 127.0.0.1 --port 5173

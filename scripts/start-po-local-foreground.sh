#!/usr/bin/env bash
# Run backend + frontend in this terminal — keeps servers alive until you close the window.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/local-dev/logs"
PID_DIR="$ROOT/local-dev/pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "$ROOT/.local-data/warm_cache"

export WARM_CACHE_DIR="$ROOT/.local-data/warm_cache"
export GITHUB_BLOB_CACHE_DIR="$ROOT/.local-data/github_cache"
export WARM_CACHE_PO_SESSION_ONLY=1

stop_port() {
  local port=$1
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Stopping existing process on port $port"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  echo ""
  echo "Stopping PO Engine servers…"
  jobs -p 2>/dev/null | xargs kill 2>/dev/null || true
  stop_port 8000
  stop_port 5173
  rm -f "$PID_DIR/backend.pid" "$PID_DIR/frontend.pid" "$PID_DIR/stack.pids"
}
trap cleanup EXIT INT TERM

if ! [[ -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv — run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-test.txt"
  exit 1
fi

stop_port 8000
stop_port 5173

cd "$ROOT"
echo "Starting backend on http://localhost:8000 …"
"$ROOT/.venv/bin/python" -m uvicorn backend.main:app \
  --host 0.0.0.0 --port 8000 \
  >>"$LOG_DIR/backend.log" 2>&1 &
BACK_PID=$!
echo "$BACK_PID" >"$PID_DIR/backend.pid"

cd "$ROOT/frontend"
echo "Starting frontend on http://localhost:5173 …"
npm run dev -- --host 0.0.0.0 --port 5173 \
  >>"$LOG_DIR/frontend.log" 2>&1 &
FRONT_PID=$!
echo "$FRONT_PID" >"$PID_DIR/frontend.pid"
echo "$BACK_PID $FRONT_PID" >"$PID_DIR/stack.pids"

for i in {1..30}; do
  if curl -sf http://localhost:8000/api/health >/dev/null 2>&1 \
    && curl -sf http://localhost:5173/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo ""
echo "=== PO Engine is running ==="
echo "  Open:  http://localhost:5173/po-fresh"
echo "  Logs:  $LOG_DIR/"
echo ""
echo "  KEEP THIS WINDOW OPEN while testing."
echo "  Close the window or press Ctrl+C to stop."
echo ""

wait "$BACK_PID" "$FRONT_PID"

#!/usr/bin/env bash
# Start Vite frontend only (port 5173).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/local-dev/logs"
PID_DIR="$ROOT/local-dev/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

stop_port() {
  local port=$1
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
  fi
}

stop_port 5173
stop_port 5174

echo "Starting frontend → $LOG_DIR/frontend.log"
nohup bash -c "cd \"$ROOT/frontend\" && npm run dev -- --host 0.0.0.0 --port 5173" \
  >>"$LOG_DIR/frontend.log" 2>&1 </dev/null &
FRONT_PID=$!
echo "$FRONT_PID" >"$PID_DIR/frontend.pid"

for i in {1..25}; do
  if curl -sf --max-time 3 http://127.0.0.1:5173/ >/dev/null 2>&1; then
    echo "Frontend ready (pid $FRONT_PID) — http://localhost:5173/po-fresh"
    exit 0
  fi
  sleep 1
done

echo "Frontend failed to start — check $LOG_DIR/frontend.log" >&2
tail -15 "$LOG_DIR/frontend.log" 2>/dev/null || true
exit 1

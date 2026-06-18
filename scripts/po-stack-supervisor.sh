#!/usr/bin/env bash
# Long-running supervisor — keeps backend + frontend alive. Run from Terminal
# (double-click local-dev/Start PO Engine.command).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/local-dev/logs"
PID_DIR="$ROOT/local-dev/pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "$ROOT/.local-data/warm_cache"

export WARM_CACHE_DIR="$ROOT/.local-data/warm_cache"
export GITHUB_BLOB_CACHE_DIR="$ROOT/.local-data/github_cache"
export WARM_CACHE_PO_SESSION_ONLY=1

BACK_PID=""
FRONT_PID=""

stop_children() {
  [[ -n "$BACK_PID" ]] && kill "$BACK_PID" 2>/dev/null || true
  [[ -n "$FRONT_PID" ]] && kill "$FRONT_PID" 2>/dev/null || true
  for port in 8000 5173 5174; do
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
  done
}

start_backend() {
  echo "$(date '+%F %T') starting backend" | tee -a "$LOG_DIR/supervisor.log"
  cd "$ROOT"
  "$ROOT/.venv/bin/python" -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000 >>"$LOG_DIR/backend.log" 2>&1 &
  BACK_PID=$!
  echo "$BACK_PID" >"$PID_DIR/backend.pid"
}

start_frontend() {
  echo "$(date '+%F %T') starting frontend" | tee -a "$LOG_DIR/supervisor.log"
  cd "$ROOT/frontend"
  npm run dev -- --host 0.0.0.0 --port 5173 >>"$LOG_DIR/frontend.log" 2>&1 &
  FRONT_PID=$!
  echo "$FRONT_PID" >"$PID_DIR/frontend.pid"
}

health_ok() {
  curl -sf --max-time 4 http://127.0.0.1:8000/api/health >/dev/null 2>&1 \
    && curl -sf --max-time 4 http://127.0.0.1:5173/api/health >/dev/null 2>&1
}

trap 'echo "$(date "+%F %T") supervisor stopping"; stop_children; exit 0' INT TERM

echo $$ >"$PID_DIR/supervisor.pid"
echo "$(date '+%F %T') PO stack supervisor started (pid $$)" | tee -a "$LOG_DIR/supervisor.log"

stop_children
sleep 2
start_backend
for i in {1..45}; do curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break; sleep 1; done
# shellcheck source=po-engine-version.sh
source "$ROOT/scripts/po-engine-version.sh"
po_engine_write_src_hash 2>/dev/null || true
start_frontend
for i in {1..45}; do curl -sf http://127.0.0.1:5173/api/health >/dev/null 2>&1 && break; sleep 1; done

if health_ok; then
  echo ""
  echo "=== PO Engine running ==="
  echo "  Intelligence: http://localhost:5173/"
  echo "  PO Fresh:     http://localhost:5173/po-fresh"
  echo "  Backend:      http://localhost:8000/api/health"
  echo "  Leave this process running. Stop via Stop PO Engine.command"
  echo ""
  if [[ "${SUPERVISOR_NO_OPEN:-}" != "1" ]]; then
    open "http://localhost:5173/po-fresh" 2>/dev/null || true
  fi
else
  echo "WARN: stack not healthy yet — will keep retrying" | tee -a "$LOG_DIR/supervisor.log"
fi

while true; do
  if ! curl -sf --max-time 4 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "$(date '+%F %T') backend down — restarting" | tee -a "$LOG_DIR/supervisor.log"
    [[ -n "$BACK_PID" ]] && kill "$BACK_PID" 2>/dev/null || true
    for port in 8000; do
      pids=$(lsof -ti ":$port" 2>/dev/null || true)
      [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    done
    sleep 1
    start_backend
    sleep 5
  fi
  if ! curl -sf --max-time 4 http://127.0.0.1:5173/api/health >/dev/null 2>&1; then
    echo "$(date '+%F %T') frontend down — restarting" | tee -a "$LOG_DIR/supervisor.log"
    [[ -n "$FRONT_PID" ]] && kill "$FRONT_PID" 2>/dev/null || true
    for port in 5173 5174; do
      pids=$(lsof -ti ":$port" 2>/dev/null || true)
      [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    done
    sleep 1
    start_frontend
    sleep 5
  fi
  sleep 10
done

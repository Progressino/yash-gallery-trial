#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/local-dev/pids"
LOG_DIR="$ROOT/local-dev/logs"

if [[ -f "$PID_DIR/backend.pid" ]]; then
  kill "$(cat "$PID_DIR/backend.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/backend.pid"
fi
if [[ -f "$PID_DIR/frontend.pid" ]]; then
  kill "$(cat "$PID_DIR/frontend.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/frontend.pid"
fi
rm -f "$PID_DIR/stack.pids"

for port in 8000 5173 5174; do
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
done

if [[ -f "$PID_DIR/watchdog.pid" ]]; then
  kill "$(cat "$PID_DIR/watchdog.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/watchdog.pid"
fi
if [[ -f "$PID_DIR/supervisor.pid" ]]; then
  kill "$(cat "$PID_DIR/supervisor.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/supervisor.pid"
fi

echo "Stopped local PO stack (ports 8000, 5173, 5174)"

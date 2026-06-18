#!/usr/bin/env bash
# Start backend + frontend for PO Fresh local testing (supervisor keeps them alive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/local-dev/logs"
PID_DIR="$ROOT/local-dev/pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "$ROOT/.local-data/warm_cache"

export WARM_CACHE_DIR="$ROOT/.local-data/warm_cache"
export GITHUB_BLOB_CACHE_DIR="$ROOT/.local-data/github_cache"
export WARM_CACHE_PO_SESSION_ONLY=1
export SUPERVISOR_NO_OPEN=1

cd "$ROOT"

if ! [[ -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT/.venv — run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-test.txt"
  exit 1
fi

# Stop any previous stack (supervisor, watchdog, ports).
bash "$ROOT/scripts/stop-po-local-daemon.sh" 2>/dev/null || true
sleep 2

echo "Starting PO stack supervisor → $LOG_DIR/supervisor.log"
nohup bash "$ROOT/scripts/po-stack-supervisor.sh" >>"$LOG_DIR/supervisor.log" 2>&1 </dev/null &
SUP_PID=$!
disown -h "$SUP_PID" 2>/dev/null || true
disown "$SUP_PID" 2>/dev/null || true
echo "$SUP_PID" >"$PID_DIR/supervisor.pid"

for i in {1..60}; do
  if bash "$ROOT/scripts/test-po-stack-connection.sh" >/dev/null 2>&1; then
    echo ""
    echo "=== PO Fresh local stack ready ==="
    bash "$ROOT/scripts/test-po-stack-connection.sh"
    echo ""
    echo "  Supervisor pid: $SUP_PID"
    echo "  Logs:           $LOG_DIR/"
    echo "  Stop:           bash $ROOT/scripts/stop-po-local-daemon.sh"
    exit 0
  fi
  sleep 1
done

echo "Startup timed out — check $LOG_DIR/supervisor.log"
tail -30 "$LOG_DIR/supervisor.log" 2>/dev/null || true
exit 1

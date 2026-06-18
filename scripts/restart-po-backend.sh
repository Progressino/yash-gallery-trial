#!/usr/bin/env bash
# Restart FastAPI backend only (keeps Vite frontend running).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/local-dev/logs"
PID_DIR="$ROOT/local-dev/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# shellcheck source=po-engine-version.sh
source "$ROOT/scripts/po-engine-version.sh"

export WARM_CACHE_DIR="${WARM_CACHE_DIR:-$ROOT/.local-data/warm_cache}"
export GITHUB_BLOB_CACHE_DIR="${GITHUB_BLOB_CACHE_DIR:-$ROOT/.local-data/github_cache}"
export WARM_CACHE_PO_SESSION_ONLY="${WARM_CACHE_PO_SESSION_ONLY:-1}"

EXPECTED="$(po_engine_expected_version)"

if ! [[ -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT/.venv"
  exit 1
fi

stop_port() {
  local port=$1
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Stopping process on port $port ($pids)"
    kill $pids 2>/dev/null || true
    sleep 2
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

stop_port 8000

echo "Starting backend (PO engine v$EXPECTED) → $LOG_DIR/backend.log"
cd "$ROOT"
nohup "$ROOT/.venv/bin/python" -m uvicorn backend.main:app \
  --host 0.0.0.0 --port 8000 \
  >>"$LOG_DIR/backend.log" 2>&1 </dev/null &
BACK_PID=$!
echo "$BACK_PID" >"$PID_DIR/backend.pid"

for i in {1..30}; do
  RUNNING="$(po_engine_running_version)"
  if [[ "$RUNNING" == "$EXPECTED" ]]; then
    po_engine_write_src_hash
    echo "Backend ready — PO engine v$RUNNING (pid $BACK_PID)"
    exit 0
  fi
  sleep 1
done

echo "Backend started but health reports v$(po_engine_running_version) (expected v$EXPECTED)" >&2
tail -15 "$LOG_DIR/backend.log" 2>/dev/null || true
exit 1

#!/usr/bin/env bash
# Restart backend/frontend if they die (OOM during PO calc on large catalogs).
# Intended to run under launchd or nohup — must never exit.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/local-dev/pids/watchdog.pid"
LOG="$ROOT/local-dev/logs/watchdog.log"
mkdir -p "$ROOT/local-dev/logs" "$ROOT/local-dev/pids"

export WARM_CACHE_DIR="${WARM_CACHE_DIR:-$ROOT/.local-data/warm_cache}"
export GITHUB_BLOB_CACHE_DIR="${GITHUB_BLOB_CACHE_DIR:-$ROOT/.local-data/github_cache}"
export WARM_CACHE_PO_SESSION_ONLY="${WARM_CACHE_PO_SESSION_ONLY:-1}"

echo $$ >"$PID_FILE"
trap 'echo "$(date "+%F %T") watchdog signal — staying alive" >>"$LOG"' HUP INT TERM

health_ok() {
  curl -sf --max-time 3 http://127.0.0.1:8000/api/health >/dev/null 2>&1 \
    && curl -sf --max-time 3 http://127.0.0.1:5173/api/health >/dev/null 2>&1
}

sync_po_engine() {
  bash "$ROOT/scripts/ensure-po-engine-local.sh" >>"$LOG" 2>&1 || true
}

echo "$(date '+%F %T') watchdog started (pid $$)" >>"$LOG"
sync_po_engine

while true; do
  if ! health_ok; then
    echo "$(date '+%F %T') stack down — restarting" >>"$LOG"
    bash "$ROOT/scripts/start-po-local-daemon.sh" --no-watchdog >>"$LOG" 2>&1 || true
    sleep 8
  else
    sync_po_engine
  fi
  sleep 15
done

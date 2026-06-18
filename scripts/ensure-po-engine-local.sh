#!/usr/bin/env bash
# Align local backend with repo PO engine code + version. Restarts backend/frontend if stale.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=po-engine-version.sh
source "$ROOT/scripts/po-engine-version.sh"

EXPECTED="$(po_engine_expected_version)"
CURRENT_HASH="$(po_engine_src_hash)"
STORED_HASH="$(po_engine_stored_src_hash)"

po_engine_frontend_up() {
  curl -sf --max-time 3 http://127.0.0.1:5173/api/health >/dev/null 2>&1
}

if ! po_engine_backend_up; then
  echo "Backend down — starting full PO stack…"
  exec bash "$ROOT/scripts/start-po-local-daemon.sh"
fi

if ! po_engine_frontend_up; then
  echo "Frontend down — starting frontend…"
  bash "$ROOT/scripts/restart-po-frontend.sh"
  exit 0
fi

RUNNING="$(po_engine_running_version)"
NEED_RESTART=0
REASON=""

if [[ -n "$RUNNING" && "$RUNNING" != "$EXPECTED" ]]; then
  NEED_RESTART=1
  REASON="version mismatch (running v$RUNNING, repo v$EXPECTED)"
elif [[ "$CURRENT_HASH" != "$STORED_HASH" ]]; then
  NEED_RESTART=1
  REASON="PO engine source changed"
fi

if [[ "$NEED_RESTART" -eq 0 ]]; then
  echo "PO engine in sync — v${RUNNING:-$EXPECTED}"
  exit 0
fi

echo "Restarting backend: $REASON"
bash "$ROOT/scripts/restart-po-backend.sh"

#!/usr/bin/env bash
FORECAST="/Users/samraisinghani/Downloads/Development/forecast"
# shellcheck source=/dev/null
source "$FORECAST/scripts/po-engine-version.sh"

echo "=== PO Engine local status ==="
EXPECTED="$(po_engine_expected_version)"
echo "Repo PO engine version: v$EXPECTED"

if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
  RUNNING="$(po_engine_running_version)"
  echo "Backend:  UP  http://localhost:8000/api/health"
  curl -s http://localhost:8000/api/health
  echo ""
  if [[ "$RUNNING" == "$EXPECTED" ]]; then
    echo "Version:  OK (running v$RUNNING)"
  else
    echo "Version:  STALE (running v${RUNNING:-?}, repo v$EXPECTED)"
    echo "Fix:      bash scripts/ensure-po-engine-local.sh"
  fi
else
  echo "Backend:  DOWN"
fi
if curl -sf http://localhost:5173/api/health >/dev/null 2>&1; then
  echo "Frontend: UP  http://localhost:5173/po-fresh (API proxy OK)"
else
  echo "Frontend: DOWN"
fi
echo ""
echo "Sync:     bash $FORECAST/scripts/ensure-po-engine-local.sh"
echo "Logs:     $FORECAST/local-dev/logs/"
lsof -i :8000 -i :5173 2>/dev/null | grep LISTEN || echo "(no listeners on 8000 / 5173)"

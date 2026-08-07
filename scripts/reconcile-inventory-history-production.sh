#!/usr/bin/env bash
# Post-deploy: promote inventory matrix end date and roll forward to latest snapshot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

_dc() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

COMPOSE_FILE="${INV_RECONCILE_COMPOSE_FILE:-docker-compose.prod.yml}"
# Large history parquet + roll-forward can take a while; never hang the deploy job forever.
TIMEOUT_SEC="${INV_RECONCILE_TIMEOUT_SEC:-480}"

echo "==> Reconcile latest inventory history on disk (compose: ${COMPOSE_FILE}, timeout ${TIMEOUT_SEC}s)"

if ! curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping inventory reconcile"
  exit 0
fi

set +e
timeout "${TIMEOUT_SEC}s" _dc -p progressino -f "$COMPOSE_FILE" exec -T backend \
  python -m backend.scripts.reconcile_latest_inventory_history_disk
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "OK: inventory history reconcile finished"
elif [ "$rc" -eq 124 ]; then
  echo "WARN: inventory history reconcile timed out after ${TIMEOUT_SEC}s (non-fatal)"
else
  echo "WARN: inventory history reconcile failed with exit ${rc} (non-fatal)"
fi
exit 0

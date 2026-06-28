#!/usr/bin/env bash
# Post-deploy: re-finalize latest Existing PO on disk (dedupe + manual-raise sidecar).
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

COMPOSE_FILE="${PO_RECONCILE_COMPOSE_FILE:-docker-compose.prod.yml}"

echo "==> Reconcile latest Existing PO on disk (compose: ${COMPOSE_FILE})"

if ! curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping Existing PO reconcile"
  exit 0
fi

_dc -p progressino -f "$COMPOSE_FILE" exec -T backend \
  python -m backend.scripts.reconcile_latest_existing_po_disk \
  || echo "WARN: Existing PO reconcile failed (non-fatal)"

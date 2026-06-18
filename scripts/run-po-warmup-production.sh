#!/usr/bin/env bash
# Post-deploy: pre-compute shared PO cache for PO Fresh / PO Engine default profiles.
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

COMPOSE_FILE="${PO_WARMUP_COMPOSE_FILE:-docker-compose.prod.yml}"
TIMEOUT_SEC="${PO_WARMUP_TIMEOUT_SEC:-2400}"

echo "==> PO shared-cache warmup (compose: ${COMPOSE_FILE})"

echo "==> Waiting for backend health (warm disk cache may still be loading)…"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping PO warmup"
  exit 0
fi

echo "==> Running run_po_calculate_production (timeout ${TIMEOUT_SEC}s)…"
if timeout "${TIMEOUT_SEC}s" _dc -f "$COMPOSE_FILE" exec -T backend \
  python -m backend.scripts.run_po_calculate_production; then
  echo "OK: PO shared-cache warmup finished"
else
  rc=$?
  echo "WARN: PO warmup exited ${rc} — operators can still Calculate PO manually"
  exit 0
fi

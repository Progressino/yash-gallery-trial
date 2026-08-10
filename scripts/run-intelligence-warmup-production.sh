#!/usr/bin/env bash
# Post-deploy: pre-compute Intelligence dashboard bundle for default 30D window.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${INTELLIGENCE_WARMUP_COMPOSE_FILE:-docker-compose.prod.yml}"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-progressino}"
TIMEOUT_SEC="${INTELLIGENCE_WARMUP_TIMEOUT_SEC:-600}"

if docker compose version >/dev/null 2>&1; then
  DC=(docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE")
else
  DC=(docker-compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE")
fi

echo "==> Intelligence bundle warmup (compose: ${COMPOSE_FILE}, project: ${COMPOSE_PROJECT})"

echo "==> Waiting for backend health (warm disk cache may still be loading)…"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -sf --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "WARN: backend /api/health not ready — skipping Intelligence warmup"
  exit 0
fi

echo "==> Running precompute_intelligence_production (timeout ${TIMEOUT_SEC}s)…"
# Important: do not pass a shell function to ``timeout`` (exit 127 — command not found).
if command -v timeout >/dev/null 2>&1; then
  if timeout "${TIMEOUT_SEC}s" "${DC[@]}" exec -T \
    -e INTELLIGENCE_PRECOMPUTE_MODE=tier3_gapfill \
    backend python -m backend.scripts.precompute_intelligence_production; then
    echo "OK: Intelligence bundle warmup finished"
  else
    rc=$?
    echo "WARN: Intelligence warmup exited ${rc} — dashboard will build on first request"
    exit 0
  fi
else
  if "${DC[@]}" exec -T \
    -e INTELLIGENCE_PRECOMPUTE_MODE=tier3_gapfill \
    backend python -m backend.scripts.precompute_intelligence_production; then
    echo "OK: Intelligence bundle warmup finished"
  else
    rc=$?
    echo "WARN: Intelligence warmup exited ${rc} — dashboard will build on first request"
    exit 0
  fi
fi

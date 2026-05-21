#!/usr/bin/env bash
# Host-level fallback: probe /api/health and restart backend if down.
# Optional cron (on VPS): */5 * * * * /root/app/scripts/backend_watchdog.sh >> /var/log/backend-watchdog.log 2>&1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "ERROR: docker compose not found" >&2
  exit 1
fi

_ok() {
  curl -sf --connect-timeout 8 --max-time 15 "$1" >/dev/null 2>&1
}

for i in 1 2 3; do
  if _ok "http://127.0.0.1:8000/api/health" \
    || _ok "http://127.0.0.1/api/health"; then
    exit 0
  fi
  sleep 5
done

echo "$(date -Is) backend unhealthy — restarting"
"${DC[@]}" -f docker-compose.prod.yml restart backend
"${DC[@]}" -f docker-compose.prod.yml up -d autoheal 2>/dev/null || true

for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if _ok "http://127.0.0.1:8000/api/health"; then
    echo "$(date -Is) backend healthy after restart"
    exit 0
  fi
  sleep 5
done

echo "$(date -Is) ERROR: backend still down after restart" >&2
exit 1

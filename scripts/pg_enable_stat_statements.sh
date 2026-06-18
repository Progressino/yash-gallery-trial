#!/usr/bin/env bash
# Enable pg_stat_statements on the forecast PostgreSQL container.
# Requires docker-compose postgres command with shared_preload_libraries=pg_stat_statements
# (see docker-compose.prod.yml). Recreate postgres once after that change:
#   docker compose -f docker-compose.prod.yml up -d --force-recreate postgres
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONTAINER="${PG_CONTAINER:-app-postgres-1}"
PG_USER="${PG_USER:-forecast}"
PG_DB="${PG_DB:-forecast}"

_dc() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$PG_CONTAINER"; then
  echo "WARN: PostgreSQL container '$PG_CONTAINER' is not running — skipping pg_stat_statements."
  exit 0
fi

_preload() {
  docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -tAc "SHOW shared_preload_libraries;" 2>/dev/null \
    | tr -d '[:space:]'
}

preload="$(_preload)"
if [[ "$preload" != *pg_stat_statements* ]]; then
  echo "pg_stat_statements is not preloaded (shared_preload_libraries='$preload')."
  echo "Recreating postgres with compose config…"
  _dc -f "$ROOT/docker-compose.prod.yml" up -d --force-recreate postgres
  sleep 8
  for _ in $(seq 1 24); do
    if docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  preload="$(_preload)"
  if [[ "$preload" != *pg_stat_statements* ]]; then
    echo "ERROR: pg_stat_statements still not in shared_preload_libraries after recreate." >&2
    exit 1
  fi
fi

echo "Creating extension pg_stat_statements (if missing)…"
docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 -c \
  "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

echo "pg_stat_statements ready. Sample stats:"
docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -c \
  "SELECT COUNT(*) AS tracked_queries FROM pg_stat_statements;"

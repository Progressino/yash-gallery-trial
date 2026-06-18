#!/usr/bin/env bash
# Print the 20 slowest queries by average execution time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONTAINER="${PG_CONTAINER:-app-postgres-1}"
PG_USER="${PG_USER:-forecast}"
PG_DB="${PG_DB:-forecast}"

if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$PG_CONTAINER"; then
  echo "ERROR: PostgreSQL container '$PG_CONTAINER' is not running." >&2
  exit 1
fi

docker exec "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 \
  -f - < "$ROOT/scripts/pg_slow_queries.sql"

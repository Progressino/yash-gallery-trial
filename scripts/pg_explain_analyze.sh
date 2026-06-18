#!/usr/bin/env bash
# EXPLAIN ANALYZE on hot forecast normalized-table queries; flags Seq Scans.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONTAINER="${PG_CONTAINER:-app-postgres-1}"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-app-backend-1}"

_run_python() {
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$BACKEND_CONTAINER"; then
    docker exec "$BACKEND_CONTAINER" python3 "/app/scripts/pg_explain_analyze.py" 2>/dev/null \
      || docker exec "$BACKEND_CONTAINER" python3 "$ROOT/scripts/pg_explain_analyze.py"
    return
  fi
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    exec "$ROOT/.venv/bin/python" "$ROOT/scripts/pg_explain_analyze.py"
  fi
  exec python3 "$ROOT/scripts/pg_explain_analyze.py"
}

_run_python

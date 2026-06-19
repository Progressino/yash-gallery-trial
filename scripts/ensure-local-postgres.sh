#!/usr/bin/env bash
# Start docker PostgreSQL for local dev (port 5433) and backfill normalized tables once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PG_URL="${FORECAST_SESSION_DATABASE_URL:-postgresql://forecast:forecast@127.0.0.1:5433/forecast}"

if ! command -v docker >/dev/null 2>&1; then
  echo "WARN: docker not found — skipping local PostgreSQL"
  exit 0
fi

if ! docker compose ps postgres 2>/dev/null | grep -q "running"; then
  echo "Starting local PostgreSQL (docker compose postgres)…"
  docker compose up -d postgres
fi

echo "Waiting for PostgreSQL on 127.0.0.1:5433…"
for i in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U forecast -d forecast >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! docker compose exec -T postgres pg_isready -U forecast -d forecast >/dev/null 2>&1; then
  echo "ERROR: PostgreSQL not ready on port 5433" >&2
  exit 1
fi

export FORECAST_SESSION_DATABASE_URL="$PG_URL"
export FORECAST_OPS_NORMALIZED="${FORECAST_OPS_NORMALIZED:-1}"
export SESSION_SHARED_FRAMES="${SESSION_SHARED_FRAMES:-1}"
export DAILY_SALES_BACKEND="${DAILY_SALES_BACKEND:-postgres}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

# One-shot backfill when normalized tables are empty but warm cache exists.
NEED_BACKFILL="$("$ROOT/.venv/bin/python" <<'PY'
import os
os.environ.setdefault("FORECAST_SESSION_DATABASE_URL", "postgresql://forecast:forecast@127.0.0.1:5433/forecast")
os.environ.setdefault("FORECAST_OPS_NORMALIZED", "1")
from backend.db.forecast_ops_pg import init_db, ops_pg_enabled
from backend.db.forecast_ops_tables import normalized_tables_enabled, tables_status
init_db()
if not ops_pg_enabled() or not normalized_tables_enabled():
    print("skip")
else:
    st = tables_status()
    inv = int(st.get("inventory_lines") or 0)
    sales = int((st.get("sales_by_platform") or {}).get("unified") or 0)
    print("yes" if inv < 1000 and sales < 100_000 else "no")
PY
)"

if [[ "$NEED_BACKFILL" == "yes" ]]; then
  echo "Backfilling normalized PostgreSQL tables from warm cache…"
  WARM_CACHE_DIR="${WARM_CACHE_DIR:-$ROOT/.local-data/warm_cache}" \
    FORECAST_SESSION_DATABASE_URL="$PG_URL" \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/migrate_warm_cache_to_pg_tables.py" || true
fi

echo "Local PostgreSQL ready: $PG_URL"

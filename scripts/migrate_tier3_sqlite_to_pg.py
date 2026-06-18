#!/usr/bin/env bash
# Copy Tier-3 SQLite daily_uploads into PostgreSQL (one-shot ops migration).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
from backend.db.forecast_ops_pg import init_db, migrate_sqlite_daily_uploads_to_pg, ops_pg_enabled, pg_get_summary

init_db()
if not ops_pg_enabled():
    raise SystemExit("PostgreSQL ops not configured — set FORECAST_SESSION_DATABASE_URL")
n = migrate_sqlite_daily_uploads_to_pg()
print(f"Migrated {n} rows. PG summary: {pg_get_summary()}")
PY

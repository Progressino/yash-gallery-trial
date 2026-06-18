-- Runs once on fresh PostgreSQL data directories (docker-entrypoint-initdb.d).
-- Existing VPS clusters: scripts/pg_enable_stat_statements.sh after postgres restart.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

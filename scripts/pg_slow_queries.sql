-- Top slow queries by average execution time (PostgreSQL 16+ / pg_stat_statements).
-- Run: bash scripts/pg_slow_queries.sh
-- Or:  docker exec app-postgres-1 psql -U forecast -d forecast -f - < scripts/pg_slow_queries.sql

SELECT
    calls,
    round(mean_exec_time::numeric, 2) AS avg_ms,
    round(max_exec_time::numeric, 2) AS max_ms,
    rows,
    left(regexp_replace(query, E'[\\n\\r\\t]+', ' ', 'g'), 500) AS query
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

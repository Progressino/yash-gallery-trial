#!/usr/bin/env bash
# Weekly VPS maintenance: log disk usage, prune Docker cruft, trim old PG session blobs.
# Safe to run while the app is live (does not remove volumes or running containers).
set -euo pipefail

ROOT="${ROOT:-/root/app}"
APP_DATA="${APP_DATA_DIR:-/root/app-data}"
LOG_DIR="${APP_DATA}/maintenance"
KEEP_SESSIONS="${WEEKLY_MAINTENANCE_KEEP_SESSIONS:-15}"
RUNNER_DIAG_DAYS="${WEEKLY_MAINTENANCE_RUNNER_DIAG_DAYS:-7}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/weekly-$(date +%Y-%m-%d).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

log_disk() {
  log "=== disk usage ==="
  df -h / | tee -a "$LOG_FILE"
  if command -v docker >/dev/null 2>&1; then
    docker system df 2>/dev/null | tee -a "$LOG_FILE" || true
  fi
  du -sh "${APP_DATA}/postgres" "${APP_DATA}/finance" /var/lib/docker 2>/dev/null \
    | tee -a "$LOG_FILE" || true
}

prune_runner_logs() {
  local diag="${RUNNER_DIAG:-/root/actions-runner/_diag}"
  if [ -d "$diag" ]; then
    log "Pruning GitHub runner _diag older than ${RUNNER_DIAG_DAYS} days"
    find "$diag" -type f -mtime "+${RUNNER_DIAG_DAYS}" -delete 2>/dev/null || true
    find "$diag" -type f -name 'Worker_*.log' -mtime +1 -delete 2>/dev/null || true
  fi
  local work="${RUNNER_WORK:-/root/actions-runner/_work}"
  if [ -d "$work" ]; then
    find "$work" -mindepth 1 -maxdepth 1 -mtime "+${RUNNER_DIAG_DAYS}" -exec rm -rf {} + 2>/dev/null || true
  fi
}

prune_docker() {
  log "=== docker prune (containers + dangling images + build cache >7d) ==="
  docker container prune -f 2>&1 | tee -a "$LOG_FILE" || true
  docker image prune -f 2>&1 | tee -a "$LOG_FILE" || true
  docker builder prune -f --filter "until=168h" 2>&1 | tee -a "$LOG_FILE" || true
}

prune_postgres_sessions() {
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'app-postgres-1'; then
    log "Postgres container not running — skip session prune"
    return 0
  fi
  log "=== postgres forecast_app_sessions (keep ${KEEP_SESSIONS} newest) ==="
  docker exec app-postgres-1 psql -U forecast -d forecast -v ON_ERROR_STOP=1 -c "
SELECT count(*) AS sessions_before,
       pg_size_pretty(coalesce(sum(octet_length(bundle)), 0)::bigint) AS blobs_before
FROM forecast_app_sessions;
" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARN: forecast_app_sessions table missing or query failed"
    return 0
  }
  docker exec app-postgres-1 psql -U forecast -d forecast -v ON_ERROR_STOP=1 -c "
DELETE FROM forecast_app_sessions
WHERE session_id NOT IN (
  SELECT session_id FROM forecast_app_sessions
  ORDER BY updated_at DESC
  LIMIT ${KEEP_SESSIONS}
);
" 2>&1 | tee -a "$LOG_FILE" || true
  docker exec app-postgres-1 psql -U forecast -d forecast -c "VACUUM ANALYZE forecast_app_sessions;" \
    2>&1 | tee -a "$LOG_FILE" || true
  docker exec app-postgres-1 psql -U forecast -d forecast -c "
SELECT count(*) AS sessions_after,
       pg_size_pretty(coalesce(sum(octet_length(bundle)), 0)::bigint) AS blobs_after
FROM forecast_app_sessions;
" 2>&1 | tee -a "$LOG_FILE" || true
}

probe_health() {
  if curl -sf --connect-timeout 8 --max-time 15 http://127.0.0.1:8000/api/health >/dev/null 2>&1 \
    || curl -sf --connect-timeout 8 --max-time 15 http://127.0.0.1/api/health >/dev/null 2>&1; then
    log "OK: API health check passed after maintenance"
  else
    log "WARN: API health check failed after maintenance (backend may still be starting)"
  fi
}

main() {
  log "========== weekly maintenance start =========="
  log_disk
  prune_runner_logs
  prune_docker
  prune_postgres_sessions
  log "=== disk usage (after) ==="
  log_disk
  probe_health
  log "========== weekly maintenance done =========="
}

main "$@"

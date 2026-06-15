#!/usr/bin/env bash
# Keep the self-hosted GitHub Actions runner alive on the VPS.
# When the runner dies (OOM during deploy tests, host reboot), all workflows stay Queued.
#
# Install once on the VPS (as root):
#   chmod +x /root/app/scripts/ensure_github_runner.sh
#   (crontab -l 2>/dev/null; echo '*/5 * * * * /root/app/scripts/ensure_github_runner.sh >> /var/log/gh-runner-watchdog.log 2>&1') | crontab -
set -euo pipefail

RUNNER_DIR="${RUNNER_DIR:-/root/actions-runner}"
LOG_TAG="[gh-runner-watchdog]"

log() { echo "$(date -Iseconds) ${LOG_TAG} $*"; }

runner_proc_running() {
  pgrep -f "${RUNNER_DIR}/.*Runner.Listener" >/dev/null 2>&1 \
    || pgrep -f "actions-runner.*run.sh" >/dev/null 2>&1
}

find_systemd_unit() {
  systemctl list-units --type=service --all --no-legend 'actions.runner.*.service' 2>/dev/null \
    | awk '{print $1}' | head -1
}

start_via_systemd() {
  local unit
  unit="$(find_systemd_unit || true)"
  if [ -n "${unit}" ]; then
    log "starting systemd unit ${unit}"
    systemctl start "${unit}" || systemctl restart "${unit}"
    sleep 3
    if systemctl is-active --quiet "${unit}"; then
      log "OK: ${unit} is active"
      return 0
    fi
    log "WARN: ${unit} failed to start"
    return 1
  fi
  return 1
}

start_via_run_sh() {
  if [ ! -x "${RUNNER_DIR}/run.sh" ]; then
    log "ERROR: ${RUNNER_DIR}/run.sh not found — cannot start runner"
    return 1
  fi
  log "starting ${RUNNER_DIR}/run.sh in background (no systemd unit found)"
  cd "${RUNNER_DIR}"
  nohup ./run.sh >> "${RUNNER_DIR}/watchdog-run.log" 2>&1 &
  sleep 5
  if runner_proc_running; then
    log "OK: runner process started via run.sh"
    return 0
  fi
  log "ERROR: run.sh did not stay up — check ${RUNNER_DIR}/_diag"
  return 1
}

main() {
  if runner_proc_running; then
  log "runner process already running"
    exit 0
  fi
  log "runner process missing — attempting restart"
  start_via_systemd || start_via_run_sh || exit 1
}

main "$@"

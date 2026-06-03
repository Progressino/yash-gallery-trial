#!/usr/bin/env bash
# Quick API latency bench — run on VPS loopback or against public URL.
# Usage:
#   BASE=http://127.0.0.1:8000/api COOKIE='session_id=...; auth_token=...' ./scripts/bench_api_latency.sh

set -eu
BASE="${BASE:-http://127.0.0.1:8000/api}"
N="${N:-20}"
CURL_OPTS=(-s -o /dev/null -w '%{time_total}\n')
if [[ -n "${COOKIE:-}" ]]; then
  CURL_OPTS+=(-H "Cookie: ${COOKIE}")
fi

bench() {
  local path="$1"
  local label="$2"
  local times=()
  local i
  for ((i = 0; i < N; i++)); do
    t=$(curl "${CURL_OPTS[@]}" "${BASE}${path}" || echo "9.999")
    times+=("$t")
  done
  python3 - "$label" "${times[@]}" <<'PY'
import sys
vals = sorted(float(x) for x in sys.argv[2:])
n = len(vals)
p50 = vals[int(n * 0.5)]
p95 = vals[int(n * 0.95)] if n > 1 else vals[0]
print(f"{sys.argv[1]}: n={n} p50={p50:.3f}s p95={p95:.3f}s max={vals[-1]:.3f}s")
PY
}

echo "Benchmark BASE=${BASE} N=${N}"
bench "/health" "health"
bench "/data/coverage?light=1" "coverage_light"
if [[ -n "${COOKIE:-}" ]]; then
  bench "/data/job-status" "job_status"
  bench "/auth/me" "auth_me"
fi

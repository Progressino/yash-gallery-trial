#!/usr/bin/env bash
# Verify PO Fresh local stack: backend, frontend, and Vite /api proxy.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0

check() {
  local name=$1 url=$2
  local code
  code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time 8 "$url" 2>/dev/null || echo "000")
  if [[ "$code" =~ ^(200|204)$ ]]; then
    echo "OK   $name ($code) $url"
  else
    echo "FAIL $name ($code) $url"
    FAIL=1
  fi
}

echo "=== PO stack connection test ==="
# shellcheck source=po-engine-version.sh
source "$ROOT/scripts/po-engine-version.sh" 2>/dev/null || true
if command -v po_engine_expected_version >/dev/null 2>&1; then
  echo "Repo PO engine: v$(po_engine_expected_version)"
fi

check "backend health" "http://127.0.0.1:8000/api/health"
check "frontend root" "http://127.0.0.1:5173/"
check "vite /api proxy" "http://127.0.0.1:5173/api/health"
check "po-fresh route" "http://127.0.0.1:5173/po-fresh"

# Stray Vite on 5174 breaks strictPort expectations
if curl -sf --max-time 2 http://127.0.0.1:5174/ >/dev/null 2>&1; then
  echo "WARN stray server on :5174 (should use :5173 only)"
  FAIL=1
fi

RUNNING="$(curl -sf --max-time 4 http://127.0.0.1:8000/api/health 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('po_merge_version',''))" 2>/dev/null || true)"
if [[ -n "${RUNNING:-}" ]] && command -v po_engine_expected_version >/dev/null 2>&1; then
  EXP="$(po_engine_expected_version)"
  if [[ "$RUNNING" == "$EXP" ]]; then
    echo "OK   PO engine version v$RUNNING"
  else
    echo "FAIL PO engine version (running v$RUNNING, repo v$EXP)"
    FAIL=1
  fi
fi

if [[ "$FAIL" -eq 0 ]]; then
  echo ""
  echo "All checks passed — open http://localhost:5173/po-fresh"
  exit 0
fi

echo ""
echo "Stack unhealthy — run: bash scripts/start-po-local-daemon.sh --no-watchdog"
exit 1

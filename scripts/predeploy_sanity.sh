#!/usr/bin/env bash
# One-command predeploy sanity: backend regressions + frontend chromium smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-tests"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
# Runner can have a stale/partial venv (python present, pip missing).
if [[ ! -x "$VENV/bin/pip" ]]; then
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi
# Ensure test deps exist even when an older venv is already present.
if ! "$VENV/bin/python" -c "import pytest" >/dev/null 2>&1; then
  "$VENV/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

echo "==> Backend sanity tests"
"$VENV/bin/python" -m pytest tests/test_inventory_amazon_parser.py -q
"$VENV/bin/python" -m pytest tests/test_sales_services.py -k "myntra" -q
"$VENV/bin/python" -m pytest tests/test_myntra_resolve.py -q
"$VENV/bin/python" -m pytest tests/test_api_data_po_sales.py -q

echo "==> Frontend build + chromium smoke"
pushd "$ROOT/frontend" >/dev/null
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
if ! npm ls @playwright/test >/dev/null 2>&1; then
  npm install -D @playwright/test
fi
npm run build
npx playwright install chromium

PORT="${E2E_PORT:-4173}"
E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:${PORT}}"
LOG_FILE="/tmp/forecast-e2e-ui.log"

npm run preview -- --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 &
UI_PID=$!
cleanup() {
  kill "$UI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY=0
for _ in {1..40}; do
  if curl -sf "$E2E_BASE_URL/login" >/dev/null; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "UI preview server did not become ready at $E2E_BASE_URL"
  echo "--- preview log ---"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

E2E_BASE_URL="$E2E_BASE_URL" npm run test:smoke
popd >/dev/null

echo ""
echo "All predeploy sanity checks passed."

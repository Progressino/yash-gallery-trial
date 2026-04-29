#!/usr/bin/env bash
# One-command predeploy sanity: backend regressions + frontend chromium smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-tests"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

echo "==> Backend sanity tests"
"$VENV/bin/python" -m pytest tests/test_inventory_amazon_parser.py -q
"$VENV/bin/python" -m pytest tests/test_sales_services.py -k "myntra" -q
"$VENV/bin/python" -m pytest tests/test_myntra_resolve.py -q
"$VENV/bin/python" -m pytest tests/test_api_data_po_sales.py -q

echo "==> Frontend build + chromium smoke"
pushd "$ROOT/frontend" >/dev/null
if [[ ! -d node_modules ]]; then
  npm install
fi
if ! npm ls @playwright/test >/dev/null 2>&1; then
  npm install -D @playwright/test
fi
npm run build
npx playwright install chromium

PORT="${E2E_PORT:-4173}"
E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:${PORT}}"

npm run dev -- --host 127.0.0.1 --port "$PORT" >/tmp/forecast-e2e-ui.log 2>&1 &
UI_PID=$!
cleanup() {
  kill "$UI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in {1..40}; do
  if curl -sf "$E2E_BASE_URL/login" >/dev/null; then
    break
  fi
  sleep 1
done

E2E_BASE_URL="$E2E_BASE_URL" npm run test:smoke
popd >/dev/null

echo ""
echo "All predeploy sanity checks passed."

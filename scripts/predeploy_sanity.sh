#!/usr/bin/env bash
# One-command predeploy sanity: backend regressions + frontend chromium smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ensure_python_packaging() {
  if python3 -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing python packaging tools (python3-pip/python3-venv)..."
    apt-get update -y >/dev/null
    apt-get install -y python3-pip python3-venv >/dev/null
    python3 -m pip --version >/dev/null 2>&1
    return 0
  fi
  echo "ERROR: python3 pip is unavailable and auto-install is unsupported on this runner." >&2
  return 1
}

ensure_python_packaging

VENV="$ROOT/.venv-tests"
PYTHON_BIN="python3"
PIP_CMD=(python3 -m pip)
if [[ -x "$VENV/bin/python" && -x "$VENV/bin/pip" ]]; then
  PYTHON_BIN="$VENV/bin/python"
  PIP_CMD=("$VENV/bin/pip")
else
  rm -rf "$VENV" >/dev/null 2>&1 || true
  if python3 -m venv "$VENV" >/dev/null 2>&1 && [[ -x "$VENV/bin/python" && -x "$VENV/bin/pip" ]]; then
    PYTHON_BIN="$VENV/bin/python"
    PIP_CMD=("$VENV/bin/pip")
  else
    echo "Warning: python venv unavailable; falling back to system python/pip." >&2
  fi
fi
# Ensure test deps exist even when an older env is present.
if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  "${PIP_CMD[@]}" install -q -r "$ROOT/backend/requirements-test.txt"
fi

echo "==> Backend sanity tests"
"$PYTHON_BIN" -m pytest tests/test_inventory_amazon_parser.py -q
"$PYTHON_BIN" -m pytest tests/test_sales_services.py -k "myntra" -q
"$PYTHON_BIN" -m pytest tests/test_myntra_resolve.py -q
"$PYTHON_BIN" -m pytest tests/test_api_data_po_sales.py -q
"$PYTHON_BIN" -m pytest tests/test_finance_sales_entries.py -q
"$PYTHON_BIN" -m pytest tests/test_finance_accountant_dry_run.py -q
"$PYTHON_BIN" -m pytest tests/test_finance_api.py -q

# Frontend smoke uses npm/npx; missing Node yields shell exit 127 on self-hosted runners.
if [[ "${SKIP_E2E:-}" == "1" ]]; then
  echo "==> SKIP_E2E=1 — skipping frontend build/playwright smoke."
elif ! command -v npm >/dev/null 2>&1; then
  echo "WARN: npm not on PATH — skipping frontend smoke (install Node.js to run Playwright here)."
else
  echo "==> Frontend build + chromium smoke"
  pushd "$ROOT/frontend" >/dev/null
  # Load JWT secret for Playwright auth-cookie generation in deep checklist.
  if [[ -z "${JWT_SECRET:-}" && -f "$ROOT/.env" ]]; then
    _jwt_from_env="$(python3 - <<'PY'
import os
from pathlib import Path

env_path = Path(os.environ.get("ROOT", ".")) / ".env"
val = ""
if env_path.exists():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "JWT_SECRET":
            val = v.strip().strip('"').strip("'")
            break
print(val)
PY
)"
    if [[ -n "$_jwt_from_env" ]]; then
      export JWT_SECRET="$_jwt_from_env"
    fi
  fi

  npm_install_with_retry() {
    local mode="$1" # ci|install
    if [[ "$mode" == "ci" ]]; then
      npm ci && return 0
      echo "WARN: npm ci failed; cleaning node_modules and retrying once..." >&2
      rm -rf node_modules
      npm ci
    else
      npm install && return 0
      echo "WARN: npm install failed; cleaning node_modules and retrying once..." >&2
      rm -rf node_modules
      npm install
    fi
  }
  node_modules_integrity_ok() {
    [[ -f node_modules/typescript/lib/lib.es2022.d.ts ]] \
      && [[ -f node_modules/typescript/lib/lib.dom.d.ts ]] \
      && [[ -f node_modules/vite/package.json ]] \
      && [[ -f node_modules/@playwright/test/package.json ]]
  }

  if [[ -f package-lock.json ]]; then
    npm_install_with_retry ci
  else
    npm_install_with_retry install
  fi
  if ! node_modules_integrity_ok; then
    echo "WARN: node_modules integrity check failed; reinstalling cleanly..." >&2
    rm -rf node_modules
    if [[ -f package-lock.json ]]; then
      npm_install_with_retry ci
    else
      npm_install_with_retry install
    fi
    if ! node_modules_integrity_ok; then
      echo "ERROR: node_modules still incomplete after reinstall." >&2
      exit 1
    fi
  fi
  if ! npm ls @playwright/test >/dev/null 2>&1; then
    npm install -D @playwright/test
  fi
  npm run build
  npx playwright install chromium

  PORT="${E2E_PORT:-4173}"
  E2E_BASE_URL="${E2E_BASE_URL:-http://127.0.0.1:${PORT}}"
  LOG_FILE="/tmp/forecast-e2e-ui.log"
  API_LOG_FILE="/tmp/forecast-e2e-api.log"

  "$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 >"$API_LOG_FILE" 2>&1 &
  API_PID=$!
  npm run preview -- --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 &
  UI_PID=$!
  cleanup() {
    kill "$UI_PID" >/dev/null 2>&1 || true
    kill "$API_PID" >/dev/null 2>&1 || true
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

  # Keep broad smoke and also pin the deep finance checklist explicitly so
  # deploy gates cannot accidentally bypass it if smoke script changes later.
  E2E_BASE_URL="$E2E_BASE_URL" npm run test:smoke
  E2E_BASE_URL="$E2E_BASE_URL" npx playwright test tests/e2e/manual-finance-checklist.spec.js
  popd >/dev/null
fi

echo ""
echo "All predeploy sanity checks passed."

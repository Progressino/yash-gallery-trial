#!/usr/bin/env bash
# One-command predeploy guard for Myntra + sales dashboard regressions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-tests"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi

echo "Running Myntra + sales regression tests..."
"$VENV/bin/python" -m pytest tests/test_sales_services.py -k "myntra" -q
"$VENV/bin/python" -m pytest tests/test_myntra_resolve.py -q
"$VENV/bin/python" -m pytest tests/test_api_data_po_sales.py -q

echo ""
echo "All Myntra predeploy checks passed."

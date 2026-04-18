#!/usr/bin/env bash
# Create .venv + install backend/requirements-test.txt if needed, then run pytest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/backend/requirements-test.txt"
fi
exec "$ROOT/.venv/bin/python" -m pytest tests/ "$@"

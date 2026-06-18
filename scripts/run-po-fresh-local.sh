#!/usr/bin/env bash
# Local PO Fresh smoke: backend pytest + frontend build + Playwright route test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Backend PO Fresh API tests =="
"$ROOT/.venv/bin/python" -m pytest tests/test_po_fresh_api.py -q

echo "== Frontend build =="
(cd frontend && npm run build)

echo "== Playwright po-fresh route =="
(cd frontend && npx playwright test tests/e2e/po-fresh.spec.js)

echo "OK — start dev stack with: docker compose up"
echo "Then open http://localhost:5173/po-fresh (login required)"

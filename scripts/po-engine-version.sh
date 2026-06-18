#!/usr/bin/env bash
# PO engine version helpers — single source: backend/services/po_shared_cache.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="$ROOT/backend/services/po_shared_cache.py"
STAMP_FILE="$ROOT/local-dev/pids/po-engine-src.sha256"
HEALTH_URL="${PO_HEALTH_URL:-http://127.0.0.1:8000/api/health}"

PO_ENGINE_SRC_FILES=(
  "$ROOT/backend/services/po_shared_cache.py"
  "$ROOT/backend/services/po_engine.py"
  "$ROOT/backend/services/po_calculate_run.py"
  "$ROOT/backend/services/po_session_hydrate.py"
  "$ROOT/backend/services/po_calculate_result_api.py"
)

po_engine_expected_version() {
  python3 -c "
import re, pathlib
text = pathlib.Path('$VERSION_FILE').read_text(encoding='utf-8')
m = re.search(r'^PO_MERGE_LOGIC_VERSION\s*=\s*(\d+)', text, re.M)
if not m:
    raise SystemExit('PO_MERGE_LOGIC_VERSION not found')
print(m.group(1))
"
}

po_engine_running_version() {
  curl -sf --max-time 4 "$HEALTH_URL" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('po_merge_version') or '')" 2>/dev/null \
    || true
}

po_engine_src_hash() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${PO_ENGINE_SRC_FILES[@]}" 2>/dev/null | shasum -a 256 | awk '{print $1}'
  else
    sha256sum "${PO_ENGINE_SRC_FILES[@]}" 2>/dev/null | sha256sum | awk '{print $1}'
  fi
}

po_engine_stored_src_hash() {
  [[ -f "$STAMP_FILE" ]] && cat "$STAMP_FILE" || echo ""
}

po_engine_write_src_hash() {
  mkdir -p "$(dirname "$STAMP_FILE")"
  po_engine_src_hash >"$STAMP_FILE"
}

po_engine_backend_up() {
  curl -sf --max-time 3 "$HEALTH_URL" >/dev/null 2>&1
}

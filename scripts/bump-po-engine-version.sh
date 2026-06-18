#!/usr/bin/env bash
# Bump PO_MERGE_LOGIC_VERSION by 1 (invalidates shared PO cache).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILE="$ROOT/backend/services/po_shared_cache.py"

# shellcheck source=po-engine-version.sh
source "$ROOT/scripts/po-engine-version.sh"

OLD="$(po_engine_expected_version)"
NEW=$((OLD + 1))

python3 - "$FILE" "$NEW" <<'PY'
import re, sys
path, new = sys.argv[1], int(sys.argv[2])
text = open(path, encoding="utf-8").read()
out, n = re.subn(
    r"^(PO_MERGE_LOGIC_VERSION\s*=\s*)\d+",
    rf"\g<1>{new}",
    text,
    count=1,
    flags=re.M,
)
if n != 1:
    raise SystemExit("PO_MERGE_LOGIC_VERSION not found")
open(path, "w", encoding="utf-8").write(out)
print(new)
PY

echo "PO_MERGE_LOGIC_VERSION: v$OLD → v$NEW"

# Update test fixture if present
TEST="$ROOT/tests/test_po_fresh_api.py"
if [[ -f "$TEST" ]] && grep -q "po_merge_version.*== $OLD" "$TEST"; then
  sed -i '' "s/po_merge_version\") == $OLD/po_merge_version\") == $NEW/" "$TEST" 2>/dev/null \
    || sed -i "s/po_merge_version\") == $OLD/po_merge_version\") == $NEW/" "$TEST"
  echo "Updated tests/test_po_fresh_api.py"
fi

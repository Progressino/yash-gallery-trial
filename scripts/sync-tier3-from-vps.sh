#!/usr/bin/env bash
# Copy production Tier-3 daily_sales.db to local repo for parity with app.progressino.com.
# Requires SSH access to the VPS (adjust HOST and remote path).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${VPS_HOST:-progressino-vps}"
REMOTE_DB="${REMOTE_DAILY_DB:-/root/app-data/finance/daily_sales.db}"
LOCAL_DB="$ROOT/daily_sales.db"
BACKUP="$ROOT/daily_sales.db.bak.$(date +%Y%m%d-%H%M%S)"

if [[ ! -f "$LOCAL_DB" ]]; then
  echo "No local $LOCAL_DB — will create from remote."
else
  cp "$LOCAL_DB" "$BACKUP"
  echo "Backed up local DB → $BACKUP"
fi

echo "Pulling $HOST:$REMOTE_DB → $LOCAL_DB"
scp "$HOST:$REMOTE_DB" "$LOCAL_DB"

ROWS=$(sqlite3 "$LOCAL_DB" "SELECT COUNT(*) FROM daily_uploads;" 2>/dev/null || echo "?")
echo "daily_uploads rows: $ROWS"
if [[ "$ROWS" == "0" ]]; then
  echo "WARNING: remote DB has 0 uploads — check REMOTE_DAILY_DB path on VPS." >&2
  exit 1
fi

echo "Restart backend so Tier-3 path + session merge pick up the new file:"
echo "  bash scripts/restart-po-backend.sh"
echo "Then in the app: Server & cache → Reload, or POST /api/cache/reload-fresh"

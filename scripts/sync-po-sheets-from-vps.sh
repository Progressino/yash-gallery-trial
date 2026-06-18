#!/usr/bin/env bash
# Sync production PO operational sheets (existing PO, inventory, sidecars) for local parity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${VPS_HOST:-root@app.progressino.com}"
REMOTE="${REMOTE_WARM_CACHE:-/root/app-data/finance/warm_cache}"
LOCAL="${WARM_CACHE_DIR:-$ROOT/.local-data/warm_cache}"
mkdir -p "$LOCAL"

FILES=(
  existing_po_df.parquet
  existing_po_meta.json
  inventory_df_variant.parquet
  inventory_df_parent.parquet
  inventory_session_meta.json
  sku_status_lead_df.parquet
  daily_inventory_history_df.parquet
  po_return_overlay_df.parquet
  return_overlay_meta.json
)

for f in "${FILES[@]}"; do
  if scp -o ConnectTimeout=15 "$HOST:$REMOTE/$f" "$LOCAL/$f" 2>/dev/null; then
    echo "synced $f"
  else
    echo "skip $f (not on server)"
  fi
done

echo "Restart backend: bash scripts/restart-po-backend.sh"
echo "Then Force recalculate on PO Fresh (use_shared_cache=false)."

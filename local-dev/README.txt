PO Engine — local development shortcuts
========================================

All files live in: forecast/local-dev/

START (recommended)
  Double-click:  Start PO Engine.command  (in this folder)
  Servers run in the background — safe to close the Terminal window after start.

TROUBLESHOOTING "still down"
  1. Stop: Stop PO Engine.command
  2. Start again: Start PO Engine.command
  3. Wait ~10s, open http://localhost:5173/po-fresh
  4. Check: bash local-dev/status.sh
  Local mode uses WARM_CACHE_PO_SESSION_ONLY (lighter RAM — avoids Mac OOM crashes).
  A Terminal window opens and STAYS OPEN — leave it open while testing.
  Open: http://localhost:5173/po-fresh

STOP
  • Close the Terminal window that says "PO Engine is running", OR
  • Double-click: Stop PO Engine.command

STATUS
  bash local-dev/status.sh
  bash scripts/test-po-stack-connection.sh
  (shows repo vs running PO engine version + API proxy)

SYNC PO ENGINE (after code changes)
  bash scripts/ensure-po-engine-local.sh
  Restarts backend if version or PO source files changed.

BUMP VERSION (when PO logic changes)
  bash scripts/bump-po-engine-version.sh

LOGS
  local-dev/logs/backend.log
  local-dev/logs/frontend.log

Note: Servers stop if you close the start Terminal window. That is normal.

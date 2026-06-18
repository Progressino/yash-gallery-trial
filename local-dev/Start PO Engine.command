#!/bin/bash
# Double-click to start — opens Terminal supervisor (survives Cursor closing).
FORECAST="/Users/samraisinghani/Downloads/Development/forecast"

if bash "$FORECAST/scripts/test-po-stack-connection.sh" >/dev/null 2>&1; then
  echo "PO Engine is already running and healthy."
  bash "$FORECAST/scripts/test-po-stack-connection.sh"
  open "http://localhost:5173/po-fresh" 2>/dev/null || true
  read -r -p "Press Enter to close…" _
  exit 0
fi

# If supervisor already running in another Terminal, wait for it to recover.
if [[ -f "$FORECAST/local-dev/pids/supervisor.pid" ]]; then
  sup="$(cat "$FORECAST/local-dev/pids/supervisor.pid" 2>/dev/null || true)"
  if [[ -n "$sup" ]] && kill -0 "$sup" 2>/dev/null; then
    echo "Supervisor running (pid $sup) but stack unhealthy — waiting…"
    for i in {1..30}; do
      if bash "$FORECAST/scripts/test-po-stack-connection.sh" >/dev/null 2>&1; then
        bash "$FORECAST/scripts/test-po-stack-connection.sh"
        open "http://localhost:5173/po-fresh" 2>/dev/null || true
        read -r -p "Press Enter to close…" _
        exit 0
      fi
      sleep 2
    done
  fi
fi

osascript <<EOF
tell application "Terminal"
  activate
  do script "bash \"$FORECAST/scripts/po-stack-supervisor.sh\""
end tell
EOF

echo "Opening Terminal to run PO Engine supervisor…"
for i in {1..45}; do
  if bash "$FORECAST/scripts/test-po-stack-connection.sh" >/dev/null 2>&1; then
    bash "$FORECAST/scripts/test-po-stack-connection.sh"
    open "http://localhost:5173/po-fresh" 2>/dev/null || true
    read -r -p "Press Enter to close…" _
    exit 0
  fi
  sleep 1
done

echo "Stack did not become healthy — check $FORECAST/local-dev/logs/supervisor.log"
read -r -p "Press Enter to close…" _
exit 1

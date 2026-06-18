#!/bin/bash
FORECAST="/Users/samraisinghani/Downloads/Development/forecast"
cd "$FORECAST" || exit 1
bash "$FORECAST/scripts/stop-po-local-daemon.sh"
echo ""
read -r -p "Press Enter to close this window…" _

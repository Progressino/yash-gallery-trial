#!/usr/bin/env bash
# Install macOS LaunchAgent so PO stack survives Cursor/terminal exit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.yashgallery.po-engine"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/local-dev/logs" "$ROOT/local-dev/pids"

cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/keep-po-local-alive.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>StandardOutPath</key>
  <string>${ROOT}/local-dev/logs/launchd-watchdog.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT}/local-dev/logs/launchd-watchdog.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>WARM_CACHE_DIR</key>
    <string>${ROOT}/.local-data/warm_cache</string>
    <key>GITHUB_BLOB_CACHE_DIR</key>
    <string>${ROOT}/.local-data/github_cache</string>
    <key>WARM_CACHE_PO_SESSION_ONLY</key>
    <string>1</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID_NUM}" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/${UID_NUM}" "$PLIST"
launchctl enable "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true

echo "Installed LaunchAgent: $PLIST"
echo "Logs: $ROOT/local-dev/logs/launchd-watchdog.log"
echo "Unload: launchctl bootout gui/${UID_NUM} \"$PLIST\""

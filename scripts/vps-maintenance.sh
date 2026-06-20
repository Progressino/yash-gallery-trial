#!/usr/bin/env bash
# Safe VPS cleanup — run on the host (not inside a container).
# Usage: bash scripts/vps-maintenance.sh
set -euo pipefail

echo "=== Memory ==="
free -h
echo ""
echo "=== Disk ==="
df -h /
echo ""
echo "=== Top memory (host) ==="
ps aux --sort=-%mem | head -12
echo ""
echo "=== Docker disk ==="
docker system df 2>/dev/null || true
echo ""
echo "Pruning dangling images and build cache (keeps running containers)…"
docker image prune -f
docker builder prune -f --filter 'until=72h' 2>/dev/null || docker builder prune -f 2>/dev/null || true
echo ""
echo "=== App stack ==="
docker compose -f /root/app/docker-compose.prod.yml ps 2>/dev/null || true
echo ""
echo "Optional (if snap packages unused): sudo systemctl disable --now snapd"

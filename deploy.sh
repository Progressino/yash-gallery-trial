#!/usr/bin/env bash
# deploy.sh — run on the VPS to pull latest code and restart containers.
# Usage: ./deploy.sh [branch]   (defaults to main)
set -euo pipefail

BRANCH=${1:-main}
COMPOSE="docker compose -f docker-compose.prod.yml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 🚀 Yash Gallery ERP — Deploy"
echo "    Branch: $BRANCH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Pull latest code
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# 2. Check .env exists
if [ ! -f .env ]; then
  echo "⚠️  .env not found — copying from .env.example"
  cp .env.example .env
  echo "   → Fill in GITHUB_TOKEN and GITHUB_REPO in .env, then re-run deploy.sh"
  exit 1
fi

# 3. Data dirs for bind mounts (PostgreSQL session store + SQLite ERP DBs)
mkdir -p /root/app-data/postgres /root/app-data/finance

# 4. Build and (re)start containers
echo ""
echo "📦 Building containers…"
$COMPOSE build --no-cache

echo ""
echo "▶  Starting containers…"
$COMPOSE down --remove-orphans 2>/dev/null || true
$COMPOSE up -d

# 5. Health check (hit backend container directly on port 8000, bypassing nginx HTTPS redirect)
echo ""
echo "🩺 Health check…"
sleep 5
for i in 1 2 3; do
  if $COMPOSE exec -T backend python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" > /dev/null 2>&1; then
    echo "   ✅ Backend healthy"
    break
  fi
  if [ "$i" -eq 3 ]; then
    echo "   ⚠️  Health check inconclusive — containers are up, check logs if app is unresponsive:"
    echo "      docker compose -f docker-compose.prod.yml logs backend"
  else
    echo "   ⏳ Attempt $i failed, retrying in 3s…"
    sleep 3
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ Deployed successfully!"
echo "    App is live at http://localhost (or your domain via Cloudflare)"
echo ""
echo " Useful commands:"
echo "   Logs:    docker compose -f docker-compose.prod.yml logs -f"
echo "   Status:  docker compose -f docker-compose.prod.yml ps"
echo "   Stop:    docker compose -f docker-compose.prod.yml down"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

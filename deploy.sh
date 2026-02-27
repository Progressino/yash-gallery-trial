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

# 3. Build and (re)start containers
echo ""
echo "📦 Building containers…"
$COMPOSE build --no-cache

echo ""
echo "▶  Starting containers…"
$COMPOSE up -d

# 4. Health check
echo ""
echo "🩺 Health check…"
sleep 3
if curl -sf http://localhost/api/health > /dev/null; then
  echo "   ✅ Backend healthy"
else
  echo "   ❌ Backend not responding — check logs:"
  echo "      docker compose -f docker-compose.prod.yml logs backend"
  exit 1
fi

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

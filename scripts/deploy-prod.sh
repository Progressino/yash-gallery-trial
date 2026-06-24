#!/usr/bin/env bash
# Deploy forecast app to production VPS (app.progressino.com).
# Sets a visible build tag in /api/health and the UI footer.
set -euo pipefail

HOST="${DEPLOY_HOST:-root@app.progressino.com}"
APP_DIR="${DEPLOY_APP_DIR:-/root/app}"
DEPLOY_TAG="${APP_GIT_SHA:-deploy-$(date -u +%Y%m%d-%H%M)}"
BUILT_AT="${APP_BUILT_AT:-$(date -u +%Y-%m-%dT%H:%MZ)}"

echo "Deploying ${DEPLOY_TAG} to ${HOST}:${APP_DIR}"

rsync -az --delete \
  --exclude '.git' \
  --exclude '.local-data' \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '.env' \
  --exclude 'po_raise_archive_dev' \
  "$(cd "$(dirname "$0")/.." && pwd)/" \
  "${HOST}:${APP_DIR}/"

ssh "${HOST}" "cd ${APP_DIR} && \
  export APP_GIT_SHA='${DEPLOY_TAG}' APP_BUILT_AT='${BUILT_AT}' && \
  docker compose -f docker-compose.prod.yml build frontend backend stitching-backend && \
  docker compose -f docker-compose.prod.yml up -d frontend backend stitching-backend"

echo "Done. Build tag: ${DEPLOY_TAG} (${BUILT_AT})"
echo "Verify: curl -s https://app.progressino.com/api/health | jq .git_sha"

#!/usr/bin/env bash
# Deploy forecast app to production VPS (app.progressino.com).
# Uses isolated Docker project name `progressino` — does not clash with progressino-dev.
set -euo pipefail

HOST="${DEPLOY_HOST:-root@app.progressino.com}"
APP_DIR="${DEPLOY_APP_DIR:-/root/app}"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-progressino}"
COMPOSE_FILE="docker-compose.prod.yml"
DEPLOY_TAG="${APP_GIT_SHA:-deploy-$(date -u +%Y%m%d-%H%M)}"
BUILT_AT="${APP_BUILT_AT:-$(date -u +%Y-%m-%dT%H:%MZ)}"

echo "Deploying ${DEPLOY_TAG} to ${HOST}:${APP_DIR} (project=${COMPOSE_PROJECT})"

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
  export APP_GIT_SHA='${DEPLOY_TAG}' APP_BUILT_AT='${BUILT_AT}' COMPOSE_PROJECT_NAME='${COMPOSE_PROJECT}' && \
  docker compose -p app -f ${COMPOSE_FILE} down 2>/dev/null || true && \
  docker compose -p ${COMPOSE_PROJECT} -f ${COMPOSE_FILE} build frontend backend stitching-backend && \
  docker compose -p ${COMPOSE_PROJECT} -f ${COMPOSE_FILE} up -d frontend backend stitching-backend"

echo "Done. Build tag: ${DEPLOY_TAG} (${BUILT_AT})"
echo "Verify: curl -s https://app.progressino.com/api/health | jq .git_sha"

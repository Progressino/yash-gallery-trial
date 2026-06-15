#!/usr/bin/env bash
# Run on the VPS when GitHub Actions deploy is stuck/failed.
# Mirrors deploy.yml: fast tests → rebuild → health check.
set -euo pipefail

ROOT="${ROOT:-/root/app}"
cd "$ROOT"

_dc() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

echo "==> Sync origin/main"
git fetch origin main
git checkout -f main
git reset --hard origin/main
git log -1 --oneline

_sha="$(git rev-parse --short HEAD)"
_built="$(date -u +%Y-%m-%dT%H:%MZ)"
for _KV in "APP_GIT_SHA=${_sha}" "APP_BUILT_AT=${_built}"; do
  _K="${_KV%%=*}"
  _V="${_KV#*=}"
  if grep -q "^${_K}=" .env 2>/dev/null; then
    sed -i "s|^${_K}=.*|${_K}=${_V}|" .env
  else
    echo "${_K}=${_V}" >> .env
  fi
done

echo "==> Stop backend during fast tests"
_dc -f docker-compose.prod.yml stop backend stitching-backend 2>/dev/null || true

echo "==> Fast predeploy sanity"
SKIP_E2E=1 DEPLOY_FAST=1 bash scripts/predeploy_sanity.sh

echo "==> Rebuild and restart"
_dc -f docker-compose.prod.yml build backend stitching-backend frontend
docker rm -f app-backend-1 app-stitching-backend-1 app-frontend-1 app-autoheal-1 2>/dev/null || true
_dc -f docker-compose.prod.yml up -d --remove-orphans

echo "==> Health check"
for i in $(seq 1 12); do
  if curl -sf --connect-timeout 5 http://127.0.0.1:8000/api/health; then
    echo ""
    echo "OK: deploy healthy (attempt $i) — build ${_sha}"
    exit 0
  fi
  echo "… waiting for /api/health ($i/12)"
  sleep 5
done
echo "ERROR: health check failed" >&2
_dc -f docker-compose.prod.yml logs --tail=80 backend 2>&1 || true
exit 1

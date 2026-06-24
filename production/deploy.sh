#!/usr/bin/env bash
# Run on the VPS inside the production app directory.
# Wrapper so production deploy always uses the isolated compose project name.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-progressino}"
exec "${ROOT}/deploy.sh" "$@"

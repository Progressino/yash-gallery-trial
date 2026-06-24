# Production deployment (isolated stack)

This folder documents the **production-only** Docker stack for `app.progressino.com`.
It is intentionally separate from local development so deploys do not clash with
`progressino-dev` or other compose projects on the same VPS.

## Stack names

| Environment | Compose project | Compose file | Host ports |
|-------------|-----------------|--------------|------------|
| **Production** | `progressino` | `docker-compose.prod.yml` | 80, 443, 127.0.0.1:8000 |
| **Local dev** | `progressino-dev` | `docker-compose.yml` | 5173, 8000, 5433 |

Container names are prefixed by the project, e.g. `progressino-backend-1` vs
`progressino-dev-backend-1`.

## Deploy from your laptop

```bash
./scripts/deploy-prod.sh
```

Environment overrides:

```bash
DEPLOY_HOST=root@app.progressino.com \
DEPLOY_APP_DIR=/root/progressino \
APP_DATA_DIR=/root/progressino-data \
./scripts/deploy-prod.sh
```

## Deploy on the VPS (git pull)

```bash
cd /root/progressino   # or /root/app during migration
./deploy.sh main
```

Both scripts use `docker compose -p progressino -f docker-compose.prod.yml`.

## Data directories

- Default prod data: `/root/app-data` (postgres + finance SQLite)
- Recommended isolated path: `/root/progressino-data`

Set in `.env` or shell:

```bash
export APP_DATA_DIR=/root/progressino-data
```

## Verify

```bash
docker compose -p progressino -f docker-compose.prod.yml ps
curl -s http://127.0.0.1:8000/api/health | jq .git_sha
```

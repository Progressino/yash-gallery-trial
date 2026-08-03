# Progressino ERP

[![Deploy to VPS](https://github.com/Progressino/yash-gallery-trial/actions/workflows/deploy.yml/badge.svg)](https://github.com/Progressino/yash-gallery-trial/actions/workflows/deploy.yml)
[![Monitor production deploy](https://github.com/Progressino/yash-gallery-trial/actions/workflows/monitor-production-deploy.yml/badge.svg)](https://github.com/Progressino/yash-gallery-trial/actions/workflows/monitor-production-deploy.yml)

Production: **https://app.progressino.com** · Health: `/api/health` (includes `git_sha`)

## Deployments (where to look in GitHub)

| What | Where |
|------|--------|
| Workflow file | [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) |
| Deploy runs | [Actions → **Deploy to VPS**](https://github.com/Progressino/yash-gallery-trial/actions/workflows/deploy.yml) |
| Environment history | [Environments → **production**](https://github.com/Progressino/yash-gallery-trial/deployments/production) |
| Post-deploy monitor | [Actions → **Monitor production deploy**](https://github.com/Progressino/yash-gallery-trial/actions/workflows/monitor-production-deploy.yml) |
| Emergency SSH deploy | [Remote deploy (SSH)](https://github.com/Progressino/yash-gallery-trial/actions/workflows/remote-deploy.yml) |

### Triggers

- **Automatic:** every push to `main` runs **Deploy to VPS** (self-hosted runner → rebuild containers on the VPS).
- **Manual:** Actions → Deploy to VPS → Run workflow.
- **Monitor:** every 15 minutes + after each successful Deploy to VPS; checks public health and whether `git_sha` matches `main`.

### Commit checks

Successful deploys publish:

- Check run: job name **`deploy`**
- Commit status: **`Deploy to VPS / production`**
- Environment: **`production`** (Deployments timeline)

### Verify live version

```bash
curl -sS https://app.progressino.com/api/health | jq '{git_sha, built_at, status}'
```

Local branch tip should match prod `git_sha` after a green Deploy run.

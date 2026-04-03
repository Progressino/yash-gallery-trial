# Developer Setup — Production Module

This guide gets you running the app locally so you can work on the **Production module** (MRP engine, Job Orders).

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop |
| Git | Any | https://git-scm.com |
| VS Code (recommended) | Any | https://code.visualstudio.com |

---

## 1. Clone and switch to your branch

```bash
git clone https://github.com/Progressino/yash-gallery-trial.git
cd yash-gallery-trial
git checkout dev/production
```

---

## 2. Set up environment variables

```bash
cp .env.dev.example .env
```

Your `.env` comes pre-filled with dev credentials. You can log in immediately with:

| Field | Value |
|-------|-------|
| Username | `devuser` |
| Password | `DevProd2026!` |

> **Change the password** after your first login by updating `AUTH_PASSWORD_HASH` in `.env`.
> Generate a new hash:
> ```bash
> python3 -c "import bcrypt; print(bcrypt.hashpw(b'YourNewPassword'.encode(), bcrypt.gensalt()).decode())"
> ```

---

## 3. Start the app

```bash
docker compose up
```

Both services start with **hot reload** — edits to Python or TypeScript files reflect instantly without restarting.

| Service | URL |
|---------|-----|
| Frontend (React) | http://localhost:5173 |
| Backend (FastAPI) | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

> First start takes ~2 minutes to download images and install dependencies.

---

## 4. Your scope — files you work on

You are responsible for **exactly 3 files**:

```
backend/routers/production.py     ← API endpoints (MRP, job orders, reservations)
backend/db/production_db.py       ← SQLite schema and query functions
frontend/src/pages/Production.tsx ← React UI (tabs: Dashboard, MRP, Orders, etc.)
```

**Do not modify any other file.** If a change in another file is needed for the production module to work (e.g. a shared utility), flag it in the PR description and the reviewer will handle it.

---

## 5. How the production module works

```
Sales Orders (sales_db) → MRP Run → BOM explosion (items_db)
                                  ↓
                         Material Requirements
                                  ↓
                    Soft Reservations + Job Orders (production_db)
```

- **MRP (Material Requirements Planning):** Selects open sales orders, explodes Bills of Materials, calculates net requirements after deducting stock.
- **Job Orders:** Manufacturing work orders assigned to processes (Cutting, Stitching, etc.) either in-house or via vendor.
- **Soft Reservations:** Temporary material reservations per SO until the order is fulfilled.

Database file in local dev: `backend/production.db` (auto-created on first run).

---

## 6. Making and pushing changes

```bash
# Always start from the latest version
git pull origin dev/production

# Make your changes to the 3 production files, then:
git add backend/routers/production.py
git add backend/db/production_db.py
git add frontend/src/pages/Production.tsx

git commit -m "feat: describe what you changed"
git push origin dev/production
```

---

## 7. Opening a Pull Request for review

1. Go to https://github.com/Progressino/yash-gallery-trial
2. Click **"Compare & pull request"** (appears after you push)
3. Set: **base** `main` ← **compare** `dev/production`
4. Fill in the PR template (appears automatically)
5. Submit — the reviewer will be notified

> Your changes **will not go to production** until the PR is reviewed and merged by the repo owner.

---

## 8. Useful commands

```bash
# Stop the app
docker compose down

# Restart only the backend (after Python changes if hot-reload doesn't catch it)
docker compose restart backend

# View backend logs live
docker compose logs -f backend

# Reset the production database (careful — deletes all local job orders)
rm backend/production.db
docker compose restart backend

# Run a quick API test
curl http://localhost:8000/api/production/stats
```

---

## 9. Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 5173 already in use | Stop other dev servers, or change port in `docker-compose.yml` |
| Login fails | Check `.env` has correct `AUTH_USERNAME` and `AUTH_PASSWORD_HASH` |
| Database errors on startup | Delete `backend/production.db` and restart |
| Changes not showing | Hard-refresh the browser (Cmd+Shift+R) |
| Docker out of disk space | `docker system prune -f` |

---

## Questions?

Open a GitHub issue on the `dev/production` branch or contact the repo owner.

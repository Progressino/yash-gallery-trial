"""
Yash Gallery ERP — FastAPI backend
Serves all business logic as a REST API.
Session state is stored server-side keyed by a UUID cookie.
"""
from dotenv import load_dotenv
load_dotenv()   # loads .env from cwd (run from repo root or backend/)

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .session import store
from .routers import upload, data, cache, po, auth as auth_router
from .routers.auth import verify_token
from .routers.finance import router as finance_router
from .routers.item import router as item_router
from .routers.sales import router as sales_router
from .routers.purchase import router as purchase_router
from .routers.tna import router as tna_router
from .routers.production import router as production_router
from .routers.grey import router as grey_router
from .routers.erp_admin import router as erp_admin_router
from .routers.marketplace_connect import router as marketplace_router
from .db.finance_db import init_db
from .db.item_db import init_db as init_item_db
from .db.marketplace_db import init_db as init_marketplace_db
from .db.sales_db import init_db as init_sales_db
from .db.purchase_db import init_db as init_purchase_db
from .db.tna_db import init_db as init_tna_db
from .db.production_db import init_db as init_production_db
from .db.grey_db import init_db as init_grey_db
from .db.users_db import init_db as init_users_db

init_db()
init_item_db()
init_marketplace_db()
init_sales_db()
init_purchase_db()
init_tna_db()
init_production_db()
init_grey_db()
init_users_db()

log = logging.getLogger("erp.cache_warmer")

# ── Shared warm cache (app-level, not per-session) ────────────────
# Populated at startup and refreshed daily at 06:00 IST.
# New/empty sessions copy from here so users never wait for a GitHub download.
_warm_cache: dict = {}
_warm_cache_loaded_at: datetime | None = None

IST = timezone(timedelta(hours=5, minutes=30))


def _do_load_warm_cache() -> bool:
    """Download from GitHub and store in module-level _warm_cache. Thread-safe (GIL)."""
    global _warm_cache, _warm_cache_loaded_at
    try:
        from .services.github_cache import load_cache_from_drive
        from .services.daily_store import load_all_platforms, merge_platform_data as _merge
        import pandas as pd

        ok, msg, loaded = load_cache_from_drive()
        if not ok or not loaded:
            log.warning("Warm-cache load failed: %s", msg)
            return False

        # Sanitise Snapdeal — strip invalid SKUs
        if "snapdeal_df" in loaded and isinstance(loaded["snapdeal_df"], pd.DataFrame):
            snap = loaded["snapdeal_df"]
            if not snap.empty and "OMS_SKU" in snap.columns:
                bad = snap["OMS_SKU"].astype(str).str.upper()
                loaded["snapdeal_df"] = snap[
                    ~(bad.isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])
                      | snap["OMS_SKU"].astype(str).str.match(r'^\d+$'))
                ].reset_index(drop=True)

        # Merge daily SQLite store on top
        try:
            daily = load_all_platforms()
            for plat, key in [("amazon","mtr_df"),("myntra","myntra_df"),
                               ("meesho","meesho_df"),("flipkart","flipkart_df")]:
                if daily.get(plat) is not None and not daily[plat].empty:
                    loaded[key] = _merge(loaded.get(key, pd.DataFrame()), daily[plat], plat)
        except Exception as e:
            log.warning("Daily-store merge warning: %s", e)

        _warm_cache = loaded
        _warm_cache_loaded_at = datetime.now(IST)
        log.info("Warm cache loaded at %s — %s", _warm_cache_loaded_at.strftime("%H:%M IST"), msg)
        return True

    except Exception as e:
        log.exception("Warm cache load error: %s", e)
        return False


def _copy_warm_cache_to_session(sess) -> bool:
    """Copy _warm_cache into an AppSession. Returns True if data was available."""
    if not _warm_cache:
        return False
    for key, val in _warm_cache.items():
        setattr(sess, key, val)
    sess._quarterly_cache.clear()
    return True


def _do_amazon_sync_all() -> None:
    """
    Called daily at 6AM IST. Pulls last 2 days of Amazon data via SP-API
    and merges results into the warm cache + SQLite daily store.
    """
    from .db.marketplace_db import get_credentials, save_sync_log
    from .services.amazon_sp_api import sync_amazon_data
    from .services.daily_store import merge_platform_data, save_daily_file
    import pandas as pd

    creds = get_credentials("amazon")
    if not creds:
        return  # Not connected — nothing to do

    log.info("Scheduled Amazon SP-API sync starting…")
    try:
        df, msg = sync_amazon_data(creds, days_back=2)
    except Exception as e:
        log.exception("Scheduled Amazon sync failed: %s", e)
        save_sync_log("amazon", "error", 0, message=str(e))
        return

    if df.empty:
        save_sync_log("amazon", "partial", 0, message=msg)
        log.info("Scheduled Amazon sync: no new data — %s", msg)
        return

    # Persist to SQLite
    try:
        from datetime import date as _date
        save_daily_file("amazon", f"sp-api-{_date.today()}.csv", df)
    except Exception as e:
        log.warning("Scheduled Amazon sync: daily store save failed: %s", e)

    # Merge into warm cache
    try:
        existing = _warm_cache.get("mtr_df", pd.DataFrame())
        _warm_cache["mtr_df"] = merge_platform_data(existing, df, "amazon")
        log.info("Scheduled Amazon sync: warm cache updated with %d rows", len(df))
    except Exception as e:
        log.warning("Scheduled Amazon sync: warm cache merge failed: %s", e)

    date_from = str(df["Date"].min().date()) if "Date" in df.columns else ""
    date_to   = str(df["Date"].max().date()) if "Date" in df.columns else ""
    save_sync_log("amazon", "success", len(df), date_from, date_to, msg)
    log.info("Scheduled Amazon SP-API sync complete: %s", msg)


async def _warm_cache_scheduler():
    """Background task: refresh cache + run Amazon sync at 06:00 IST every day."""
    while True:
        now = datetime.now(IST)
        target = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        log.info("Next warm-cache refresh at %s IST (in %.0fs)", target.strftime("%Y-%m-%d %H:%M"), wait_seconds)
        await asyncio.sleep(wait_seconds)
        log.info("Running scheduled warm-cache refresh…")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_load_warm_cache)
        # Run Amazon SP-API sync after cache refresh
        await loop.run_in_executor(None, _do_amazon_sync_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load cache in background so server is ready immediately
    asyncio.get_event_loop().run_in_executor(None, _do_load_warm_cache)
    # Schedule daily 6AM IST refresh
    task = asyncio.create_task(_warm_cache_scheduler())
    yield
    task.cancel()


app = FastAPI(
    title="Yash Gallery ERP API",
    version="1.0.0",
    description="FastAPI backend for the Yash Gallery ERP system",
    lifespan=lifespan,
)

# ── CORS (allow Vite dev server + production domain) ─────────
import os as _os
_extra = _os.environ.get("EXTRA_CORS_ORIGIN", "").strip()
_origins = [
    "http://localhost:5173",        # Vite dev
    "http://localhost:3000",        # alternate dev
    "https://progressino.com",      # production root
    "https://www.progressino.com",  # www
    "https://app.progressino.com",  # app subdomain
    "https://yashgallery.com",      # yashgallery production
    "https://www.yashgallery.com",  # www
    "https://app.yashgallery.com",  # app subdomain
]
if _extra:
    _origins.append(_extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,       # needed for the session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth middleware (outermost — runs first) ──────────────────
_AUTH_EXEMPT = {"/api/auth/login", "/api/auth/logout", "/api/health"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT or not request.url.path.startswith("/api/"):
        return await call_next(request)
    token = request.cookies.get("auth_token")
    if not token or not verify_token(token):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
    return await call_next(request)


# ── Session cookie middleware ─────────────────────────────────
SESSION_COOKIE = "session_id"

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE)
    is_new = not (sid and store.get(sid))
    sid, session = store.get_or_create(sid)
    request.state.session_id = sid
    request.state.session = session

    # If this is a brand-new session and warm cache is ready, pre-populate it
    # so the user has data immediately without a manual cache load.
    if is_new and session.mtr_df.empty and session.sales_df.empty:
        _copy_warm_cache_to_session(session)

    response: Response = await call_next(request)

    # Set / refresh cookie on every response
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        max_age=12 * 3600,  # 12 hours
    )
    return response


# ── Routers ───────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/api/auth",       tags=["auth"])
app.include_router(upload.router,      prefix="/api/upload",     tags=["upload"])
app.include_router(data.router,        prefix="/api/data",       tags=["data"])
app.include_router(cache.router,       prefix="/api/cache",      tags=["cache"])
app.include_router(po.router,          prefix="/api/po",         tags=["po"])
app.include_router(finance_router,     prefix="/api/finance",    tags=["finance"])
app.include_router(item_router,        prefix="/api/items",      tags=["items"])
app.include_router(sales_router,       prefix="/api/sales",      tags=["sales"])
app.include_router(purchase_router,    prefix="/api/purchase",   tags=["purchase"])
app.include_router(tna_router,         prefix="/api/tna",        tags=["tna"])
app.include_router(production_router,  prefix="/api/production", tags=["production"])
app.include_router(grey_router,        prefix="/api/grey",       tags=["grey"])
app.include_router(erp_admin_router,   prefix="/api/erp-admin",  tags=["erp-admin"])
app.include_router(marketplace_router, prefix="/api/marketplace", tags=["marketplace"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sessions": store.count,
        "warm_cache": bool(_warm_cache),
        "warm_cache_loaded_at": _warm_cache_loaded_at.isoformat() if _warm_cache_loaded_at else None,
    }

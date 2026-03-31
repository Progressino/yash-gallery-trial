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
        from .services.daily_store import load_all_platforms, merge_platform_data as _merge, save_daily_file
        import pandas as pd

        ok, msg, loaded = load_cache_from_drive()
        if not ok or not loaded:
            log.warning("Warm-cache load failed: %s — falling back to SQLite only", msg)
            # GitHub failed — build cache entirely from SQLite daily store
            try:
                daily = load_all_platforms()
                if daily:
                    loaded = {
                        "mtr_df":      daily.get("amazon",   pd.DataFrame()),
                        "myntra_df":   daily.get("myntra",   pd.DataFrame()),
                        "meesho_df":   daily.get("meesho",   pd.DataFrame()),
                        "flipkart_df": daily.get("flipkart", pd.DataFrame()),
                        "snapdeal_df": pd.DataFrame(),
                        "sales_df":    pd.DataFrame(),
                        "sku_mapping": {},
                        "inventory_df_variant": pd.DataFrame(),
                        "inventory_df_parent":  pd.DataFrame(),
                    }
                    # Rebuild sales_df from what SQLite has
                    if any(not v.empty for v in [loaded["mtr_df"], loaded["myntra_df"],
                                                  loaded["meesho_df"], loaded["flipkart_df"]]):
                        from .services.sales import build_sales_df
                        loaded["sales_df"] = build_sales_df(
                            mtr_df=loaded["mtr_df"], myntra_df=loaded["myntra_df"],
                            meesho_df=loaded["meesho_df"], flipkart_df=loaded["flipkart_df"],
                            snapdeal_df=loaded["snapdeal_df"],
                        )
                    log.info("Warm cache rebuilt from SQLite: %d sales rows",
                             len(loaded.get("sales_df", pd.DataFrame())))
                else:
                    return False
            except Exception as e2:
                log.warning("SQLite fallback also failed: %s", e2)
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

        # Merge daily SQLite store on top of GitHub cache
        try:
            daily = load_all_platforms()
            for plat, key in [("amazon","mtr_df"),("myntra","myntra_df"),
                               ("meesho","meesho_df"),("flipkart","flipkart_df")]:
                if daily.get(plat) is not None and not daily[plat].empty:
                    loaded[key] = _merge(loaded.get(key, pd.DataFrame()), daily[plat], plat)
        except Exception as e:
            log.warning("Daily-store merge warning: %s", e)

        # Back up combined data to SQLite so future restarts survive GitHub failures
        try:
            for plat, key in [("amazon","mtr_df"),("myntra","myntra_df"),
                               ("meesho","meesho_df"),("flipkart","flipkart_df")]:
                df = loaded.get(key)
                if df is not None and not df.empty:
                    sqlite_rows = len(daily.get(plat, pd.DataFrame())) if 'daily' in dir() else 0
                    if len(df) > sqlite_rows:
                        save_daily_file(plat, f"_cache_backup_{plat}", df)
                        log.info("SQLite backup updated for %s: %d rows", plat, len(df))
        except Exception as e:
            log.warning("SQLite backup warning: %s", e)

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


def _do_marketplace_sync_all() -> None:
    """
    Called daily at 6AM IST. Pulls last 2 days of data from all connected
    marketplace APIs and merges results into warm cache + SQLite daily store.
    """
    from .db.marketplace_db import get_credentials, save_sync_log
    from .services.daily_store import merge_platform_data, save_daily_file
    from datetime import date as _date
    import pandas as pd

    # Map: platform → (sync_fn, cache_key, db_platform)
    _PLATFORM_MAP = {
        "amazon":   ("amazon_sp_api",  "sync_amazon_data",   "mtr_df",      "amazon"),
        "flipkart": ("flipkart_api",   "sync_flipkart_data", "flipkart_df", "flipkart"),
        "myntra":   ("myntra_api",     "sync_myntra_data",   "myntra_df",   "myntra"),
        "meesho":   ("meesho_api",     "sync_meesho_data",   "meesho_df",   "meesho"),
    }

    for platform, (module, fn_name, cache_key, db_platform) in _PLATFORM_MAP.items():
        creds = get_credentials(platform)
        if not creds:
            continue
        log.info("Scheduled %s sync starting…", platform)
        try:
            import importlib
            mod = importlib.import_module(f".services.{module}", package="backend")
            sync_fn = getattr(mod, fn_name)
            df, msg = sync_fn(creds, days_back=2)
        except Exception as e:
            log.exception("Scheduled %s sync failed: %s", platform, e)
            save_sync_log(platform, "error", 0, message=str(e))
            continue

        if df.empty:
            save_sync_log(platform, "partial", 0, message=msg)
            log.info("Scheduled %s: no new data — %s", platform, msg)
            continue

        try:
            save_daily_file(db_platform, f"api-{platform}-{_date.today()}.csv", df)
        except Exception as e:
            log.warning("Scheduled %s: daily store save failed: %s", platform, e)

        try:
            existing = _warm_cache.get(cache_key, pd.DataFrame())
            _warm_cache[cache_key] = merge_platform_data(existing, df, db_platform)
            log.info("Scheduled %s: warm cache updated with %d rows", platform, len(df))
        except Exception as e:
            log.warning("Scheduled %s: warm cache merge failed: %s", platform, e)

        date_col  = "Date"
        date_from = str(df[date_col].min().date()) if date_col in df.columns and not df.empty else ""
        date_to   = str(df[date_col].max().date()) if date_col in df.columns and not df.empty else ""
        save_sync_log(platform, "success", len(df), date_from, date_to, msg)
        log.info("Scheduled %s sync complete: %s", platform, msg)


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
        # Run all marketplace API syncs after cache refresh
        await loop.run_in_executor(None, _do_marketplace_sync_all)


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

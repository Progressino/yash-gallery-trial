"""
Yash Gallery ERP — FastAPI backend
Serves all business logic as a REST API.
Session state is stored server-side keyed by a UUID cookie.
"""
from dotenv import load_dotenv
load_dotenv()   # loads .env from cwd (run from repo root or backend/)

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .session import store
from .routers import upload, data, cache, po, shipment, auth as auth_router
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
from .db.forecast_session_pg import init_db as init_forecast_session_pg

init_db()
init_item_db()
init_marketplace_db()
init_sales_db()
init_purchase_db()
init_tna_db()
init_production_db()
init_grey_db()
init_users_db()
init_forecast_session_pg()

log = logging.getLogger("erp.cache_warmer")

# ── Shared warm cache (app-level, not per-session) ────────────────
# Populated at startup and refreshed daily at 06:00 IST.
# New/empty sessions copy from here so users never wait for a GitHub download.
_warm_cache: dict = {}
_warm_cache_loaded_at: datetime | None = None
# Monotonically increasing counter: increments on every successful warm-cache
# update (Phase 1 = 1, Phase 2 = 2, next day refresh = 3, …).  Sessions track
# which generation they last received so the middleware can re-copy Phase-2 data
# to sessions that only got Phase-1 data without requiring a full page reload.
_warm_cache_generation: int = 0
# Set (signalled) once Phase 1 (SQLite) finishes. Cleared by clear_warm_cache().
_warm_cache_ready = threading.Event()

IST = timezone(timedelta(hours=5, minutes=30))

# Keys persisted to GitHub data-cache release (must match github_cache._CACHE_FILES)
_WARM_CACHE_KEYS = (
    "sales_df",
    "mtr_df",
    "meesho_df",
    "myntra_df",
    "flipkart_df",
    "snapdeal_df",
    "sku_mapping",
    "inventory_df_variant",
    "inventory_df_parent",
)


def clear_warm_cache() -> None:
    """Drop app-level warm cache so the next load re-downloads from GitHub / rebuilds from SQLite."""
    global _warm_cache, _warm_cache_loaded_at, _warm_cache_generation
    _warm_cache = {}
    _warm_cache_loaded_at = None
    _warm_cache_generation = 0
    _warm_cache_ready.clear()  # Reset so next load is awaited correctly


def publish_warm_cache_from_session(sess) -> None:
    """Mirror session dataframes into _warm_cache after a full reload or rebuild."""
    global _warm_cache, _warm_cache_loaded_at
    import pandas as pd

    out: dict = {}
    for key in _WARM_CACHE_KEYS:
        val = getattr(sess, key, None)
        if val is None:
            continue
        if key == "sku_mapping" and not isinstance(val, dict):
            out[key] = {}
        elif hasattr(val, "empty"):
            out[key] = val if isinstance(val, pd.DataFrame) else pd.DataFrame()
        else:
            out[key] = val
    _warm_cache = out
    _warm_cache_loaded_at = datetime.now(IST)


def _do_load_warm_cache() -> bool:
    """
    Two-phase warm-cache load. Thread-safe (GIL).

    Phase 1 (~2 s): Load platform DataFrames from the local SQLite daily store,
    publish to _warm_cache, and immediately signal _warm_cache_ready so the first
    page-load after a deploy returns data within a couple of seconds.

    Phase 2 (~60–90 s): Download GitHub historical cache in the same thread (no
    extra thread needed — lifespan already runs this in an executor). Strip any
    GitHub rows whose dates are already covered by the SQLite store (daily store
    always wins), merge, rebuild sales_df, and update _warm_cache.  Existing
    sessions that received Phase-1 data get the fuller historical view on their
    next fresh request (new session or page reload).
    """
    global _warm_cache, _warm_cache_loaded_at, _warm_cache_generation
    try:
        from .services.github_cache import load_cache_from_drive
        from .services.daily_store import load_all_platforms, merge_platform_data as _merge
        from .services.sales import build_sales_df
        import pandas as pd

        def _sanitise_snapdeal(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "OMS_SKU" not in df.columns:
                return df
            bad = df["OMS_SKU"].astype(str).str.upper()
            return df[
                ~(bad.isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])
                  | df["OMS_SKU"].astype(str).str.match(r'^\d+$'))
            ].reset_index(drop=True)

        # ── Phase 1: recent SQLite data (local, fast ~5-15s) ─────────────────
        # Load only the last 4 months of uploads so this completes quickly.
        # Phase 2 provides the full history from GitHub for older date ranges.
        from .services.daily_store import load_platform_data as _load_plat
        _P1_MONTHS = 4   # covers last ~120 days — typical dashboard range

        phase1_ok = False
        try:
            p1_raw: dict = {}
            for _plat in ("amazon", "myntra", "meesho", "flipkart", "snapdeal"):
                _df = _load_plat(_plat, months=_P1_MONTHS)
                if not _df.empty:
                    p1_raw[_plat] = _df
            if p1_raw:
                p1: dict = {
                    "mtr_df":               p1_raw.get("amazon",   pd.DataFrame()),
                    "myntra_df":            p1_raw.get("myntra",   pd.DataFrame()),
                    "meesho_df":            p1_raw.get("meesho",   pd.DataFrame()),
                    "flipkart_df":          p1_raw.get("flipkart", pd.DataFrame()),
                    "snapdeal_df":          _sanitise_snapdeal(p1_raw.get("snapdeal", pd.DataFrame())),
                    "sales_df":             pd.DataFrame(),
                    "sku_mapping":          {},
                    "inventory_df_variant": pd.DataFrame(),
                    "inventory_df_parent":  pd.DataFrame(),
                }
                has_sales = any(
                    not v.empty
                    for v in [p1["mtr_df"], p1["myntra_df"], p1["meesho_df"],
                               p1["flipkart_df"], p1["snapdeal_df"]]
                )
                if has_sales:
                    p1["sales_df"] = build_sales_df(
                        mtr_df=p1["mtr_df"],       myntra_df=p1["myntra_df"],
                        meesho_df=p1["meesho_df"], flipkart_df=p1["flipkart_df"],
                        sku_mapping={},            snapdeal_df=p1["snapdeal_df"],
                    )
                _warm_cache = p1
                _warm_cache_loaded_at = datetime.now(IST)
                _warm_cache_generation += 1   # generation 1 = Phase-1 SQLite data
                _warm_cache_ready.set()       # ← unblocks first page-load immediately
                phase1_ok = True
                log.info(
                    "Warm-cache Phase 1 ready: %d sales rows (last %d months from SQLite). "
                    "Fetching GitHub historical cache…",
                    len(p1.get("sales_df", pd.DataFrame())), _P1_MONTHS,
                )
        except Exception as e:
            log.warning("Warm-cache Phase 1 (SQLite) failed: %s — continuing to GitHub", e)

        # ── Phase 2: GitHub historical cache (network, slow) ──────────────────
        # Provides data for dates not yet in the SQLite daily store.
        ok, msg, loaded = load_cache_from_drive()
        if not ok or not loaded:
            log.warning("Warm-cache Phase 2 (GitHub) failed: %s", msg)
            if not phase1_ok:
                # Nothing at all — try a pure SQLite fallback so we return something
                if phase1_daily:
                    log.info("Using Phase 1 SQLite data only (GitHub unavailable).")
                    return True
                return False
            return True   # Phase 1 data is still good

        # Sanitise Snapdeal from GitHub
        loaded["snapdeal_df"] = _sanitise_snapdeal(loaded.get("snapdeal_df", pd.DataFrame()))

        # Date-superseding merge: for every date covered by the SQLite daily store,
        # strip that date from the GitHub cache before merging so the freshly-parsed
        # SQLite rows always win (prevents double-counting after parser fixes).
        try:
            # Re-read SQLite so we pick up any upload that arrived during Phase 2.
            daily = load_all_platforms()
            merged_any = False
            for plat, key in [
                ("amazon",   "mtr_df"),
                ("myntra",   "myntra_df"),
                ("meesho",   "meesho_df"),
                ("flipkart", "flipkart_df"),
                ("snapdeal", "snapdeal_df"),
            ]:
                if daily.get(plat) is not None and not daily[plat].empty:
                    daily_df  = daily[plat]
                    github_df = loaded.get(key, pd.DataFrame())
                    if (not github_df.empty
                            and "Date" in github_df.columns
                            and "Date" in daily_df.columns):
                        try:
                            daily_dates  = set(pd.to_datetime(daily_df["Date"],  errors="coerce").dt.normalize())
                            github_dates =     pd.to_datetime(github_df["Date"], errors="coerce").dt.normalize()
                            github_df = github_df[~github_dates.isin(daily_dates)].copy()
                            loaded[key] = github_df
                        except Exception:
                            pass
                    loaded[key] = _merge(loaded.get(key, pd.DataFrame()), daily_df, plat)
                    merged_any = True
            if merged_any:
                log.info("Phase 2: SQLite merged on top of GitHub for %d platform(s).", merged_any)
        except Exception as e:
            log.warning("Phase 2 daily-store merge warning: %s", e)

        # Rebuild unified sales_df from the fully-merged platform data
        try:
            loaded["sales_df"] = build_sales_df(
                mtr_df=loaded.get("mtr_df", pd.DataFrame()),
                myntra_df=loaded.get("myntra_df", pd.DataFrame()),
                meesho_df=loaded.get("meesho_df", pd.DataFrame()),
                flipkart_df=loaded.get("flipkart_df", pd.DataFrame()),
                sku_mapping=loaded.get("sku_mapping") or {},
                snapdeal_df=loaded.get("snapdeal_df"),
            )
            log.info("Phase 2 sales_df: %d rows", len(loaded["sales_df"]))
        except Exception as e:
            log.warning("Phase 2 sales_df rebuild failed: %s", e)
            if phase1_ok:
                loaded.setdefault("sales_df", _warm_cache.get("sales_df", pd.DataFrame()))

        _warm_cache = loaded
        _warm_cache_loaded_at = datetime.now(IST)
        _warm_cache_generation += 1   # generation 2+ = Phase-2 GitHub+SQLite data
        log.info("Warm cache fully loaded (Phase 2, gen=%d) at %s — %s",
                 _warm_cache_generation, _warm_cache_loaded_at.strftime("%H:%M IST"), msg)
        return True

    except Exception as e:
        log.exception("Warm cache load error: %s", e)
        return False

    finally:
        # Always signal so waiting code is never blocked forever
        # (no-op if Phase 1 already called set())
        _warm_cache_ready.set()


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

        api_sync_fname = f"api-{platform}-{_date.today()}.csv"
        try:
            save_daily_file(db_platform, api_sync_fname, df)
        except Exception as e:
            log.warning("Scheduled %s: daily store save failed: %s", platform, e)

        try:
            existing = _warm_cache.get(cache_key, pd.DataFrame())
            _warm_cache[cache_key] = merge_platform_data(
                existing, df, db_platform, source_filename=api_sync_fname,
            )
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
    setattr(session, "_persist_sid", sid)

    # Pre-populate session from warm cache whenever the session has no data, OR
    # when the warm cache has moved to a newer generation (Phase 2 after Phase 1)
    # and the session still has only auto-copied data (no user uploads).
    # _warm_cache_only=True means the session data came exclusively from an auto-copy
    # of the warm cache — no explicit upload or load has run since.  Upload routes
    # call resume_auto_data_restore() which sets _warm_cache_only=False.
    copied_warm = False
    _session_gen   = getattr(session, "_warm_cache_gen", 0)
    _wc_only       = getattr(session, "_warm_cache_only", False)
    _phase2_ready  = _warm_cache_generation >= 2 and _session_gen < _warm_cache_generation
    if not getattr(session, "pause_auto_data_restore", False):
        if session.mtr_df.empty and session.sales_df.empty:
            # Session has no data — copy from warm cache (available ~2 s after restart)
            copied_warm = _copy_warm_cache_to_session(session)
            if copied_warm:
                session._warm_cache_gen  = _warm_cache_generation
                session._warm_cache_only = True
        elif _wc_only and _phase2_ready:
            # Session has Phase-1 SQLite data only; Phase 2 (GitHub historical) is ready.
            # Seamlessly upgrade — no user-uploaded data to protect.
            copied_warm = _copy_warm_cache_to_session(session)
            if copied_warm:
                session._warm_cache_gen  = _warm_cache_generation
                session._warm_cache_only = True

    try:
        from .services.sku_mapping import ensure_default_sku_mapping_from_bundle

        ensure_default_sku_mapping_from_bundle(session)
    except Exception:
        pass

    response: Response = await call_next(request)

    try:
        from .db.forecast_session_pg import debounced_persist_session, pg_session_persist_enabled

        if pg_session_persist_enabled() and sid:
            if copied_warm:
                debounced_persist_session(sid, session, delay=5.0)
            elif request.method in frozenset({"POST", "PUT", "DELETE", "PATCH"}) and request.url.path.startswith(
                "/api/"
            ):
                debounced_persist_session(sid, session, delay=12.0)
    except Exception:
        pass

    # Set / refresh cookie on every response
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        max_age=14 * 24 * 3600,  # 14 days — must stay stable while PostgreSQL stores session blobs by this id
    )
    return response


# ── Routers ───────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/api/auth",       tags=["auth"])
app.include_router(upload.router,      prefix="/api/upload",     tags=["upload"])
app.include_router(data.router,        prefix="/api/data",       tags=["data"])
app.include_router(cache.router,       prefix="/api/cache",      tags=["cache"])
app.include_router(po.router,          prefix="/api/po",         tags=["po"])
app.include_router(shipment.router,    prefix="/api/shipment",   tags=["shipment"])
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
        # generation: 0=loading, 1=Phase1 SQLite ready, 2=Phase2 GitHub+SQLite ready
        "warm_cache_generation": _warm_cache_generation,
    }

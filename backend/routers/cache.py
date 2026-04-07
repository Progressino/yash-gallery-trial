"""
GitHub Releases cache router.
POST /api/cache/save   → upload parquet files to GitHub Release
POST /api/cache/load   → download from GitHub Release into session
POST /api/cache/reload-fresh → clear warm + session, full GitHub download, daily merge, rebuild sales
GET  /api/cache/status → check if cache exists
"""
import logging
from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel

import pandas as pd

from ..services.sales import build_sales_df
from ..services.github_cache import save_cache_to_drive, load_cache_from_drive, get_cache_manifest, delete_github_cache_assets
from ..services.daily_store import (
    load_all_platforms,
    merge_platform_data as _merge_platform_data,
    clear_all_daily_uploads,
)
from ..session import wipe_app_session, resume_auto_data_restore

router = APIRouter()
_log = logging.getLogger(__name__)


class CacheStatusResponse(BaseModel):
    ok: bool
    message: str


class CacheReloadResponse(BaseModel):
    ok: bool
    message: str
    sales_rows: int = 0


class ResetAllBody(BaseModel):
    """Wipe this session clean so the user can upload from scratch."""
    clear_tier3_sqlite: bool = False  # delete all rows in server daily_sales.db (Tier-3 history)
    clear_warm_cache: bool = True     # clear shared in-memory cache for all sessions on this server
    clear_github_cache: bool = False  # delete all assets from the GitHub Release cache


class ResetAllResponse(BaseModel):
    ok: bool
    message: str
    tier3_deleted: int = 0


def _sanitize_snapdeal_in_loaded(loaded: dict) -> None:
    if "snapdeal_df" not in loaded or not isinstance(loaded["snapdeal_df"], pd.DataFrame):
        return
    snap = loaded["snapdeal_df"]
    if snap.empty or "OMS_SKU" not in snap.columns:
        return
    _bad = snap["OMS_SKU"].astype(str).str.upper()
    loaded["snapdeal_df"] = snap[
        ~(_bad.isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])
          | snap["OMS_SKU"].astype(str).str.match(r"^\d+$"))
    ].reset_index(drop=True)


def _merge_daily_store_into_session(sess) -> str:
    """Layer SQLite daily uploads onto session platform DFs. Returns suffix for messages."""
    try:
        daily_data = load_all_platforms()
        if daily_data.get("amazon") is not None and not daily_data["amazon"].empty:
            sess.mtr_df = _merge_platform_data(sess.mtr_df, daily_data["amazon"], "amazon")
        if daily_data.get("myntra") is not None and not daily_data["myntra"].empty:
            sess.myntra_df = _merge_platform_data(sess.myntra_df, daily_data["myntra"], "myntra")
        if daily_data.get("meesho") is not None and not daily_data["meesho"].empty:
            sess.meesho_df = _merge_platform_data(sess.meesho_df, daily_data["meesho"], "meesho")
        if daily_data.get("flipkart") is not None and not daily_data["flipkart"].empty:
            sess.flipkart_df = _merge_platform_data(sess.flipkart_df, daily_data["flipkart"], "flipkart")
        n = sum(1 for v in daily_data.values() if hasattr(v, "empty") and not v.empty)
        return f" + {n} platform(s) daily store merged." if n else ""
    except Exception as e:
        return f" (daily store warning: {e})"


def _rebuild_sales_in_session(sess) -> int:
    """Rebuild unified sales_df with current server code; returns row count."""
    if not sess.sku_mapping:
        return 0
    try:
        sess.sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
        )
        return len(sess.sales_df)
    except Exception as e:
        _log.exception("rebuild sales: %s", e)
        return len(sess.sales_df)


def _auto_save_cache(sess) -> None:
    session_data = {
        "sales_df":             sess.sales_df,
        "mtr_df":               sess.mtr_df,
        "meesho_df":            sess.meesho_df,
        "myntra_df":            sess.myntra_df,
        "flipkart_df":          sess.flipkart_df,
        "snapdeal_df":          sess.snapdeal_df,
        "sku_mapping":          sess.sku_mapping,
        "inventory_df_variant": sess.inventory_df_variant,
        "inventory_df_parent":  sess.inventory_df_parent,
    }
    ok, msg = save_cache_to_drive(session_data)
    if ok:
        _log.info("reload-fresh cache save: %s", msg)
    else:
        _log.warning("reload-fresh cache save skipped: %s", msg)


@router.get("/status")
def cache_status(request: Request):
    manifest = get_cache_manifest()
    if manifest is None:
        return {"configured": False, "cached": False}
    return {
        "configured":  True,
        "cached":      True,
        "saved_at":    manifest.get("saved_at_display", "?"),
        "row_counts":  manifest.get("row_counts", {}),
    }


@router.post("/save", response_model=CacheStatusResponse)
def cache_save(request: Request):
    sess = request.state.session
    if sess is None:
        return CacheStatusResponse(ok=False, message="No session")

    session_data = {
        "sales_df":    sess.sales_df,
        "mtr_df":      sess.mtr_df,
        "meesho_df":   sess.meesho_df,
        "myntra_df":   sess.myntra_df,
        "flipkart_df": sess.flipkart_df,
        "snapdeal_df": sess.snapdeal_df,
        "sku_mapping": sess.sku_mapping,
    }
    ok, msg = save_cache_to_drive(session_data)
    return CacheStatusResponse(ok=ok, message=msg)


@router.post("/load", response_model=CacheStatusResponse)
def cache_load(request: Request):
    sess = request.state.session
    if sess is None:
        return CacheStatusResponse(ok=False, message="No session")

    # ── Fast path: warm cache already in memory — no GitHub download needed ──
    try:
        import backend.main as _main
        if _main._warm_cache:
            _main._copy_warm_cache_to_session(sess)
            daily_note = _merge_daily_store_into_session(sess)
            n_sales = _rebuild_sales_in_session(sess)
            sess._quarterly_cache.clear()
            _main.publish_warm_cache_from_session(sess)
            resume_auto_data_restore(sess)
            return CacheStatusResponse(
                ok=True,
                message=f"Loaded from warm cache; sales rebuilt ({n_sales:,} rows).{daily_note}",
            )
    except Exception:
        pass  # fall through to GitHub download

    ok, msg, loaded = load_cache_from_drive()
    if ok:
        _sanitize_snapdeal_in_loaded(loaded)
        for key, val in loaded.items():
            if hasattr(sess, key):
                setattr(sess, key, val)
        sess._quarterly_cache.clear()
        daily_note = _merge_daily_store_into_session(sess)
        n_sales = _rebuild_sales_in_session(sess)
        msg = f"{msg}{daily_note} Sales rebuilt: {n_sales:,} rows."
        try:
            import backend.main as _main
            _main.publish_warm_cache_from_session(sess)
        except Exception:
            pass
        resume_auto_data_restore(sess)

    return CacheStatusResponse(ok=ok, message=msg)


@router.delete("", response_model=CacheStatusResponse)
def cache_clear(request: Request, include_warm: bool = False):
    """Clear this browser session (all platforms, inventory, PO sheet, sales). Optional: server warm cache."""
    sess = request.state.session
    if sess is None:
        return CacheStatusResponse(ok=False, message="No session")
    if include_warm:
        import backend.main as _main
        _main.clear_warm_cache()
    wipe_app_session(sess)
    msg = "Session wiped (all data). Server warm cache cleared too." if include_warm else "Session wiped (all data)."
    return CacheStatusResponse(ok=True, message=msg)


@router.post("/reset-all", response_model=ResetAllResponse)
def cache_reset_all(request: Request, body: ResetAllBody = ResetAllBody()):
    """
    Full fresh start: clear session + optionally Tier-3 SQLite + optionally warm cache.
    Does not delete GitHub Release cache — next Save Cache updates the cloud after re-upload.
    """
    sess = request.state.session
    if sess is None:
        return ResetAllResponse(ok=False, message="No session", tier3_deleted=0)

    tier3_n = 0
    if body.clear_tier3_sqlite:
        try:
            tier3_n = clear_all_daily_uploads()
        except Exception as e:
            _log.warning("clear_all_daily_uploads: %s", e)
            return ResetAllResponse(
                ok=False,
                message=f"Could not clear Tier-3 database: {e}",
                tier3_deleted=0,
            )

    if body.clear_warm_cache:
        import backend.main as _main
        _main.clear_warm_cache()

    wipe_app_session(sess)

    gh_msg = ""
    if body.clear_github_cache:
        gh_ok, gh_result = delete_github_cache_assets()
        gh_msg = f" GitHub cache: {gh_result}"

    bits = ["Session wiped (SKU map, all platforms, inventory, PO imports, sales)."]
    if body.clear_warm_cache:
        bits.append("Server warm cache cleared.")
    if body.clear_tier3_sqlite:
        bits.append(f"Tier-3 daily store: removed {tier3_n} saved file(s).")
    if gh_msg:
        bits.append(gh_msg.strip())
    return ResetAllResponse(ok=True, message=" ".join(bits), tier3_deleted=tier3_n)


@router.post("/reload-fresh", response_model=CacheReloadResponse)
def cache_reload_fresh(request: Request, background_tasks: BackgroundTasks):
    """
    For operators who already uploaded to GitHub / SQLite: wipe in-memory warm + session,
    pull fresh from GitHub, merge Tier-3 SQLite daily store, rebuild sales_df with current
    server code, re-publish warm cache, and queue Save Cache to GitHub.
    """
    sess = request.state.session
    if sess is None:
        return CacheReloadResponse(ok=False, message="No session", sales_rows=0)

    import backend.main as _main

    _main.clear_warm_cache()
    sess.sales_df    = pd.DataFrame()
    sess.mtr_df      = pd.DataFrame()
    sess.meesho_df   = pd.DataFrame()
    sess.myntra_df   = pd.DataFrame()
    sess.flipkart_df = pd.DataFrame()
    sess.snapdeal_df = pd.DataFrame()
    sess.sku_mapping = {}
    sess._quarterly_cache.clear()

    ok, msg, loaded = load_cache_from_drive()
    if ok:
        _sanitize_snapdeal_in_loaded(loaded)
        for key, val in loaded.items():
            if hasattr(sess, key):
                setattr(sess, key, val)
        daily_note = _merge_daily_store_into_session(sess)
        n_sales = _rebuild_sales_in_session(sess)
        msg = f"Full reload from GitHub.{daily_note} {msg}"
    else:
        if not _main._do_load_warm_cache():
            return CacheReloadResponse(
                ok=False,
                message=msg or "No GitHub cache and could not rebuild from SQLite.",
                sales_rows=0,
            )
        _main._copy_warm_cache_to_session(sess)
        n_sales = len(sess.sales_df)
        msg = "Reloaded from warm pipeline (GitHub unavailable; used SQLite and/or partial cache)."

    resume_auto_data_restore(sess)
    sess._quarterly_cache.clear()
    _main.publish_warm_cache_from_session(sess)
    background_tasks.add_task(_auto_save_cache, sess)
    return CacheReloadResponse(
        ok=True,
        message=f"{msg} Saving to GitHub in background…",
        sales_rows=n_sales,
    )

"""
GitHub Releases cache router.
POST /api/cache/save   → upload parquet files to GitHub Release
POST /api/cache/load   → download from GitHub Release into session
GET  /api/cache/status → check if cache exists
"""
import dataclasses
from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..services.github_cache import save_cache_to_drive, load_cache_from_drive, get_cache_manifest
from ..services.daily_store import load_all_platforms, merge_platform_data as _merge_platform_data

router = APIRouter()


class CacheStatusResponse(BaseModel):
    ok: bool
    message: str


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

    ok, msg, loaded = load_cache_from_drive()
    if ok:
        import pandas as pd
        # Sanitise snapdeal_df — remove stale "UNKNOWN" / invalid SKUs from cached data
        if "snapdeal_df" in loaded and isinstance(loaded["snapdeal_df"], pd.DataFrame):
            snap = loaded["snapdeal_df"]
            if not snap.empty and "OMS_SKU" in snap.columns:
                _bad = snap["OMS_SKU"].astype(str).str.upper()
                loaded["snapdeal_df"] = snap[
                    ~(_bad.isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])
                      | snap["OMS_SKU"].astype(str).str.match(r'^\d+$'))
                ].reset_index(drop=True)
        for key, val in loaded.items():
            setattr(sess, key, val)
        # Invalidate quarterly cache — sales data changed
        sess._quarterly_cache.clear()

        # Merge saved daily uploads into the session (last 30 days, from SQLite)
        # Use _merge_platform_data (not pd.concat) to prevent duplicates when
        # the GitHub cache and SQLite daily store overlap for the same period.
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
            if daily_data:
                n_days = sum(1 for v in daily_data.values() if not v.empty)
                msg += f" + {n_days} platform(s) of daily data loaded from local store."
        except Exception as e:
            msg += f" (daily store warning: {e})"

    return CacheStatusResponse(ok=ok, message=msg)


@router.delete("", response_model=CacheStatusResponse)
def cache_clear(request: Request):
    # Clear from session only (GitHub assets stay until next save)
    sess = request.state.session
    if sess is None:
        return CacheStatusResponse(ok=False, message="No session")
    import pandas as pd
    sess.sales_df    = pd.DataFrame()
    sess.mtr_df      = pd.DataFrame()
    sess.meesho_df   = pd.DataFrame()
    sess.myntra_df   = pd.DataFrame()
    sess.flipkart_df = pd.DataFrame()
    sess.snapdeal_df = pd.DataFrame()
    sess.sku_mapping = {}
    return CacheStatusResponse(ok=True, message="Session data cleared.")

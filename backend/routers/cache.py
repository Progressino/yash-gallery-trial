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
from ..services.daily_store import load_all_platforms

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
        for key, val in loaded.items():
            setattr(sess, key, val)

        # Merge saved daily uploads into the session (last 30 days, from SQLite)
        try:
            import pandas as pd
            daily_data = load_all_platforms()
            if daily_data.get("amazon") is not None and not daily_data["amazon"].empty:
                sess.mtr_df = pd.concat([sess.mtr_df, daily_data["amazon"]], ignore_index=True) if not sess.mtr_df.empty else daily_data["amazon"]
            if daily_data.get("myntra") is not None and not daily_data["myntra"].empty:
                sess.myntra_df = pd.concat([sess.myntra_df, daily_data["myntra"]], ignore_index=True) if not sess.myntra_df.empty else daily_data["myntra"]
            if daily_data.get("meesho") is not None and not daily_data["meesho"].empty:
                sess.meesho_df = pd.concat([sess.meesho_df, daily_data["meesho"]], ignore_index=True) if not sess.meesho_df.empty else daily_data["meesho"]
            if daily_data.get("flipkart") is not None and not daily_data["flipkart"].empty:
                sess.flipkart_df = pd.concat([sess.flipkart_df, daily_data["flipkart"]], ignore_index=True) if not sess.flipkart_df.empty else daily_data["flipkart"]
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

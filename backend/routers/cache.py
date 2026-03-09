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

"""
GitHub Releases cache router.
POST /api/cache/save   → upload parquet files to GitHub Release
POST /api/cache/load   → download from GitHub Release into session
DELETE /api/cache      → delete all cached assets
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CacheStatusResponse(BaseModel):
    ok: bool
    message: str


@router.post("/save", response_model=CacheStatusResponse)
def cache_save(request: Request):
    # TODO: extract save_cache_to_drive() from app.py
    return CacheStatusResponse(ok=False, message="Cache save not yet implemented")


@router.post("/load", response_model=CacheStatusResponse)
def cache_load(request: Request):
    # TODO: extract load_cache_from_drive() from app.py
    return CacheStatusResponse(ok=False, message="Cache load not yet implemented")


@router.delete("", response_model=CacheStatusResponse)
def cache_clear(request: Request):
    # TODO: extract clear_drive_cache() from app.py
    return CacheStatusResponse(ok=False, message="Cache clear not yet implemented")

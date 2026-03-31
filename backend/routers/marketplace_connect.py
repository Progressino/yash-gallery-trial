"""
Marketplace connections router.
Supports Amazon (SP-API), Flipkart, Myntra, Meesho.

Endpoints per platform:
  POST   /api/marketplace/{platform}/connect     — save/update credentials
  DELETE /api/marketplace/{platform}/disconnect  — remove credentials
  POST   /api/marketplace/{platform}/sync        — manual sync trigger
  GET    /api/marketplace/{platform}/sync-log    — recent sync history
  GET    /api/marketplace/status                 — all platforms status
"""
import logging
from datetime import date

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from ..db.marketplace_db import (
    delete_credentials,
    get_credentials,
    get_last_sync,
    list_connected_platforms,
    list_sync_log,
    save_credentials,
    save_sync_log,
)

log    = logging.getLogger("erp.marketplace")
router = APIRouter()

SUPPORTED_PLATFORMS = ["amazon", "flipkart", "myntra", "meesho"]


# ── Pydantic models ───────────────────────────────────────────────────────────

class AmazonCredentials(BaseModel):
    client_id:      str
    client_secret:  str
    refresh_token:  str
    seller_id:      str
    marketplace_id: str = "A21TJRUUN4KGV"


class FlipkartCredentials(BaseModel):
    app_id:     str   # client_id in DB
    app_secret: str   # client_secret in DB
    seller_id:  str


class MyntraCredentials(BaseModel):
    username:  str   # stored in refresh_token field (encrypted)
    password:  str   # stored in client_secret field (encrypted)
    api_key:   str   # stored in client_id field
    seller_id: str = ""


class MeeshoCredentials(BaseModel):
    client_id:     str
    client_secret: str
    supplier_id:   str  # stored in seller_id


# ── Generic sync helper ───────────────────────────────────────────────────────

def _run_sync(platform: str, creds: dict, days_back: int, session=None) -> None:
    """
    Dispatch to the correct API service, then persist + update warm cache.
    Called as a background task.
    """
    import pandas as pd
    from ..services.daily_store import merge_platform_data, save_daily_file

    # Attach SKU mapping so API clients can map seller SKUs → OMS SKUs
    if session and not creds.get("sku_mapping"):
        creds = dict(creds, sku_mapping=session.sku_mapping or {})

    # --- call the right service ---
    try:
        if platform == "amazon":
            from ..services.amazon_sp_api import sync_amazon_data
            df, msg = sync_amazon_data(creds, days_back=days_back)
            db_platform = "amazon"
            session_attr = "mtr_df"
        elif platform == "flipkart":
            from ..services.flipkart_api import sync_flipkart_data
            df, msg = sync_flipkart_data(creds, days_back=days_back)
            db_platform = "flipkart"
            session_attr = "flipkart_df"
        elif platform == "myntra":
            from ..services.myntra_api import sync_myntra_data
            df, msg = sync_myntra_data(creds, days_back=days_back)
            db_platform = "myntra"
            session_attr = "myntra_df"
        elif platform == "meesho":
            from ..services.meesho_api import sync_meesho_data
            df, msg = sync_meesho_data(creds, days_back=days_back)
            db_platform = "meesho"
            session_attr = "meesho_df"
        else:
            log.warning("Unknown platform for sync: %s", platform)
            return
    except Exception as e:
        log.exception("%s sync error: %s", platform, e)
        save_sync_log(platform, "error", 0, message=str(e))
        return

    if df.empty:
        save_sync_log(platform, "partial", 0, message=msg)
        log.info("%s sync: no new data — %s", platform, msg)
        return

    # Persist to SQLite daily store
    try:
        filename = f"api-{platform}-{date.today()}.csv"
        save_daily_file(db_platform, filename, df)
    except Exception as e:
        log.warning("%s: daily store save failed: %s", platform, e)

    # Merge into current session
    if session is not None:
        try:
            existing = getattr(session, session_attr, pd.DataFrame())
            merged   = merge_platform_data(existing, df, db_platform)
            setattr(session, session_attr, merged)
            # Rebuild unified sales_df
            from ..services.sales import build_sales_df
            session.sales_df = build_sales_df(
                mtr_df=session.mtr_df,
                myntra_df=session.myntra_df,
                meesho_df=session.meesho_df,
                flipkart_df=session.flipkart_df,
                snapdeal_df=session.snapdeal_df,
                sku_mapping=session.sku_mapping,
            )
            session._quarterly_cache.clear()
        except Exception as e:
            log.warning("%s: session merge failed: %s", platform, e)

    # Update app-level warm cache
    try:
        import backend.main as _main
        cache_key = "mtr_df" if platform == "amazon" else f"{platform}_df"
        existing  = _main._warm_cache.get(cache_key, pd.DataFrame())
        _main._warm_cache[cache_key] = merge_platform_data(existing, df, db_platform)
    except Exception as e:
        log.warning("%s: warm cache update failed: %s", platform, e)

    date_col  = "Date" if platform == "amazon" else "Date"
    date_from = str(df[date_col].min().date()) if date_col in df.columns else ""
    date_to   = str(df[date_col].max().date()) if date_col in df.columns else ""
    save_sync_log(platform, "success", len(df), date_from, date_to, msg)
    log.info("%s sync complete: %s", platform, msg)


# ── Status endpoint ───────────────────────────────────────────────────────────

@router.get("/status")
def connection_status():
    """Returns connection + last sync info for all platforms."""
    connected = {p["platform"] for p in list_connected_platforms()}
    result = {}
    for platform in SUPPORTED_PLATFORMS:
        last = get_last_sync(platform)
        result[platform] = {
            "connected":    platform in connected,
            "last_sync":    last["synced_at"]  if last else None,
            "last_status":  last["status"]     if last else None,
            "last_rows":    last["rows_added"] if last else 0,
            "last_message": last["message"]    if last else "",
        }
    return result


# ── Amazon ────────────────────────────────────────────────────────────────────

@router.post("/amazon/connect")
def amazon_connect(creds: AmazonCredentials):
    try:
        from ..services.amazon_sp_api import get_access_token
        get_access_token(creds.client_id, creds.client_secret, creds.refresh_token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Credential test failed: {e}")
    save_credentials(
        platform="amazon",
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        refresh_token=creds.refresh_token,
        seller_id=creds.seller_id,
        marketplace_id=creds.marketplace_id,
    )
    return {"ok": True, "message": "Amazon credentials saved and verified."}


@router.post("/amazon/sync")
def amazon_sync(request: Request, background_tasks: BackgroundTasks, days_back: int = 7):
    creds = get_credentials("amazon")
    if not creds:
        raise HTTPException(status_code=400, detail="Amazon not connected.")
    background_tasks.add_task(_run_sync, "amazon", creds, days_back, getattr(request.state, "session", None))
    return {"ok": True, "message": f"Amazon sync started (last {days_back} days)."}


@router.get("/amazon/sync-log")
def amazon_sync_log(limit: int = 20):
    return list_sync_log("amazon", limit=limit)


@router.delete("/amazon/disconnect")
def amazon_disconnect():
    if not delete_credentials("amazon"):
        raise HTTPException(status_code=404, detail="Amazon credentials not found.")
    return {"ok": True, "message": "Amazon disconnected."}


# ── Flipkart ──────────────────────────────────────────────────────────────────

@router.post("/flipkart/connect")
def flipkart_connect(creds: FlipkartCredentials):
    try:
        from ..services.flipkart_api import get_fk_access_token
        get_fk_access_token(creds.app_id, creds.app_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Flipkart credential test failed: {e}")
    save_credentials(
        platform="flipkart",
        client_id=creds.app_id,
        client_secret=creds.app_secret,
        seller_id=creds.seller_id,
    )
    return {"ok": True, "message": "Flipkart credentials saved and verified."}


@router.post("/flipkart/sync")
def flipkart_sync(request: Request, background_tasks: BackgroundTasks, days_back: int = 7):
    creds = get_credentials("flipkart")
    if not creds:
        raise HTTPException(status_code=400, detail="Flipkart not connected.")
    background_tasks.add_task(_run_sync, "flipkart", creds, days_back, getattr(request.state, "session", None))
    return {"ok": True, "message": f"Flipkart sync started (last {days_back} days)."}


@router.get("/flipkart/sync-log")
def flipkart_sync_log(limit: int = 20):
    return list_sync_log("flipkart", limit=limit)


@router.delete("/flipkart/disconnect")
def flipkart_disconnect():
    if not delete_credentials("flipkart"):
        raise HTTPException(status_code=404, detail="Flipkart credentials not found.")
    return {"ok": True, "message": "Flipkart disconnected."}


# ── Myntra ────────────────────────────────────────────────────────────────────

@router.post("/myntra/connect")
def myntra_connect(creds: MyntraCredentials):
    try:
        from ..services.myntra_api import test_myntra_connection
        ok = test_myntra_connection(creds.username, creds.password, creds.api_key)
        if not ok:
            raise ValueError("Connection test returned failure — check credentials")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Myntra credential test failed: {e}")
    # client_id = api_key, client_secret = password (encrypted), refresh_token = username (encrypted)
    save_credentials(
        platform="myntra",
        client_id=creds.api_key,
        client_secret=creds.password,
        refresh_token=creds.username,
        seller_id=creds.seller_id or "",
    )
    return {"ok": True, "message": "Myntra credentials saved and verified."}


@router.post("/myntra/sync")
def myntra_sync(request: Request, background_tasks: BackgroundTasks, days_back: int = 7):
    creds = get_credentials("myntra")
    if not creds:
        raise HTTPException(status_code=400, detail="Myntra not connected.")
    background_tasks.add_task(_run_sync, "myntra", creds, days_back, getattr(request.state, "session", None))
    return {"ok": True, "message": f"Myntra sync started (last {days_back} days)."}


@router.get("/myntra/sync-log")
def myntra_sync_log(limit: int = 20):
    return list_sync_log("myntra", limit=limit)


@router.delete("/myntra/disconnect")
def myntra_disconnect():
    if not delete_credentials("myntra"):
        raise HTTPException(status_code=404, detail="Myntra credentials not found.")
    return {"ok": True, "message": "Myntra disconnected."}


# ── Meesho ────────────────────────────────────────────────────────────────────

@router.post("/meesho/connect")
def meesho_connect(creds: MeeshoCredentials):
    try:
        from ..services.meesho_api import test_meesho_connection
        ok = test_meesho_connection(creds.client_id, creds.client_secret, creds.supplier_id)
        if not ok:
            raise ValueError("Connection test returned failure — check credentials")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Meesho credential test failed: {e}")
    save_credentials(
        platform="meesho",
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        seller_id=creds.supplier_id,
    )
    return {"ok": True, "message": "Meesho credentials saved and verified."}


@router.post("/meesho/sync")
def meesho_sync(request: Request, background_tasks: BackgroundTasks, days_back: int = 7):
    creds = get_credentials("meesho")
    if not creds:
        raise HTTPException(status_code=400, detail="Meesho not connected.")
    background_tasks.add_task(_run_sync, "meesho", creds, days_back, getattr(request.state, "session", None))
    return {"ok": True, "message": f"Meesho sync started (last {days_back} days)."}


@router.get("/meesho/sync-log")
def meesho_sync_log(limit: int = 20):
    return list_sync_log("meesho", limit=limit)


@router.delete("/meesho/disconnect")
def meesho_disconnect():
    if not delete_credentials("meesho"):
        raise HTTPException(status_code=404, detail="Meesho credentials not found.")
    return {"ok": True, "message": "Meesho disconnected."}

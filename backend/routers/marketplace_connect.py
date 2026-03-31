"""
Marketplace connections router.
POST   /api/marketplace/amazon/connect     — save / update credentials
GET    /api/marketplace/status             — connection status + last sync per platform
POST   /api/marketplace/amazon/sync        — manual sync trigger (background task)
GET    /api/marketplace/amazon/sync-log    — recent sync history
DELETE /api/marketplace/amazon/disconnect  — remove credentials
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


# ── Pydantic models ───────────────────────────────────────────────────────────

class AmazonCredentials(BaseModel):
    client_id:      str
    client_secret:  str
    refresh_token:  str
    seller_id:      str
    marketplace_id: str = "A21TJRUUN4KGV"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _do_amazon_sync(days_back: int = 7, session=None) -> None:
    """
    Background-safe sync function. Pulls SP-API data, merges into session + stores.
    Called from both the manual endpoint and the 6AM IST scheduler.
    """
    import pandas as pd
    from ..services.amazon_sp_api import sync_amazon_data
    from ..services.daily_store import merge_platform_data, save_daily_file

    creds = get_credentials("amazon")
    if not creds:
        log.warning("Amazon sync triggered but no credentials found")
        return

    df, msg = sync_amazon_data(creds, days_back=days_back)
    rows_added = len(df)

    if df.empty:
        save_sync_log("amazon", "partial", 0, message=msg)
        log.info("Amazon SP-API sync: no rows — %s", msg)
        return

    # Persist to SQLite daily store
    try:
        filename = f"sp-api-{date.today()}.csv"
        save_daily_file("amazon", filename, df)
    except Exception as e:
        log.warning("SP-API: failed to save to daily store: %s", e)

    # Merge into current session if available
    if session is not None:
        try:
            session.mtr_df = merge_platform_data(session.mtr_df, df, "amazon")
            # Rebuild sales_df
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
            log.warning("SP-API: failed to merge into session: %s", e)

    # Update warm cache (app-level)
    try:
        import backend.main as _main
        existing = _main._warm_cache.get("mtr_df", pd.DataFrame())
        _main._warm_cache["mtr_df"] = merge_platform_data(existing, df, "amazon")
        log.info("Amazon SP-API: warm cache updated with %d new rows", rows_added)
    except Exception as e:
        log.warning("SP-API: failed to update warm cache: %s", e)

    # Record sync success
    date_from = str(df["Date"].min().date()) if "Date" in df.columns and not df.empty else ""
    date_to   = str(df["Date"].max().date()) if "Date" in df.columns and not df.empty else ""
    save_sync_log("amazon", "success", rows_added, date_from, date_to, msg)
    log.info("Amazon SP-API sync complete: %s", msg)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/amazon/connect")
def amazon_connect(creds: AmazonCredentials):
    """
    Save or update Amazon SP-API credentials.
    Performs a quick token test to verify credentials before saving.
    """
    # Validate credentials with a test token call
    try:
        from ..services.amazon_sp_api import get_access_token
        get_access_token(creds.client_id, creds.client_secret, creds.refresh_token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Credential test failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not verify credentials: {e}")

    save_credentials(
        platform="amazon",
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        refresh_token=creds.refresh_token,
        seller_id=creds.seller_id,
        marketplace_id=creds.marketplace_id,
    )
    return {"ok": True, "message": "Amazon credentials saved. Connection verified."}


@router.get("/status")
def connection_status():
    """Returns connection status and last sync info for all platforms."""
    platforms = ["amazon", "myntra", "meesho", "flipkart", "snapdeal"]
    result = {}
    connected = {p["platform"] for p in list_connected_platforms()}

    for platform in platforms:
        last = get_last_sync(platform)
        result[platform] = {
            "connected":    platform in connected,
            "last_sync":    last["synced_at"] if last else None,
            "last_status":  last["status"]    if last else None,
            "last_rows":    last["rows_added"] if last else 0,
            "last_message": last["message"]   if last else "",
        }
    return result


@router.post("/amazon/sync")
def amazon_sync(
    request: Request,
    background_tasks: BackgroundTasks,
    days_back: int = 7,
):
    """
    Trigger a manual Amazon SP-API sync in the background.
    Returns immediately — sync runs asynchronously.
    """
    creds = get_credentials("amazon")
    if not creds:
        raise HTTPException(
            status_code=400,
            detail="Amazon not connected. Add credentials first via /marketplace/amazon/connect."
        )
    session = getattr(request.state, "session", None)
    background_tasks.add_task(_do_amazon_sync, days_back=days_back, session=session)
    return {
        "ok":      True,
        "message": f"Amazon sync started (last {days_back} days). Check sync-log for status.",
    }


@router.get("/amazon/sync-log")
def amazon_sync_log(limit: int = 20):
    """Return recent Amazon sync history (newest first)."""
    return list_sync_log("amazon", limit=limit)


@router.delete("/amazon/disconnect")
def amazon_disconnect():
    """Remove Amazon SP-API credentials."""
    deleted = delete_credentials("amazon")
    if not deleted:
        raise HTTPException(status_code=404, detail="Amazon credentials not found.")
    return {"ok": True, "message": "Amazon credentials removed."}

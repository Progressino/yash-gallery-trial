"""
Myntra Partner API — outbound webhook receiver (PPMP / Omni stage + prod).

Myntra pushes order lifecycle events to POST /api/myntra/partner/webhook.
Authenticated via x-api-key (MYNTRA_PARTNER_WEBHOOK_API_KEY).
"""
from __future__ import annotations

import logging
import os
import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..db.myntra_partner_db import (
    count_events,
    init_db,
    insert_webhook_event,
    list_recent_events,
)
from ..services.myntra_partner_webhook import process_webhook_payload

log = logging.getLogger("erp.myntra_partner")
router = APIRouter()

MYNTRA_PARTNER_CODE = os.environ.get("MYNTRA_PARTNER_CODE", "PROGRESSINO").strip() or "PROGRESSINO"

init_db()


def _webhook_api_key() -> str:
    return os.environ.get("MYNTRA_PARTNER_WEBHOOK_API_KEY", "").strip()


def _verify_api_key(request: Request) -> None:
    expected = _webhook_api_key()
    if not expected:
        log.error("MYNTRA_PARTNER_WEBHOOK_API_KEY is not configured")
        raise HTTPException(503, "Webhook receiver not configured")
    got = (request.headers.get("x-api-key") or "").strip()
    if not got or not secrets.compare_digest(got, expected):
        raise HTTPException(401, "Invalid or missing x-api-key")


@router.get("/webhook/health")
def webhook_health():
    """Public liveness check for Myntra / load balancers (no auth)."""
    return {
        "ok": True,
        "service": "myntra-partner-webhook",
        "partner_code": MYNTRA_PARTNER_CODE,
        "configured": bool(_webhook_api_key()),
        "events_received": count_events(),
    }


@router.post("/webhook")
async def receive_webhook(request: Request):
    """
    Receive Myntra outbound pushes (Create Order, Shipped, Delivered, IC, etc.).
    Returns HTTP 200 quickly; idempotent on duplicate keys.
    """
    _verify_api_key(request)

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, "Invalid JSON body") from exc

    headers = {k: v for k, v in request.headers.items()}
    meta = process_webhook_payload(payload, headers)

    row_id, is_dup = insert_webhook_event(
        idempotency_key=meta["idempotency_key"],
        event_type=meta["event_type"],
        seller_order_id=meta["seller_order_id"],
        packet_id=meta["packet_id"],
        order_line_id=meta["order_line_id"],
        payload=payload,
        headers={
            "mocking-partner-name": meta.get("partner_name") or "",
            "content-type": headers.get("content-type", ""),
        },
    )

    log.info(
        "Myntra webhook event=%s seller_order=%s packet=%s duplicate=%s id=%s",
        meta["event_type"],
        meta["seller_order_id"] or "-",
        meta["packet_id"] or "-",
        is_dup,
        row_id,
    )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "received": True,
            "duplicate": is_dup,
            "event_id": row_id,
            "event_type": meta["event_type"],
            "seller_order_id": meta["seller_order_id"] or None,
            "packet_id": meta["packet_id"] or None,
            "order_line_id": meta["order_line_id"] or None,
            "partner_code": MYNTRA_PARTNER_CODE,
        },
    )


@router.get("/config")
def partner_config(request: Request):
    """Integration metadata for onboarding (requires ERP login)."""
    auth = getattr(request.state, "auth", None)
    if not auth:
        raise HTTPException(401, "Not authenticated")
    base = os.environ.get("MYNTRA_PARTNER_WEBHOOK_BASE_URL", "").strip()
    if not base:
        base = "https://app.progressino.com/api/myntra/partner/webhook"
    return {
        "partner_code": MYNTRA_PARTNER_CODE,
        "mocking_partner_name": MYNTRA_PARTNER_CODE,
        "integration_model": "PPMP",
        "webhook_url": base,
        "webhook_method": "POST",
        "webhook_headers": {
            "content-type": "application/json",
            "x-api-key": "<configured on server — share with Myntra separately>",
        },
        "webhook_health_url": base.replace("/webhook", "/webhook/health"),
        "events_received": count_events(),
        "configured": bool(_webhook_api_key()),
    }


@router.get("/events")
def recent_events(request: Request, limit: int = 50):
    """Recent webhook events for debugging (requires ERP login)."""
    auth = getattr(request.state, "auth", None)
    if not auth:
        raise HTTPException(401, "Not authenticated")
    limit = max(1, min(int(limit or 50), 200))
    return {"events": list_recent_events(limit=limit), "total": count_events()}

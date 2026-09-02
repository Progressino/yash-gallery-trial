"""Parse and classify Myntra Partner outbound webhook payloads."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


_EVENT_HEADER = "x-myntra-event-type"
_PARTNER_HEADER = "mocking-partner-name"

# Common Myntra outbound event labels (header or payload hints).
KNOWN_EVENT_TYPES = frozenset(
    {
        "create_order",
        "shipped",
        "delivered",
        "lost",
        "item_cancellation",
        "onhold",
        "unhold",
        "assignment_update",
        "rto",
        "rtv",
        "rto_update",
        "rtv_update",
        "download_invoice",
        "unknown",
    }
)


def _first_str(*values: Any) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("none", "null", "nan"):
            return s
    return ""


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _extract_ids(payload: Any) -> tuple[str, str, str]:
    seller_order_id = ""
    packet_id = ""
    order_line_id = ""

    if isinstance(payload, dict):
        seller_order_id = _first_str(
            payload.get("sellerOrderId"),
            payload.get("seller_order_id"),
            payload.get("sellerOrderID"),
        )
        packet_id = _first_str(
            payload.get("packetId"),
            payload.get("packet_id"),
            payload.get("packetID"),
        )
        order_line_id = _first_str(
            payload.get("orderLineId"),
            payload.get("order_line_id"),
            payload.get("orderLineID"),
        )
        lines = payload.get("orderLines") or payload.get("orderLineEntries") or []
        if isinstance(lines, list) and lines:
            first = lines[0] if isinstance(lines[0], dict) else {}
            seller_order_id = seller_order_id or _first_str(
                first.get("sellerOrderId"), first.get("seller_order_id")
            )
            order_line_id = order_line_id or _first_str(
                first.get("orderLineId"), first.get("order_line_id")
            )
            packet_id = packet_id or _first_str(
                first.get("packetId"), first.get("packet_id")
            )

    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        seller_order_id = seller_order_id or _first_str(
            node.get("sellerOrderId"), node.get("seller_order_id")
        )
        packet_id = packet_id or _first_str(node.get("packetId"), node.get("packet_id"))
        order_line_id = order_line_id or _first_str(
            node.get("orderLineId"), node.get("order_line_id")
        )

    return seller_order_id, packet_id, order_line_id


def _normalize_event_type(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        return "unknown"
    aliases = {
        "createorder": "create_order",
        "order_create": "create_order",
        "order_created": "create_order",
        "itemcancellation": "item_cancellation",
        "item_cancellation": "item_cancellation",
        "ic": "item_cancellation",
        "courier_return": "rto",
        "customer_return": "rtv",
        "assignmentupdate": "assignment_update",
        "on_hold": "onhold",
        "un_hold": "unhold",
    }
    return aliases.get(s, s if s in KNOWN_EVENT_TYPES else s)


def detect_event_type(payload: Any, headers: dict[str, str]) -> str:
    lowered = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    for key in (_EVENT_HEADER, "x-event-type", "event-type", "eventtype"):
        if key in lowered and lowered[key].strip():
            return _normalize_event_type(lowered[key])

    if isinstance(payload, dict):
        for key in ("eventType", "event_type", "type", "updateType", "status"):
            val = payload.get(key)
            if val:
                return _normalize_event_type(str(val))

    text = json.dumps(payload, default=str).lower()
    for hint, et in (
        ("createorder", "create_order"),
        ("create_order", "create_order"),
        ("itemcancellation", "item_cancellation"),
        ("assignmentupdate", "assignment_update"),
        ("unhold", "unhold"),
        ("onhold", "onhold"),
        ("delivered", "delivered"),
        ("shipped", "shipped"),
        ("lost", "lost"),
        ("rtv", "rtv"),
        ("rto", "rto"),
    ):
        if hint in text:
            return et
    return "unknown"


def build_idempotency_key(
    event_type: str,
    seller_order_id: str,
    packet_id: str,
    order_line_id: str,
    payload: Any,
) -> str:
    """Stable key for duplicate detection."""
    base = "|".join(
        [
            event_type or "unknown",
            seller_order_id or "",
            packet_id or "",
            order_line_id or "",
        ]
    )
    if base.replace("|", "").strip():
        return hashlib.sha256(base.encode()).hexdigest()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return digest


def partner_name_from_headers(headers: dict[str, str]) -> str:
    lowered = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    return lowered.get(_PARTNER_HEADER.lower(), lowered.get("mocking-partner-name", ""))


def process_webhook_payload(payload: Any, headers: dict[str, str]) -> dict:
    event_type = detect_event_type(payload, headers)
    seller_order_id, packet_id, order_line_id = _extract_ids(payload)
    idem = build_idempotency_key(
        event_type, seller_order_id, packet_id, order_line_id, payload
    )
    return {
        "event_type": event_type,
        "seller_order_id": seller_order_id,
        "packet_id": packet_id,
        "order_line_id": order_line_id,
        "idempotency_key": idem,
        "partner_name": partner_name_from_headers(headers),
    }

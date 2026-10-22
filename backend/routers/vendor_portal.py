"""Vendor portal — read own purchase / job-work documents. No stock or ledger writes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..db.vendor_portal_db import (
    acknowledge_material_issue,
    create_delivery_intimation,
    dashboard_for_link,
    get_job_work_order,
    get_link_for_user,
    get_purchase_order,
    list_delivery_intimations,
    list_job_work_orders,
    list_links,
    list_material_issues,
    list_production_job_orders,
    list_purchase_orders,
    log_event,
    record_document_response,
    set_link_active,
    show_commercial,
    upsert_link,
)
from ..services.permissions import may_access_erp_admin

router = APIRouter()


class LinkIn(BaseModel):
    user_id: int
    party_kind: str
    party_id: int
    portal_role: str = "vendor_admin"


class LinkActiveIn(BaseModel):
    active: bool = Field(default=False)


class DocResponseIn(BaseModel):
    action: str
    reason: str = ""


class AckLineIn(BaseModel):
    line_id: int
    received_qty: float
    reason: str = ""


class AckIn(BaseModel):
    lines: list[AckLineIn]


class IntimationIn(BaseModel):
    doc_type: str
    doc_id: int
    qty: float
    challan_no: str = ""
    delivery_date: str = ""
    remarks: str = ""


def _actor(request: Request) -> tuple[int | None, str]:
    auth = getattr(request.state, "auth", None) or {}
    uid = auth.get("user_id")
    return (int(uid) if uid else None, auth.get("sub") or "")


def _vendor_action(link: dict, fn):
    try:
        return fn()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_vendor(request: Request) -> dict:
    auth = getattr(request.state, "auth", None) or {}
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=403, detail="Vendor account is not linked to a user id")
    link = get_link_for_user(int(user_id))
    if not link:
        raise HTTPException(status_code=403, detail="Vendor portal access is not active for this login")
    return link


def _require_admin(request: Request) -> dict:
    auth = getattr(request.state, "auth", None) or {}
    role = auth.get("role") or ""
    if not may_access_erp_admin(role):
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth


@router.get("/me")
def vendor_me(request: Request):
    link = _require_vendor(request)
    return {
        "user_id": link["user_id"],
        "party_kind": link["party_kind"],
        "party_id": link["party_id"],
        "party_name": link.get("party_name") or "",
        "party_code": link.get("party_code") or "",
        "portal_role": link["portal_role"],
        "show_commercial": show_commercial(link["portal_role"]),
    }


@router.get("/dashboard")
def vendor_dashboard(request: Request):
    return dashboard_for_link(_require_vendor(request))


@router.get("/purchase-orders")
def vendor_pos(request: Request):
    link = _require_vendor(request)
    if link["party_kind"] != "supplier":
        return []
    return list_purchase_orders(link["party_id"], show_rates=show_commercial(link["portal_role"]))


@router.get("/purchase-orders/{po_id}")
def vendor_po(po_id: int, request: Request):
    link = _require_vendor(request)
    if link["party_kind"] != "supplier":
        raise HTTPException(status_code=404, detail="Purchase order not found")
    row = get_purchase_order(link["party_id"], po_id, show_rates=show_commercial(link["portal_role"]))
    if not row:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    return row


@router.get("/job-work-orders")
def vendor_jwos(request: Request):
    link = _require_vendor(request)
    if link["party_kind"] != "processor":
        return []
    return list_job_work_orders(link["party_id"], show_rates=show_commercial(link["portal_role"]))


@router.get("/job-work-orders/{jwo_id}")
def vendor_jwo(jwo_id: int, request: Request):
    link = _require_vendor(request)
    if link["party_kind"] != "processor":
        raise HTTPException(status_code=404, detail="Job work order not found")
    row = get_job_work_order(link["party_id"], jwo_id, show_rates=show_commercial(link["portal_role"]))
    if not row:
        raise HTTPException(status_code=404, detail="Job work order not found")
    return row


@router.post("/purchase-orders/{po_id}/response")
def vendor_po_response(po_id: int, body: DocResponseIn, request: Request):
    link = _require_vendor(request)
    uid, username = _actor(request)
    return _vendor_action(
        link,
        lambda: record_document_response(
            link, doc_type="po", doc_id=po_id, action=body.action, reason=body.reason,
            user_id=uid, username=username,
        ),
    )


@router.post("/job-work-orders/{jwo_id}/response")
def vendor_jwo_response(jwo_id: int, body: DocResponseIn, request: Request):
    link = _require_vendor(request)
    uid, username = _actor(request)
    return _vendor_action(
        link,
        lambda: record_document_response(
            link, doc_type="jwo", doc_id=jwo_id, action=body.action, reason=body.reason,
            user_id=uid, username=username,
        ),
    )


@router.post("/job-orders/{jo_id}/response")
def vendor_jo_response(jo_id: int, body: DocResponseIn, request: Request):
    link = _require_vendor(request)
    uid, username = _actor(request)
    return _vendor_action(
        link,
        lambda: record_document_response(
            link, doc_type="jo", doc_id=jo_id, action=body.action, reason=body.reason,
            user_id=uid, username=username,
        ),
    )


@router.get("/material-issues")
def vendor_material_issues(request: Request):
    link = _require_vendor(request)
    return list_material_issues(link.get("party_name") or "")


@router.post("/material-issues/{min_id}/acknowledge")
def vendor_ack(min_id: int, body: AckIn, request: Request):
    link = _require_vendor(request)
    uid, username = _actor(request)
    return _vendor_action(
        link,
        lambda: acknowledge_material_issue(
            link, min_id=min_id, lines=[ln.model_dump() for ln in body.lines],
            user_id=uid, username=username,
        ),
    )


@router.get("/delivery-intimations")
def vendor_intimations(request: Request):
    return list_delivery_intimations(_require_vendor(request))


@router.post("/delivery-intimations")
def vendor_create_intimation(body: IntimationIn, request: Request):
    link = _require_vendor(request)
    uid, username = _actor(request)
    return _vendor_action(
        link,
        lambda: create_delivery_intimation(
            link,
            doc_type=body.doc_type.strip().lower(),
            doc_id=body.doc_id,
            qty=body.qty,
            challan_no=body.challan_no,
            delivery_date=body.delivery_date,
            remarks=body.remarks,
            user_id=uid,
            username=username,
        ),
    )


@router.get("/job-orders")
def vendor_jos(request: Request):
    link = _require_vendor(request)
    return list_production_job_orders(
        link.get("party_name") or "",
        show_rates=show_commercial(link["portal_role"]),
    )


@router.get("/admin/links")
def admin_list_links(request: Request):
    _require_admin(request)
    return list_links()


@router.post("/admin/links")
def admin_upsert_link(body: LinkIn, request: Request):
    auth = _require_admin(request)
    try:
        link = upsert_link(
            user_id=body.user_id,
            party_kind=body.party_kind.strip().lower(),
            party_id=body.party_id,
            portal_role=body.portal_role.strip().lower(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_event(
        user_id=auth.get("user_id"),
        username=auth.get("sub") or "",
        party_kind=link["party_kind"],
        party_id=link["party_id"],
        action="portal_link_saved",
        record_type="vendor_portal_link",
        record_id=str(link["id"]),
        details={"target_user_id": body.user_id, "portal_role": link["portal_role"]},
    )
    return link


@router.post("/admin/links/{link_id}/active")
def admin_set_active(link_id: int, body: LinkActiveIn, request: Request):
    auth = _require_admin(request)
    row = set_link_active(link_id, body.active)
    if not row:
        raise HTTPException(status_code=404, detail="Vendor link not found")
    log_event(
        user_id=auth.get("user_id"),
        username=auth.get("sub") or "",
        party_kind=row["party_kind"],
        party_id=row["party_id"],
        action="portal_link_activated" if body.active else "portal_link_deactivated",
        record_type="vendor_portal_link",
        record_id=str(link_id),
    )
    return row

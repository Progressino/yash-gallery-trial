"""Accounts document verification API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..db import document_audit_db as audit
from ..services.permissions import can_document_verify, may_access_erp_admin

router = APIRouter()


def _actor(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("full_name") or auth.get("username") or auth.get("sub") or "")


def _role(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("role") or "")


def _can_verify(request: Request) -> bool:
    return can_document_verify(_role(request))


class UnverifyBody(BaseModel):
    reason: str = Field(min_length=3)
    force: bool = False


@router.get("")
def list_docs(
    audit_status: str = "",
    doc_type: str = "",
    date_from: str = "",
    date_to: str = "",
    search: str = "",
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return audit.list_documents(
        audit_status=audit_status,
        doc_type=doc_type,
        date_from=date_from,
        date_to=date_to,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get("/meta")
def audit_meta():
    return {
        "doc_types": list(audit.DOC_TYPES),
        "labels": audit.DOC_TYPE_LABELS,
    }


@router.post("/backfill")
def backfill(request: Request, limit_per_type: int = Query(5000, ge=1, le=20000)):
    """Enroll historical PO/JWO/GRN/MIN/JO/GIN/issue/receive into the audit registry."""
    if not (may_access_erp_admin(_role(request)) or _can_verify(request)):
        raise HTTPException(status_code=403, detail="Accounts/Admin required to backfill")
    return audit.backfill_from_modules(limit_per_type=limit_per_type, actor=_actor(request) or "backfill")


@router.get("/{doc_type}/{doc_id}")
def get_doc(
    doc_type: str,
    doc_id: int,
    include_blockers: int = Query(0, ge=0, le=1),
):
    try:
        return audit.get_document_detail(
            doc_type,
            doc_id,
            include_blockers=bool(include_blockers),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{doc_type}/{doc_id}/blockers")
def get_blockers(doc_type: str, doc_id: int):
    if not audit.get_audit(doc_type, doc_id):
        raise HTTPException(status_code=404, detail="Not found in audit registry")
    return {"dependency_blockers": audit.dependency_blockers(doc_type, doc_id)}


@router.post("/{doc_type}/{doc_id}/verify")
def verify(doc_type: str, doc_id: int, request: Request):
    if not _can_verify(request):
        raise HTTPException(status_code=403, detail="Accounts/Admin role required to verify")
    try:
        if not audit.get_audit(doc_type, doc_id):
            audit.enroll_document(doc_type, doc_id, created_by=_actor(request))
        return audit.verify_document(doc_type, doc_id, actor=_actor(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{doc_type}/{doc_id}/unverify")
def unverify(doc_type: str, doc_id: int, body: UnverifyBody, request: Request):
    if not _can_verify(request):
        raise HTTPException(status_code=403, detail="Accounts/Admin role required to unverify")
    force = bool(body.force) and may_access_erp_admin(_role(request))
    if body.force and not force:
        raise HTTPException(status_code=403, detail="Only Admin can force-unverify with downstream dependencies")
    try:
        return audit.unverify_document(
            doc_type,
            doc_id,
            actor=_actor(request),
            reason=body.reason,
            force=force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{doc_type}/{doc_id}/editable")
def editable(doc_type: str, doc_id: int, include_blockers: int = Query(0, ge=0, le=1)):
    verified = audit.is_verified(doc_type, doc_id)
    blockers = audit.dependency_blockers(doc_type, doc_id) if include_blockers else []
    return {
        "editable": not verified,
        "verified": verified,
        "dependency_blockers": blockers,
    }

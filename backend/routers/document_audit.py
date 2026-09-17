"""Accounts document verification API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..db import document_audit_db as audit
from ..services.permissions import may_access_erp_admin

router = APIRouter()


def _actor(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("full_name") or auth.get("username") or auth.get("sub") or "")


def _role(request: Request) -> str:
    auth = getattr(request.state, "auth", None) or {}
    return str(auth.get("role") or "")


def _can_verify(request: Request) -> bool:
    role = _role(request).lower()
    if may_access_erp_admin(role):
        return True
    # Accounts / finance / manager / hod can verify
    return any(k in role for k in ("admin", "manager", "accounts", "finance", "hod", "audit"))


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


@router.get("/{doc_type}/{doc_id}")
def get_doc(doc_type: str, doc_id: int):
    row = audit.get_audit(doc_type, doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found in audit registry")
    blockers = audit.dependency_blockers(doc_type, doc_id)
    return {
        **row,
        "events": audit.list_audit_events(doc_type, doc_id),
        "dependency_blockers": blockers,
        "editable": row.get("audit_status") != "Verified",
    }


@router.post("/{doc_type}/{doc_id}/verify")
def verify(doc_type: str, doc_id: int, request: Request):
    if not _can_verify(request):
        raise HTTPException(status_code=403, detail="Accounts/Admin role required to verify")
    try:
        # Auto-enroll if missing (backfill from lookup is best-effort)
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
def editable(doc_type: str, doc_id: int):
    verified = audit.is_verified(doc_type, doc_id)
    blockers = audit.dependency_blockers(doc_type, doc_id)
    return {
        "editable": not verified,
        "verified": verified,
        "dependency_blockers": blockers,
    }

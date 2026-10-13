"""User usage analytics — client tracking + Admin → Usage report."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.usage_analytics import REPORT_ADMIN_ROLES, record_client_events, usage_report

track_router = APIRouter()
report_router = APIRouter()


class UsageEvent(BaseModel):
    kind: str
    path: str = ""
    module: str = ""
    tab: str = ""


class TrackBody(BaseModel):
    events: list[UsageEvent] = Field(default_factory=list, max_length=50)
    active: bool = False


@track_router.post("/track")
def track(body: TrackBody, request: Request):
    auth = getattr(request.state, "auth", None) or {}
    n = record_client_events(
        auth.get("sub") or "",
        auth.get("role") or "",
        [e.model_dump() for e in body.events],
        active=body.active,
    )
    return {"ok": True, "recorded": n}


@report_router.get("/usage")
def admin_usage(
    request: Request,
    days: int = 30,
    role: str = "",
    department: str = "",
    username: str = "",
):
    auth = getattr(request.state, "auth", None) or {}
    if (auth.get("role") or "") not in REPORT_ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Usage report is limited to Super Admin and Admin")
    return usage_report(
        days,
        role=role.strip() or None,
        department=department.strip() or None,
        username=username.strip() or None,
    )

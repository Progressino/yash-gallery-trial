"""Admin performance dashboard API."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..services.perf_metrics import build_dashboard

router = APIRouter()


@router.get("/performance")
def admin_performance(request: Request, hours: float = 48.0):
    """Slow endpoints, queries, PO timings, cache hit rate, PG stats — no SSH required."""
    hours = max(6.0, min(float(hours or 48.0), 168.0))
    return build_dashboard(hours=hours)

"""Shared Material Issue Note tolerance + reason rules (Purchase JWO MIN and Production JO MIN).

Issue qty may differ from the frozen BOM requirement. Within ``tolerance %`` a Draft can be
confirmed by the issuer; beyond it the note moves to ``Pending Approval`` and only
Manager / Admin / Super Admin / Sir may approve it before it can be confirmed.

Interim approval (maker-checker, approver != issuer; approve / reject / send back with remarks) is a
stop-gap that will be replaced by the centralized ERP Approval Workflow — see ``jo_min_control``.

Tolerance is configurable via ``MIN_ISSUE_TOLERANCE_PCT`` (default 10).
"""
from __future__ import annotations

import os

DEFAULT_TOLERANCE_PCT = 10.0
EPS = 0.0005

STATUS_DRAFT = "Draft"
STATUS_PENDING = "Pending Approval"
STATUS_CONFIRMED = "Confirmed"
STATUS_CANCELLED = "Cancelled"
STATUS_REJECTED = "Rejected"  # additional MIN rejected by the approver (interim approval)

# Production (Cutting / Stitching / Finishing) JO MIN variance reasons.
PRODUCTION_VARIANCE_REASONS = (
    "Mini CAD Average Change",
    "Wastage",
    "Roll/Bundle Adjustment",
    "Rework",
    "Stock Shortage",
    "Other",
)


def r3(x: float | None) -> float:
    return round(float(x or 0), 3)


def get_tolerance_pct() -> float:
    """Configured tolerance % (absolute, applied above and below BOM)."""
    raw = os.environ.get("MIN_ISSUE_TOLERANCE_PCT", "")
    try:
        val = float(str(raw).strip()) if str(raw).strip() else DEFAULT_TOLERANCE_PCT
    except ValueError:
        val = DEFAULT_TOLERANCE_PCT
    if val < 0:
        val = DEFAULT_TOLERANCE_PCT
    return min(val, 100.0)


def variance_qty(bom_qty: float, issue_qty: float) -> float:
    return r3(float(issue_qty or 0) - float(bom_qty or 0))


def variance_pct(bom_qty: float, issue_qty: float) -> float | None:
    """Signed variance % vs BOM; ``None`` when BOM is zero (extra issue with no requirement)."""
    bom = float(bom_qty or 0)
    if bom <= EPS:
        return None
    return round((float(issue_qty or 0) - bom) / bom * 100.0, 2)


def differs_from_bom(bom_qty: float, issue_qty: float) -> bool:
    return abs(float(issue_qty or 0) - float(bom_qty or 0)) > EPS


def exceeds_tolerance(bom_qty: float, issue_qty: float, tolerance_pct: float | None = None) -> bool:
    """True when |issue − BOM| / BOM is above tolerance (or any issue against a zero BOM)."""
    if not differs_from_bom(bom_qty, issue_qty):
        return False
    tol = get_tolerance_pct() if tolerance_pct is None else float(tolerance_pct)
    pct = variance_pct(bom_qty, issue_qty)
    if pct is None:
        return float(issue_qty or 0) > EPS
    return abs(pct) > tol + 1e-9


def validate_reason(
    reason: str,
    remarks: str,
    *,
    required: bool,
    allowed: tuple[str, ...] | list[str],
    subject: str = "Issue Qty",
) -> tuple[str, str]:
    """Reason is mandatory when qty differs from BOM; ``Other`` also needs remarks."""
    reason = str(reason or "").strip()
    remarks = str(remarks or "").strip()
    if not required:
        return reason, remarks
    if not reason:
        raise ValueError(f"Reason is required when {subject} differs from the BOM requirement")
    if reason not in allowed:
        raise ValueError(f"Reason must be one of: {', '.join(allowed)}")
    if reason == "Other" and not remarks:
        raise ValueError("Remarks are required when reason is Other")
    return reason, remarks


def may_approve(role: str | None) -> bool:
    """Manager / Admin / Super Admin / Sir."""
    from .permissions import may_access_erp_admin

    return may_access_erp_admin(str(role or ""))


def tolerance_summary() -> dict:
    return {"tolerance_pct": get_tolerance_pct(), "reasons": list(PRODUCTION_VARIANCE_REASONS)}

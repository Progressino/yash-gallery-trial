"""Resolve PO calculate lead time from the most recent confirmed raise."""
from __future__ import annotations

from typing import Any, Optional

DEFAULT_PO_LEAD_TIME_DAYS = 45


def apply_lead_time_from_last_raise(sess, body: dict) -> tuple[dict, dict[str, Any]]:
    """
    When a raise exists in the lookback window, use that day's stored lead time
  for PO calculate (operator workflow: same lead time as the last PO batch).
    """
    if body.get("ignore_raise_lead_time"):
        return body, {}

    planning = str(body.get("planning_date") or "").strip()[:10] or None
    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)

    from ..db.po_raised_db import latest_raise_lead_time_before

    hit = latest_raise_lead_time_before(planning, lookback_days=lookback)
    if hit and int(hit.get("lead_time") or 0) > 0:
        lt = int(hit["lead_time"])
        out = dict(body)
        out["lead_time"] = lt
        sess.po_calculate_lead_time = lt
        return out, {
            "lead_time_applied": lt,
            "lead_time_source": "last_raise",
            "lead_time_raise_date": str(hit.get("raised_date") or ""),
        }

    lt_raw = body.get("lead_time")
    if lt_raw not in (None, ""):
        try:
            lt = int(lt_raw)
            if lt > 0:
                sess.po_calculate_lead_time = lt
                return body, {"lead_time_applied": lt, "lead_time_source": "request"}
        except (TypeError, ValueError):
            pass

    lt = DEFAULT_PO_LEAD_TIME_DAYS
    out = dict(body)
    out["lead_time"] = lt
    sess.po_calculate_lead_time = lt
    return out, {"lead_time_applied": lt, "lead_time_source": "default"}


def lead_time_for_raise_record(sess, body: Optional[dict] = None) -> int:
    """Lead time to stamp when recording a raise (confirm / manual Existing PO seed)."""
    body = body or {}
    for src in (
        body.get("lead_time"),
        getattr(sess, "po_calculate_lead_time", None),
    ):
        if src in (None, ""):
            continue
        try:
            lt = int(src)
            if lt > 0:
                return lt
        except (TypeError, ValueError):
            continue

    from ..db.po_raised_db import latest_raise_lead_time_before

    planning = str(body.get("planning_date") or "").strip()[:10] or None
    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)
    hit = latest_raise_lead_time_before(planning, lookback_days=lookback)
    if hit and int(hit.get("lead_time") or 0) > 0:
        return int(hit["lead_time"])
    return DEFAULT_PO_LEAD_TIME_DAYS


def persist_raise_day_lead_time(
    raised_date: str,
    lead_time: int,
    *,
    period_days: Optional[int] = None,
    target_days: Optional[int] = None,
    source: str = "",
) -> None:
    from ..db.po_raised_db import save_raise_day_meta, update_raises_lead_time_for_date

    day = str(raised_date).strip()[:10]
    if not day or int(lead_time) <= 0:
        return
    save_raise_day_meta(
        day,
        int(lead_time),
        period_days=period_days,
        target_days=target_days,
        source=source,
    )
    update_raises_lead_time_for_date(day, int(lead_time))

"""Merge Tier-3 daily uploads into session platform frames (PO + Intelligence parity)."""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .platform_session_window import (
    _PLATFORM_ATTRS,
    platform_df_date_bounds,
    session_platform_shorter_than_tier3,
)

_log = logging.getLogger(__name__)


def tier3_months() -> int:
    try:
        return max(1, int((os.environ.get("INTELLIGENCE_TIER3_MONTHS") or "6").strip()))
    except ValueError:
        return 6


def tier3_max_files() -> int:
    try:
        v = int((os.environ.get("INTELLIGENCE_TIER3_MAX_FILES") or "60").strip())
        return max(5, v)
    except ValueError:
        return 60


def mark_tier3_sync_applied(sess) -> None:
    from .daily_store import get_tier3_sync_token

    sess._tier3_sync_token_applied = get_tier3_sync_token()


def tier3_token_mismatch(sess) -> bool:
    from .daily_store import get_tier3_sync_token

    store = get_tier3_sync_token()
    applied: dict[str, str] = getattr(sess, "_tier3_sync_token_applied", None) or {}
    return store != applied


def platforms_with_tier3_token_mismatch(sess) -> list[str]:
    from .daily_store import get_tier3_sync_token

    store = get_tier3_sync_token()
    applied: dict[str, str] = getattr(sess, "_tier3_sync_token_applied", None) or {}
    out: list[str] = []
    for plat, _attr in _PLATFORM_ATTRS:
        if store.get(plat) != applied.get(plat):
            out.append(plat)
    return out


def merge_tier3_for_report_range(
    sess,
    platforms: list[str],
    start_date: str,
    end_date: str,
) -> bool:
    from .daily_store import load_platform_data_for_report_range, merge_platform_data

    changed = False
    for platform, attr in _PLATFORM_ATTRS:
        if platform not in platforms:
            continue
        df = load_platform_data_for_report_range(platform, start_date, end_date, dedup=False)
        if df.empty:
            continue
        cur = getattr(sess, attr, pd.DataFrame())
        merged = merge_platform_data(cur, df, platform)
        if len(merged) != len(cur) or (getattr(cur, "empty", True) and not merged.empty):
            setattr(sess, attr, merged)
            changed = True
    if changed:
        mark_tier3_sync_applied(sess)
    return changed


def merge_tier3_light(sess, *, only_platforms: list[str] | None = None) -> bool:
    from .daily_store import load_platform_data, merge_platform_data

    months = tier3_months()
    max_files = tier3_max_files()
    changed = False
    for platform, attr in _PLATFORM_ATTRS:
        if only_platforms is not None and platform not in only_platforms:
            continue
        df = load_platform_data(
            platform,
            months=months,
            dedup=False,
            max_files=max_files,
        )
        if df.empty:
            continue
        cur = getattr(sess, attr, pd.DataFrame())
        merged = merge_platform_data(cur, df, platform)
        if len(merged) != len(cur) or (getattr(cur, "empty", True) and not merged.empty):
            setattr(sess, attr, merged)
            changed = True
    if changed:
        mark_tier3_sync_applied(sess)
    return changed


def _po_ads_horizon_days(period_days: int, use_seasonality: bool, use_ly_fallback: bool) -> int:
    horizon = max(int(period_days), 90)
    if use_seasonality or use_ly_fallback:
        horizon = max(horizon, 400)
    return horizon


def _normalize_planning_date(planning_date: str | None) -> str:
    if planning_date:
        try:
            return str(pd.Timestamp(pd.to_datetime(str(planning_date).strip()).normalize()).date())
        except Exception:
            pass
    try:
        from zoneinfo import ZoneInfo

        IST = ZoneInfo("Asia/Kolkata")
        return str(pd.Timestamp.now(tz=IST).normalize().date())
    except Exception:
        return str(date.today())


def session_sales_through(sess) -> str:
    try:
        sdf = getattr(sess, "sales_df", None)
        if sdf is None or sdf.empty or "TxnDate" not in sdf.columns:
            return ""
        t = pd.to_datetime(sdf["TxnDate"], errors="coerce").max()
        if pd.notna(t):
            return str(pd.Timestamp(t).date())
    except Exception:
        pass
    return ""


def platform_session_bounds(sess) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for plat, attr in _PLATFORM_ATTRS:
        cur = getattr(sess, attr, None)
        lo, hi = platform_df_date_bounds(cur) if cur is not None else (None, None)
        out[plat] = {
            "min": str(lo.date()) if lo is not None and pd.notna(lo) else "",
            "max": str(hi.date()) if hi is not None and pd.notna(hi) else "",
        }
    return out


def build_parity_report(sess, *, planning_date: str | None = None) -> dict[str, Any]:
    """Diagnostics for live vs local / dashboard vs PO mismatches."""
    from .daily_store import get_summary, get_tier3_sync_token

    plan = _normalize_planning_date(planning_date)
    tier3 = get_summary() or {}
    tier3_files = sum(int((tier3.get(p) or {}).get("file_count") or 0) for p in tier3)
    sales_through = session_sales_through(sess)
    warnings: list[str] = []
    if tier3_files > 0 and tier3_token_mismatch(sess):
        warnings.append(
            "Tier-3 daily uploads exist but are not fully merged into this session — "
            "Intelligence and PO may disagree until data reloads."
        )
    if session_platform_shorter_than_tier3(sess):
        warnings.append(
            "Session platform history is shorter than Tier-3 SQLite — use Reload from server "
            "or run PO calculate to merge recent dailies."
        )
    if sales_through and plan and sales_through < plan:
        warnings.append(
            f"Platform data ends {sales_through} but planning date is {plan} — "
            "reload data for a complete sales window."
        )
    if tier3_files == 0 and not getattr(sess, "sales_df", pd.DataFrame()).empty:
        warnings.append(
            "No Tier-3 daily uploads on this server — dashboard totals use bulk/warm cache only "
            "(production may differ if daily uploads exist there)."
        )
    return {
        "planning_date": plan,
        "sales_through": sales_through,
        "tier3_file_count": tier3_files,
        "tier3_summary": tier3,
        "tier3_sync_mismatch": tier3_token_mismatch(sess),
        "tier3_platforms_mismatch": platforms_with_tier3_token_mismatch(sess),
        "session_platform_bounds": platform_session_bounds(sess),
        "tier3_sync_token": get_tier3_sync_token(),
        "session_tier3_token_applied": getattr(sess, "_tier3_sync_token_applied", None) or {},
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }


def ensure_tier3_merged_for_po(
    sess,
    *,
    planning_date: str | None = None,
    period_days: int = 30,
    use_seasonality: bool = False,
    use_ly_fallback: bool = True,
) -> dict[str, Any]:
    """
    Legacy session merge — prefer ``build_po_ads_platform_sales`` for PO calculate
    (in-memory Tier-3 overlay; does not mutate session or rebuild unified sales).
    """
    _ = (sess, planning_date, period_days, use_seasonality, use_ly_fallback)
    return {"merged": False, "reason": "use_build_po_ads_platform_sales", "platforms": []}


def build_po_ads_platform_sales(
    sess,
    *,
    planning_date: str | None = None,
    period_days: int = 30,
    use_seasonality: bool = False,
    use_ly_fallback: bool = True,
) -> pd.DataFrame:
    """
    Build platform sales for PO ADS from session bulk + Tier-3 window overlay.
    Does not mutate session frames or rebuild unified sales_df (fast path).
    """
    from .daily_store import get_summary, load_platform_data_for_report_range, merge_platform_data, platforms_with_uploads_in_range

    summary = get_summary() or {}
    tier3_any = any(int((summary.get(p) or {}).get("file_count") or 0) > 0 for p in summary)
    if not tier3_any:
        return pd.DataFrame()

    plan = _normalize_planning_date(planning_date)
    horizon = _po_ads_horizon_days(period_days, use_seasonality, use_ly_fallback)
    end = plan
    start = str((pd.Timestamp(plan) - pd.Timedelta(days=horizon)).date())

    window_plats = platforms_with_uploads_in_range(start, end)
    if not window_plats:
        return pd.DataFrame()

    frame_overrides: dict[str, pd.DataFrame] = {}
    used_tier3 = False
    for platform, attr in _PLATFORM_ATTRS:
        cur = getattr(sess, attr, pd.DataFrame())
        if platform not in window_plats:
            if cur is not None and not getattr(cur, "empty", True):
                frame_overrides[attr] = cur
            continue
        tier = load_platform_data_for_report_range(platform, start, end, dedup=False)
        if tier.empty:
            if cur is not None and not getattr(cur, "empty", True):
                frame_overrides[attr] = cur
            continue
        base = cur if cur is not None and not getattr(cur, "empty", True) else pd.DataFrame()
        frame_overrides[attr] = merge_platform_data(base, tier, platform)
        used_tier3 = True

    if not used_tier3:
        return pd.DataFrame()

    from .po_calculate_run import _build_platform_sales_df

    return _build_platform_sales_df(sess, frame_overrides=frame_overrides)

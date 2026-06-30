"""Merge Tier-3 daily uploads into session platform frames (PO + Intelligence parity)."""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .platform_session_window import (
    _PLATFORM_ATTRS,
    platform_date_column,
    platform_df_date_bounds,
    session_platform_shorter_than_tier3,
)

_log = logging.getLogger(__name__)

# ── Process-level cache for the expensive platform-sales build ──────────────
# Key = (frame-fingerprint, tier3-token-repr). Invalidated naturally when tier3
# sync token changes (any daily upload). Avoids rebuilding raw platform DFs for
# every PO calculate when data hasn't changed (e.g. user changes lead time).
_PLAT_BUILD_CACHE: dict[str, pd.DataFrame] = {}
_PLAT_BUILD_LOCK = threading.Lock()


def _plat_build_cache_key(bulk_frames: dict[str, pd.DataFrame], tier3_token: dict) -> str:
    parts: list[str] = []
    for attr in sorted(bulk_frames.keys()):
        df = bulk_frames[attr]
        n = len(df) if df is not None else 0
        parts.append(f"{attr}:{n}")
    parts.append(repr(sorted(tier3_token.items()) if isinstance(tier3_token, dict) else tier3_token))
    return "|".join(parts)


def invalidate_platform_build_cache() -> None:
    """Call when any platform upload changes data (tier3 token will also change, but belt+suspenders)."""
    with _PLAT_BUILD_LOCK:
        _PLAT_BUILD_CACHE.clear()


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
    from .po_ads_horizon import po_ads_history_horizon_days

    return po_ads_history_horizon_days(
        period_days,
        use_seasonality=use_seasonality,
        use_ly_fallback=use_ly_fallback,
    )


def _trim_sales_to_ads_window(
    sales_df: pd.DataFrame,
    planning_date: str,
    period_days: int,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> pd.DataFrame:
    """Keep only rows needed for PO ADS / seasonality on the unified sales frame."""
    if sales_df is None or getattr(sales_df, "empty", True):
        return pd.DataFrame()
    if "TxnDate" not in sales_df.columns:
        return pd.DataFrame()
    horizon = _po_ads_horizon_days(period_days, use_seasonality, use_ly_fallback)
    plan = pd.Timestamp(_normalize_planning_date(planning_date)).normalize()
    txn = pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    end = txn.max()
    if pd.notna(end):
        end = min(pd.Timestamp(end).normalize(), plan)
    else:
        end = plan
    start = end - pd.Timedelta(days=horizon)
    mask = (txn >= start) & (txn <= plan)
    if not bool(mask.any()):
        return pd.DataFrame()
    out = sales_df.loc[mask].reset_index(drop=True)
    if "TxnDate" in out.columns:
        out = out.copy()
        out["TxnDate"] = pd.to_datetime(out["TxnDate"], errors="coerce")
    return out


def _sales_has_ads_history(
    sales_df: pd.DataFrame,
    planning_date: str,
    period_days: int,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> bool:
    if sales_df is None or getattr(sales_df, "empty", True) or "TxnDate" not in sales_df.columns:
        return False
    txn = pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    mn = txn.min()
    if pd.isna(mn):
        return False
    plan = pd.Timestamp(_normalize_planning_date(planning_date)).normalize()
    horizon = _po_ads_horizon_days(period_days, use_seasonality, use_ly_fallback)
    need = plan - pd.Timedelta(days=horizon)
    return pd.Timestamp(mn).normalize() <= need


def _merge_sales_tail(
    base: pd.DataFrame,
    tail: pd.DataFrame,
    planning_date: str,
    period_days: int,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> pd.DataFrame:
    """Append recent Tier-3 rows and drop overlapping base dates."""
    if tail is None or getattr(tail, "empty", True):
        return _trim_sales_to_ads_window(base, planning_date, period_days, use_seasonality, use_ly_fallback)
    if base is None or getattr(base, "empty", True):
        return _trim_sales_to_ads_window(tail, planning_date, period_days, use_seasonality, use_ly_fallback)
    tail_dates = pd.to_datetime(tail["TxnDate"], errors="coerce")
    gap_min = tail_dates.min()
    if pd.notna(gap_min):
        base_dates = pd.to_datetime(base["TxnDate"], errors="coerce")
        base = base.loc[base_dates < gap_min]
    combined = pd.concat([base, tail], ignore_index=True)
    return _trim_sales_to_ads_window(
        combined, planning_date, period_days, use_seasonality, use_ly_fallback
    )


def _build_tier3_gap_sales(sess, gap_start: str, gap_end: str) -> pd.DataFrame:
    """Load only the missing tail window from Tier-3 (seconds, not minutes)."""
    from .daily_store import load_platform_data_for_report_range, platforms_with_uploads_in_range

    plats = platforms_with_uploads_in_range(gap_start, gap_end)
    if not plats:
        return pd.DataFrame()
    overrides: dict[str, pd.DataFrame] = {}
    for platform, attr in _PLATFORM_ATTRS:
        if platform not in plats:
            continue
        tier = load_platform_data_for_report_range(
            platform, gap_start, gap_end, dedup=True, columns_only=True
        )
        if not tier.empty:
            overrides[attr] = tier
    if not overrides:
        return pd.DataFrame()
    from .po_calculate_run import _build_platform_sales_df

    return _build_platform_sales_df(sess, frame_overrides=overrides)


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


def sales_data_lag_days(planning_date: str | None, sales_through: str | None) -> int | None:
    """Calendar days between sales_through and planning_date (positive = sales end before plan)."""
    if not planning_date or not sales_through:
        return None
    try:
        plan = pd.Timestamp(pd.to_datetime(str(planning_date).strip()[:10]).normalize()).date()
        thru = pd.Timestamp(pd.to_datetime(str(sales_through).strip()[:10]).normalize()).date()
        return int((plan - thru).days)
    except Exception:
        return None


def sales_data_gap_needs_warning(
    planning_date: str | None,
    sales_through: str | None,
    *,
    max_expected_lag_days: int = 1,
) -> bool:
    """
    True when sales end materially before planning day.
    Daily uploads are always T-1, so a 1-day gap is normal and should not warn.
    """
    lag = sales_data_lag_days(planning_date, sales_through)
    if lag is None:
        return False
    return lag > max(0, int(max_expected_lag_days))


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


def tier3_sales_through() -> str:
    """Latest calendar day across Tier-3 SQLite upload metadata (Upload tab source of truth)."""
    try:
        from .daily_store import get_summary

        summary = get_summary() or {}
    except Exception:
        return ""
    bests: list[str] = []
    for info in summary.values():
        if not isinstance(info, dict):
            continue
        if int(info.get("file_count") or 0) <= 0:
            continue
        md = str(info.get("max_date") or "")[:10]
        if len(md) == 10:
            bests.append(md)
    return max(bests) if bests else ""


def effective_sales_through(sess, ads_df: pd.DataFrame | None = None) -> str:
    """
    Latest sales day for PO gap warnings — session, Tier-3 metadata, and ADS overlay.
    Avoids false “4d gap” warnings when dailies are saved but session sales is stale.
    """
    candidates: list[str] = []
    st = session_sales_through(sess)
    if st:
        candidates.append(st[:10])
    t3 = tier3_sales_through()
    if t3:
        candidates.append(t3[:10])
    if ads_df is not None and not getattr(ads_df, "empty", True):
        try:
            if "TxnDate" in ads_df.columns:
                t = pd.to_datetime(ads_df["TxnDate"], errors="coerce").max()
                if pd.notna(t):
                    candidates.append(str(pd.Timestamp(t).date()))
        except Exception:
            pass
    candidates = [c for c in candidates if len(c) >= 10]
    return max(candidates) if candidates else ""


def refresh_po_sales_through_meta(sess, meta: dict | None) -> dict:
    """Bump stale PO result metadata when Tier-3 dailies are newer than session sales."""
    out = dict(meta or {})
    fresh = effective_sales_through(sess)
    if not fresh:
        return out
    old = str(out.get("sales_through") or "")[:10]
    if not old or fresh >= old:
        out["sales_through"] = fresh
    return out


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
    sales_through = effective_sales_through(sess)
    warnings: list[str] = []
    if tier3_files > 0 and tier3_token_mismatch(sess):
        warnings.append(
            "Tier-3 daily uploads exist but are not fully merged into this session — "
            "Intelligence and PO may disagree until data reloads."
        )
    ep_fn = str(getattr(sess, "existing_po_filename", "") or "")
    ep_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    if ep_fn and "17-Jun" not in ep_fn and "Jun-26" not in ep_fn and ep_gen <= 1:
        warnings.append(
            f"Existing PO sheet looks stale ({ep_fn or 'unknown'}) — production uses "
            "Po 17-Jun-26.xlsx; reload warm cache or re-upload for matching PO totals."
        )
    if session_platform_shorter_than_tier3(sess):
        warnings.append(
            "Session platform history is shorter than Tier-3 SQLite — use Reload from server "
            "or run PO calculate to merge recent dailies."
        )
    if sales_data_gap_needs_warning(plan, sales_through):
        lag = sales_data_lag_days(plan, sales_through)
        warnings.append(
            f"Platform data ends {sales_through} but planning date is {plan} "
            f"({lag}d gap) — reload data for a complete sales window."
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


def _session_platform_frames(sess) -> dict[str, pd.DataFrame]:
    """Tier-1 bulk platform history from session or process warm cache."""
    from .shared_frames import warm_frame

    out: dict[str, pd.DataFrame] = {}
    for _plat, attr in _PLATFORM_ATTRS:
        df = warm_frame(attr, sess)
        if df is not None and not getattr(df, "empty", True):
            out[attr] = df
    return out


def _overlay_tier3_on_platform_frames(
    sess,
    frame_overrides: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    window_plats: list[str],
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Merge Tier-3 authoritative rows into bulk platform frames for ``start..end``."""
    from .daily_store import load_platform_data_for_report_range, merge_platform_data

    used_tier3 = False
    out = dict(frame_overrides)
    for platform, attr in _PLATFORM_ATTRS:
        tier = pd.DataFrame()
        if platform in window_plats:
            tier = load_platform_data_for_report_range(
                platform, start, end, dedup=True, columns_only=True
            )
        if tier.empty:
            if attr not in out:
                cur = getattr(sess, attr, pd.DataFrame())
                if cur is not None and not getattr(cur, "empty", True):
                    out[attr] = cur
            continue
        used_tier3 = True
        cur = out.get(attr, getattr(sess, attr, pd.DataFrame()))
        lo, _hi = platform_df_date_bounds(tier)
        if (
            cur is not None
            and not getattr(cur, "empty", True)
            and lo is not None
            and pd.notna(lo)
        ):
            col = platform_date_column(cur)
            if col:
                cur_dates = pd.to_datetime(cur[col], errors="coerce")
                older = cur.loc[cur_dates < lo].copy()
                out[attr] = merge_platform_data(older, tier, platform) if not older.empty else tier
            else:
                out[attr] = tier
        else:
            out[attr] = tier
    return out, used_tier3


def _build_po_ads_from_platform_history(
    sess,
    *,
    planning_date: str,
    period_days: int,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> pd.DataFrame:
    """
    ADS input from Tier-1 platform bulk (+ Tier-3 overlay when uploads exist).

    Used when LY or seasonal signals need full marketplace history — unified
    ``sales_df`` is often shallow (OMS-only) while quarterly / PO columns use
  platform parquets.
    """
    from .daily_store import get_summary, get_tier3_sync_token, platforms_with_uploads_in_range
    from .po_calculate_run import _build_platform_sales_df

    plan = _normalize_planning_date(planning_date)
    bulk_frames = _session_platform_frames(sess)
    if not bulk_frames:
        return pd.DataFrame()

    horizon = _po_ads_horizon_days(period_days, use_seasonality, use_ly_fallback)
    end = plan
    start = str((pd.Timestamp(plan) - pd.Timedelta(days=horizon)).date())

    summary = get_summary() or {}
    tier3_any = any(int((summary.get(p) or {}).get("file_count") or 0) > 0 for p in summary)
    frame_overrides = bulk_frames
    if tier3_any:
        window_plats = platforms_with_uploads_in_range(start, end)
        if window_plats:
            frame_overrides, _used = _overlay_tier3_on_platform_frames(
                sess,
                bulk_frames,
                start=start,
                end=end,
                window_plats=window_plats,
            )

    # Check process-level cache before the expensive _build_platform_sales_df call.
    # Key captures frame sizes + tier3 state — any upload shifts the token → miss → rebuild.
    tier3_token = get_tier3_sync_token() or {}
    _cache_key = _plat_build_cache_key(frame_overrides, tier3_token)
    with _PLAT_BUILD_LOCK:
        base = _PLAT_BUILD_CACHE.get(_cache_key)
    if base is not None:
        _log.info("PO ADS platform build CACHE HIT (%d rows)", len(base))
    else:
        base = _build_platform_sales_df(sess, frame_overrides=frame_overrides)
        if not base.empty:
            with _PLAT_BUILD_LOCK:
                _PLAT_BUILD_CACHE.clear()  # keep only one entry — avoid memory bloat
                _PLAT_BUILD_CACHE[_cache_key] = base
            _log.info("PO ADS platform build cached (%d rows)", len(base))

    if base.empty:
        return pd.DataFrame()

    thru = session_sales_through(sess)
    lag = sales_data_lag_days(plan, thru)
    if lag is not None and 1 < lag <= 21 and thru:
        gap_start = str((pd.Timestamp(thru) + pd.Timedelta(days=1)).date())
        if gap_start <= plan:
            gap_sales = _build_tier3_gap_sales(sess, gap_start, plan)
            if not gap_sales.empty:
                trimmed = _trim_sales_to_ads_window(
                    base, plan, period_days, use_seasonality, use_ly_fallback
                )
                return _merge_sales_tail(
                    trimmed,
                    gap_sales,
                    plan,
                    period_days,
                    use_seasonality,
                    use_ly_fallback,
                )

    return _trim_sales_to_ads_window(
        base, plan, period_days, use_seasonality, use_ly_fallback
    )


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
    Does not mutate session frames or rebuild unified sales_df.

    Fast path: reuse unified ``sales_df`` when fresh (T-1) and deep enough.
    Incremental path: append only the missing tail from Tier-3 (few days of blobs).
    Slow path: full Tier-3 authoritative overlay when session has no sales history.
    """
    from .daily_store import (
        get_summary,
        platforms_with_uploads_in_range,
    )

    plan = _normalize_planning_date(planning_date)
    sdf = getattr(sess, "sales_df", None)
    thru = session_sales_through(sess)
    lag = sales_data_lag_days(plan, thru)

    needs_platform_history = use_ly_fallback or use_seasonality
    if needs_platform_history:
        bulk = _build_po_ads_from_platform_history(
            sess,
            planning_date=plan,
            period_days=period_days,
            use_seasonality=use_seasonality,
            use_ly_fallback=use_ly_fallback,
        )
        if not bulk.empty:
            _log.info(
                "PO ADS platform-history path (LY/seasonal): %s rows (lag=%sd)",
                len(bulk),
                lag,
            )
            return bulk

    if sdf is not None and not getattr(sdf, "empty", True):
        if _sales_has_ads_history(sdf, plan, period_days, use_seasonality, use_ly_fallback):
            if lag is not None and lag <= 1:
                out = _trim_sales_to_ads_window(
                    sdf, plan, period_days, use_seasonality, use_ly_fallback
                )
                _log.info(
                    "PO ADS fast path: session sales_df %s rows (lag=%sd)",
                    len(out),
                    lag,
                )
                return out

            _try_incremental_tier3 = lag is not None and 1 < lag <= 21 and thru
            try:
                from ..local_dev import local_dev_mode

                if local_dev_mode():
                    _try_incremental_tier3 = False
            except Exception:
                pass
            if _try_incremental_tier3:
                gap_start = str((pd.Timestamp(thru) + pd.Timedelta(days=1)).date())
                if gap_start <= plan:
                    gap_sales = _build_tier3_gap_sales(sess, gap_start, plan)
                    if not gap_sales.empty:
                        trimmed = _trim_sales_to_ads_window(
                            sdf, plan, period_days, use_seasonality, use_ly_fallback
                        )
                        out = _merge_sales_tail(
                            trimmed,
                            gap_sales,
                            plan,
                            period_days,
                            use_seasonality,
                            use_ly_fallback,
                        )
                        _log.info(
                            "PO ADS incremental tier3 %s..%s → %s rows (was %sd behind)",
                            gap_start,
                            plan,
                            len(out),
                            lag,
                        )
                        return out

            out = _trim_sales_to_ads_window(
                sdf, plan, period_days, use_seasonality, use_ly_fallback
            )
            _log.info(
                "PO ADS bulk path: session sales_df → %s rows (lag=%s, tier3 overlay skipped)",
                len(out),
                lag,
            )
            return out

    summary = get_summary() or {}
    tier3_any = any(int((summary.get(p) or {}).get("file_count") or 0) > 0 for p in summary)
    if not tier3_any:
        return pd.DataFrame()

    horizon = _po_ads_horizon_days(period_days, use_seasonality, use_ly_fallback)
    end = plan
    start = str((pd.Timestamp(plan) - pd.Timedelta(days=horizon)).date())

    window_plats = platforms_with_uploads_in_range(start, end)
    if not window_plats:
        return pd.DataFrame()

    bulk_frames = _session_platform_frames(sess)
    frame_overrides, used_tier3 = _overlay_tier3_on_platform_frames(
        sess,
        bulk_frames,
        start=start,
        end=end,
        window_plats=window_plats,
    )

    if not used_tier3 and not frame_overrides:
        return pd.DataFrame()

    from .po_calculate_run import _build_platform_sales_df

    _log.info("PO ADS slow path: full tier3 overlay %s..%s", start, end)
    return _build_platform_sales_df(sess, frame_overrides=frame_overrides)

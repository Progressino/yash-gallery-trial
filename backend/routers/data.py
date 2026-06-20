"""
Data query router — analytics endpoints.
GET /api/data/coverage, sales-summary, sales-export, sales-by-source, daily-dsr, daily-dsr-export,
dsr-brand-monthly, dsr-brand-monthly-export, top-skus,
mtr-analytics, myntra-analytics, meesho-analytics, flipkart-analytics, inventory
"""
import csv
import datetime
import io
import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional, Set
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from ..models.schemas import CoverageResponse, JobStatusResponse, RestoreFullResponse, IntelligenceReadinessResponse, DashboardSummaryResponse
from ..services.helpers import (
    clean_sku,
    get_parent_sku,
    is_likely_non_sku_notes_value,
    map_to_oms_sku,
    mapping_lookup_sets,
    normalize_id_token_for_mapping,
    sku_recognized_in_master,
)
from ..services.meesho import apply_meesho_listing_sku_recovery_for_export
from ..services.sales import (
    _filter_by_reporting_days,
    apply_upload_report_day_gate,
    canonical_sales_sku,
    canonical_sales_sku_series,
    daily_dsr_report_to_csv_rows,
    filter_sales_for_export,
    get_anomalies,
    get_daily_dsr_report,
    dsr_brand_monthly_to_csv_rows,
    get_dsr_brand_monthly_comparison,
    get_platform_summary,
    get_sales_by_source,
    get_sales_summary,
    get_top_skus,
    txn_reporting_naive_ist,
)
from ..services.daily_store import list_uploads, get_summary, delete_upload
from ..session import AppSession
from ..services.inventory import (
    ensure_inventory_snapshot_metadata,
    inventory_marketplace_breakdown,
    inventory_missing_marketplace_warnings,
    inventory_rows_for_api,
    inventory_snapshot_meta_for_api,
    refresh_inventory_api_cache,
    sync_inventory_snapshot_from_warm,
)

router = APIRouter()

# Process-wide Intelligence bundle cache (PG-restore shells share the same Tier-3 window).
# Bump when bundle shape/semantics change (invalidates persisted intel_bundle_*.json keys).
_INTELLIGENCE_BUNDLE_CACHE_GEN = "v3"
_GLOBAL_INTELLIGENCE_BUNDLE_CACHE: dict = {}

# How long a cached bundle is served without recomputation. Real data changes
# invalidate this cache directly (see _invalidate_intelligence_bundle_cache),
# so this TTL only needs to bound staleness from sources that don't trigger
# invalidation (e.g. anomaly/DSR drift over time) — keep it generous.
_INTELLIGENCE_BUNDLE_TTL_SEC = 1800.0

_INTEL_BUNDLE_DISK_DIR = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")


def _intel_bundle_disk_path(global_key: tuple) -> str:
    import hashlib

    digest = hashlib.sha1(repr(global_key).encode("utf-8")).hexdigest()[:16]
    return os.path.join(_INTEL_BUNDLE_DISK_DIR, f"intel_bundle_{digest}.json")


def _save_intelligence_bundle_to_disk(global_key: tuple, entry: dict) -> None:
    """Best-effort persist of one global bundle-cache entry so the first
    Intelligence request after a restart can serve instantly."""
    try:
        os.makedirs(_INTEL_BUNDLE_DISK_DIR, exist_ok=True)
        path = _intel_bundle_disk_path(global_key)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"key": list(global_key), "entry": entry}, f, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_intelligence_bundle_cache_from_disk() -> None:
    """Populate _GLOBAL_INTELLIGENCE_BUNDLE_CACHE from persisted entries at
    startup so the first request after a restart doesn't recompute."""
    try:
        if not os.path.isdir(_INTEL_BUNDLE_DISK_DIR):
            return
        for name in os.listdir(_INTEL_BUNDLE_DISK_DIR):
            if not (name.startswith("intel_bundle_") and name.endswith(".json")):
                continue
            try:
                with open(os.path.join(_INTEL_BUNDLE_DISK_DIR, name)) as f:
                    data = json.load(f)
                global_key = tuple(
                    tuple(x) if isinstance(x, list) else x for x in data.get("key") or []
                )
                entry = data.get("entry")
                if global_key and isinstance(entry, dict):
                    _GLOBAL_INTELLIGENCE_BUNDLE_CACHE[global_key] = entry
            except Exception:
                continue
    except Exception:
        pass


def _invalidate_intelligence_bundle_cache() -> None:
    """Drop the process-wide Intelligence bundle cache (memory + disk) after
    sales/platform data actually changes."""
    _GLOBAL_INTELLIGENCE_BUNDLE_CACHE.clear()
    try:
        if os.path.isdir(_INTEL_BUNDLE_DISK_DIR):
            for name in os.listdir(_INTEL_BUNDLE_DISK_DIR):
                if name.startswith("intel_bundle_") and name.endswith(".json"):
                    try:
                        os.remove(os.path.join(_INTEL_BUNDLE_DISK_DIR, name))
                    except OSError:
                        pass
    except Exception:
        pass
    schedule_intelligence_bundle_precompute()


_intel_precompute_lock = threading.Lock()
_intel_precompute_running = False


def schedule_intelligence_bundle_precompute() -> None:
    """Queue background precompute of default Intelligence bundles (one at a time)."""
    if os.environ.get("INTELLIGENCE_PRECOMPUTE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    global _intel_precompute_running
    with _intel_precompute_lock:
        if _intel_precompute_running:
            return
        _intel_precompute_running = True
    try:
        from ..concurrency import HEAVY_EXECUTOR

        HEAVY_EXECUTOR.submit(_precompute_intelligence_bundles_worker)
    except Exception:
        with _intel_precompute_lock:
            _intel_precompute_running = False


def _precompute_intelligence_bundles_worker() -> None:
    global _intel_precompute_running
    _log = logging.getLogger(__name__)
    try:
        precompute_default_intelligence_bundles()
    except Exception:
        _log.exception("intelligence bundle precompute failed")
    finally:
        with _intel_precompute_lock:
            _intel_precompute_running = False


def precompute_default_intelligence_bundles() -> None:
    """Pre-build default Dashboard Intelligence window into global/disk cache."""
    import backend.main as _main

    _log = logging.getLogger(__name__)
    if not _main._warm_cache_ready.wait(timeout=600.0):
        _log.warning("intelligence precompute: warm cache not ready after 600s")
        return
    if not _main._warm_cache:
        _log.warning("intelligence precompute: warm cache empty")
        return

    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    today = datetime.datetime.now(ist).date()
    end_date = today.isoformat()
    start_date = (today - datetime.timedelta(days=30)).isoformat()

    sess = AppSession()
    try:
        _main.try_attach_shared_frames_fast(sess)
    except Exception:
        _log.exception("intelligence precompute: attach shared frames failed")
        return

    bundle_cache: dict = {}
    now = time.time()
    cache_key = (start_date, end_date, "gross", 10, False)

    if _bundle_cache_lookup(
        cache_key,
        bundle_cache,
        start_date=start_date,
        end_date=end_date,
        allow_sparse=True,
        sess=sess,
    ):
        _log.info("intelligence precompute: cache hit for %s..%s", start_date, end_date)
        return

    session_build_unsafe = _intelligence_session_build_unsafe(sess)
    if not session_build_unsafe or _disk_bulk_history_available():
        payload = _build_intelligence_bundle_payload_from_session(
            sess, start_date, end_date, 10, "gross", False
        )
        if payload and _bundle_payload_has_display_data(payload):
            payload["status"] = "ready"
            _bundle_cache_store(cache_key, bundle_cache, payload, ts=now)
            units = int((payload.get("sales_summary") or {}).get("total_units") or 0)
            _log.warning(
                "intelligence precompute: stored session bundle %s..%s (%d units)",
                start_date,
                end_date,
                units,
            )
            return

    tier3_payload = _try_serve_tier3_intelligence_bundle(
        sess,
        cache_key,
        bundle_cache,
        start_date,
        end_date,
        10,
        "gross",
        False,
        ts=now,
    )
    if tier3_payload is not None:
        units = int((tier3_payload.get("sales_summary") or {}).get("total_units") or 0)
        _log.warning(
            "intelligence precompute: stored tier3 bundle %s..%s (%d units)",
            start_date,
            end_date,
            units,
        )


def _sku_deepdive_aliases(raw: str) -> Set[str]:
    """Return SKU tokens that should match the same row after PL/YK normalisation."""
    from ..services.sku_deepdive_data import deepdive_sku_alias_tokens

    return deepdive_sku_alias_tokens(raw)


def _sess(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


_PLATFORM_ATTRS = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)

# Restore success / warnings ignore optional platforms (e.g. Snapdeal not used by all tenants).
_ESSENTIAL_RESTORE_PLATFORMS = frozenset({"amazon", "myntra", "meesho", "flipkart"})


def _essential_missing_platforms(missing: list[str]) -> list[str]:
    return [p for p in missing if p in _ESSENTIAL_RESTORE_PLATFORMS]


def _session_operational_data_complete(sess: AppSession) -> bool:
    """True when SKU + all platforms + sales + inventory are loaded in session."""
    if not getattr(sess, "sku_mapping", None):
        return False
    sales = getattr(sess, "sales_df", None)
    if sales is None or not hasattr(sales, "empty") or sales.empty:
        return False
    for _plat, attr in _PLATFORM_ATTRS:
        df = getattr(sess, attr, None)
        if df is None or not hasattr(df, "empty") or df.empty:
            return False
    inv = getattr(sess, "inventory_df_variant", None)
    if inv is None or not hasattr(inv, "empty") or inv.empty:
        return False
    return True


def _missing_platform_names(sess: AppSession) -> list[str]:
    out: list[str] = []
    for plat, attr in _PLATFORM_ATTRS:
        df = getattr(sess, attr, None)
        if df is None or not hasattr(df, "empty") or df.empty:
            out.append(plat)
    return out


def _merge_disk_warm_into_session(sess: AppSession) -> str:
    """Merge on-disk warm snapshot into session platform frames (fills gaps vs GitHub-only)."""
    import pandas as pd

    try:
        from backend.routers.cache import _disk_recovery_payload, _sanitize_snapdeal_in_loaded
        from ..services.daily_store import merge_platform_data
    except Exception:
        return ""
    try:
        disk_data = _disk_recovery_payload()
    except Exception:
        return ""
    if not disk_data:
        return ""
    _sanitize_snapdeal_in_loaded(disk_data)
    notes: list[str] = []
    for plat, attr in _PLATFORM_ATTRS:
        dsk = disk_data.get(attr)
        if not isinstance(dsk, pd.DataFrame) or dsk.empty:
            continue
        label = _RESTORE_STEP_LABEL.get(_GITHUB_STEP_BY_PLATFORM.get(plat, ""), plat)
        _set_restore_step(sess, "disk", f"Disk backup — {label or plat}…")

        def _merge_one(cur_df, dsk_df=dsk, platform=plat, attribute=attr):
            merged = merge_platform_data(cur_df, dsk_df, platform)
            setattr(sess, attribute, merged)
            return len(merged)

        cur = getattr(sess, attr)
        if not isinstance(cur, pd.DataFrame):
            cur = pd.DataFrame()
        before = len(cur) if not cur.empty else 0
        after = _run_with_restore_heartbeat(
            sess, "disk", f"Disk — {plat}", lambda c=cur: _merge_one(c)
        )
        if after > before:
            notes.append(f"{attr} +{after - before:,}")
    dm = disk_data.get("sku_mapping")
    if isinstance(dm, dict) and dm and not sess.sku_mapping:
        sess.sku_mapping = dm.copy()
        notes.append("sku_mapping from disk")
    if not notes:
        return ""
    return " Disk snapshot merged (" + "; ".join(notes) + ")."


def _restore_daily_if_needed(
    sess: AppSession,
    *,
    force: bool = False,
    lock_timeout: float | None = None,
    restore_full_mode: bool = False,
    skip_sales_rebuild: bool = False,
) -> None:
    """
    On first coverage check per session, merge any persisted daily SQLite data
    into the session platform DFs (fills gaps after restart and folds Tier-3
    rows into warm-cache copies without replacing bulk history).
    Also auto-restores SKU mapping from GitHub cache if missing.

    ``force=True`` (restore-full): blocking lock, full Tier-3 pass, ignores pause flag.
    """
    import pandas as pd

    if force:
        try:
            from ..session import resume_auto_data_restore

            sess.pause_auto_data_restore = False
            resume_auto_data_restore(sess)
        except Exception:
            pass
        sess.daily_restored = False

    if getattr(sess, "daily_inventory_upload_status", "idle") == "running" and not force:
        return
    if getattr(sess, "inventory_upload_status", "idle") == "running" and not force:
        return
    if getattr(sess, "tier1_bulk_status", "idle") == "running" and not force:
        return
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
    except Exception:
        pass

    needs_data = False
    try:
        import backend.main as _main

        needs_data = _main.session_needs_operational_data(sess)
    except Exception:
        needs_data = False

    # After "Clear all app data": warm-cache fill above may still be empty; do not pull
    # Tier-3 SQLite into the session until the user uploads or clicks Load Cache.
    if not force and getattr(sess, "pause_auto_data_restore", False) and needs_data:
        return

    if not force and getattr(sess, "pause_auto_data_restore", False) and not needs_data:
        return
    needs_tier3_refresh = (
        _tier3_session_needs_topup(sess)
        or _session_sales_stale_vs_platforms(sess)
        or (force and bool(_missing_platform_names(sess)))
    )
    if sess.daily_restored and not needs_data and not force:
        if not needs_tier3_refresh:
            return
    if force or needs_data or needs_tier3_refresh:
        sess.daily_restored = False

    if force:
        acquired = sess._daily_restore_lock.acquire(
            blocking=True, timeout=lock_timeout if lock_timeout is not None else 300.0
        )
    else:
        acquired = sess._daily_restore_lock.acquire(blocking=False)
    if not acquired:
        return

    lock_held = True
    try:
        if sess.daily_restored and not needs_tier3_refresh and not force:
            return

        import pandas as pd
        from ..services.daily_store import load_platform_data, merge_platform_data
        from ..services.sales import build_sales_df

        try:
            from ..services.sku_mapping import restore_sku_mapping_to_session

            restore_sku_mapping_to_session(sess)
        except Exception:
            pass

        # Default restore should be full history. Operators can opt-in caps via env when
        # they prefer faster startup over completeness.
        # 0 or negative disables each cap (loads all history).
        try:
            from ..services.platform_session_window import AUTO_RESTORE_MONTHS_DEFAULT

            _auto_months = int((os.environ.get("AUTO_RESTORE_MONTHS") or str(AUTO_RESTORE_MONTHS_DEFAULT)).strip())
        except Exception:
            from ..services.platform_session_window import AUTO_RESTORE_MONTHS_DEFAULT

            _auto_months = AUTO_RESTORE_MONTHS_DEFAULT
        _auto_months = None if _auto_months <= 0 else _auto_months
        try:
            _auto_max_files = int((os.environ.get("AUTO_RESTORE_MAX_FILES") or "0").strip())
        except Exception:
            _auto_max_files = 0
        _auto_max_files = None if _auto_max_files <= 0 else _auto_max_files

        # Restore-full must load all Tier-3 history — bulk uploads live in SQLite and a
        # 12-month window leaves Myntra/Meesho/Flipkart at ~40–50% of stored rows.
        _tier3_months = None if (force or restore_full_mode) else _auto_months
        _tier3_max_files = None if (force or restore_full_mode) else _auto_max_files

        changed = False
        platform_attrs = [
            ("amazon",   "mtr_df"),
            ("myntra",   "myntra_df"),
            ("meesho",   "meesho_df"),
            ("flipkart", "flipkart_df"),
            ("snapdeal", "snapdeal_df"),
        ]
        lock_reacquire_failed = False
        for idx, (platform, attr) in enumerate(platform_attrs):
            if lock_reacquire_failed:
                break
            if restore_full_mode:
                _set_restore_step(sess, "tier3", f"Tier-3 — loading {platform}…")
                if lock_held:
                    sess._daily_restore_lock.release()
                    lock_held = False
                try:
                    df = load_platform_data(
                        platform,
                        months=_tier3_months,
                        dedup=False,
                        max_files=_tier3_max_files,
                    )
                finally:
                    if not lock_held:
                        if sess._daily_restore_lock.acquire(blocking=True, timeout=120.0):
                            lock_held = True
                        else:
                            lock_reacquire_failed = True
            else:
                df = load_platform_data(
                    platform,
                    months=_tier3_months,
                    dedup=False,
                    max_files=_tier3_max_files,
                )
            if not df.empty:
                cur = getattr(sess, attr)

                def _merge_t3(cur_df=cur, new_df=df, plat=platform, attribute=attr):
                    merged = merge_platform_data(cur_df, new_df, plat)
                    setattr(sess, attribute, merged)
                    return len(merged)

                if restore_full_mode:
                    _run_with_restore_heartbeat(
                        sess,
                        "tier3",
                        f"Tier-3 merge {platform}",
                        _merge_t3,
                    )
                else:
                    setattr(sess, attr, merge_platform_data(cur, df, platform))
                changed = True

        # Safety net: if bounded restore found nothing, do one full-history pass.
        if not changed and not restore_full_mode:
            for _p, _a in platform_attrs:
                if not getattr(sess, _a).empty:
                    changed = True
                    break
        if not changed:
            for platform, attr in platform_attrs:
                if getattr(sess, attr).empty:
                    if restore_full_mode:
                        _set_restore_step(sess, "tier3", f"Tier-3 — full load {platform} (empty only)…")
                    df = load_platform_data(
                        platform,
                        months=None if not restore_full_mode else _auto_months,
                        dedup=False,
                        max_files=None if not restore_full_mode else _auto_max_files,
                    )
                    if not df.empty:
                        cur = getattr(sess, attr)
                        setattr(sess, attr, merge_platform_data(cur, df, platform))
                        changed = True

        if changed and not skip_sales_rebuild:
            try:
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                    **_sales_overlay_build_kwargs(sess),
                )
                sess._quarterly_cache.clear()
                _invalidate_shared_quarterly_cache()
                _invalidate_intelligence_bundle_cache()
            except Exception:
                pass

        # Restore inventory from warm cache only (fast — already in memory). Inventory has no
        # SQLite backing so it's lost on server restart until warm cache is populated at startup
        # or the user runs Load Cache from the UI.
        need_inventory = sess.inventory_df_variant.empty
        if need_inventory:
            try:
                import backend.main as _main
                if _main._warm_cache:
                    # Fast path: copy from in-memory warm cache — no network call
                    for key in ["inventory_df_variant", "inventory_df_parent"]:
                        val = _main._warm_cache.get(key)
                        if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                            setattr(sess, key, val)
                    if not sess.sku_mapping and _main._warm_cache.get("sku_mapping"):
                        sess.sku_mapping = _main._warm_cache["sku_mapping"]
                # Intentionally do **not** call load_cache_from_drive() here: it can pull many
                # large parquet assets from GitHub and block this request (and the whole session
                # lock) for minutes — nginx/Cloudflare return 504 while the dashboard stays on
                # "Loading…". Warm cache fills shortly after startup; users can use Load Cache /
                # Fresh reload for an explicit GitHub restore.
            except Exception:
                pass

        # Same story for optional PO sheets: they are not rebuilt from SQLite Tier-1/2 data.
        # Without restoring from warm cache, every new session shows "No sheet loaded" until
        # the user re-uploads even though ``extend_history_with_sales`` already rolls forward
        # from the baseline in memory on the shared cache.
        try:
            import backend.main as _main
            if _main._warm_cache:
                for key in ("daily_inventory_history_df", "sku_status_lead_df", "po_raise_ledger_df", "po_return_overlay_df"):
                    if getattr(sess, key, None) is not None and not getattr(sess, key).empty:
                        continue
                    val = _main._warm_cache.get(key)
                    if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                        setattr(sess, key, val)
        except Exception:
            pass

        sess.daily_restored = True
    finally:
        if lock_held:
            sess._daily_restore_lock.release()


_SOURCE_BY_ATTR = {
    "mtr_df": "Amazon",
    "myntra_df": "Myntra",
    "meesho_df": "Meesho",
    "flipkart_df": "Flipkart",
    "snapdeal_df": "Snapdeal",
}


def _platform_df_max_iso(df) -> Optional[str]:
    import pandas as pd

    if df is None or not hasattr(df, "empty") or df.empty:
        return None
    for col in ("Date", "TxnDate", "_Date"):
        if col not in df.columns:
            continue
        d = pd.to_datetime(df[col], errors="coerce").max()
        if pd.notna(d):
            return str(pd.Timestamp(d).date())
    return None


def _platform_df_min_iso(df) -> Optional[str]:
    import pandas as pd

    from ..services.platform_session_window import platform_df_date_bounds

    lo, _ = platform_df_date_bounds(df)
    if lo is None:
        return None
    return str(pd.Timestamp(lo).date())


def _sales_max_for_source(sales_df, source: str) -> Optional[str]:
    """Latest unified-sales calendar day for a marketplace ``Source``."""
    import pandas as pd

    from ..services.sales import txn_reporting_naive_ist

    if sales_df is None or not hasattr(sales_df, "empty") or sales_df.empty:
        return None
    if "Source" not in sales_df.columns or "TxnDate" not in sales_df.columns:
        return None
    sub = sales_df[sales_df["Source"].astype(str).str.strip() == source]
    if sub.empty:
        return None
    mx = txn_reporting_naive_ist(sub["TxnDate"]).max()
    if pd.isna(mx):
        return None
    return str(pd.Timestamp(mx).date())


def _mark_tier3_sync_applied(sess: AppSession) -> None:
    from ..services.daily_store import get_tier3_sync_token

    sess._tier3_sync_token_applied = get_tier3_sync_token()


def _tier3_token_mismatch(sess: AppSession) -> bool:
    from ..services.daily_store import get_tier3_sync_token

    store = get_tier3_sync_token()
    applied: Dict[str, str] = getattr(sess, "_tier3_sync_token_applied", None) or {}
    return store != applied


def _platforms_with_tier3_token_mismatch(sess: AppSession) -> list[str]:
    from ..services.daily_store import get_tier3_sync_token

    store = get_tier3_sync_token()
    applied: Dict[str, str] = getattr(sess, "_tier3_sync_token_applied", None) or {}
    out: list[str] = []
    for plat, _attr in _PLATFORM_ATTRS:
        if store.get(plat) != applied.get(plat):
            out.append(plat)
    return out


def _report_span_days(
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[int]:
    """Inclusive calendar days in the reporting window, or None if open-ended."""
    s0 = str(start_date or "").strip()[:10]
    s1 = str(end_date or "").strip()[:10]
    if len(s0) != 10 and len(s1) != 10:
        return None
    if len(s0) != 10:
        s0 = s1
    if len(s1) != 10:
        s1 = s0
    try:
        d0 = datetime.date.fromisoformat(s0)
        d1 = datetime.date.fromisoformat(s1)
    except ValueError:
        return None
    if d1 < d0:
        d0, d1 = d1, d0
    return (d1 - d0).days + 1


def _intelligence_fast_window_days() -> int:
    try:
        return max(7, int((os.environ.get("INTELLIGENCE_FAST_WINDOW_DAYS") or "45").strip()))
    except ValueError:
        return 45


def _refresh_sales_days_by_source(sess: AppSession) -> Dict[str, Set[str]]:
    """Per-marketplace calendar days present in unified sales (cheap gap checks)."""
    sales = getattr(sess, "sales_df", None)
    out: Dict[str, Set[str]] = {}
    for _attr, src in _SOURCE_BY_ATTR.items():
        out[src] = set()
    if sales is None or not hasattr(sales, "empty") or sales.empty:
        sess._sales_days_by_source = out
        return out
    if "Source" not in sales.columns or "TxnDate" not in sales.columns:
        sess._sales_days_by_source = out
        return out
    src_col = sales["Source"].astype(str).str.strip()
    days = txn_reporting_naive_ist(sales["TxnDate"]).dt.normalize().dt.strftime("%Y-%m-%d")
    for src in out:
        mask = src_col == src
        if mask.any():
            out[src] = set(days.loc[mask].dropna().unique())
    sess._sales_days_by_source = out
    return out


def _sales_days_by_source(sess: AppSession) -> Dict[str, Set[str]]:
    cached = getattr(sess, "_sales_days_by_source", None)
    if isinstance(cached, dict):
        return cached
    return _refresh_sales_days_by_source(sess)


def _platforms_with_sales_gaps_fast(
    sess: AppSession,
    start_date: str,
    end_date: str,
) -> list[str]:
    """
    Platforms where Tier-3 coverage includes a day in the window but unified sales
    has no rows that day (e.g. June 1 backfill while session max is June 4).
    Uses cached per-source day sets — O(window × platforms), not O(sales rows).
    """
    from ..services.daily_store import get_upload_report_day_coverage

    s0 = str(start_date or "").strip()[:10]
    s1 = str(end_date or "").strip()[:10]
    if len(s0) != 10 or len(s1) != 10:
        return []
    try:
        d0 = datetime.date.fromisoformat(s0)
        d1 = datetime.date.fromisoformat(s1)
    except ValueError:
        return []
    if d1 < d0:
        d0, d1 = d1, d0
    if (d1 - d0).days > 120:
        return []

    cov = get_upload_report_day_coverage()
    have_by_src = _sales_days_by_source(sess)
    out: list[str] = []
    cur = d0
    while cur <= d1:
        day = cur.isoformat()
        cur += datetime.timedelta(days=1)
        for plat, attr in _PLATFORM_ATTRS:
            if day not in (cov.get(plat) or set()):
                continue
            src = _SOURCE_BY_ATTR.get(attr, "")
            if not src:
                continue
            if day not in (have_by_src.get(src) or set()) and plat not in out:
                out.append(plat)
    return out


def _ensure_sku_mapping_for_dashboard(sess: AppSession) -> bool:
    if getattr(sess, "sku_mapping", None):
        return True
    try:
        import backend.main as _main

        warm = getattr(_main, "_warm_cache", None) or {}
        cmap = warm.get("sku_mapping")
        if cmap:
            sess.sku_mapping = cmap
            return True
    except Exception:
        pass
    return False


def _platform_shipment_units_in_slice(
    sales_slice: "pd.DataFrame",
    source: str,
) -> int:
    import pandas as pd

    if sales_slice is None or sales_slice.empty:
        return 0
    if "Source" not in sales_slice.columns:
        return 0
    sub = sales_slice[sales_slice["Source"].astype(str).str.strip() == source]
    if sub.empty or "Transaction Type" not in sub.columns:
        return 0
    txn = sub["Transaction Type"].astype(str).str.strip()
    qty = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0)
    return int(qty[txn == "Shipment"].sum())


def _platforms_needing_auto_tier3_pull(
    sess: AppSession,
    start_date: str,
    end_date: str,
    sales_slice: "pd.DataFrame",
) -> list[str]:
    """
    Pull Tier-3 when uploads exist for the window but unified sales are missing or zero.
    """
    from ..services.daily_store import (
        load_platform_data_for_report_range,
        platforms_with_uploads_in_range,
    )

    uploaded = set(platforms_with_uploads_in_range(start_date, end_date))
    if not uploaded:
        return []

    need: set[str] = set(_platforms_with_sales_gaps_fast(sess, start_date, end_date))
    need |= set(_platforms_with_tier3_token_mismatch(sess)) & uploaded

    sales = getattr(sess, "sales_df", None)
    if sales is None or not hasattr(sales, "empty") or sales.empty:
        return sorted(uploaded)

    for plat, attr in _PLATFORM_ATTRS:
        if plat not in uploaded:
            continue
        src = _SOURCE_BY_ATTR.get(attr, "")
        if not src:
            continue
        if _platform_shipment_units_in_slice(sales_slice, src) > 0:
            continue
        chunk = load_platform_data_for_report_range(plat, start_date, end_date, dedup=True)
        if not chunk.empty:
            need.add(plat)
    return sorted(need)


def _load_tier3_frames_for_platforms(
    platforms: list[str],
    start_date: str,
    end_date: str,
    *,
    dedup: bool = False,
    columns_only: bool = False,
) -> dict[str, "pd.DataFrame"]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ..services.daily_store import load_platform_data_for_report_range

    plats = [str(p).strip().lower() for p in platforms if str(p).strip()]
    if not plats:
        return {}

    def _one(plat: str):
        chunk = load_platform_data_for_report_range(
            plat,
            start_date,
            end_date,
            dedup=dedup,
            columns_only=columns_only,
        )
        if chunk.empty and columns_only:
            chunk = load_platform_data_for_report_range(
                plat,
                start_date,
                end_date,
                dedup=dedup,
                columns_only=False,
            )
        return plat, chunk

    out: dict[str, pd.DataFrame] = {}
    workers = min(5, len(plats))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, p) for p in plats]
        for fut in as_completed(futs):
            try:
                plat, chunk = fut.result()
                if chunk is not None and not chunk.empty:
                    out[plat] = chunk
            except Exception:
                pass
    return out


def _bundle_platform_frames(
    sess: AppSession,
    tier3_frames: dict[str, "pd.DataFrame"],
) -> tuple:
    """Session platform frames merged with Tier-3 window slices (for platform cards)."""
    import pandas as pd
    from ..services.daily_store import merge_platform_data

    mtr = getattr(sess, "mtr_df", pd.DataFrame())
    myntra = getattr(sess, "myntra_df", pd.DataFrame())
    meesho = getattr(sess, "meesho_df", pd.DataFrame())
    flipkart = getattr(sess, "flipkart_df", pd.DataFrame())
    snapdeal = getattr(sess, "snapdeal_df", pd.DataFrame())

    for plat, chunk in tier3_frames.items():
        if plat == "amazon":
            mtr = merge_platform_data(mtr, chunk, plat)
        elif plat == "myntra":
            myntra = merge_platform_data(myntra, chunk, plat)
        elif plat == "meesho":
            meesho = merge_platform_data(meesho, chunk, plat)
        elif plat == "flipkart":
            flipkart = merge_platform_data(flipkart, chunk, plat)
        elif plat == "snapdeal":
            snapdeal = merge_platform_data(snapdeal, chunk, plat)
    return mtr, myntra, meesho, flipkart, snapdeal


def _build_sales_from_tier3_frames(
    sess: AppSession,
    tier3_frames: dict[str, "pd.DataFrame"],
) -> "pd.DataFrame":
    import pandas as pd
    from ..services.sales import build_sales_df

    if not tier3_frames:
        return pd.DataFrame()
    return build_sales_df(
        mtr_df=tier3_frames.get("amazon", pd.DataFrame()),
        myntra_df=tier3_frames.get("myntra", pd.DataFrame()),
        meesho_df=tier3_frames.get("meesho", pd.DataFrame()),
        flipkart_df=tier3_frames.get("flipkart", pd.DataFrame()),
        snapdeal_df=tier3_frames.get("snapdeal", pd.DataFrame()),
        sku_mapping=sess.sku_mapping or {},
        **_sales_overlay_build_kwargs(sess),
    )


def _slice_sales_for_bundle(
    sales_df: "pd.DataFrame",
    start_date: Optional[str],
    end_date: Optional[str],
) -> "pd.DataFrame":
    from ..services.sales import apply_upload_report_day_gate

    if sales_df is None or not hasattr(sales_df, "empty") or sales_df.empty:
        return sales_df if sales_df is not None else __import__("pandas").DataFrame()
    # Clip to the reporting window before upload-day gating (much faster on large sales_df).
    clipped = sales_df
    if (start_date or end_date) and "TxnDate" in sales_df.columns:
        clipped = _filter_by_reporting_days(sales_df, "TxnDate", start_date, end_date)
    return apply_upload_report_day_gate(clipped)


def _platform_max_reporting_day_in_window(
    df: "pd.DataFrame",
    start_date: str,
    end_date: str,
    *,
    date_col: str = "Date",
) -> str:
    """Latest reporting day present in a platform frame within ``[start, end]``."""
    import pandas as pd

    w = _filter_platform_df_by_window(df, start_date, end_date, date_col=date_col)
    if w.empty:
        return ""
    col = "_Date" if "_Date" in w.columns else date_col
    if col not in w.columns:
        return ""
    t = txn_reporting_naive_ist(pd.to_datetime(w[col], errors="coerce")).dropna()
    if t.empty:
        return ""
    return str(t.max().normalize())[:10]


def _warm_disk_platform_frame(
    attr: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> "pd.DataFrame":
    """Read one platform parquet from warm-cache disk (PO-session-only skips RAM load)."""
    import os
    from pathlib import Path

    import pandas as pd

    try:
        import backend.main as _main

        mem = (_main._warm_cache or {}).get(attr)
        if mem is not None and hasattr(mem, "empty") and not mem.empty:
            return mem
    except Exception:
        pass

    disk_dir = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    path = disk_dir / f"{attr}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        s = str(start_date or "")[:10]
        e = str(end_date or "")[:10]
        if len(s) == 10 and len(e) == 10:
            try:
                return pd.read_parquet(
                    path,
                    filters=[
                        ("Date", ">=", pd.Timestamp(s)),
                        ("Date", "<=", pd.Timestamp(e) + pd.Timedelta(days=1)),
                    ],
                )
            except Exception:
                pass
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _platform_df_for_intelligence_bundle(
    sess: AppSession,
    platform_key: str,
    attr: str,
    start_date: str,
    end_date: str,
) -> "pd.DataFrame":
    """
    Session platform frame plus Tier-3 gap-fill when uploads extend past session data
    (e.g. daily files saved through June while warm-cache session stops at May 29).
    """
    import pandas as pd
    from ..services.daily_store import (
        load_platform_data_for_report_range,
        merge_platform_data,
        platforms_with_uploads_in_range,
    )

    raw = getattr(sess, attr, None)
    if raw is None or not hasattr(raw, "empty"):
        raw = pd.DataFrame()
    s = str(start_date)[:10]
    e = str(end_date)[:10]
    if len(s) != 10 or len(e) != 10:
        return raw

    in_w = _filter_platform_df_by_window(raw, s, e)
    disk = _warm_disk_platform_frame(attr, s, e)
    in_disk = _filter_platform_df_by_window(disk, s, e) if not disk.empty else pd.DataFrame()
    if not in_disk.empty and (in_w.empty or len(in_w) < max(len(in_disk) // 2, 1000)):
        raw = disk
        in_w = in_disk
    elif in_w.empty and not in_disk.empty:
        raw = disk
        in_w = in_disk

    if platform_key not in platforms_with_uploads_in_range(s, e):
        return raw

    in_w = _filter_platform_df_by_window(raw, s, e)
    sess_max = _platform_max_reporting_day_in_window(raw, s, e)
    need_topup = in_w.empty or (bool(sess_max) and sess_max < e)
    if not need_topup:
        return raw

    gap_start = s
    if sess_max and sess_max < e:
        try:
            gap_start = (
                pd.Timestamp(sess_max) + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        except Exception:
            gap_start = s
    if gap_start > e:
        return raw

    t3 = load_platform_data_for_report_range(
        platform_key,
        gap_start,
        e,
        dedup=False,
        columns_only=True,
    )
    if t3.empty:
        t3 = load_platform_data_for_report_range(
            platform_key,
            gap_start,
            e,
            dedup=False,
            columns_only=False,
        )
    if t3.empty:
        return raw
    if in_w.empty:
        merged = t3
    else:
        merged = merge_platform_data(raw, t3, platform_key)
    if _filter_platform_df_by_window(merged, s, e).empty and not in_w.empty:
        return raw
    return merged if not merged.empty else raw


def _filter_platform_df_by_window(
    df: "pd.DataFrame",
    start_date: str,
    end_date: str,
    *,
    date_col: str = "Date",
) -> "pd.DataFrame":
    """In-memory platform frame clipped to the dashboard window (no Tier-3 blob load)."""
    import pandas as pd

    if df is None or not hasattr(df, "empty") or df.empty:
        return pd.DataFrame()
    if date_col not in df.columns:
        return df
    d = df.copy()
    d["_Date"] = txn_reporting_naive_ist(
        pd.to_datetime(d[date_col], errors="coerce")
    )
    d = d.dropna(subset=["_Date"])
    return _filter_by_reporting_days(d, "_Date", start_date, end_date)


def _session_has_operational_frames(sess: AppSession) -> bool:
    sales = getattr(sess, "sales_df", None)
    if sales is not None and hasattr(sales, "empty") and not sales.empty:
        return True
    return _session_has_platform_data(sess)


def _repair_platform_loaded_flags(payload: dict) -> dict:
    """Legacy bundles: unified sales had per-platform units but ``loaded`` stayed false."""
    platforms = payload.get("platform_summary")
    if not isinstance(platforms, list):
        return payload
    for p in platforms:
        if not isinstance(p, dict) or p.get("loaded"):
            continue
        units = int(p.get("total_units") or 0)
        if units > 0 or (p.get("daily") or []) or (p.get("monthly") or []):
            p["loaded"] = True
    return payload


def _intelligence_session_build_unsafe(sess: AppSession) -> bool:
    """Unified sales builds on the request thread can OOM small VPS hosts (~8 GB RAM)."""
    try:
        cap = int((os.environ.get("INTELLIGENCE_SESSION_BUILD_MAX_ROWS") or "750000").strip())
    except ValueError:
        cap = 750_000
    sales = getattr(sess, "sales_df", None)
    if sales is None or not hasattr(sales, "__len__"):
        return False
    return len(sales) >= cap


def _try_serve_tier3_intelligence_bundle(
    sess: AppSession,
    cache_key: tuple,
    bundle_cache: dict,
    s_win: str,
    e_win: str,
    limit: int,
    basis: Optional[str],
    include_extras: bool,
    *,
    ts: float | None = None,
) -> dict | None:
    """Tier-3 direct bundle — memory-safe on production VPS."""
    tier3 = _build_intelligence_bundle_payload_from_tier3(
        sess, s_win, e_win, limit, basis, include_extras
    )
    if not tier3:
        return None
    tier3 = _repair_platform_loaded_flags(tier3)
    if not _bundle_payload_has_display_data(tier3):
        units = int((tier3.get("sales_summary") or {}).get("total_units") or 0)
        if units <= 0:
            return None
    tier3_units = int((tier3.get("sales_summary") or {}).get("total_units") or 0)
    if _tier3_only_undercounts_bulk(tier3_units, s_win, e_win):
        return None
    tier3["status"] = "ready"
    _bundle_cache_store(cache_key, bundle_cache, tier3, ts=ts if ts is not None else time.time())
    return tier3


def _bundle_payload_has_display_data(payload: dict) -> bool:
    _repair_platform_loaded_flags(payload)
    platforms = payload.get("platform_summary") or []
    has_loaded_units = any(
        p.get("loaded") and int(p.get("total_units") or 0) > 0
        for p in platforms
    )
    if int((payload.get("sales_summary") or {}).get("total_units") or 0) > 0:
        return has_loaded_units or not platforms
    return has_loaded_units


def _platform_summary_has_units(platform_summary: list) -> bool:
    return any(int(p.get("total_units") or 0) > 0 for p in (platform_summary or []))


def _session_platform_frame(sess: AppSession, attr: str) -> "pd.DataFrame":
    import pandas as pd

    raw = getattr(sess, attr, None)
    if raw is None or not hasattr(raw, "empty"):
        return pd.DataFrame()
    return raw


def _resolve_bundle_platform_frames(
    sess: AppSession,
    start_date: str,
    end_date: str,
) -> tuple:
    """Gap-filled session frames; fall back to raw session when gap-fill yields no window rows."""
    specs = (
        ("amazon", "mtr_df"),
        ("myntra", "myntra_df"),
        ("meesho", "meesho_df"),
        ("flipkart", "flipkart_df"),
        ("snapdeal", "snapdeal_df"),
    )
    out = []
    for pk, attr in specs:
        raw = _session_platform_frame(sess, attr)
        aug = _platform_df_for_intelligence_bundle(sess, pk, attr, start_date, end_date)
        in_aug = _filter_platform_df_by_window(aug, start_date, end_date)
        in_raw = _filter_platform_df_by_window(raw, start_date, end_date)
        if not in_aug.empty:
            out.append(aug if not aug.empty else raw)
        elif not in_raw.empty:
            out.append(raw)
        else:
            out.append(aug if not aug.empty else raw)
    return tuple(out)


def _ensure_return_overlay_hydrated(sess: AppSession) -> None:
    """Load saved PO return overlay from warm cache / disk for dashboard + marketplace tabs."""
    try:
        cur = getattr(sess, "po_return_overlay_df", None)
        if cur is not None and hasattr(cur, "empty") and not cur.empty:
            from ..services.po_return_import import ensure_return_overlay_meta_hydrated

            ensure_return_overlay_meta_hydrated(sess)
            return
        from ..services.po_session_hydrate import ensure_po_return_overlay_from_server

        ensure_po_return_overlay_from_server(sess)
    except Exception:
        pass


def _apply_return_overlay_to_intelligence_bundle(
    sess: AppSession,
    platform_summary: list,
    sales_for_returns,
    start_date: str,
    end_date: str,
) -> tuple[list, dict]:
    """Merge PO return-sheet overlay into platform cards and headline totals."""
    import pandas as pd

    from ..services.po_return_import import aggregate_return_overlay_for_use
    from ..services.sales import (
        RETURN_SHEET_SOURCE,
        merge_return_data_into_platform_summaries,
    )

    _ensure_return_overlay_hydrated(sess)
    s = str(start_date)[:10]
    e = str(end_date)[:10]
    sales_slice = (
        sales_for_returns
        if sales_for_returns is not None
        and hasattr(sales_for_returns, "empty")
        and not sales_for_returns.empty
        else None
    )
    platform_summary = merge_return_data_into_platform_summaries(
        platform_summary,
        aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None)),
        sales_slice,
        s,
        e,
        fallback_as_of=str(getattr(sess, "return_overlay_as_of", "") or "")[:10] or None,
    )
    shipped = sum(int(p.get("total_units") or 0) for p in platform_summary)
    returned = sum(int(p.get("total_returns") or 0) for p in platform_summary)
    net = sum(int(p.get("net_units") or 0) for p in platform_summary)
    rate = round(returned / shipped * 100, 1) if shipped > 0 else 0.0
    sales_summary = {
        "total_units": shipped,
        "total_returns": returned,
        "net_units": net,
        "return_rate": rate,
    }
    if sales_slice is not None and "Source" in sales_slice.columns:
        txn = sales_slice["Transaction Type"].astype(str).str.strip() == "Refund"
        src = sales_slice["Source"].astype(str).str.strip()
        overlay_refunds = int(
            pd.to_numeric(sales_slice.loc[txn & src.eq(RETURN_SHEET_SOURCE), "Quantity"], errors="coerce")
            .fillna(0)
            .sum()
        )
        if overlay_refunds > 0:
            sales_summary["return_sheet_units"] = overlay_refunds
            sales_summary["marketplace_return_units"] = max(0, returned - overlay_refunds)
    return platform_summary, sales_summary


def _hydrate_session_for_intelligence(sess: AppSession) -> bool:
    """
  Lightweight prep for Intelligence metrics — never copy the full warm cache here.
  ``GET /coverage?light=1`` and the background hydrate worker fill platform frames;
  blocking on a 1M+ row copy made the dashboard feel stuck at 5%.
    """
    try:
        import backend.main as _main

        _main.restore_po_sidecars_from_warm(sess)
        _ensure_return_overlay_hydrated(sess)
    except Exception:
        pass
    _ensure_sku_mapping_for_dashboard(sess)
    return _session_has_operational_frames(sess)


def _session_sales_reporting_range(sess: AppSession) -> tuple[str, str] | None:
    """IST calendar min/max from unified ``sales_df`` (cheap; for empty-window hints)."""
    import pandas as pd

    from ..services.sales import txn_reporting_naive_ist

    sales = getattr(sess, "sales_df", None)
    if sales is None or not hasattr(sales, "empty") or sales.empty:
        return None
    if "TxnDate" not in sales.columns:
        return None
    t = txn_reporting_naive_ist(sales["TxnDate"]).dropna()
    if t.empty:
        return None
    return str(t.min().normalize())[:10], str(t.max().normalize())[:10]


def _intelligence_empty_window_payload(
    sess: AppSession,
    start_date: str,
    end_date: str,
) -> dict:
    """Session has sales but none in the requested window — return ready (not warming)."""
    bounds = _session_sales_reporting_range(sess)
    msg = f"No marketplace units for {start_date} → {end_date}."
    if bounds:
        msg += f" Loaded sales span {bounds[0]} → {bounds[1]} — adjust the date range above."
    payload = {
        "status": "ready",
        "message": msg,
        "empty_window": True,
        "sales_summary": {
            "total_units": 0,
            "total_returns": 0,
            "net_units": 0,
            "return_rate": 0.0,
        },
        "platform_summary": [],
        "top_skus": [],
        "anomalies": [],
        "dsr_brand_monthly": {"rows": [], "totals": {}, "note": ""},
    }
    if bounds:
        payload["session_data_range"] = {"min": bounds[0], "max": bounds[1]}
    return payload


def _session_has_units_in_window(sess: AppSession, start_date: str, end_date: str) -> bool:
    """True when unified sales or any platform frame has shipment rows in the window."""
    import pandas as pd

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    if len(s) != 10 or len(e) != 10:
        return False
    raw_sales = getattr(sess, "sales_df", None)
    if raw_sales is None or not hasattr(raw_sales, "empty") or raw_sales.empty:
        sales = pd.DataFrame()
    else:
        sales = _slice_sales_for_bundle(raw_sales, s, e)
    if not sales.empty and "Transaction Type" in sales.columns:
        txn = sales["Transaction Type"].astype(str).str.strip() == "Shipment"
        if int(pd.to_numeric(sales.loc[txn, "Quantity"], errors="coerce").fillna(0).sum()) > 0:
            return True
    for attr, txn_col in (
        ("mtr_df", "Transaction_Type"),
        ("myntra_df", "TxnType"),
        ("meesho_df", "TxnType"),
        ("flipkart_df", "TxnType"),
        ("snapdeal_df", "TxnType"),
    ):
        raw = getattr(sess, attr, None)
        if raw is None or not hasattr(raw, "empty") or raw.empty:
            continue
        w = _filter_platform_df_by_window(raw, s, e)
        if w.empty or txn_col not in w.columns:
            continue
        ship = w[txn_col].astype(str).str.strip() == "Shipment"
        if int(pd.to_numeric(w.loc[ship, "Quantity"], errors="coerce").fillna(0).sum()) > 0:
            return True
    return False


def _build_platform_summary_for_bundle(
    sess: AppSession,
    mtr_b,
    myntra_b,
    meesho_b,
    flipkart_b,
    snapdeal_b,
    start_date: str,
    end_date: str,
) -> list:
    """Platform cards from session/Tier-3 frames, with unified-sales fallback."""
    from ..services.sales import apply_upload_report_day_gate, get_platform_summary

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    kwargs = dict(start_date=s, end_date=e)
    platform_summary = get_platform_summary(
        mtr_b,
        myntra_b,
        meesho_b,
        flipkart_b,
        snapdeal_b,
        sales_df=None,
        **kwargs,
    )
    if _platform_summary_has_units(platform_summary):
        return platform_summary

    direct = (
        _session_platform_frame(sess, "mtr_df"),
        _session_platform_frame(sess, "myntra_df"),
        _session_platform_frame(sess, "meesho_df"),
        _session_platform_frame(sess, "flipkart_df"),
        _session_platform_frame(sess, "snapdeal_df"),
    )
    platform_summary = get_platform_summary(*direct, sales_df=None, **kwargs)
    if _platform_summary_has_units(platform_summary):
        return platform_summary

    sales = apply_upload_report_day_gate(sess.sales_df)
    if sales is not None and not sales.empty:
        platform_summary = get_platform_summary(*direct, sales_df=sales, **kwargs)
        if _platform_summary_has_units(platform_summary):
            return platform_summary

    t3_tuple = _tier3_direct_has_units(s, e, sess, 10, "gross")
    if t3_tuple is not None:
        return list(t3_tuple[1])
    return platform_summary


def _schedule_persist_tier3_window(
    session_id: str | None,
    start_date: str,
    end_date: str,
    platforms: list[str],
) -> None:
    if not session_id or not platforms:
        return

    def _worker() -> None:
        from ..session import store

        sess = store.get(session_id)
        if sess is None:
            return
        try:
            if sess._daily_restore_lock.acquire(timeout=30.0):
                try:
                    if _merge_tier3_for_report_range(sess, platforms, start_date, end_date):
                        _rebuild_session_sales(sess)
                        _mark_tier3_sync_applied(sess)
                finally:
                    sess._daily_restore_lock.release()
        except Exception:
            pass

    try:
        from ..concurrency import DAILY_UPLOAD_EXECUTOR

        DAILY_UPLOAD_EXECUTOR.submit(_worker)
    except Exception:
        pass


def _auto_dashboard_bundle_data(
    sess: AppSession,
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple:
    """
    Build dashboard sales slice + platform frames, auto-pulling Tier-3 when uploads
    exist but Intelligence would otherwise show zero.
    Returns (sales_for_bundle, mtr_df, myntra_df, meesho_df, flipkart_df, snapdeal_df, pulled_platforms).
    """
    import pandas as pd
    from ..services.daily_store import platforms_with_uploads_in_range
    from ..services.sales import patch_sales_df_after_daily_upload

    s = str(start_date or end_date or "")[:10]
    e = str(end_date or start_date or "")[:10]
    if len(s) != 10 or len(e) != 10:
        raw = getattr(sess, "sales_df", None) or pd.DataFrame()
        return (
            _slice_sales_for_bundle(raw, start_date, end_date),
            sess.mtr_df,
            sess.myntra_df,
            sess.meesho_df,
            sess.flipkart_df,
            sess.snapdeal_df,
            [],
        )

    _ensure_sku_mapping_for_dashboard(sess)
    uploaded = platforms_with_uploads_in_range(s, e)
    raw = getattr(sess, "sales_df", None)
    if raw is None or not hasattr(raw, "empty"):
        raw = pd.DataFrame()

    sales_slice = _slice_sales_for_bundle(raw, start_date, end_date)
    need = _platforms_needing_auto_tier3_pull(sess, s, e, sales_slice)

    tier3_frames = _load_tier3_frames_for_platforms(need, s, e) if need else {}
    working = raw

    if tier3_frames and (sess.sku_mapping or _ensure_sku_mapping_for_dashboard(sess)):
        fresh = _build_sales_from_tier3_frames(sess, tier3_frames)
        if not fresh.empty:
            if raw.empty:
                working = fresh
            else:
                working = patch_sales_df_after_daily_upload(
                    raw,
                    fresh,
                    list(tier3_frames.keys()),
                    pd.Timestamp(s),
                    pd.Timestamp(e),
                )

    # Uploaded data but still no rows in window — build only from Tier-3 for this window.
    if uploaded and _slice_sales_for_bundle(working, start_date, end_date).empty:
        all_frames = _load_tier3_frames_for_platforms(uploaded, s, e)
        if all_frames and (sess.sku_mapping or _ensure_sku_mapping_for_dashboard(sess)):
            built = _build_sales_from_tier3_frames(sess, all_frames)
            if not built.empty:
                working = built
                tier3_frames = all_frames
                need = uploaded

    sales_for_bundle = _slice_sales_for_bundle(working, start_date, end_date)
    mtr, myntra, meesho, flipkart, snapdeal = _bundle_platform_frames(sess, tier3_frames)
    pulled = list(tier3_frames.keys())
    return sales_for_bundle, mtr, myntra, meesho, flipkart, snapdeal, pulled


def _intelligence_payload_from_tier3_direct(
    sess: AppSession,
    start_date: str,
    end_date: str,
    limit: int,
    basis: Optional[str],
    *,
    platforms: Optional[list[str]] = None,
) -> tuple:
    """
    Build dashboard metrics straight from Tier-3 SQLite (Upload tab source of truth).
    Used when session unified sales exist but show zero for a window that has uploads.
    """
    import pandas as pd
    from ..services.daily_store import platforms_with_uploads_in_range
    from ..services.sales import (
        _compute_platform_metrics,
        _unified_platform_stub,
        get_top_skus,
    )

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    uploaded = set(platforms or platforms_with_uploads_in_range(s, e))
    all_frames = _load_tier3_frames_for_platforms(
        sorted(uploaded), s, e, dedup=False, columns_only=True
    )
    if not all_frames:
        all_frames = _load_tier3_frames_for_platforms(
            sorted(uploaded), s, e, dedup=False, columns_only=False
        )

    metrics_specs = (
        ("amazon", "Amazon", "Date", "SKU", "Transaction_Type"),
        ("myntra", "Myntra", "Date", "OMS_SKU", "TxnType"),
        ("meesho", "Meesho", "Date", "OMS_SKU", "TxnType"),
        ("flipkart", "Flipkart", "Date", "OMS_SKU", "TxnType"),
        ("snapdeal", "Snapdeal", "Date", "OMS_SKU", "TxnType"),
    )
    platform_summary: list[dict] = []
    for plat, name, _dc, sku_col, txn_col in metrics_specs:
        if plat not in uploaded:
            platform_summary.append(_unified_platform_stub(name, False))
            continue
        df = all_frames.get(plat, pd.DataFrame())
        if df.empty:
            platform_summary.append(_unified_platform_stub(name, False))
            continue
        try:
            platform_summary.append(
                _compute_platform_metrics(
                    df,
                    name,
                    sku_col,
                    txn_col,
                    start_date=s,
                    end_date=e,
                )
            )
        except Exception:
            platform_summary.append(_unified_platform_stub(name, True))

    shipped = sum(int(p.get("total_units") or 0) for p in platform_summary)
    returned = sum(int(p.get("total_returns") or 0) for p in platform_summary)
    net = sum(int(p.get("net_units") or 0) for p in platform_summary)
    rate = round(returned / shipped * 100, 1) if shipped > 0 else 0.0
    sales_summary = {
        "total_units": shipped,
        "total_returns": returned,
        "net_units": net,
        "return_rate": rate,
        "date_basis_note": (
            "Totals from saved Tier-3 daily uploads for this date range "
            "(auto-synced with the Upload tab)."
        ),
    }

    sales_for_bundle = pd.DataFrame()
    _ensure_sku_mapping_for_dashboard(sess)
    span = _report_span_days(s, e) or 999
    if span <= _intelligence_fast_window_days() and all_frames and sess.sku_mapping:
        try:
            built = _build_sales_from_tier3_frames(sess, all_frames)
            sales_for_bundle = _slice_sales_for_bundle(built, s, e)
        except Exception:
            _log.exception("tier3 direct sales build failed for %s..%s", s, e)
            sales_for_bundle = pd.DataFrame()

    top_skus = (
        get_top_skus(
            sales_for_bundle,
            limit=limit,
            start_date=None,
            end_date=None,
            basis=basis or "gross",
        )
        if not sales_for_bundle.empty
        else []
    )

    mtr, myntra, meesho, flipkart, snapdeal = _bundle_platform_frames(sess, all_frames)
    return (
        sales_summary,
        platform_summary,
        top_skus,
        sales_for_bundle,
        mtr,
        myntra,
        meesho,
        flipkart,
        snapdeal,
        sorted(uploaded),
    )


def _bundle_payload_chart_sparse(
    payload: dict,
    start_date: Optional[str],
    end_date: Optional[str],
) -> bool:
    """True when a short-range chart has headline units but almost no daily points."""
    from ..services.sales import intelligence_daily_chart_enabled

    if not intelligence_daily_chart_enabled(start_date, end_date):
        return False
    span = _report_span_days(start_date, end_date) or 0
    if span < 7:
        return False
    min_days = max(7, span // 3)
    for p in payload.get("platform_summary") or []:
        if not p.get("loaded"):
            continue
        units = int(p.get("total_units") or 0)
        if units < 200:
            continue
        if len(p.get("daily") or []) < min_days:
            return True
    return False


def _cached_bundle_stale_vs_tier3_uploads(
    payload: dict,
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    sess: AppSession | None = None,
) -> bool:
    """True when a cached bundle should not be served — Tier-3 daily uploads are newer."""
    from ..services.daily_store import platforms_with_uploads_in_range

    s = str(start_date or end_date or "")[:10]
    e = str(end_date or start_date or "")[:10]
    if len(s) != 10 or len(e) != 10:
        return False
    if not platforms_with_uploads_in_range(s, e):
        return False
    applied = getattr(sess, "_tier3_sync_token_applied", None) if sess else None
    if not applied:
        # Global / precomputed disk cache — key already embeds get_tier3_sync_token().
        return False
    if payload.get("tier3_auto_pull"):
        return bool(sess and _tier3_token_mismatch(sess))
    return bool(sess and _tier3_token_mismatch(sess))


def _tier3_direct_has_units(
    start_date: str,
    end_date: str,
    sess: AppSession,
    limit: int,
    basis: Optional[str],
) -> tuple | None:
    """
    If Tier-3 has shipment units for the window, return the direct payload tuple; else None.
    """
    from ..services.daily_store import platforms_with_uploads_in_range

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    uploaded = platforms_with_uploads_in_range(s, e)
    if not uploaded:
        # Metadata overlap query can lag; still attempt all channels for the window.
        uploaded = ["amazon", "myntra", "meesho", "flipkart", "snapdeal"]
    out = _intelligence_payload_from_tier3_direct(sess, s, e, limit, basis, platforms=uploaded)
    if int(out[0].get("total_units") or 0) <= 0:
        return None
    return out


def _bundle_cache_global_key(
    cache_key: tuple,
) -> tuple:
    from ..services.daily_store import get_tier3_sync_token

    return (*cache_key, _INTELLIGENCE_BUNDLE_CACHE_GEN, tuple(sorted((get_tier3_sync_token() or {}).items())))


def _bundle_cache_lookup(
    cache_key: tuple,
    sess_cache: dict | None,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    allow_sparse: bool = False,
    sess: AppSession | None = None,
) -> dict | None:
    """Return a fresh cached payload if we have display data for this window."""
    now = time.time()
    candidates: list = []
    if sess_cache:
        hit = sess_cache.get(cache_key)
        if hit:
            candidates.append(hit)
    ghit = _GLOBAL_INTELLIGENCE_BUNDLE_CACHE.get(_bundle_cache_global_key(cache_key))
    if ghit:
        candidates.append(ghit)
    for hit in candidates:
        age = now - float(hit.get("_ts", 0))
        payload = hit.get("payload") or {}
        if age >= _INTELLIGENCE_BUNDLE_TTL_SEC or not _bundle_payload_has_display_data(payload):
            continue
        if _cached_bundle_stale_vs_tier3_uploads(
            payload, start_date, end_date, sess=sess
        ):
            continue
        if sess is not None and payload.get("tier3_auto_pull"):
            s = str(start_date or end_date or "")[:10]
            e = str(end_date or start_date or "")[:10]
            if len(s) == 10 and len(e) == 10:
                tier3_units = int((payload.get("sales_summary") or {}).get("total_units") or 0)
                if _tier3_only_undercounts_bulk(tier3_units, s, e):
                    continue
        if not allow_sparse and _bundle_payload_chart_sparse(payload, start_date, end_date):
            continue
        try:
            from ..services.perf_metrics import record_cache

            record_cache(hit=True, source="intelligence_bundle", name="bundle_lookup")
        except Exception:
            pass
        return _repair_platform_loaded_flags(payload)
    return None


def _bundle_cache_store(
    cache_key: tuple,
    sess_cache: dict,
    payload: dict,
    *,
    ts: float | None = None,
) -> None:
    payload = _repair_platform_loaded_flags(payload)
    entry = {"_ts": ts if ts is not None else time.time(), "payload": payload}
    sess_cache[cache_key] = entry
    global_key = _bundle_cache_global_key(cache_key)
    _GLOBAL_INTELLIGENCE_BUNDLE_CACHE[global_key] = entry
    _save_intelligence_bundle_to_disk(global_key, entry)


def _intelligence_warming_payload(message: str = "Loading marketplace data on the server…") -> dict:
    return {
        "status": "warming",
        "message": message,
        "sales_summary": {
            "total_units": 0,
            "total_returns": 0,
            "net_units": 0,
            "return_rate": 0.0,
        },
        "platform_summary": [],
        "top_skus": [],
        "anomalies": [],
        "dsr_brand_monthly": {"rows": [], "totals": {}, "note": ""},
    }


def _build_intelligence_bundle_payload_from_session(
    sess: AppSession,
    start_date: str,
    end_date: str,
    limit: int,
    basis: Optional[str],
    include_extras: bool,
) -> dict | None:
    """
    Fast session frames first; Tier-3 direct metrics only when the window is empty.
    Avoids ``_auto_dashboard_bundle_data`` (full parquet + sales rebuild) on every GET.
    """
    import pandas as pd
    from ..services.sales import (
        apply_upload_report_day_gate,
        get_anomalies,
        get_dsr_brand_monthly_comparison,
        get_platform_summary,
        get_top_skus,
        merge_return_data_into_platform_summaries,
    )

    _hydrate_session_for_intelligence(sess)

    s = str(start_date)[:10]
    e = str(end_date)[:10]

    mtr_b, myntra_b, meesho_b, flipkart_b, snapdeal_b = _resolve_bundle_platform_frames(
        sess, s, e
    )
    gated_sales = (
        apply_upload_report_day_gate(sess.sales_df)
        if (
            not _intelligence_session_build_unsafe(sess)
            and getattr(sess, "sales_df", None) is not None
            and hasattr(sess.sales_df, "empty")
            and not sess.sales_df.empty
        )
        else pd.DataFrame()
    )
    win_gated = _slice_sales_for_bundle(gated_sales, s, e)
    sales_slice = win_gated

    if not win_gated.empty:
        platform_summary = get_platform_summary(
            mtr_b,
            myntra_b,
            meesho_b,
            flipkart_b,
            snapdeal_b,
            start_date=s,
            end_date=e,
            sales_df=gated_sales,
        )
    else:
        platform_summary = get_platform_summary(
            mtr_b,
            myntra_b,
            meesho_b,
            flipkart_b,
            snapdeal_b,
            start_date=s,
            end_date=e,
            sales_df=None,
        )

    if not _platform_summary_has_units(platform_summary):
        platform_summary = _build_platform_summary_for_bundle(
            sess, mtr_b, myntra_b, meesho_b, flipkart_b, snapdeal_b, s, e
        )

    if not _platform_summary_has_units(platform_summary):
        t3_tuple = _tier3_direct_has_units(s, e, sess, limit, basis)
        if t3_tuple is not None:
            platform_summary = list(t3_tuple[1])
            if not t3_tuple[2].empty:
                sales_slice = t3_tuple[2]
    sales_for_returns = (
        win_gated
        if win_gated is not None and not win_gated.empty
        else sales_slice
    )
    platform_summary, sales_summary = _apply_return_overlay_to_intelligence_bundle(
        sess,
        platform_summary,
        sales_for_returns,
        s,
        e,
    )
    top_skus = (
        get_top_skus(
            sales_slice,
            limit=limit,
            start_date=None,
            end_date=None,
            basis=basis or "gross",
        )
        if not sales_slice.empty
        else []
    )
    sales_for_extras = sales_slice if not sales_slice.empty else pd.DataFrame()

    span_days = _report_span_days(s, e)
    fast_window = span_days is not None and span_days <= _intelligence_fast_window_days()
    payload: dict = {
        "sales_summary": sales_summary,
        "platform_summary": platform_summary,
        "top_skus": top_skus,
        "status": "ready",
        "session_fast_path": True,
    }
    if include_extras:
        _empty_plat = pd.DataFrame()
        if fast_window and not sales_for_extras.empty:
            payload["anomalies"] = get_anomalies(
                _empty_plat,
                _empty_plat,
                _empty_plat,
                _empty_plat,
                _empty_plat,
                sess.inventory_df_variant,
                sales_for_extras,
                start_date=None,
                end_date=None,
            )
        else:
            payload["anomalies"] = []
        payload["dsr_brand_monthly"] = (
            get_dsr_brand_monthly_comparison(
                sales_for_extras, start_date=None, end_date=None
            )
            if not sales_for_extras.empty
            else {"rows": [], "totals": {}, "note": ""}
        )
    else:
        payload["anomalies"] = []
        payload["dsr_brand_monthly"] = {"rows": [], "totals": {}, "note": ""}
    if not _bundle_payload_has_display_data(payload):
        return None
    return payload


def _build_intelligence_bundle_payload_from_tier3(
    sess: AppSession,
    start_date: str,
    end_date: str,
    limit: int,
    basis: Optional[str],
    include_extras: bool,
) -> dict | None:
    """Fast path: full Intelligence bundle JSON from Tier-3 only (no session sales scan)."""
    import pandas as pd

    t3 = _tier3_direct_has_units(start_date, end_date, sess, limit, basis)
    if t3 is None:
        return None
    (
        sales_summary,
        platform_summary,
        top_skus,
        sales_for_bundle,
        _mtr,
        _myntra,
        _meesho,
        _flipkart,
        _snapdeal,
        pulled_platforms,
    ) = t3
    platform_summary, sales_summary = _apply_return_overlay_to_intelligence_bundle(
        sess,
        list(platform_summary),
        None,
        start_date,
        end_date,
    )
    _empty_plat = pd.DataFrame()
    payload: dict = {
        "sales_summary": sales_summary,
        "platform_summary": platform_summary,
        "top_skus": top_skus,
        "status": "ready",
        "tier3_auto_pull": True,
    }
    if include_extras:
        payload["anomalies"] = get_anomalies(
            _empty_plat,
            _empty_plat,
            _empty_plat,
            _empty_plat,
            _empty_plat,
            sess.inventory_df_variant,
            sales_for_bundle,
            start_date=None,
            end_date=None,
        )
        payload["dsr_brand_monthly"] = get_dsr_brand_monthly_comparison(
            sales_for_bundle, start_date=None, end_date=None
        )
    else:
        payload["anomalies"] = []
        payload["dsr_brand_monthly"] = {"rows": [], "totals": {}, "note": ""}
    return payload


def _tier3_session_needs_topup(sess: AppSession) -> bool:
    """True when SQLite has newer/more daily uploads than session or unified sales."""
    if _tier3_token_mismatch(sess):
        return True
    try:
        from ..services.daily_store import get_summary

        summary = get_summary()
    except Exception:
        return False
    sales = getattr(sess, "sales_df", None)
    for plat, attr in _PLATFORM_ATTRS:
        plat_sum = summary.get(plat) or {}
        if int(plat_sum.get("file_count") or 0) <= 0:
            continue
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            return True
        tier_max = str(plat_sum.get("max_date") or "")[:10]
        tier_min = str(plat_sum.get("min_date") or "")[:10]
        sess_max = _platform_df_max_iso(cur) or ""
        sess_min = _platform_df_min_iso(cur) or ""
        src = _SOURCE_BY_ATTR.get(attr, "")
        sales_max = _sales_max_for_source(sales, src) or ""
        if tier_max and (not sess_max or tier_max > sess_max):
            return True
        if tier_min and (not sess_min or tier_min < sess_min):
            return True
        if tier_max and (not sales_max or tier_max > sales_max):
            return True
        tier_rows = int(plat_sum.get("total_rows") or 0)
        sess_rows = len(cur) if cur is not None and hasattr(cur, "__len__") else 0
        # total_rows is pre-dedup sum of uploads; session should reach ~70%+ after full merge.
        if tier_rows > 500 and sess_rows < int(tier_rows * 0.70):
            return True
    return False


def _session_sales_stale_vs_platforms(sess: AppSession) -> bool:
    """True when unified sales_df is missing or lags loaded in-memory platform frames."""
    sales = getattr(sess, "sales_df", None)
    sources: set[str] = set()
    if sales is not None and not sales.empty and "Source" in sales.columns:
        sources = set(sales["Source"].astype(str).str.strip())
    for attr, src in _SOURCE_BY_ATTR.items():
        raw = getattr(sess, attr, None)
        if raw is not None and hasattr(raw, "empty") and not raw.empty and src not in sources:
            return True
        if raw is not None and hasattr(raw, "empty") and not raw.empty:
            plat_max = _platform_df_max_iso(raw) or ""
            sales_max = _sales_max_for_source(sales, src) or ""
            if plat_max and (not sales_max or sales_max < plat_max):
                return True
    return sales is None or (hasattr(sales, "empty") and sales.empty and _session_has_platform_data(sess))


def _sales_overlay_build_kwargs(sess: AppSession) -> dict:
    from ..services.po_return_import import aggregate_return_overlay_for_use

    kw: dict = {}
    ov = aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None))
    if ov is not None and not getattr(ov, "empty", True):
        kw["return_overlay_df"] = ov
    as_of = getattr(sess, "return_overlay_as_of", None)
    if as_of and str(as_of).strip():
        kw["return_overlay_as_of"] = str(as_of).strip()[:10]
    return kw


def _invalidate_shared_quarterly_cache() -> None:
    """Drop the server-wide PO quarterly cache (memory + disk) after sales/
    platform data actually changes. The next ``/po/quarterly`` request already
    rebuilds on a cache miss (``start_quarterly_background``), so no separate
    rewarm thread is needed here."""
    try:
        from ..services.po_quarterly_cache import invalidate_shared_quarterly

        invalidate_shared_quarterly()
    except Exception:
        pass


def _rebuild_session_sales(sess: AppSession) -> None:
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        return
    if not sess.sku_mapping:
        return
    if not _session_has_platform_data(sess):
        return
    try:
        from ..services.sales import build_sales_df

        sess.sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
            **_sales_overlay_build_kwargs(sess),
        )
        sess._quarterly_cache.clear()
        sess._intelligence_bundle_cache.clear()
        _invalidate_shared_quarterly_cache()
        _invalidate_intelligence_bundle_cache()
        _refresh_sales_days_by_source(sess)
        _mark_tier3_sync_applied(sess)
    except Exception:
        pass


def _intelligence_tier3_months() -> int:
    try:
        return max(1, int((os.environ.get("INTELLIGENCE_TIER3_MONTHS") or "6").strip()))
    except ValueError:
        return 6


def _intelligence_tier3_max_files() -> int:
    try:
        v = int((os.environ.get("INTELLIGENCE_TIER3_MAX_FILES") or "60").strip())
        return max(5, v)
    except ValueError:
        return 60


def _merge_tier3_light(sess: AppSession, *, only_platforms: list[str] | None = None) -> bool:
    """
    Bounded Tier-3 merge for dashboard APIs — recent uploads only, no full-history scan.
    """
    from ..services.daily_store import load_platform_data, merge_platform_data

    months = _intelligence_tier3_months()
    max_files = _intelligence_tier3_max_files()
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
        cur = getattr(sess, attr)
        merged = merge_platform_data(cur, df, platform)
        if len(merged) != len(cur) or (cur.empty and not merged.empty):
            setattr(sess, attr, merged)
            changed = True
    if changed:
        _mark_tier3_sync_applied(sess)
    return changed


def _merge_tier3_for_report_range(
    sess: AppSession,
    platforms: list[str],
    start_date: str,
    end_date: str,
) -> bool:
    from ..services.daily_store import (
        load_platform_data_for_report_range,
        merge_platform_data,
    )

    changed = False
    for platform, attr in _PLATFORM_ATTRS:
        if platform not in platforms:
            continue
        df = load_platform_data_for_report_range(platform, start_date, end_date, dedup=False)
        if df.empty:
            continue
        cur = getattr(sess, attr)
        merged = merge_platform_data(cur, df, platform)
        if len(merged) != len(cur) or (cur.empty and not merged.empty):
            setattr(sess, attr, merged)
            changed = True
    if changed:
        _mark_tier3_sync_applied(sess)
    return changed


def _platforms_needing_tier3_topup(sess: AppSession) -> list[str]:
    token_gap = _platforms_with_tier3_token_mismatch(sess)
    if token_gap:
        return token_gap
    out: list[str] = []
    try:
        from ..services.daily_store import get_summary

        summary = get_summary()
    except Exception:
        return out
    sales = getattr(sess, "sales_df", None)
    for plat, attr in _PLATFORM_ATTRS:
        plat_sum = summary.get(plat) or {}
        if int(plat_sum.get("file_count") or 0) <= 0:
            continue
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            out.append(plat)
            continue
        tier_max = str(plat_sum.get("max_date") or "")[:10]
        sess_max = _platform_df_max_iso(cur) or ""
        src = _SOURCE_BY_ATTR.get(attr, "")
        sales_max = _sales_max_for_source(sales, src) or ""
        if tier_max and (not sess_max or tier_max > sess_max):
            out.append(plat)
            continue
        if tier_max and (not sales_max or tier_max > sales_max):
            out.append(plat)
    return out


def _schedule_intelligence_refresh_async(session_id: str | None) -> None:
    """Tier-3 top-up / sales rebuild without blocking dashboard GET handlers."""
    if not session_id:
        return
    try:
        from ..concurrency import DAILY_UPLOAD_EXECUTOR

        DAILY_UPLOAD_EXECUTOR.submit(_intelligence_refresh_worker, session_id)
    except Exception:
        pass


def _intelligence_refresh_worker(session_id: str) -> None:
    from ..session import store

    sess = store.get(session_id)
    if sess is None:
        return
    try:
        _ensure_intelligence_session_fresh(sess)
    except Exception:
        pass


def _ensure_intelligence_session_fresh(sess: AppSession) -> None:
    """
    Fast path for Intelligence GET handlers: rebuild sales from session frames,
    optionally merge **recent** Tier-3 rows (non-blocking lock). Never run a full
    multi-minute ``_restore_daily_if_needed`` on the request thread.
    """
    if getattr(sess, "pause_auto_data_restore", False):
        return
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return
    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        return
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        return
    # Dashboard fires several queries in parallel (sales-summary/platform/top-skus/anomalies).
    # Keep only one expensive freshness pass every few seconds per session.
    if getattr(sess, "_intelligence_refresh_running", False):
        return
    now = time.time()
    last = float(getattr(sess, "_intelligence_refresh_ts", 0.0) or 0.0)
    if now - last < 5.0:
        return
    sess._intelligence_refresh_running = True
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
    except Exception:
        pass
    try:
        if _session_sales_stale_vs_platforms(sess):
            _rebuild_session_sales(sess)

        need_plat = _platforms_needing_tier3_topup(sess)
        if need_plat and sess._daily_restore_lock.acquire(blocking=False):
            try:
                if _merge_tier3_light(sess, only_platforms=need_plat):
                    _rebuild_session_sales(sess)
            finally:
                sess._daily_restore_lock.release()
        elif _session_sales_stale_vs_platforms(sess):
            _rebuild_session_sales(sess)
        else:
            _ensure_sales_rebuilt(sess)
    finally:
        sess._intelligence_refresh_ts = time.time()
        sess._intelligence_refresh_running = False


def _ensure_warm_session_data(sess: AppSession) -> None:
    """Warm cache fill for empty sessions; Intelligence routes use ``_ensure_intelligence_session_fresh``."""
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
    except Exception:
        pass


def _restore_inventory_from_warm(sess: AppSession) -> None:
    """Sync snapshot inventory with warm cache (newest ``uploaded_at`` wins)."""
    try:
        import backend.main as _main

        sync_inventory_snapshot_from_warm(sess)
        if not sess.sku_mapping and _main._warm_cache and _main._warm_cache.get("sku_mapping"):
            sess.sku_mapping = _main._warm_cache["sku_mapping"]
        ensure_inventory_snapshot_metadata(sess)
    except Exception:
        pass


def _session_window_gross_units(sess: AppSession, start_date: str, end_date: str) -> int:
    """Fast gross shipment units from unified session/warm sales for a reporting window."""
    import pandas as pd

    from ..services.sales import apply_upload_report_day_gate, get_sales_summary
    from ..services.shared_frames import session_sales_df, warm_frame

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    if len(s) != 10 or len(e) != 10:
        return 0
    sales = session_sales_df(sess)
    if sales is None or sales.empty:
        sales = warm_frame("sales_df", sess)
    if sales is None or sales.empty:
        return 0
    gated = apply_upload_report_day_gate(sales)
    clipped = _slice_sales_for_bundle(gated, s, e)
    if clipped.empty:
        return 0
    try:
        summary = get_sales_summary(clipped, start_date=s, end_date=e)
        return int(summary.get("total_units") or 0)
    except Exception:
        txn_col = "Transaction Type" if "Transaction Type" in clipped.columns else "Transaction_Type"
        if txn_col not in clipped.columns:
            return 0
        ship = clipped[clipped[txn_col].astype(str).str.strip().str.lower() == "shipment"]
        return int(pd.to_numeric(ship.get("Quantity", ship.get("Units_Effective", 0)), errors="coerce").fillna(0).sum())


_PLATFORM_ROW_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mtr_df", "mtr_rows", "amazon"),
    ("myntra_df", "myntra_rows", "myntra"),
    ("meesho_df", "meesho_rows", "meesho"),
    ("flipkart_df", "flipkart_rows", "flipkart"),
    ("snapdeal_df", "snapdeal_rows", "snapdeal"),
)

_disk_row_count_cache: dict[str, int] = {}


def _disk_parquet_row_count(frame_key: str) -> int:
    if frame_key in _disk_row_count_cache:
        return _disk_row_count_cache[frame_key]
    try:
        import os

        import pyarrow.parquet as pq

        import backend.main as _main

        path = os.path.join(_main._DISK_CACHE_DIR, f"{frame_key}.parquet")
        if os.path.isfile(path):
            n = int(pq.read_metadata(path).num_rows)
            _disk_row_count_cache[frame_key] = n
            return n
    except Exception:
        pass
    return 0


def _tier3_platform_row_count(platform: str) -> int:
    try:
        from ..services.daily_store import get_summary

        summary = get_summary() or {}
        plat = summary.get(platform) or {}
        return int(plat.get("total_rows") or 0)
    except Exception:
        return 0


def _coverage_platform_row_count(sess: AppSession, frame_key: str, platform: str) -> int:
    """Row count for Upload/coverage UI — session frame, warm RAM, disk, or Tier-3 summary."""
    from ..services.shared_frames import frame_row_count

    rows = int(frame_row_count(frame_key, sess))
    disk_rows = _disk_parquet_row_count(frame_key)
    tier3_rows = _tier3_platform_row_count(platform)
    return max(rows, disk_rows, tier3_rows)


def _disk_bulk_history_available() -> bool:
    """True when on-disk warm-cache parquets hold bulk Tier-1 platform history."""
    return any(
        _disk_parquet_row_count(k) >= 50_000
        for k in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df")
    )


def _tier3_only_undercounts_bulk(
    tier3_units: int,
    start_date: str,
    end_date: str,
) -> bool:
    """Fast check: Tier-3 dailies alone vs bulk parquets on disk for multi-week windows."""
    if tier3_units <= 0:
        return False
    span = _report_span_days(start_date, end_date) or 0
    if span < 7 or not _disk_bulk_history_available():
        return False
    # When bulk parquets exist, 30D Tier-3-only totals below ~35K are almost always incomplete.
    floor = min(35_000, max(7_000, span * 900))
    return tier3_units < floor


def _gapfill_window_gross_units(sess: AppSession, start_date: str, end_date: str) -> int:
    """Gross shipments from gap-filled platform frames (bulk disk + Tier-3 dailies)."""
    from ..services.sales import _compute_platform_metrics, _unified_platform_stub

    s = str(start_date)[:10]
    e = str(end_date)[:10]
    if len(s) != 10 or len(e) != 10:
        return 0
    specs = (
        ("amazon", "mtr_df", "Amazon", "SKU", "Transaction_Type"),
        ("myntra", "myntra_df", "Myntra", "OMS_SKU", "TxnType"),
        ("meesho", "meesho_df", "Meesho", "OMS_SKU", "TxnType"),
        ("flipkart", "flipkart_df", "Flipkart", "OMS_SKU", "TxnType"),
        ("snapdeal", "snapdeal_df", "Snapdeal", "OMS_SKU", "TxnType"),
    )
    total = 0
    for pk, attr, name, sku_col, txn_col in specs:
        try:
            df = _platform_df_for_intelligence_bundle(sess, pk, attr, s, e)
            win = _filter_platform_df_by_window(df, s, e)
            if win.empty:
                continue
            metrics = _compute_platform_metrics(
                win, name, sku_col, txn_col, start_date=s, end_date=e
            )
            total += int(metrics.get("total_units") or 0)
        except Exception:
            continue
    return total


def _best_non_tier3_window_units(sess: AppSession, start_date: str, end_date: str) -> int:
    return max(
        _session_window_gross_units(sess, start_date, end_date),
        _gapfill_window_gross_units(sess, start_date, end_date),
    )


def _session_has_platform_data(sess: AppSession) -> bool:
    from ..services.shared_frames import frame_row_count

    if any(
        _coverage_platform_row_count(sess, fk, plat) > 0
        for fk, _, plat in _PLATFORM_ROW_SPECS
    ):
        return True
    return any(
        frame_row_count(attr, sess) > 0
        for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df")
    )


def _coverage_sales_ready(sess: AppSession) -> bool:
    """
    Unified sales_df exists, or bulk platform history is loaded (derived sales rebuilds in background).
    Prevents the app staying at 7/8 after warm-cache hydrate copies platform frames only.
    """
    from ..services.shared_frames import session_sales_df

    sales = session_sales_df(sess)
    if sales is not None and hasattr(sales, "empty") and not sales.empty:
        return True
    return bool(sess.sku_mapping) and _session_has_platform_data(sess)


def _ensure_sales_rebuilt(sess: AppSession) -> None:
    """Rebuild unified sales when platform history is loaded but sales_df is still empty."""
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        return
    if not sess.sku_mapping or not sess.sales_df.empty:
        return
    if not _session_has_platform_data(sess):
        return
    try:
        from ..services.sales import build_sales_df

        sess.sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
            **_sales_overlay_build_kwargs(sess),
        )
        sess._quarterly_cache.clear()
        _invalidate_shared_quarterly_cache()
        _invalidate_intelligence_bundle_cache()
    except Exception:
        pass


def _maybe_restore_daily_for_empty_sales(sess: AppSession) -> None:
    """When sales is missing but history exists, attempt Tier-3 merge (non-blocking lock)."""
    if getattr(sess, "pause_auto_data_restore", False):
        return
    if not sess.sales_df.empty:
        return
    if not _session_has_platform_data(sess) and not get_summary():
        return
    _restore_daily_if_needed(sess)


# (step_id, progress 0–100, default label for UI) — GitHub before disk/Tier-3 (full MTR priority).
_RESTORE_STEP_DEFS: tuple[tuple[str, int, str], ...] = (
    ("queued", 1, "Queued"),
    ("waiting", 3, "Waiting for server memory (warm cache or upload)"),
    ("sku", 6, "SKU mapping"),
    ("warm", 12, "Warm cache — Amazon, platforms, sales"),
    ("github_download", 18, "Downloading GitHub cache"),
    ("github_amazon", 32, "GitHub — Amazon (MTR)"),
    ("github_myntra", 42, "GitHub — Myntra"),
    ("github_meesho", 52, "GitHub — Meesho"),
    ("github_flipkart", 62, "GitHub — Flipkart"),
    ("github_snapdeal", 66, "GitHub — Snapdeal"),
    ("github_inventory", 70, "GitHub — inventory"),
    ("disk", 74, "On-disk backup snapshot"),
    ("inventory", 76, "Inventory snapshot"),
    ("tier3", 84, "Tier-3 daily history (SQLite)"),
    ("daily_store", 90, "Daily upload store"),
    ("publish", 92, "Saving warm cache"),
    ("sales_queue", 94, "Queuing combined sales rebuild"),
    ("sales", 98, "Rebuilding combined sales"),
    ("done", 100, "Complete"),
)
_RESTORE_STEP_PCT = {s[0]: s[1] for s in _RESTORE_STEP_DEFS}
_RESTORE_STEP_LABEL = {s[0]: s[2] for s in _RESTORE_STEP_DEFS}
_GITHUB_STEP_BY_PLATFORM = {
    "amazon": "github_amazon",
    "myntra": "github_myntra",
    "meesho": "github_meesho",
    "flipkart": "github_flipkart",
    "snapdeal": "github_snapdeal",
}


def _set_restore_step(sess: AppSession, step_id: str, detail: str | None = None) -> None:
    sess.session_restore_step = step_id
    sess.session_restore_progress = int(_RESTORE_STEP_PCT.get(step_id, 0))
    sess.session_restore_message = detail or _RESTORE_STEP_LABEL.get(step_id, step_id)


def _run_with_restore_heartbeat(sess: AppSession, step_id: str, label: str, fn):
    """Keep UI progress alive during long CPU/IO (merge of millions of rows)."""
    import threading

    stop = threading.Event()

    def _beat() -> None:
        elapsed = 0
        while not stop.wait(3.0):
            elapsed += 3
            _set_restore_step(sess, step_id, f"{label} ({elapsed}s)…")

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop.set()
        t.join(timeout=1.0)


def _merge_github_bulk_into_session(sess: AppSession, *, progress=None) -> bool:
    """Load history: local server cache first, then selective GitHub; merge into session."""
    import pandas as pd

    from ..services.daily_store import merge_platform_data
    from ..services.github_cache import load_history_for_restore
    from backend.routers.cache import (
        _merge_disk_warm_cache_into_loaded,
        _sanitize_snapdeal_in_loaded,
    )

    _set_restore_step(
        sess,
        "github_download",
        "Loading history (server disk first — GitHub only if needed)…",
    )

    def _gh_progress(done: int, total: int, msg: str) -> None:
        span = max(1, total)
        pct = 18 + int(14 * min(done, span) / span)
        sess.session_restore_progress = pct
        sess.session_restore_step = "github_download"
        sess.session_restore_message = f"{msg} ({done}/{total})"

    ok, summary, loaded, _used_github = load_history_for_restore(progress_callback=_gh_progress)
    if summary:
        sess.session_restore_message = summary
    if not ok and not loaded:
        return False
    if not loaded:
        return False
    _sanitize_snapdeal_in_loaded(loaded)
    loaded, _disk_note = _merge_disk_warm_cache_into_loaded(loaded)
    changed = False
    for _plat, attr in _PLATFORM_ATTRS:
        gh = loaded.get(attr)
        if not isinstance(gh, pd.DataFrame) or gh.empty:
            continue
        step_id = _GITHUB_STEP_BY_PLATFORM.get(_plat, "github_download")
        n_rows = len(gh)
        _set_restore_step(sess, step_id, f"{_RESTORE_STEP_LABEL[step_id]} ({n_rows:,} rows)…")
        cur = getattr(sess, attr, None)
        if not isinstance(cur, pd.DataFrame):
            cur = pd.DataFrame()
        def _merge_gh(cur_df=cur, gh_df=gh, plat=_plat, attribute=attr):
            merged = merge_platform_data(cur_df, gh_df, plat)
            setattr(sess, attribute, merged)
            return len(merged)

        after = _run_with_restore_heartbeat(
            sess,
            step_id,
            f"GitHub merge { _plat }",
            _merge_gh,
        )
        if after > len(cur) or (cur.empty and after > 0):
            changed = True
    gm = loaded.get("sku_mapping")
    if isinstance(gm, dict) and gm and len(gm) > len(sess.sku_mapping or {}):
        sess.sku_mapping = {**gm, **(sess.sku_mapping or {})}
        changed = True
    for inv_key in ("inventory_df_variant", "inventory_df_parent"):
        iv = loaded.get(inv_key)
        if isinstance(iv, pd.DataFrame) and not iv.empty:
            _set_restore_step(sess, "github_inventory", f"GitHub — inventory ({len(iv):,} rows)…")
            cur_iv = getattr(sess, inv_key, None)
            if not isinstance(cur_iv, pd.DataFrame) or cur_iv.empty or len(iv) > len(cur_iv):
                setattr(sess, inv_key, iv)
                changed = True
    return changed


def full_restore_session(
    sess: AppSession,
    *,
    defer_sales_rebuild: bool = False,
) -> tuple[list[str], list[str], str]:
    """
  Operator restore: warm → **GitHub bulk (priority)** → disk → Tier-3 → daily store.
    Unlike GET coverage, this may download GitHub assets and rebuild sales (defer for async).
    """
    import backend.main as _main
    from ..session import resume_auto_data_restore

    steps: list[str] = []
    sess.pause_auto_data_restore = False
    resume_auto_data_restore(sess)
    sess.daily_restored = False

    _set_restore_step(sess, "sku")
    try:
        from ..services.sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass

    _set_restore_step(sess, "warm")
    try:
        _main.bootstrap_warm_cache_if_empty()
        _main.restore_po_sidecars_from_warm(sess)
        _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
        steps.append("warm")
    except Exception:
        pass

    skip_github = (
        _session_operational_data_complete(sess)
        and not _main.session_needs_warm_cache_topup(sess)
    )
    if skip_github:
        _set_restore_step(
            sess,
            "sales_queue",
            "All 8 datasets already loaded — skipping GitHub download…",
        )
        steps.append("warm_only")
        if defer_sales_rebuild:
            _set_restore_step(sess, "sales_queue")
        else:
            _set_restore_step(sess, "sales")
            _rebuild_session_sales(sess)
        try:
            _main.publish_warm_cache_from_session(sess)
        except Exception:
            pass
        missing = _missing_platform_names(sess)
        essential_missing = _essential_missing_platforms(missing)
        if essential_missing:
            labels = ", ".join(p.capitalize() for p in essential_missing)
            msg = f"Fast refresh complete. Still missing: {labels}."
        else:
            msg = "Fast refresh complete — all 8 datasets already loaded (skipped GitHub download)."
        return missing, steps, msg

    try:
        if _merge_github_bulk_into_session(sess):
            steps.append("github")
    except Exception:
        pass

    _set_restore_step(sess, "disk")
    disk_note = _merge_disk_warm_into_session(sess)
    if disk_note.strip():
        steps.append("disk")

    _set_restore_step(sess, "inventory")
    _restore_inventory_from_warm(sess)
    _set_restore_step(sess, "tier3", "Tier-3 daily history — merging recent uploads…")
    _restore_daily_if_needed(
        sess,
        force=True,
        lock_timeout=600.0,
        restore_full_mode=True,
        skip_sales_rebuild=defer_sales_rebuild,
    )
    if "tier3" not in steps:
        steps.append("tier3")

    _set_restore_step(sess, "daily_store")
    try:
        from backend.routers.cache import _merge_daily_store_into_session

        if _merge_daily_store_into_session(sess).strip():
            if "daily_store" not in steps:
                steps.append("daily_store")
    except Exception:
        pass

    _set_restore_step(sess, "publish")
    if defer_sales_rebuild:
        _set_restore_step(sess, "sales_queue")
    else:
        _set_restore_step(sess, "sales")
        _rebuild_session_sales(sess)
    try:
        _main.publish_warm_cache_from_session(sess)
    except Exception:
        pass

    missing = _missing_platform_names(sess)
    essential_missing = _essential_missing_platforms(missing)
    step_txt = ", ".join(steps) if steps else "none"
    if essential_missing:
        labels = ", ".join(p.capitalize() for p in essential_missing)
        msg = (
            f"Restored from server ({step_txt}). Still missing: {labels} — "
            "not found in warm cache, disk, Tier-3, or GitHub. Re-upload those files."
        )
    elif missing:
        opt = ", ".join(p.capitalize() for p in missing)
        msg = f"Full restore complete ({step_txt}). Optional data not on server: {opt}."
    else:
        msg = f"Full restore complete ({step_txt})."
    return missing, steps, msg


def _apply_light_coverage_hydrate(sess: AppSession) -> None:
    """Attach shared frames / sku map for fast coverage — never block on full DataFrame copy."""
    try:
        import backend.main as _main

        if not _main._warm_cache:
            _main.bootstrap_warm_cache_if_empty()
        else:
            _main._top_up_warm_cache_from_disk()
        if not _main._warm_cache:
            return
        _main.try_attach_shared_frames_fast(sess)
        if _shared_frames_operational(sess):
            _mark_tier3_sync_applied(sess)
    except Exception:
        pass


def _shared_frames_operational(sess: AppSession) -> bool:
    """Warm cache attached — skip session rebuild workers (partial attach OK for Intelligence)."""
    try:
        from ..services.shared_frames import frame_row_count, session_uses_shared_frames

        import backend.main as _main

        if not session_uses_shared_frames(sess):
            return False
        if not _main.session_needs_operational_data(sess):
            return True
        return (
            frame_row_count("sales_df", sess) >= 100_000
            and frame_row_count("inventory_df_variant", sess) >= 1_000
        )
    except Exception:
        return False


def _build_coverage_response(sess: AppSession, *, light: bool = False) -> CoverageResponse:
    """Build coverage flags from current session state (no restore side effects)."""
    import pandas as pd

    from ..services.shared_frames import (
        frame_row_count,
        session_inventory_variant,
        session_sales_df,
    )

    if light:
        _apply_light_coverage_hydrate(sess)

    paused = getattr(sess, "pause_auto_data_restore", False)
    from ..services.existing_po import (
        count_per_size_pipeline_skus,
        existing_po_looks_aggregated_bundled_only,
        existing_po_needs_recalc as _existing_po_needs_recalc,
    )

    if not light:
        from ..services.daily_store import get_summary
        from ..services.existing_po import ensure_existing_po_hydrated

        try:
            ensure_existing_po_hydrated(sess)
        except Exception:
            pass
        tier3_any = bool(get_summary())
    else:
        tier3_any = False
        try:
            from ..services.intelligence_readiness import _tier3_all_platforms_have_uploads

            tier3_any = _tier3_all_platforms_have_uploads()
        except Exception:
            pass

    _po_ledger = getattr(sess, "po_raise_ledger_df", None)
    _po_ledger_ok = _po_ledger is not None and not getattr(_po_ledger, "empty", True)
    _ret_ov = getattr(sess, "po_return_overlay_df", None)
    _ret_ok = _ret_ov is not None and not getattr(_ret_ov, "empty", True) and not light
    _ret_agg = None
    _ret_sources: list[dict] = []
    _ret_by_platform: list[dict] = []
    if _ret_ok:
        try:
            from ..services.po_return_import import (
                aggregate_return_overlay_for_use,
                ensure_return_overlay_meta_hydrated,
                rebuild_return_overlay_sources,
                summarize_return_overlay_by_platform,
            )

            ensure_return_overlay_meta_hydrated(sess)
            _ret_agg = aggregate_return_overlay_for_use(_ret_ov)
            _ret_sources = list(getattr(sess, "return_overlay_sources", None) or [])
            if not _ret_sources:
                _ret_sources = rebuild_return_overlay_sources(sess)
                sess.return_overlay_sources = _ret_sources
            _ret_by_platform = summarize_return_overlay_by_platform(_ret_ov)
        except Exception:
            _ret_agg = _ret_ov
    _ingest = getattr(sess, "daily_auto_ingest_result", None) or {}
    _has_ingest = bool(_ingest)
    try:
        import backend.main as _main
        from ..services.intelligence_readiness import _tier3_all_platforms_have_uploads

        _po_only = _main.warm_cache_po_session_only() and frame_row_count("sales_df", sess) > 0
        if not _po_only and frame_row_count("sales_df", sess) >= 100_000:
            _po_only = True
        if not _po_only and _tier3_all_platforms_have_uploads() and (
            frame_row_count("sales_df", sess) > 0 or _session_has_platform_data(sess)
        ):
            _po_only = True
    except Exception:
        _po_only = False
    _plat_ok = _po_only or None  # when True, all marketplace flags count as loaded
    _inv = session_inventory_variant(sess)
    _has_sku_map = bool(sess.sku_mapping)
    if not _has_sku_map:
        try:
            import backend.main as _main

            _has_sku_map = bool((_main._warm_cache or {}).get("sku_mapping"))
        except Exception:
            pass
    cov = CoverageResponse(
        sku_mapping=_has_sku_map,
        mtr=_plat_ok or frame_row_count("mtr_df", sess) > 0,
        sales=_coverage_sales_ready(sess),
        myntra=_plat_ok or frame_row_count("myntra_df", sess) > 0,
        meesho=_plat_ok or frame_row_count("meesho_df", sess) > 0,
        flipkart=_plat_ok or frame_row_count("flipkart_df", sess) > 0,
        snapdeal=_plat_ok or frame_row_count("snapdeal_df", sess) > 0,
        inventory=not _inv.empty or frame_row_count("inventory_df_variant", sess) > 0,
        daily_orders=len(sess.daily_sales_sources) > 0 or tier3_any,
        existing_po=not sess.existing_po_df.empty,
        sku_status_lead=not sess.sku_status_lead_df.empty,
        daily_inventory_history=not sess.daily_inventory_history_df.empty,
        manual_intransit_sheet=not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty,
        po_raise_ledger=bool(_po_ledger_ok),
        return_sheet=bool(_ret_ok),
        mtr_rows=_coverage_platform_row_count(sess, "mtr_df", "amazon"),
        sales_rows=frame_row_count("sales_df", sess) or _disk_parquet_row_count("sales_df"),
        myntra_rows=_coverage_platform_row_count(sess, "myntra_df", "myntra"),
        meesho_rows=_coverage_platform_row_count(sess, "meesho_df", "meesho"),
        flipkart_rows=_coverage_platform_row_count(sess, "flipkart_df", "flipkart"),
        snapdeal_rows=_coverage_platform_row_count(sess, "snapdeal_df", "snapdeal"),
        inventory_rows=int(len(_inv)),
        sku_status_lead_rows=int(len(sess.sku_status_lead_df)),
        daily_inventory_history_rows=int(len(sess.daily_inventory_history_df)),
        daily_inventory_history_skus=(
            int(sess.daily_inventory_history_df["OMS_SKU"].nunique())
            if not sess.daily_inventory_history_df.empty
            else 0
        ),
        manual_intransit_skus=(
            int(len(sess.manual_intransit_overlay_df))
            if not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty
            else 0
        ),
        manual_intransit_units=(
            int(pd.to_numeric(sess.manual_intransit_overlay_df.get("Manual_InTransit"), errors="coerce").fillna(0).sum())
            if not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty
            else 0
        ),
        manual_not_in_inventory_units=(
            int(
                pd.to_numeric(sess.manual_intransit_overlay_df.get("Not_In_Inventory_Qty"), errors="coerce")
                .fillna(0)
                .sum()
            )
            if not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty
            else 0
        ),
        manual_intransit_uploaded_at=(
            (getattr(sess, "manual_intransit_uploaded_at", "") or None) or None
        ),
        manual_intransit_filename=(
            (getattr(sess, "manual_intransit_filename", "") or None) or None
        ),
        manual_intransit_parse_report=(
            dict(getattr(sess, "manual_intransit_parse_report", None) or {}) or None
        ),
        finishing_receipt_uploaded_at=(
            (getattr(sess, "finishing_receipt_uploaded_at", "") or None) or None
        ),
        finishing_receipt_filename=(
            (getattr(sess, "finishing_receipt_filename", "") or None) or None
        ),
        finishing_receipt_report=(
            dict(getattr(sess, "finishing_receipt_report", None) or {}) or None
        ),
        po_raise_ledger_rows=int(len(_po_ledger)) if _po_ledger_ok else 0,
        return_sheet_skus=(
            int(len(_ret_agg))
            if _ret_ok and _ret_agg is not None and not getattr(_ret_agg, "empty", True)
            else (int(len(_ret_ov)) if _ret_ok else 0)
        ),
        return_sheet_units=(
            int(_ret_agg["Return_Units"].sum())
            if _ret_ok and _ret_agg is not None and not getattr(_ret_agg, "empty", True)
            else (int(_ret_ov["Return_Units"].sum()) if _ret_ok else 0)
        ),
        return_overlay_uploaded_at=(
            (getattr(sess, "return_overlay_uploaded_at", "") or None) or None
        ),
        return_overlay_filename=(
            (getattr(sess, "return_overlay_filename", "") or None) or None
        ),
        return_overlay_sources=_ret_sources or None,
        return_overlay_by_platform=_ret_by_platform or None,
        pause_auto_data_restore=paused,
        returns_import_status=getattr(sess, "returns_import_status", "idle") or "idle",
        returns_import_message=getattr(sess, "returns_import_message", "") or "",
        returns_import_progress=int(getattr(sess, "returns_import_progress", 0) or 0),
        returns_import_warnings=list(getattr(sess, "returns_import_warnings", None) or []) or None,
        sales_rebuild=getattr(sess, "sales_rebuild_status", "idle") or "idle",
        sales_rebuild_message=getattr(sess, "sales_rebuild_message", "") or "",
        sales_data_revision=int(getattr(sess, "sales_data_revision", 0) or 0),
        daily_auto_ingest_status=getattr(sess, "daily_auto_ingest_status", "idle") or "idle",
        daily_auto_ingest_message=getattr(sess, "daily_auto_ingest_message", "") or "",
        daily_auto_ingest_detected_platforms=(
            list(_ingest.get("detected_platforms") or []) if _has_ingest else None
        ),
        daily_auto_ingest_warnings=(
            list(_ingest.get("warnings") or []) if _has_ingest else None
        ),
        daily_auto_ingest_processed_files=(
            int(_ingest["processed_files"]) if _has_ingest else None
        ),
        daily_auto_ingest_detected_files=(
            int(_ingest["detected_files"]) if _has_ingest else None
        ),
        daily_auto_ingest_unknown_files=(
            int(_ingest["unknown_files"]) if _has_ingest else None
        ),
        daily_auto_ingest_expanded_files=(
            int(_ingest["expanded_files"]) if _has_ingest and _ingest.get("expanded_files") is not None else None
        ),
        daily_auto_ingest_saved_files=(
            int(_ingest["saved_files"]) if _has_ingest and _ingest.get("saved_files") is not None else None
        ),
        daily_auto_ingest_file_results=(
            list(_ingest.get("file_results") or []) if _has_ingest else None
        ),
        inventory_upload_status=getattr(sess, "inventory_upload_status", "idle") or "idle",
        inventory_upload_message=getattr(sess, "inventory_upload_message", "") or "",
        inventory_upload_progress=int(getattr(sess, "inventory_upload_progress", 0) or 0),
        tier1_bulk_status=getattr(sess, "tier1_bulk_status", "idle") or "idle",
        tier1_bulk_message=getattr(sess, "tier1_bulk_message", "") or "",
        tier1_bulk_platform=getattr(sess, "tier1_bulk_platform", "") or "",
        inventory_upload_rows=(
            int(sess.inventory_upload_result["rows"])
            if getattr(sess, "inventory_upload_result", None)
            and sess.inventory_upload_result.get("rows") is not None
            else None
        ),
        inventory_upload_warnings=(
            list(sess.inventory_upload_result.get("warnings") or [])
            if getattr(sess, "inventory_upload_result", None)
            else None
        ),
        inventory_upload_file_results=(
            list(sess.inventory_upload_result.get("file_results") or [])
            if getattr(sess, "inventory_upload_result", None)
            else None
        ),
        inventory_upload_sources=(
            list(sess.inventory_upload_result.get("sources_summary") or [])
            if getattr(sess, "inventory_upload_result", None)
            else None
        ),
        inventory_upload_amz_disclaimer=(
            (sess.inventory_upload_result.get("debug") or {}).get("amz_disclaimer")
            if getattr(sess, "inventory_upload_result", None)
            else None
        ),
        inventory_snapshot_date=(
            getattr(sess, "inventory_snapshot_date", "") or None
        ) or None,
        inventory_snapshot_date_label=(
            getattr(sess, "inventory_snapshot_date_label", "") or None
        ) or None,
        inventory_snapshot_date_sources=(
            list(getattr(sess, "inventory_snapshot_date_sources", None) or [])
            or None
        ),
        inventory_snapshot_uploaded_at=(
            getattr(sess, "inventory_snapshot_uploaded_at", "") or None
        ) or None,
        existing_po_uploaded_at=(getattr(sess, "existing_po_uploaded_at", "") or None) or None,
        existing_po_filename=(getattr(sess, "existing_po_filename", "") or None) or None,
        existing_po_generation=int(getattr(sess, "existing_po_generation", 0) or 0),
        existing_po_rows=(
            int(len(sess.existing_po_df))
            if getattr(sess, "existing_po_df", None) is not None
            and not getattr(sess.existing_po_df, "empty", True)
            else 0
        ),
        existing_po_needs_recalc=_existing_po_needs_recalc(sess),
        existing_po_per_size_skus=count_per_size_pipeline_skus(getattr(sess, "existing_po_df", None)),
        existing_po_looks_aggregated=existing_po_looks_aggregated_bundled_only(
            getattr(sess, "existing_po_df", None)
        ),
        existing_po_upload_status=getattr(sess, "existing_po_upload_status", "idle") or "idle",
        existing_po_upload_message=getattr(sess, "existing_po_upload_message", "") or "",
        existing_po_upload_progress=int(getattr(sess, "existing_po_upload_progress", 0) or 0),
        daily_inventory_upload_status=getattr(sess, "daily_inventory_upload_status", "idle") or "idle",
        daily_inventory_upload_message=getattr(sess, "daily_inventory_upload_message", "") or "",
        session_restore_status=getattr(sess, "session_restore_status", "idle") or "idle",
        session_restore_message=getattr(sess, "session_restore_message", "") or "",
        session_restore_step=getattr(sess, "session_restore_step", "") or "",
        session_restore_progress=int(getattr(sess, "session_restore_progress", 0) or 0),
    )
    from ..services.po_readiness import augment_coverage

    return augment_coverage(sess, cov, light=light)


def _acquire_upload_lock_with_progress(sess: AppSession, *, timeout_sec: float = 3600.0) -> bool:
    """Wait for upload-memory lock; update UI progress so restore does not look stuck at 0%."""
    import time

    from ..concurrency import _UPLOAD_MEMORY_LOCK

    start = time.monotonic()
    while True:
        if _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
            return True
        elapsed = int(time.monotonic() - start)
        _set_restore_step(
            sess,
            "waiting",
            f"Waiting for server memory lock ({elapsed}s) — warm cache or another upload may be running…",
        )
        if time.monotonic() - start > timeout_sec:
            return False
        time.sleep(2.0)


def queue_session_restore_if_needed(sess: AppSession, session_id: str, *, reason: str = "") -> bool:
    """Queue async full restore when the session has no operational data."""
    import time

    from ..concurrency import SESSION_RESTORE_EXECUTOR

    if not session_id:
        return False
    status = getattr(sess, "session_restore_status", "idle") or "idle"
    if status == "running":
        return True
    if status == "error" and reason.startswith("Auto-restore"):
        return False
    if getattr(sess, "_auto_restore_queued", False) and reason.startswith("Auto-restore"):
        return False
    try:
        import backend.main as _main

        if not _main.session_needs_operational_data(sess):
            return False
    except Exception:
        pass
    if getattr(sess, "pause_auto_data_restore", False):
        return False

    sess.session_restore_status = "running"
    sess.session_restore_started = time.monotonic()
    _set_restore_step(
        sess,
        "queued",
        reason or "Queued full restore — loading server history…",
    )
    sess.session_restore_result = {}
    SESSION_RESTORE_EXECUTOR.submit(_run_session_restore_worker, session_id)
    return True


def _server_has_recoverable_history() -> bool:
    """True when disk, warm cache, GitHub manifest, or Tier-3 may repopulate a session."""
    try:
        import backend.main as _main
        from ..services.github_cache import get_cache_manifest

        _main.bootstrap_warm_cache_if_empty()
        if _main._warm_cache:
            for key in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
                df = _main._warm_cache.get(key)
                if df is not None and hasattr(df, "empty") and not df.empty:
                    return True
        manifest = get_cache_manifest()
        if manifest and (manifest.get("row_counts") or manifest.get("saved_at")):
            return True
        if get_summary():
            return True
    except Exception:
        pass
    return False


def _maybe_queue_full_restore_when_empty(sess: AppSession, session_id: str | None) -> None:
    """Queue fast warm-cache hydrate when session is empty — never auto-start slow GitHub restore."""
    if not session_id:
        return
    if getattr(sess, "pause_auto_data_restore", False):
        return
    if getattr(sess, "_auto_restore_queued", False):
        return
    try:
        import backend.main as _main

        if not _main.session_needs_operational_data(sess):
            return
        if not _server_has_recoverable_history():
            return
        _main.bootstrap_warm_cache_if_empty()
        if _main._warm_cache:
            _maybe_queue_light_session_hydrate(sess, session_id)
            sess._auto_restore_queued = True
            return
    except Exception:
        pass
    # No warm cache in memory/disk — last resort full restore (GitHub), operator-triggered only.
    # Do not auto-queue here; empty sessions rely on frontend hydrate-warm + coverage worker.


def _run_session_restore_worker(session_id: str) -> None:
    """Background full restore — disk/GitHub/Tier-3 merge without blocking on warm-cache memory lock."""
    import logging
    import time

    from ..session import store

    _log = logging.getLogger(__name__)
    sess = store.get(session_id)
    if sess is None:
        return

    try:
        sess.session_restore_status = "running"
        sess.session_restore_started = time.monotonic()
        _set_restore_step(sess, "queued", "Restore worker started…")
        _set_restore_step(sess, "sku")

        missing, steps, msg = full_restore_session(sess, defer_sales_rebuild=True)
        essential_missing = _essential_missing_platforms(missing)
        sess.session_restore_result = {
            "missing_platforms": missing,
            "restore_steps": steps,
            "message": msg,
            "essential_missing": essential_missing,
        }

        if _session_has_platform_data(sess) and sess.sku_mapping:
            from ..concurrency import SESSION_RESTORE_EXECUTOR

            _set_restore_step(sess, "sales", "Rebuilding combined sales (large history)…")
            sess.sales_rebuild_status = "running"
            sess.sales_rebuild_message = sess.session_restore_message
            SESSION_RESTORE_EXECUTOR.submit(
                _run_session_restore_sales_worker,
                session_id,
            )
        else:
            _rebuild_session_sales(sess)
            _set_restore_step(sess, "done", msg)
            sess.session_restore_status = "done"
            try:
                from ..services.perf_metrics import record_session_restore

                started = float(getattr(sess, "session_restore_started", 0) or 0)
                if started:
                    record_session_restore("full_restore", time.monotonic() - started, ok=True)
            except Exception:
                pass
    except Exception as e:
        _log.exception("background session restore failed")
        sess.session_restore_status = "error"
        sess.session_restore_message = str(e)[:500]
        sess.session_restore_progress = 0


def _run_session_restore_sales_worker(session_id: str) -> None:
    """Finish restore after async sales rebuild."""
    import logging

    from ..session import store
    from .upload import _run_sales_rebuild_worker

    _log = logging.getLogger(__name__)
    sess = store.get(session_id)
    if sess is None:
        return
    try:
        _run_sales_rebuild_worker(session_id, refresh_sqlite=True)
        msg = (sess.session_restore_result or {}).get("message") or "Full restore complete."
        _set_restore_step(sess, "done", msg)
        sess.session_restore_status = "done"
        try:
            import time

            from ..services.perf_metrics import record_session_restore

            started = float(getattr(sess, "session_restore_started", 0) or 0)
            if started:
                record_session_restore("full_restore+sales", time.monotonic() - started, ok=True)
        except Exception:
            pass
        try:
            import backend.main as _main

            _main.publish_warm_cache_from_session(sess)
        except Exception:
            _log.exception("publish warm cache after restore sales")
    except Exception as e:
        _log.exception("restore sales worker failed")
        sess.session_restore_status = "error"
        sess.session_restore_message = str(e)[:500]


# ── Coverage / job status ─────────────────────────────────────

_hydrate_queued: set[str] = set()  # legacy — prefer session_hydrate.schedule_session_hydrate


def _session_needs_background_hydrate(sess: AppSession) -> bool:
    """True when session should top up from warm cache / Tier-3 in a background worker."""
    if getattr(sess, "pause_auto_data_restore", False):
        return False
    if any(
        getattr(sess, attr, "idle") == "running"
        for attr in (
            "daily_auto_ingest_status",
            "session_restore_status",
            "inventory_upload_status",
            "daily_inventory_upload_status",
        )
    ):
        return False
    try:
        from ..services.shared_frames import frame_row_count, session_uses_shared_frames

        import backend.main as _main

        if session_uses_shared_frames(sess):
            # Shared warm-cache pointers are enough for Intelligence / coverage polls.
            if not _main.session_needs_operational_data(sess):
                return False
            if (
                frame_row_count("sales_df", sess) >= 100_000
                and frame_row_count("inventory_df_variant", sess) >= 1_000
            ):
                return False
    except Exception:
        pass
    try:
        import backend.main as _main

        if _main.session_needs_warm_cache_topup(sess):
            return True
        if _main.session_needs_operational_data(sess):
            return True
    except Exception:
        pass
    if not sess.sales_df.empty:
        return False
    if _session_has_platform_data(sess):
        return True
    try:
        if get_summary():
            return True
    except Exception:
        pass
    return False


def _run_light_session_hydrate_worker(session_id: str) -> None:
    """Background Tier-3 / warm-cache hydrate — never blocks GET /coverage?light=1."""
    import logging

    from ..session import store

    _log = logging.getLogger(__name__)
    try:
        sess = store.get(session_id)
        if sess is None:
            return
        if getattr(sess, "pause_auto_data_restore", False):
            return
        try:
            import backend.main as _main
            from ..services.shared_frames import frame_row_count, shared_frames_enabled

            if shared_frames_enabled():
                _main.try_attach_shared_frames_fast(sess)
                if not _main.session_needs_operational_data(sess):
                    return
                if (
                    frame_row_count("sales_df", sess) >= 100_000
                    and frame_row_count("inventory_df_variant", sess) >= 1_000
                ):
                    return
        except Exception:
            pass
        try:
            import backend.main as _main
            from ..services.po_session_hydrate import ensure_po_return_overlay_from_server

            _main.restore_po_sidecars_from_warm(sess)
            if _main.session_needs_warm_cache_topup(sess):
                if not _main._warm_cache:
                    _main.bootstrap_warm_cache_if_empty()
                if _main._warm_cache:
                    _main._copy_warm_cache_to_session(sess)
                    sess._warm_cache_gen = int(
                        getattr(_main, "_warm_cache_generation", 0) or 0
                    )
                    sess._warm_cache_only = True
                elif _main.session_needs_operational_data(sess):
                    _main.force_restore_session_from_server_cache(
                        sess, _main._warm_cache_generation,
                    )
                else:
                    _main._apply_warm_cache_if_needed(
                        sess, _main._warm_cache_generation,
                    )
        except Exception:
            _log.exception("light hydrate warm copy failed session=%s", session_id[:8])
        try:
            from backend.routers.cache import _persist_pg_session_bg

            _persist_pg_session_bg(session_id, sess)
        except Exception:
            pass
        if not sess._daily_restore_lock.acquire(blocking=False):
            return
        try:
            _restore_daily_if_needed(sess)
            _ensure_sales_rebuilt(sess)
        finally:
            sess._daily_restore_lock.release()
    except Exception:
        _log.exception("light session hydrate worker failed session=%s", session_id[:8])
    finally:
        _hydrate_queued.discard(session_id)


def _maybe_queue_light_session_hydrate(sess: AppSession, session_id: str | None) -> None:
    if not session_id or not _session_needs_background_hydrate(sess):
        return
    from ..concurrency import SESSION_RESTORE_EXECUTOR
    from ..services.session_hydrate import HydrateSchedule, schedule_session_hydrate, session_hydrate_inflight

    if session_hydrate_inflight(session_id):
        return
    status = schedule_session_hydrate(
        session_id,
        _run_light_session_hydrate_worker,
        executor=SESSION_RESTORE_EXECUTOR,
    )
    if status in (HydrateSchedule.SCHEDULED, HydrateSchedule.INFLIGHT):
        _hydrate_queued.add(session_id)


_tier3_sync_queued: set[str] = set()


def _run_tier3_sales_sync_worker(session_id: str, platforms: list[str] | None) -> None:
    """Merge Tier-3 SQLite into session when uploads saved but sales rebuild never finished."""
    from .upload import _run_sales_rebuild_worker

    try:
        plats = set(platforms) if platforms else None
        _run_sales_rebuild_worker(
            session_id,
            refresh_sqlite=not plats,
            platforms_touched=plats,
        )
    finally:
        _tier3_sync_queued.discard(session_id)


def _maybe_queue_tier3_sales_sync(sess: AppSession, session_id: str | None) -> None:
    """Background sales rebuild when Tier-3 SQLite is ahead of session (upload saved, no rebuild)."""
    if not session_id:
        return
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        return
    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        return
    if not _tier3_token_mismatch(sess):
        return
    if session_id in _tier3_sync_queued:
        return
    _tier3_sync_queued.add(session_id)
    from ..concurrency import AUX_EXECUTOR

    plats = _platforms_with_tier3_token_mismatch(sess)
    AUX_EXECUTOR.submit(_run_tier3_sales_sync_worker, session_id, plats or None)


_unified_sales_build_queued: set[str] = set()


def _maybe_queue_unified_sales_build(sess: AppSession, session_id: str | None) -> None:
    """Build unified sales_df when platform history is in session but sales_df is still empty."""
    if not session_id:
        return
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        return
    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        return
    if not sess.sku_mapping:
        return
    sales = getattr(sess, "sales_df", None)
    if sales is not None and hasattr(sales, "empty") and not sales.empty:
        return
    if not _session_has_platform_data(sess):
        return
    if session_id in _unified_sales_build_queued or session_id in _tier3_sync_queued:
        return
    _unified_sales_build_queued.add(session_id)
    from ..concurrency import AUX_EXECUTOR
    from .upload import _run_sales_rebuild_worker

    def _worker(sid: str) -> None:
        try:
            _run_sales_rebuild_worker(sid, refresh_sqlite=False)
        finally:
            _unified_sales_build_queued.discard(sid)

    AUX_EXECUTOR.submit(_worker, session_id)


@router.get("/job-status", response_model=JobStatusResponse)
def get_job_status(request: Request):
    """Fast job snapshot for UI polling — no restore, Tier-3 merge, or sales rebuild."""
    from datetime import datetime, timezone

    import backend.main as _main
    from ..concurrency import upload_memory_lock_held

    sess = getattr(request.state, "session", None)
    return JobStatusResponse(
        server_time=datetime.now(timezone.utc).isoformat(),
        warm_cache=bool(_main._warm_cache),
        warm_cache_generation=int(getattr(_main, "_warm_cache_generation", 0) or 0),
        upload_memory_lock_held=upload_memory_lock_held(),
        daily_auto_ingest_status=getattr(sess, "daily_auto_ingest_status", "idle") or "idle",
        sales_rebuild_status=getattr(sess, "sales_rebuild_status", "idle") or "idle",
        session_restore_status=getattr(sess, "session_restore_status", "idle") or "idle",
        inventory_upload_status=getattr(sess, "inventory_upload_status", "idle") or "idle",
        daily_inventory_upload_status=getattr(sess, "daily_inventory_upload_status", "idle") or "idle",
        tier1_bulk_status=getattr(sess, "tier1_bulk_status", "idle") or "idle",
        tier1_bulk_message=getattr(sess, "tier1_bulk_message", "") or "",
    )


@router.get("/coverage", response_model=CoverageResponse)
async def get_coverage(request: Request, light: bool = False):
    """Session coverage flags. ``light=1`` is read-only + queues background hydrate when needed."""
    from ..concurrency import run_aux

    return await run_aux(_get_coverage_sync, request, light)


def _get_coverage_sync(request: Request, light: bool = False) -> CoverageResponse:
    sess = _sess(request)
    sid = getattr(request.state, "session_id", None) or ""
    try:
        from ..routers.upload import clear_stale_background_jobs

        clear_stale_background_jobs(sess)
    except Exception:
        pass

    if light:
        _apply_light_coverage_hydrate(sess)
        if not _shared_frames_operational(sess):
            _maybe_queue_light_session_hydrate(sess, sid or None)
            _maybe_queue_tier3_sales_sync(sess, sid or None)
            _maybe_queue_unified_sales_build(sess, sid or None)
        return _build_coverage_response(sess, light=True)

    try:
        from ..services.sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass
    try:
        import backend.main as _main

        _main.restore_po_sidecars_from_warm(sess)
        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(
                sess, _main._warm_cache_generation
            )
    except Exception:
        pass
    _restore_inventory_from_warm(sess)
    _maybe_queue_tier3_sales_sync(sess, sid or None)
    _maybe_queue_unified_sales_build(sess, sid or None)
    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        light = True
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        light = True
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        light = True
    if getattr(sess, "daily_inventory_upload_status", "idle") == "running":
        light = True
    if getattr(sess, "session_restore_status", "idle") == "running":
        light = True
    if light:
        _maybe_queue_light_session_hydrate(sess, sid or None)
        return _build_coverage_response(sess)
    try:
        from ..services.po_raise_import import hydrate_session_ledger_from_db

        hydrate_session_ledger_from_db(sess, lookback_days=30, authoritative=False)
    except Exception:
        pass
    _restore_daily_if_needed(sess)
    _ensure_sales_rebuilt(sess)
    return _build_coverage_response(sess)


@router.get("/intelligence/readiness", response_model=IntelligenceReadinessResponse)
def intelligence_readiness(request: Request):
    """Dashboard gate — 8/8 + platform history (ignores sales_rebuild)."""
    sess = _sess(request)
    sid = getattr(request.state, "session_id", None) or ""
    try:
        from ..routers.upload import clear_stale_background_jobs

        clear_stale_background_jobs(sess)
    except Exception:
        pass
    if not _shared_frames_operational(sess):
        _maybe_queue_light_session_hydrate(sess, sid or None)
    cov = _build_coverage_response(sess, light=True)
    from ..services.intelligence_readiness import build_intelligence_readiness

    return IntelligenceReadinessResponse(**build_intelligence_readiness(sess, cov, session_id=sid))


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
):
    """Compact dashboard aggregates from Tier-3 / PostgreSQL (no session platform copies)."""
    from ..services.dashboard_summary import build_dashboard_summary

    sess = _sess(request)
    payload = build_dashboard_summary(sess, start_date=start_date, end_date=end_date, limit=limit)
    return DashboardSummaryResponse(**payload)


@router.get("/parity")
def data_parity(request: Request, planning_date: Optional[str] = None):
    """
    Lightweight live vs local / dashboard vs PO parity diagnostics.
    When Tier-3 SQLite is ahead of the session, queue a background sales sync.
    """
    from ..services.tier3_session_merge import build_parity_report

    sess = _sess(request)
    sid = getattr(request.state, "session_id", None) or ""
    report = build_parity_report(sess, planning_date=planning_date)
    if report.get("tier3_sync_mismatch") or report.get("tier3_platforms_mismatch"):
        _maybe_queue_tier3_sales_sync(sess, sid or None)
    return report


@router.post("/restore-full", response_model=RestoreFullResponse)
def restore_full(request: Request, sync: bool = False):
    """
    Full session restore: warm cache, disk snapshot, Tier-3 SQLite, GitHub bulk merge.

    Default is **async** (returns immediately; poll ``session_restore_status`` on coverage).
    Pass ``sync=1`` only for tests / debugging.
    """
    import os

    import time

    from ..concurrency import SESSION_RESTORE_EXECUTOR

    sess = _sess(request)
    sid = getattr(request.state, "session_id", None) or ""

    if getattr(sess, "session_restore_status", "idle") == "running":
        cov = _build_coverage_response(sess)
        return RestoreFullResponse(
            **cov.model_dump(),
            ok=True,
            message=sess.session_restore_message or "Restore already in progress…",
            missing_platforms=[],
            restore_steps=[],
            restore_async=True,
        )

    sess._auto_restore_queued = False

    use_sync = sync or (os.environ.get("RESTORE_FULL_SYNC", "").strip() == "1")
    if use_sync:
        missing, steps, msg = full_restore_session(sess)
        cov = _build_coverage_response(sess)
        essential_missing = _essential_missing_platforms(missing)
        ok = (
            not essential_missing
            and bool(sess.sku_mapping)
            and not sess.mtr_df.empty
            and not sess.sales_df.empty
        )
        return RestoreFullResponse(
            **cov.model_dump(),
            ok=ok,
            message=msg,
            missing_platforms=missing,
            restore_steps=steps,
            restore_async=False,
        )

    sess.session_restore_status = "running"
    sess.session_restore_started = time.monotonic()
    _set_restore_step(
        sess,
        "queued",
        "Queued full restore — starting worker (not blocked behind warm-cache load)…",
    )
    sess.session_restore_result = {}
    # SESSION_RESTORE_EXECUTOR — its own queue so this isn't stuck behind a
    # backlog of per-session hydrate-warm/upload jobs on DAILY_UPLOAD_EXECUTOR.
    SESSION_RESTORE_EXECUTOR.submit(_run_session_restore_worker, sid)
    cov = _build_coverage_response(sess)
    return RestoreFullResponse(
        **cov.model_dump(),
        ok=True,
        message=(
            "Restore running in background — stay on this page. "
            "Large Amazon history may take several minutes."
        ),
        missing_platforms=[],
        restore_steps=[],
        restore_async=True,
    )


# ── Data quality (duplicate / sanity checks) ──────────────────

@router.get("/data-quality")
def data_quality_report(request: Request):
    """
    Lightweight diagnostics to spot overlapping uploads and sanity-check totals.
    Does not modify session data.
    """
    import pandas as pd
    sess = _sess(request)
    _restore_daily_if_needed(sess)

    hints = [
        "Pick one SKU and date range, then in SKU Deepdive choose “Amazon only” (or another channel) and compare units to your marketplace export for the same window.",
        "If “All channels” is much higher than a single channel export, you are summing every marketplace — that is expected, not a bug.",
        "Amazon MTR “potential duplicate rows” uses the same collapse rules as uploads: if this number is large, overlapping Tier‑1 ZIPs were merging extra lines (now deduped when you re-upload or Rebuild).",
        "After “Reset all data”, upload Tier‑1 first, click Build Sales, then add Tier‑3 dailies so history is not loaded twice from SQLite + bulk.",
    ]

    checks: dict = {}
    loaded = (
        bool(sess.sku_mapping)
        or not sess.mtr_df.empty
        or not sess.sales_df.empty
        or not sess.myntra_df.empty
    )

    if not sess.mtr_df.empty and {
        "Invoice_Number", "Order_Id", "SKU", "Transaction_Type", "Quantity", "Date",
    }.issubset(sess.mtr_df.columns):
        from ..services.mtr import dedup_amazon_mtr_dataframe

        raw_n = len(sess.mtr_df)
        ded_n = len(dedup_amazon_mtr_dataframe(sess.mtr_df.copy()))
        checks["amazon_mtr"] = {
            "rows_in_session":     raw_n,
            "rows_after_dedup_key": ded_n,
            "rows_collapsible":    max(0, raw_n - ded_n),
            "note": (
                "Rows that would merge if dedup rules were applied again to the current "
                "Amazon frame (overlapping ZIPs / duplicate lines)."
            ),
        }

    if not sess.sales_df.empty and "Sku" in sess.sales_df.columns:
        s = sess.sales_df
        key = [c for c in ("Sku", "TxnDate", "Transaction Type", "Quantity", "Source", "OrderId") if c in s.columns]
        dup_extra = 0
        if key:
            uniq = s.drop_duplicates(subset=key, keep="first")
            dup_extra = int(len(s) - len(uniq))
        ship = s[s["Transaction Type"].astype(str).str.strip() == "Shipment"]
        ship_units = int(pd.to_numeric(ship["Quantity"], errors="coerce").fillna(0).sum()) if not ship.empty else 0
        checks["unified_sales_df"] = {
            "rows":              len(s),
            "exact_duplicate_rows": dup_extra,
            "shipment_units_sum": ship_units,
            "by_source":         get_sales_by_source(sess.sales_df),
        }

    try:
        checks["tier3_sqlite_summary"] = get_summary()
    except Exception:
        checks["tier3_sqlite_summary"] = {}

    return {
        "loaded": loaded,
        "checks": checks,
        "hints":  hints,
    }


# ── Sales Dashboard KPIs ──────────────────────────────────────

@router.get("/sales-summary")
def sales_summary(
    request: Request,
    months: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _ensure_intelligence_session_fresh(sess)
    return get_sales_summary(sess.sales_df, months=months, start_date=start_date, end_date=end_date)


@router.get("/intelligence-bundle")
def intelligence_bundle(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    basis: Optional[str] = None,
    include_extras: bool = False,
):
    """
    One round-trip for the Intelligence dashboard (summary + platforms + top SKUs + anomalies + DSR brands).
    Runs session freshness once instead of five parallel handlers each rebuilding sales.
    """
    from ..services.sales import (
        apply_upload_report_day_gate,
        txn_reporting_naive_ist,
        _filter_by_reporting_days,
    )

    sess = _sess(request)
    sid = getattr(request.state, "session_id", None) or ""
    try:
        import backend.main as _main

        if not _main._warm_cache:
            _main.bootstrap_warm_cache_if_empty()
        else:
            _main._top_up_warm_cache_from_disk()
        _main.try_attach_shared_frames_fast(sess)
    except Exception:
        pass
    has_dates = bool(start_date or end_date)
    s_win = str(start_date or end_date or "")[:10] if has_dates else ""
    e_win = str(end_date or start_date or "")[:10] if has_dates else ""
    if not sess:
        return {"ok": False, "computing": True, "message": "Session data still loading"}
    sales_empty = hasattr(sess, "sales_df") and sess.sales_df.empty
    if sales_empty:
        tier3_can_serve = False
        if has_dates and len(s_win) == 10 and len(e_win) == 10:
            try:
                from ..services.daily_store import platforms_with_uploads_in_range

                tier3_can_serve = bool(platforms_with_uploads_in_range(s_win, e_win))
            except Exception:
                tier3_can_serve = False
        if not tier3_can_serve and not _session_has_platform_data(sess):
            return {"ok": False, "computing": True, "message": "Session data still loading"}
    try:
        from ..routers.upload import clear_stale_background_jobs

        clear_stale_background_jobs(sess)
    except Exception:
        pass
    _ensure_sku_mapping_for_dashboard(sess)

    span_days = _report_span_days(start_date, end_date)

    cache_key = (
        str(start_date or ""),
        str(end_date or ""),
        str(basis or "gross"),
        int(limit),
        bool(include_extras),
    )
    bundle_cache = getattr(sess, "_intelligence_bundle_cache", None)
    if bundle_cache is None:
        bundle_cache = {}
        sess._intelligence_bundle_cache = bundle_cache

    tier3_window = False
    if has_dates and len(s_win) == 10 and len(e_win) == 10:
        from ..services.daily_store import platforms_with_uploads_in_range

        tier3_window = bool(platforms_with_uploads_in_range(s_win, e_win))
    prefer_tier3 = tier3_window or _tier3_token_mismatch(sess)

    # Precomputed global/disk bundle — instant path before any Tier-3 rebuild.
    cached_instant = _bundle_cache_lookup(
        cache_key,
        bundle_cache,
        start_date=start_date,
        end_date=end_date,
        allow_sparse=True,
        sess=sess,
    )
    if cached_instant is not None:
        _maybe_queue_light_session_hydrate(sess, sid or None)
        return cached_instant

    # Tier-3 daily uploads when cache miss and window has fresh Tier-3 blobs.
    if prefer_tier3 and has_dates and len(s_win) == 10 and len(e_win) == 10:
        tier3_immediate = _try_serve_tier3_intelligence_bundle(
            sess,
            cache_key,
            bundle_cache,
            s_win,
            e_win,
            limit,
            basis,
            include_extras,
        )
        if tier3_immediate is not None:
            _maybe_queue_tier3_sales_sync(sess, sid or None)
            return tier3_immediate

    _hydrate_session_for_intelligence(sess)
    _maybe_queue_light_session_hydrate(sess, sid or None)

    now = time.time()
    cached_payload = _bundle_cache_lookup(
        cache_key, bundle_cache, start_date=start_date, end_date=end_date, sess=sess
    )
    if cached_payload is not None:
        if _tier3_token_mismatch(sess):
            _schedule_intelligence_refresh_async(sid or None)
        return cached_payload

    fast_window = span_days is not None and span_days <= _intelligence_fast_window_days()

    _schedule_intelligence_refresh_async(sid or None)

    session_payload: dict | None = None
    session_build_unsafe = _intelligence_session_build_unsafe(sess)
    if has_dates and len(s_win) == 10 and len(e_win) == 10:
        # Tier-3 before session unified build — session path OOMs on 1M+ row catalogs.
        if tier3_window or not _session_has_units_in_window(sess, s_win, e_win) or session_build_unsafe:
            tier3_first = _try_serve_tier3_intelligence_bundle(
                sess,
                cache_key,
                bundle_cache,
                s_win,
                e_win,
                limit,
                basis,
                include_extras,
                ts=now,
            )
            if tier3_first is not None:
                return tier3_first

        if not session_build_unsafe or _disk_bulk_history_available():
            session_payload = _build_intelligence_bundle_payload_from_session(
                sess, s_win, e_win, limit, basis, include_extras
            )
            if session_payload and _bundle_payload_has_display_data(session_payload):
                _bundle_cache_store(cache_key, bundle_cache, session_payload, ts=now)
                return session_payload

        if fast_window or (session_build_unsafe and not _disk_bulk_history_available()):
            tier3_payload = _try_serve_tier3_intelligence_bundle(
                sess,
                cache_key,
                bundle_cache,
                s_win,
                e_win,
                limit,
                basis,
                include_extras,
                ts=now,
            )
            if tier3_payload is not None:
                from ..services.daily_store import platforms_with_uploads_in_range

                need_persist = platforms_with_uploads_in_range(s_win, e_win)
                if need_persist:
                    _schedule_persist_tier3_window(sid or None, s_win, e_win, need_persist)
                return tier3_payload

        from ..services.daily_store import platforms_with_uploads_in_range

        need_persist = platforms_with_uploads_in_range(s_win, e_win)
        if need_persist:
            _schedule_persist_tier3_window(sid or None, s_win, e_win, need_persist)

    elif _session_has_operational_frames(sess) and (
        not session_build_unsafe or _disk_bulk_history_available()
    ):
        session_payload = _build_intelligence_bundle_payload_from_session(
            sess,
            s_win or str(start_date or "")[:10],
            e_win or str(end_date or "")[:10],
            limit,
            basis,
            include_extras,
        )
        if session_payload is not None and _bundle_payload_has_display_data(session_payload):
            _bundle_cache_store(cache_key, bundle_cache, session_payload, ts=now)
            return session_payload

    if has_dates and len(s_win) == 10 and len(e_win) == 10:
        tier3_any = _try_serve_tier3_intelligence_bundle(
            sess,
            cache_key,
            bundle_cache,
            s_win,
            e_win,
            limit,
            basis,
            include_extras,
            ts=now,
        )
        if tier3_any is not None:
            return tier3_any

    if not _session_has_operational_frames(sess):
        busy = any(
            getattr(sess, a, "idle") == "running"
            for a in (
                "daily_auto_ingest_status",
                "sales_rebuild_status",
                "session_restore_status",
            )
        )
        msg = (
            getattr(sess, "sales_rebuild_message", "")
            or getattr(sess, "daily_auto_ingest_message", "")
            or getattr(sess, "session_restore_message", "")
            or "Loading sales data on the server…"
        )
        return {
            "status": "warming",
            "message": msg,
            "sales_summary": {"total_units": 0, "total_gmv": 0, "sku_count": 0},
            "platform_summary": [],
            "top_skus": [],
            "anomalies": [],
            "dsr_brand_monthly": {"rows": []},
            "busy": busy,
        }

    if has_dates and len(s_win) == 10 and len(e_win) == 10 and _session_sales_reporting_range(sess):
        return _intelligence_empty_window_payload(sess, s_win, e_win)

    return _intelligence_warming_payload(
        "Marketplace data is still loading — retrying shortly."
    )


@router.get("/sales-export")
def sales_export(
    request: Request,
    months: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platforms: Optional[str] = None,
):
    """CSV of unified `sales_df` rows for the dashboard date range (and optional platform list)."""
    import pandas as pd

    sess = _sess(request)
    _restore_daily_if_needed(sess)
    if sess.sales_df.empty:
        raise HTTPException(status_code=404, detail="No sales data loaded — upload or rebuild sales first.")

    src_list: Optional[List[str]] = None
    if platforms and platforms.strip():
        src_list = [p.strip() for p in platforms.split(",") if p.strip()]

    df = filter_sales_for_export(
        sess.sales_df,
        months=months,
        start_date=start_date,
        end_date=end_date,
        sources=src_list,
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No rows in this date range / platform filter — widen dates or include more platforms.",
        )

    out = df.copy()
    out["TxnDate"] = pd.to_datetime(out["TxnDate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    base_cols = ["TxnDate", "Sku", "Transaction Type", "Quantity", "Units_Effective", "Source"]
    extra = [c for c in ("OrderId",) if c in out.columns]
    cols = [c for c in base_cols + extra if c in out.columns]
    export_df = out[cols].copy()
    export_df = apply_meesho_listing_sku_recovery_for_export(export_df, sess.meesho_df)
    cmap = sess.sku_mapping or {}
    _map_keys, _map_vals, _map_num = mapping_lookup_sets(cmap) if cmap else (set(), set(), {})

    def _export_oms_sku_cell(v) -> str:
        if pd.isna(v):
            return ""
        s = normalize_id_token_for_mapping(str(v).strip())
        if s.lower() in ("", "nan", "none"):
            return ""
        if is_likely_non_sku_notes_value(s):
            return ""
        resolved = canonical_sales_sku(map_to_oms_sku(s, cmap))
        if is_likely_non_sku_notes_value(resolved):
            return ""
        # If lookup is a no-op and this token never appears on the master (key or OMS),
        # leave OMS_Sku blank so exports don't fake a match (common for raw Myntra style IDs).
        if (
            cmap
            and clean_sku(resolved) == clean_sku(s)
            and not sku_recognized_in_master(
                s, cmap, key_set=_map_keys, val_set=_map_vals, numeric_embed=_map_num
            )
        ):
            return ""
        return resolved

    export_df["OMS_Sku"] = export_df["Sku"].apply(_export_oms_sku_cell)
    sku_pos = cols.index("Sku") + 1
    ordered = cols[:sku_pos] + ["OMS_Sku"] + cols[sku_pos:]
    export_df = export_df[ordered]

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    body = buf.getvalue().encode("utf-8")

    part_start = (start_date or "all").replace(":", "")
    part_end = (end_date or "all").replace(":", "")
    fname = f"intelligence-sales_{part_start}_{part_end}_{len(export_df)}_rows.csv"

    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/sales-by-source")
def sales_by_source(request: Request):
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_sales_by_source(sess.sales_df)


@router.get("/top-skus")
def top_skus(
    request: Request,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    basis: Optional[str] = None,
):
    """``basis=gross`` (default): rank by shipment quantity. ``basis=net``: by ``Units_Effective`` sum."""
    sess = _sess(request)
    _ensure_intelligence_session_fresh(sess)
    return get_top_skus(
        sess.sales_df,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        basis=basis or "gross",
    )


# ── SKU List (for search autocomplete) ───────────────────────

@router.get("/sku-list")
def sku_list(request: Request, q: Optional[str] = None, limit: int = 100, include_parents: bool = False):
    """Return unique SKUs in sales_df, optionally filtered by search query.
    When include_parents=True also returns deduplicated parent/base SKUs marked with a flag."""
    import pandas as pd

    sess = _sess(request)
    df   = sess.sales_df
    if df.empty or "Sku" not in df.columns:
        return []
    shipped = df[df["Transaction Type"].astype(str) == "Shipment"]["Sku"].astype(str)
    skus = shipped.unique().tolist()
    # Filter out noise rows
    skus = [s for s in skus if s and s.lower() not in ("nan", "none", "") and not s.lower().endswith("_total")]
    if q:
        q_lower = q.strip().lower()
        skus = [s for s in skus if q_lower in s.lower()]
    skus = sorted(skus)

    if include_parents:
        parents = sorted({get_parent_sku(s) for s in skus if get_parent_sku(s) != s})
        if q:
            q_lower = q.strip().lower()
            parents = [p for p in parents if q_lower in p.lower()]
        # Return as dicts with type flag so frontend can distinguish
        result = [{"sku": s, "type": "variant"} for s in skus[:limit]]
        result = [{"sku": p, "type": "parent"} for p in parents[:20]] + result
        return result[:limit]

    return skus[:limit]


# ── SKU Deepdive ──────────────────────────────────────────────

@router.get("/sku-deepdive")
def sku_deepdive(
    request: Request,
    sku: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    all_sizes:  bool = False,   # if True, match all SKUs that share the same base (parent) SKU
    source: Optional[str] = None,  # if set (e.g. "Amazon"), only rows from that channel — matches single-market exports
):
    """
    Full sales breakdown for a single SKU (or all sizes of a base SKU).
    Returns: summary KPIs, monthly trend, platform breakdown, daily trend, sizes breakdown.
    Default window: last 90 days.
    When all_sizes=True the `sku` param is treated as the base/parent SKU and all
    size variants (e.g. 1898YKYELLOW-3XL, 1898YKYELLOW-2XL …) are aggregated together.
    """
    import pandas as pd

    from ..services.shared_frames import session_sales_df
    from ..services.sku_deepdive_data import build_deepdive_sales_frame

    sess = _sess(request)
    unified = session_sales_df(sess)
    if unified is None or unified.empty:
        return {"loaded": False, "message": "No sales data loaded"}

    sales = build_deepdive_sales_frame(sess, sku, all_sizes=all_sizes)
    df0 = apply_upload_report_day_gate(sales) if sales is not None and not sales.empty else pd.DataFrame()

    # Detect whether Meesho is loaded but has no per-SKU data (TCS ZIP format)
    meesho_note: str | None = None
    if not sess.meesho_df.empty:
        unified_gated = apply_upload_report_day_gate(unified)
        meesho_skus_in_sales = (
            unified_gated[unified_gated["Source"].astype(str) == "Meesho"]["Sku"]
            .dropna().unique().tolist()
            if not unified_gated.empty and "Source" in unified_gated.columns else []
        )
        if meesho_skus_in_sales == ["MEESHO_TOTAL"] or set(meesho_skus_in_sales) == {"MEESHO_TOTAL"}:
            meesho_total_units = int(
                unified_gated[
                    (unified_gated["Source"].astype(str) == "Meesho") &
                    (unified_gated["Transaction Type"].astype(str) == "Shipment")
                ]["Quantity"].sum()
            ) if not unified_gated.empty else 0
            meesho_note = (
                f"Meesho data loaded ({meesho_total_units:,} total units) but your uploaded "
                f"Meesho TCS ZIP reports don't include per-SKU data. "
                f"To see Meesho in SKU breakdown, upload the Meesho Order Report CSV "
                f"(Supplier Panel → Reports → Order Reports)."
            )

    if df0.empty:
        return {
            "loaded":       True,
            "sku":          sku,
            "all_sizes":    all_sizes,
            "matched_skus": [],
            "summary":      {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":      [],
            "by_platform":  [],
            "by_size":      [],
            "daily":        [],
            "first_sale":   None,
            "last_sale":    None,
            "meesho_note":  meesho_note,
            "source_filter": None,
            "filter_note":  None,
        }

    # Parse dates once; avoid copying the full sales table (was the main latency on large sessions).
    txn_dates = txn_reporting_naive_ist(df0["TxnDate"])

    valid_dt = txn_dates.notna()
    source_filter: Optional[str] = None
    if source and str(source).strip():
        source_filter = str(source).strip()
        src_mask = df0["Source"].astype(str).str.strip() == source_filter
    else:
        src_mask = pd.Series(True, index=df0.index)

    base_mask = valid_dt & src_mask
    if not base_mask.any():
        return {
            "loaded":        bool(source_filter),
            "sku":           sku,
            "all_sizes":     all_sizes,
            "matched_skus":  [],
            "summary":       {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":       [],
            "by_platform":   [],
            "by_size":       [],
            "daily":         [],
            "first_sale":    None,
            "last_sale":     None,
            "meesho_note":   meesho_note,
            "source_filter": source_filter,
            "filter_note":   (
                f"No unified sales rows for channel “{source_filter}”. Try “All channels”."
                if source_filter
                else "No sales rows after date filter."
            ),
        }

    # Default: full loaded history (matches Excel "total sales" exports). Use explicit
    # start_date / end_date query params for a shorter window (e.g. last 90 days).
    if not start_date and not end_date:
        start_ts = txn_dates.loc[base_mask].min()
        end_ts = txn_dates.loc[base_mask].max()
    else:
        start_ts = pd.Timestamp(start_date) if start_date else txn_dates.loc[base_mask].min()
        end_ts = pd.Timestamp(end_date) if end_date else txn_dates.loc[base_mask].max()

    end_inclusive = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    date_mask = (txn_dates >= start_ts) & (txn_dates <= end_inclusive)
    pre_mask = base_mask & date_mask

    sku_df = df0.loc[pre_mask].copy()
    sku_df["TxnDate"] = txn_dates.loc[pre_mask]

    if sku_df.empty:
        return {
            "loaded":       True,
            "sku":          sku,
            "all_sizes":    all_sizes,
            "matched_skus": [],
            "summary":      {"shipped": 0, "returns": 0, "net_units": 0, "return_rate": 0.0, "ads": 0.0},
            "monthly":      [],
            "by_platform":  [],
            "by_size":      [],
            "daily":        [],
            "first_sale":   None,
            "last_sale":    None,
            "meesho_note":  meesho_note,
            "source_filter": source_filter,
            "filter_note":  None,
        }

    qty    = pd.to_numeric(sku_df["Quantity"],       errors="coerce").fillna(0)
    eff    = pd.to_numeric(sku_df["Units_Effective"], errors="coerce").fillna(0)
    txn    = sku_df["Transaction Type"].astype(str).str.strip()
    shipped  = int(qty[txn == "Shipment"].sum())
    returns  = int(qty[txn == "Refund"].sum())
    net_units = int(eff.sum())
    rr       = round(returns / shipped * 100, 1) if shipped > 0 else 0.0
    period_days = max((end_ts - start_ts).days, 1)
    # Active demand days (first→last shipment in filter window, inclusive), matching PO engine.
    ship_dates = sku_df.loc[txn == "Shipment", "TxnDate"]
    if not ship_dates.empty:
        first_ship_ts = ship_dates.min()
        last_ship_ts = ship_dates.max()
        eff_days = max((last_ship_ts - first_ship_ts).days + 1, 7)
        eff_days = min(eff_days, period_days)
    else:
        eff_days = period_days
    ads      = round(shipped / eff_days, 2) if shipped > 0 else 0.0

    # Monthly trend
    sku_df["_month"] = sku_df["TxnDate"].dt.to_period("M").astype(str)
    monthly_raw = (
        sku_df.assign(_qty=qty, _eff=eff)
        .groupby(["_month", "Transaction Type"])
        .agg(units=("_qty", "sum"))
        .reset_index()
        .pivot_table(index="_month", columns="Transaction Type", values="units", fill_value=0)
        .reset_index()
    )
    monthly_raw.columns.name = None
    monthly_raw = monthly_raw.rename(columns={
        "_month":   "month",
        "Shipment": "shipped",
        "Refund":   "returns",
        "Cancel":   "cancels",
    })
    for col in ["shipped", "returns", "cancels"]:
        if col not in monthly_raw.columns:
            monthly_raw[col] = 0
    monthly_raw["net"] = monthly_raw["shipped"] - monthly_raw.get("returns", 0)
    monthly = monthly_raw.sort_values("month")[["month", "shipped", "returns", "cancels", "net"]].to_dict("records")

    # Platform breakdown
    plat_grp = (
        sku_df.assign(_qty=qty)
        .groupby(["Source", "Transaction Type"])
        .agg(units=("_qty", "sum"))
        .reset_index()
        .pivot_table(index="Source", columns="Transaction Type", values="units", fill_value=0)
        .reset_index()
    )
    plat_grp.columns.name = None
    plat_grp = plat_grp.rename(columns={"Shipment": "shipped", "Refund": "returns"})
    if "shipped" not in plat_grp.columns:
        plat_grp["shipped"] = 0
    if "returns" not in plat_grp.columns:
        plat_grp["returns"] = 0
    plat_grp["return_rate"] = (plat_grp["returns"] / plat_grp["shipped"].replace(0, float("nan")) * 100).fillna(0).round(1)
    plat_grp = plat_grp.rename(columns={"Source": "platform"})
    by_platform = plat_grp[["platform", "shipped", "returns", "return_rate"]].sort_values("shipped", ascending=False).to_dict("records")

    # Daily trend (shipments only)
    _ship_m = txn == "Shipment"
    _ship_df = sku_df.loc[_ship_m]
    daily_grp = (
        _ship_df.assign(_qty=qty.loc[_ship_m], _day=_ship_df["TxnDate"].dt.strftime("%Y-%m-%d"))
        .groupby("_day", as_index=False)
        .agg(units=("_qty", "sum"))
        .rename(columns={"_day": "date"})
        .sort_values("date")
    )
    daily = daily_grp.to_dict("records")

    # Sizes breakdown (only meaningful in all_sizes mode)
    matched_skus = sorted(sku_df["Sku"].astype(str).unique().tolist())
    if all_sizes and len(matched_skus) > 1:
        sz_grp = (
            sku_df[txn == "Shipment"]
            .assign(_qty=qty[txn == "Shipment"])
            .groupby("Sku")
            .agg(shipped=("_qty", "sum"))
            .reset_index()
            .sort_values("shipped", ascending=False)
        )
        by_size = sz_grp.rename(columns={"Sku": "sku"}).to_dict("records")
    else:
        by_size = []

    return {
        "loaded":     True,
        "sku":        sku,
        "all_sizes":  all_sizes,
        "matched_skus": matched_skus,
        "start_date": str(start_ts.date()),
        "end_date":   str(end_ts.date()),
        "summary": {
            "shipped":     shipped,
            "returns":     returns,
            "net_units":   net_units,
            "return_rate": rr,
            "ads":         ads,
        },
        "monthly":     monthly,
        "by_platform": by_platform,
        "by_size":     by_size,
        "daily":       daily,
        "first_sale":  str(sku_df["TxnDate"].min().date()),
        "last_sale":   str(sku_df["TxnDate"].max().date()),
        "meesho_note": meesho_note,
        "source_filter": source_filter,
        "filter_note":  (
            f"Showing {source_filter} only — totals exclude other marketplaces."
            if source_filter
            else None
        ),
    }


# ── Daily Breakdown ───────────────────────────────────────────

@router.get("/daily-breakdown")
def daily_breakdown(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platform: Optional[str] = None,   # comma-sep list, e.g. "Amazon,Meesho"
):
    """
    Per-day shipment/refund counts broken down by platform.
    Returns [{date, platform, units, returns}] sorted by date.
    """
    import pandas as pd
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    df = apply_upload_report_day_gate(sess.sales_df.copy())
    if df.empty:
        return []

    try:
        d = df.copy()
        d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
        d = d.dropna(subset=["TxnDate"])
        if start_date or end_date:
            d = _filter_by_reporting_days(d, "TxnDate", start_date, end_date)
        if platform:
            plats = [p.strip() for p in platform.split(",")]
            d = d[d["Source"].isin(plats)]

        if d.empty:
            return []

        d["_day"] = d["TxnDate"].dt.strftime("%Y-%m-%d")
        qty = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0)
        ship_mask   = d["Transaction Type"].astype(str).str.strip() == "Shipment"
        refund_mask = d["Transaction Type"].astype(str).str.strip() == "Refund"

        grp = (
            d.assign(_qty=qty)
            .groupby(["_day", "Source"])
            .apply(lambda g: pd.Series({
                "units":   int(g.loc[ship_mask.loc[g.index], "_qty"].sum()),
                "returns": int(g.loc[refund_mask.loc[g.index], "_qty"].sum()),
            }))
            .reset_index()
            .rename(columns={"_day": "date", "Source": "platform"})
            .sort_values("date")
        )
        return grp.to_dict("records")
    except Exception:
        return []


def _resolve_daily_dsr_date(sess: AppSession, date: Optional[str]) -> tuple:
    """Return (sales_df, iso_date_str) for DSR helpers."""
    import pandas as pd

    from ..services.sales import apply_upload_report_day_gate, txn_reporting_naive_ist

    iso = (date or "").strip()
    df = apply_upload_report_day_gate(sess.sales_df.copy())

    if iso and len(iso) >= 10:
        iso = iso[:10]
        needs_tier3 = False
        if df.empty:
            needs_tier3 = True
        elif "TxnDate" in df.columns:
            d = df.copy()
            d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
            d = d.dropna(subset=["TxnDate"])
            try:
                day = pd.Timestamp(iso).normalize()
                if d.empty or not (d["TxnDate"].dt.normalize() == day).any():
                    needs_tier3 = True
            except Exception:
                pass
        if needs_tier3:
            try:
                from ..services.daily_store import platforms_with_uploads_in_range

                plats = platforms_with_uploads_in_range(iso, iso)
                if plats:
                    frames = _load_tier3_frames_for_platforms(plats, iso, iso, dedup=False)
                    if frames:
                        tier3_sales = _build_sales_from_tier3_frames(sess, frames)
                        if tier3_sales is not None and not tier3_sales.empty:
                            df = apply_upload_report_day_gate(tier3_sales)
            except Exception:
                logging.getLogger(__name__).exception("daily-dsr tier3 fallback for %s", iso)

    if df.empty:
        return df, iso

    if not iso:
        d = df.copy()
        d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
        d = d.dropna(subset=["TxnDate"])
        if d.empty:
            return df, ""
        latest = d["TxnDate"].max()
        if pd.isna(latest):
            return df, ""
        iso = str(latest.normalize().date())[:10]

    return df, iso


@router.get("/daily-dsr")
def daily_dsr(request: Request, date: Optional[str] = None):
    """
    Daily DSR-style report for one calendar day: marketplace sections with optional
    segment rows (Flipkart Brand, Snapdeal Company, etc.) and an Others bucket.
    Query: ``date`` = ISO ``YYYY-MM-DD`` (defaults to latest day with sales if omitted).
    """
    sess = _sess(request)
    _ensure_intelligence_session_fresh(sess)
    df, iso = _resolve_daily_dsr_date(sess, date)
    return get_daily_dsr_report(df, iso)


@router.get("/daily-dsr-export")
def daily_dsr_export(request: Request, date: Optional[str] = None):
    """CSV download matching the on-screen Daily DSR table."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    df, iso = _resolve_daily_dsr_date(sess, date)
    report = get_daily_dsr_report(df, iso)
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in daily_dsr_report_to_csv_rows(report):
        w.writerow(row)
    body = buf.getvalue().encode("utf-8")
    fname = f"daily-dsr_{report.get('date') or 'nodate'}.csv"
    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/dsr-brand-monthly")
def dsr_brand_monthly(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """YG vs Akiko shipment units by calendar month (DSR segment / brand labels)."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    return get_dsr_brand_monthly_comparison(sess.sales_df, start_date, end_date)


@router.get("/dsr-brand-monthly-export")
def dsr_brand_monthly_export(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """CSV of YG vs Akiko monthly comparison (same logic as ``/dsr-brand-monthly``)."""
    sess = _sess(request)
    _restore_daily_if_needed(sess)
    result = get_dsr_brand_monthly_comparison(sess.sales_df, start_date, end_date)
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in dsr_brand_monthly_to_csv_rows(result):
        w.writerow(row)
    body = buf.getvalue().encode("utf-8")
    part = f"{start_date or 'all'}_{end_date or 'all'}".replace("/", "-")
    fname = f"dsr-yg-akiko-monthly_{part}.csv"
    return StreamingResponse(
        iter([body]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _patch_analytics_monthly_returns(
    monthly_records: list,
    sess: AppSession,
    platform_key: str,
    *,
    display_name: str,
) -> None:
    """Merge PO return overlay + unified sales RETURN_SHEET refunds into platform analytics."""
    from ..services.sales import (
        RETURN_PLATFORM_TO_SOURCE,
        overlay_refunds_by_calendar_month,
        _return_sheet_refund_buckets_for_platform,
    )

    from ..services.po_return_import import aggregate_return_overlay_for_use

    overlay = aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None))
    fallback = str(getattr(sess, "return_overlay_as_of", "") or "")[:10] or None
    ov_by_month = overlay_refunds_by_calendar_month(
        overlay,
        platform_key,
        fallback_as_of=fallback,
        sales_df=None,
        subtract_sales_overlay=False,
    )
    display = RETURN_PLATFORM_TO_SOURCE.get(platform_key, display_name)
    sales_df = getattr(sess, "sales_df", None)
    _, sheet_by_month, _ = _return_sheet_refund_buckets_for_platform(
        sales_df, display, None, None
    )
    for m, extra in sheet_by_month.items():
        cur = int(ov_by_month.get(m, 0) or 0)
        if int(extra) > cur:
            ov_by_month[m] = int(extra)
    month_key = "Month"

    def _month_of(row: dict) -> str:
        return str(row.get(month_key) or row.get("month") or "")

    for row in monthly_records:
        m = _month_of(row)
        extra = int(ov_by_month.get(m, 0) or 0)
        if extra <= 0:
            continue
        row["refunds"] = int(row.get("refunds") or 0) + extra
        row["net"] = int(row.get("shipments") or 0) - int(row["refunds"] or 0)

    known = {_month_of(r) for r in monthly_records}
    for m, extra in ov_by_month.items():
        if extra <= 0 or m in known:
            continue
        monthly_records.append(
            {
                month_key: m,
                "shipments": 0,
                "refunds": int(extra),
                "net": -int(extra),
            }
        )
        known.add(m)


def _ensure_platform_analytics_frame(sess: AppSession, platform: str, attr: str):
    """Top up session platform frame from Tier-3 when warm-cache copy was trimmed."""
    import pandas as pd

    from ..services.daily_store import get_summary, load_platform_data, merge_platform_data

    cur = getattr(sess, attr, None)
    if cur is None or not hasattr(cur, "empty"):
        cur = pd.DataFrame()
    summary = get_summary().get(platform) or {}
    if int(summary.get("file_count") or 0) <= 0:
        return cur

    tier_min = str(summary.get("min_date") or "")[:10]
    tier_max = str(summary.get("max_date") or "")[:10]
    sess_min = _platform_df_min_iso(cur) or ""
    sess_max = _platform_df_max_iso(cur) or ""
    needs_topup = (
        cur.empty
        or (tier_min and (not sess_min or tier_min < sess_min))
        or (tier_max and (not sess_max or tier_max > sess_max))
    )
    if not needs_topup:
        return cur

    full = load_platform_data(platform, months=None, dedup=False)
    if full.empty:
        return cur
    merged = merge_platform_data(cur, full, platform)
    setattr(sess, attr, merged)
    return merged


def _annotate_partial_calendar_months(monthly_records: list, max_date) -> None:
    """Mark the trailing calendar month when data does not run through month-end."""
    import pandas as pd

    if not monthly_records or max_date is None or pd.isna(max_date):
        return
    end = pd.Timestamp(max_date).normalize()
    month_key = "Month" if monthly_records and "Month" in monthly_records[0] else "month"
    last_period = str(end.to_period("M"))
    month_end = (end.to_period("M").to_timestamp("M")).normalize()
    if end >= month_end:
        return
    for row in monthly_records:
        m = str(row.get(month_key) or row.get("month") or "")
        if m == last_period:
            row["partial"] = True
            row["partial_note"] = f"Through {end.date().isoformat()} only"
            break


def _finalize_platform_analytics_monthly(
    sess: AppSession,
    monthly,
    platform_key: str,
    *,
    display_name: str,
) -> tuple[list, int, int, int]:
    """Monthly records + totals after overlay / unified Refund merge."""
    import pandas as pd

    frame = monthly.copy()
    if "shipments" not in frame.columns:
        frame["shipments"] = 0
    if "refunds" not in frame.columns:
        frame["refunds"] = 0
    frame["refunds"] = pd.to_numeric(frame["refunds"], errors="coerce").fillna(0).abs()
    frame["shipments"] = pd.to_numeric(frame["shipments"], errors="coerce").fillna(0)
    frame["net"] = frame["shipments"] - frame.get("refunds", 0)
    records = frame.to_dict("records")
    _patch_analytics_monthly_returns(
        records, sess, platform_key, display_name=display_name
    )
    returned = sum(int(r.get("refunds") or 0) for r in records)
    shipped = sum(int(r.get("shipments") or 0) for r in records)
    return records, shipped, returned, shipped - returned


# ── MTR Analytics ─────────────────────────────────────────────

@router.get("/mtr-analytics")
def mtr_analytics(request: Request):
    sess = _sess(request)
    _ensure_return_overlay_hydrated(sess)
    df = sess.mtr_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd
    import numpy as np

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Monthly shipments vs refunds
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby(["Month", "Transaction_Type"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="Transaction_Type", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={
        "Shipment": "shipments",
        "Refund":   "refunds",
        "Cancel":   "cancels",
    })

    # Top SKUs
    top = (
        df[df["Transaction_Type"] == "Shipment"]
        .groupby("SKU")["Quantity"].sum()
        .sort_values(ascending=False).head(20).reset_index()
    )
    top.columns = ["sku", "units"]

    # Summary
    shipped  = float(df[df["Transaction_Type"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["Transaction_Type"] == "Refund"]["Quantity"].abs().sum())
    net_units = int(shipped - returned)

    if "shipments" in monthly.columns and "refunds" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly["refunds"]
    elif "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"]

    monthly_records = monthly.to_dict("records")
    _patch_analytics_monthly_returns(
        monthly_records,
        sess,
        "amazon",
        display_name="Amazon",
    )
    returned = int(
        sum(int(r.get("refunds") or 0) for r in monthly_records)
    )
    shipped = int(
        sum(int(r.get("shipments") or 0) for r in monthly_records)
    )
    net_units = shipped - returned

    return {
        "loaded":       True,
        "rows":         len(df),
        "date_range":   [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":      shipped,
        "returned":     returned,
        "net_units":    net_units,
        "return_rate":  round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":      monthly_records,
        "top_skus":     top.to_dict("records"),
    }


# ── Myntra Analytics ─────────────────────────────────────────

@router.get("/myntra-analytics")
def myntra_analytics(request: Request):
    sess = _sess(request)
    _ensure_return_overlay_hydrated(sess)
    df = _ensure_platform_analytics_frame(sess, "myntra", "myntra_df")
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    monthly_records, shipped, returned, net_units = _finalize_platform_analytics_monthly(
        sess, monthly, "myntra", display_name="Myntra"
    )
    _annotate_partial_calendar_months(monthly_records, df["Date"].max())

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     shipped,
        "returned":    returned,
        "net_units":   net_units,
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly_records,
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Meesho Analytics ─────────────────────────────────────────

@router.get("/meesho-analytics")
def meesho_analytics(request: Request):
    sess = _sess(request)
    _ensure_return_overlay_hydrated(sess)
    df = _ensure_platform_analytics_frame(sess, "meesho", "meesho_df")
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    monthly_records, shipped, returned, net_units = _finalize_platform_analytics_monthly(
        sess, monthly, "meesho", display_name="Meesho"
    )
    _annotate_partial_calendar_months(monthly_records, df["Date"].max())

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     shipped,
        "returned":    returned,
        "net_units":   net_units,
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly_records,
        "by_state":    by_state.to_dict("records"),
    }


# ── Flipkart Analytics ────────────────────────────────────────

@router.get("/flipkart-analytics")
def flipkart_analytics(request: Request):
    sess = _sess(request)
    _ensure_return_overlay_hydrated(sess)
    df = _ensure_platform_analytics_frame(sess, "flipkart", "flipkart_df")
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]

    monthly_records, shipped, returned, net_units = _finalize_platform_analytics_monthly(
        sess, monthly, "flipkart", display_name="Flipkart"
    )
    _annotate_partial_calendar_months(monthly_records, df["Date"].max())

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     shipped,
        "returned":    returned,
        "net_units":   net_units,
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly_records,
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Inventory ─────────────────────────────────────────────────

@router.get("/inventory")
def get_inventory(
    request: Request,
    search: Optional[str] = None,
    offset: int = 0,
    limit: int = 500,
):
    sess = _sess(request)
    import pandas as pd

    inv_status = getattr(sess, "inventory_upload_status", "idle") or "idle"
    if inv_status == "running":
        return {
            "loaded": False,
            "upload_in_progress": True,
            "inventory_upload_status": "running",
            "inventory_upload_message": getattr(sess, "inventory_upload_message", "") or "",
            "inventory_upload_progress": int(getattr(sess, "inventory_upload_progress", 0) or 0),
            "rows": [],
            "total_rows": 0,
            "offset": 0,
            "limit": max(1, min(int(limit), 5000)),
        }

    _restore_inventory_from_warm(sess)
    df = sess.inventory_df_variant
    if df.empty:
        return {
            "loaded": False,
            "rows": [],
            "total_rows": 0,
            "offset": 0,
            "limit": max(1, min(int(limit), 5000)),
        }

    ensure_inventory_snapshot_metadata(sess)
    cols = [c for c in df.columns if c != "OMS_SKU"]

    totals = getattr(sess, "inventory_api_totals", None) or {}
    if not totals or set(totals.keys()) != set(c for c in cols):
        refresh_inventory_api_cache(sess)
        totals = getattr(sess, "inventory_api_totals", None) or {}

    rows, total_rows = inventory_rows_for_api(
        df, search=search or "", offset=offset, limit=limit
    )
    dbg = getattr(sess, "inventory_debug", {}) or {}
    marketplaces = getattr(sess, "inventory_api_marketplaces", None)
    if not marketplaces:
        marketplaces = inventory_marketplace_breakdown(df, dbg)
    upload_warnings: list[str] = []
    inv_result = getattr(sess, "inventory_upload_result", None) or {}
    if isinstance(inv_result, dict):
        upload_warnings = list(inv_result.get("warnings") or [])
    return {
        "loaded": True,
        "rows": rows,
        "total_rows": total_rows,
        "offset": max(0, int(offset)),
        "limit": max(1, min(int(limit), 5000)),
        "columns": ["OMS_SKU"] + cols,
        "totals": totals,
        "debug": dbg,
        "marketplaces": marketplaces,
        "missing_marketplace_hints": inventory_missing_marketplace_warnings(dbg),
        "inventory_upload_warnings": upload_warnings,
        "manual_intransit_loaded": not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty,
        "manual_intransit_units": int(
            pd.to_numeric(
                getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).get("Manual_InTransit"),
                errors="coerce",
            ).fillna(0).sum()
        )
        if not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty
        else 0,
        "manual_not_in_inventory_units": int(
            pd.to_numeric(
                getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).get("Not_In_Inventory_Qty"),
                errors="coerce",
            ).fillna(0).sum()
        )
        if not getattr(sess, "manual_intransit_overlay_df", pd.DataFrame()).empty
        else 0,
        "manual_intransit_parse_report": getattr(sess, "manual_intransit_parse_report", None) or None,
        **inventory_snapshot_meta_for_api(sess),
    }


# ── Snapdeal Analytics ────────────────────────────────────────

@router.get("/snapdeal-analytics")
def snapdeal_analytics(request: Request, company: Optional[str] = None):
    sess = _sess(request)
    _ensure_return_overlay_hydrated(sess)
    df = _ensure_platform_analytics_frame(sess, "snapdeal", "snapdeal_df")
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # Collect unique companies before filtering
    companies: list = []
    if "Company" in df.columns:
        companies = sorted(df["Company"].dropna().str.strip().unique().tolist())
        companies = [c for c in companies if c]

    # Apply company filter
    if company and "Company" in df.columns:
        df = df[df["Company"].str.strip() == company.strip()]

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})

    top_skus = (
        df[df["TxnType"] == "Shipment"].groupby("OMS_SKU")["Quantity"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top_skus.columns = ["sku", "units"]

    by_state = (
        df[df["TxnType"] == "Shipment"].groupby("State")["Quantity"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    by_state.columns = ["state", "units"]
    by_state = by_state[by_state["state"].str.strip() != ""]

    monthly_records, shipped, returned, net_units = _finalize_platform_analytics_monthly(
        sess, monthly, "snapdeal", display_name="Snapdeal"
    )
    _annotate_partial_calendar_months(monthly_records, df["Date"].max())

    return {
        "loaded":      True,
        "rows":        len(df),
        "companies":   companies,
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     shipped,
        "returned":    returned,
        "net_units":   net_units,
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly_records,
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Snapdeal Debug (column inspection) ───────────────────────

@router.get("/snapdeal-debug")
def snapdeal_debug(request: Request):
    """Returns column names, TxnType distribution, and SKU sample from the loaded snapdeal_df."""
    sess = _sess(request)
    df = sess.snapdeal_df
    if df.empty:
        return {"loaded": False}
    return {
        "loaded":       True,
        "rows":         len(df),
        "txn_types":    df["TxnType"].value_counts().to_dict(),
        "sku_sample":   df["OMS_SKU"].value_counts().head(15).to_dict(),
        "state_sample": df["State"].value_counts().head(10).to_dict(),
        "parse_info":   sess.snapdeal_parse_info,   # raw cols + detected fields per file
        "sample_rows":  df.head(3).fillna("").to_dict("records"),
    }


# ── Daily Sales Management ───────────────────────────────────

@router.get("/daily-summary")
def daily_summary(_request: Request):
    """Per-platform summary of persisted daily uploads."""
    return get_summary()


@router.get("/daily-uploads")
def daily_uploads(_request: Request):
    """Full list of persisted daily upload records (newest first)."""
    return list_uploads()


@router.delete("/daily-uploads/{upload_id}")
def delete_daily_upload(upload_id: int, request: Request):
    from ..services.upload_policy import _DELETE_DENIED_MSG, may_delete_upload_data

    auth = getattr(request.state, "auth", None) or {}
    if not may_delete_upload_data(str(auth.get("role") or ""), str(auth.get("sub") or "")):
        raise HTTPException(status_code=403, detail=_DELETE_DENIED_MSG)
    ok = delete_upload(upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"ok": True, "message": f"Deleted upload {upload_id}"}


# ── Data Debug / Coverage ────────────────────────────────────

@router.get("/coverage/debug")
def coverage_debug_endpoint(request: Request):
    """Structured source/session/warm-cache diagnostics (no SSH)."""
    from ..services.coverage_debug import build_coverage_debug

    return build_coverage_debug(_sess(request))


@router.get("/debug-coverage")
def debug_coverage(request: Request):
    """
    Returns row counts, date ranges, and sample transaction types
    for each loaded DataFrame. Useful for diagnosing data integrity
    issues on production without redeploying.
    """
    import pandas as pd
    sess = _sess(request)

    def _df_info(df: pd.DataFrame, date_col: str, txn_col: str | None = None) -> dict:
        if df.empty:
            return {"loaded": False, "rows": 0}
        out: dict = {"loaded": True, "rows": len(df)}
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                out["min_date"] = str(dates.min().date())
                out["max_date"] = str(dates.max().date())
                out["tz_aware"] = dates.dt.tz is not None
        except Exception as e:
            out["date_error"] = str(e)
        if txn_col and txn_col in df.columns:
            out["txn_type_counts"] = df[txn_col].astype(str).value_counts().head(10).to_dict()
        return out

    from backend.main import _warm_cache, _warm_cache_loaded_at  # type: ignore
    return {
        "session": {
            "mtr_df":      _df_info(sess.mtr_df,      "Date", "Transaction_Type"),
            "myntra_df":   _df_info(sess.myntra_df,   "Date", "TxnType"),
            "meesho_df":   {**_df_info(sess.meesho_df, "Date", "TxnType"),
                            "columns": list(sess.meesho_df.columns) if not sess.meesho_df.empty else [],
                            "sku_sample": sess.meesho_df["SKU"].dropna().unique()[:5].tolist()
                                          if not sess.meesho_df.empty and "SKU" in sess.meesho_df.columns else "NO SKU COLUMN"},
            "flipkart_df": _df_info(sess.flipkart_df, "Date", "TxnType"),
            "snapdeal_df": _df_info(sess.snapdeal_df, "Date", "TxnType"),
            "sales_df":    {**_df_info(sess.sales_df, "TxnDate", "Transaction Type"),
                            "meesho_skus": sess.sales_df[sess.sales_df["Source"].astype(str) == "Meesho"]["Sku"]
                                           .value_counts().head(5).to_dict()
                                           if not sess.sales_df.empty and "Source" in sess.sales_df.columns else {}},
            "sku_mapping_len": len(sess.sku_mapping),
        },
        "warm_cache": {
            "loaded_at": _warm_cache_loaded_at.isoformat() if _warm_cache_loaded_at else None,
            "keys":      list(_warm_cache.keys()),
        },
    }


# ── AI Dashboard Endpoints ────────────────────────────────────

@router.get("/platform-summary")
def platform_summary(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _ensure_intelligence_session_fresh(sess)
    return get_platform_summary(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
        start_date=start_date, end_date=end_date,
        sales_df=sess.sales_df,
    )


@router.get("/anomalies")
def anomalies_endpoint(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    sess = _sess(request)
    _ensure_intelligence_session_fresh(sess)
    return get_anomalies(
        sess.mtr_df, sess.myntra_df, sess.meesho_df,
        sess.flipkart_df, sess.snapdeal_df,
        sess.inventory_df_variant,
        apply_upload_report_day_gate(sess.sales_df.copy()),
        start_date=start_date,
        end_date=end_date,
    )


# ── Quarterly History (for PO Engine) ────────────────────────

@router.get("/quarterly-history")
def quarterly_history(request: Request, group_by_parent: bool = False, n_quarters: int = 8):
    sess = _sess(request)
    if sess.sales_df.empty and sess.mtr_df.empty:
        return {"loaded": False, "rows": []}

    from ..services.po_quarterly_warmup import build_quarterly_payload

    payload = build_quarterly_payload(
        sess, group_by_parent=group_by_parent, n_quarters=n_quarters
    )
    if not payload.get("loaded"):
        return {"loaded": False, "rows": []}
    return payload

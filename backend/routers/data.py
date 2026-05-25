"""
Data query router — analytics endpoints.
GET /api/data/coverage, sales-summary, sales-export, sales-by-source, daily-dsr, daily-dsr-export,
dsr-brand-monthly, dsr-brand-monthly-export, top-skus,
mtr-analytics, myntra-analytics, meesho-analytics, flipkart-analytics, inventory
"""
import csv
import io
import os
import re
from typing import List, Optional, Set
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from ..models.schemas import CoverageResponse
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
)

router = APIRouter()


def _sku_deepdive_aliases(raw: str) -> Set[str]:
    """Return SKU tokens that should match the same row after PL/YK normalisation."""
    u = raw.strip().upper()
    out = {u, canonical_sales_sku(u)}
    m = re.match(r"^(\d+)(YK[A-Z0-9\-]+)$", u)
    if m and "PL" not in m.group(0):
        out.add(f"{m.group(1)}PL{m.group(2)}")
    return {x for x in out if x and x != "NAN"}


def _sess(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


def _restore_daily_if_needed(sess: AppSession) -> None:
    """
    On first coverage check per session, merge any persisted daily SQLite data
    into the session platform DFs (fills gaps after restart and folds Tier-3
    rows into warm-cache copies without replacing bulk history).
    Also auto-restores SKU mapping from GitHub cache if missing.
    """
    if getattr(sess, "daily_inventory_upload_status", "idle") == "running":
        return
    if getattr(sess, "inventory_upload_status", "idle") == "running":
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
    if getattr(sess, "pause_auto_data_restore", False) and needs_data:
        return

    if getattr(sess, "pause_auto_data_restore", False) and not needs_data:
        return
    needs_tier3_refresh = _tier3_session_needs_topup(sess) or _session_sales_stale_vs_platforms(sess)
    if sess.daily_restored and not needs_data:
        if not needs_tier3_refresh:
            return
    if needs_data or needs_tier3_refresh:
        sess.daily_restored = False

    # Never block coverage/login behind a multi-minute Tier-3 upload on the same lock.
    if not sess._daily_restore_lock.acquire(blocking=False):
        return

    try:
        if sess.daily_restored and not needs_tier3_refresh:
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

        changed = False
        platform_attrs = [
            ("amazon",   "mtr_df"),
            ("myntra",   "myntra_df"),
            ("meesho",   "meesho_df"),
            ("flipkart", "flipkart_df"),
            ("snapdeal", "snapdeal_df"),
        ]
        for platform, attr in platform_attrs:
            # Merge Tier-3 SQLite into whatever the session already holds (warm cache,
            # prior uploads). Replacing only when empty allowed a small SQLite slice to
            # leave stale partial session data from an earlier bug path.
            df = load_platform_data(
                platform,
                months=_auto_months,
                dedup=False,
                max_files=_auto_max_files,
            )
            if not df.empty:
                cur = getattr(sess, attr)
                setattr(sess, attr, merge_platform_data(cur, df, platform))
                changed = True

        # Safety net: if bounded restore found nothing, do one full-history pass.
        # This prevents "all data missing" after restart when the useful history sits
        # outside the bounded month/file window.
        if not changed:
            for _p, _a in platform_attrs:
                if not getattr(sess, _a).empty:
                    changed = True
                    break
        if not changed:
            for platform, attr in platform_attrs:
                if getattr(sess, attr).empty:
                    df = load_platform_data(
                        platform,
                        months=None,
                        dedup=False,
                        max_files=None,
                    )
                    if not df.empty:
                        cur = getattr(sess, attr)
                        setattr(sess, attr, merge_platform_data(cur, df, platform))
                        changed = True

        if changed:
            try:
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                    return_overlay_df=(
                        None
                        if getattr(sess, "po_return_overlay_df", None) is None
                        or getattr(sess.po_return_overlay_df, "empty", True)
                        else sess.po_return_overlay_df
                    ),
                )
                sess._quarterly_cache.clear()
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
        sess._daily_restore_lock.release()


_PLATFORM_ATTRS = (
    ("amazon", "mtr_df"),
    ("myntra", "myntra_df"),
    ("meesho", "meesho_df"),
    ("flipkart", "flipkart_df"),
    ("snapdeal", "snapdeal_df"),
)

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


def _tier3_session_needs_topup(sess: AppSession) -> bool:
    """True when SQLite has newer/more daily uploads than the in-memory platform frame."""
    try:
        from ..services.daily_store import get_summary

        summary = get_summary()
    except Exception:
        return False
    for plat, attr in _PLATFORM_ATTRS:
        plat_sum = summary.get(plat) or {}
        if int(plat_sum.get("file_count") or 0) <= 0:
            continue
        cur = getattr(sess, attr, None)
        if cur is None or not hasattr(cur, "empty") or cur.empty:
            return True
        tier_max = str(plat_sum.get("max_date") or "")[:10]
        sess_max = _platform_df_max_iso(cur) or ""
        if tier_max and (not sess_max or tier_max > sess_max):
            return True
    return False


def _session_sales_stale_vs_platforms(sess: AppSession) -> bool:
    """True when a non-empty platform frame is missing from unified sales_df."""
    sales = getattr(sess, "sales_df", None)
    sources: set[str] = set()
    if sales is not None and not sales.empty and "Source" in sales.columns:
        sources = set(sales["Source"].astype(str).str.strip())
    for attr, src in _SOURCE_BY_ATTR.items():
        raw = getattr(sess, attr, None)
        if raw is not None and hasattr(raw, "empty") and not raw.empty and src not in sources:
            return True
    return sales is None or (hasattr(sales, "empty") and sales.empty and _session_has_platform_data(sess))


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
            return_overlay_df=(
                None
                if getattr(sess, "po_return_overlay_df", None) is None
                or getattr(sess.po_return_overlay_df, "empty", True)
                else sess.po_return_overlay_df
            ),
        )
        sess._quarterly_cache.clear()
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
    return changed


def _platforms_needing_tier3_topup(sess: AppSession) -> list[str]:
    out: list[str] = []
    try:
        from ..services.daily_store import get_summary

        summary = get_summary()
    except Exception:
        return out
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
        if tier_max and (not sess_max or tier_max > sess_max):
            out.append(plat)
    return out


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
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
    except Exception:
        pass

    if _session_sales_stale_vs_platforms(sess):
        _rebuild_session_sales(sess)

    need_plat = _platforms_needing_tier3_topup(sess)
    if need_plat and sess._daily_restore_lock.acquire(blocking=False):
        try:
            if _merge_tier3_light(sess, only_platforms=need_plat):
                _rebuild_session_sales(sess)
        finally:
            sess._daily_restore_lock.release()
    else:
        _ensure_sales_rebuilt(sess)


def _ensure_warm_session_data(sess: AppSession) -> None:
    """Warm cache fill for empty sessions; Intelligence routes use ``_ensure_intelligence_session_fresh``."""
    try:
        import backend.main as _main

        if _main.session_needs_operational_data(sess):
            _main.force_restore_session_from_server_cache(sess, _main._warm_cache_generation)
    except Exception:
        pass


def _restore_inventory_from_warm(sess: AppSession) -> None:
    """Copy snapshot inventory from in-memory warm cache (fast; no GitHub download)."""
    try:
        import backend.main as _main
        import pandas as pd
        from ..services.inventory import apply_inventory_session_meta

        if not _main._warm_cache:
            return
        if sess.inventory_df_variant.empty:
            for key in ("inventory_df_variant", "inventory_df_parent"):
                val = _main._warm_cache.get(key)
                if val is not None and not (isinstance(val, pd.DataFrame) and val.empty):
                    setattr(sess, key, val)
        meta = _main._warm_cache.get(getattr(_main, "_INVENTORY_META_WARM_KEY", "inventory_session_meta"))
        if meta:
            apply_inventory_session_meta(sess, meta)
        if not sess.sku_mapping and _main._warm_cache.get("sku_mapping"):
            sess.sku_mapping = _main._warm_cache["sku_mapping"]
        ensure_inventory_snapshot_metadata(sess)
    except Exception:
        pass


def _session_has_platform_data(sess: AppSession) -> bool:
    return any(
        not getattr(sess, attr).empty
        for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df")
    )


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
            return_overlay_df=(
                None
                if getattr(sess, "po_return_overlay_df", None) is None
                or getattr(sess.po_return_overlay_df, "empty", True)
                else sess.po_return_overlay_df
            ),
        )
        sess._quarterly_cache.clear()
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


# ── Coverage ──────────────────────────────────────────────────

@router.get("/coverage", response_model=CoverageResponse)
def get_coverage(request: Request, light: bool = False):
    """Session coverage flags. ``light=1`` skips SQLite restore / sales rebuild (fast after PO uploads)."""
    sess = _sess(request)
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
    if not light:
        if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
            light = True
        if getattr(sess, "sales_rebuild_status", "idle") == "running":
            light = True
        if getattr(sess, "inventory_upload_status", "idle") == "running":
            light = True
        if getattr(sess, "daily_inventory_upload_status", "idle") == "running":
            light = True
    if not light:
        try:
            from ..services.po_raise_import import hydrate_session_ledger_from_db

            hydrate_session_ledger_from_db(sess, lookback_days=30)
        except Exception:
            pass
        _restore_daily_if_needed(sess)   # auto-load persisted daily data on first access
    else:
        _maybe_restore_daily_for_empty_sales(sess)
    _ensure_sales_rebuilt(sess)
    paused = getattr(sess, "pause_auto_data_restore", False)
    from ..services.daily_store import get_summary

    tier3_any = bool(get_summary())
    _po_ledger = getattr(sess, "po_raise_ledger_df", None)
    _po_ledger_ok = _po_ledger is not None and not getattr(_po_ledger, "empty", True)
    _ret_ov = getattr(sess, "po_return_overlay_df", None)
    _ret_ok = _ret_ov is not None and not getattr(_ret_ov, "empty", True)
    _ingest = getattr(sess, "daily_auto_ingest_result", None) or {}
    _has_ingest = bool(_ingest)
    return CoverageResponse(
        sku_mapping=bool(sess.sku_mapping),
        mtr=not sess.mtr_df.empty,
        sales=not sess.sales_df.empty,
        myntra=not sess.myntra_df.empty,
        meesho=not sess.meesho_df.empty,
        flipkart=not sess.flipkart_df.empty,
        snapdeal=not sess.snapdeal_df.empty,
        inventory=not sess.inventory_df_variant.empty,
        daily_orders=len(sess.daily_sales_sources) > 0 or tier3_any,
        existing_po=not sess.existing_po_df.empty,
        sku_status_lead=not sess.sku_status_lead_df.empty,
        daily_inventory_history=not sess.daily_inventory_history_df.empty,
        po_raise_ledger=bool(_po_ledger_ok),
        return_sheet=bool(_ret_ok),
        mtr_rows=len(sess.mtr_df),
        sales_rows=len(sess.sales_df),
        myntra_rows=len(sess.myntra_df),
        meesho_rows=len(sess.meesho_df),
        flipkart_rows=len(sess.flipkart_df),
        snapdeal_rows=len(sess.snapdeal_df),
        sku_status_lead_rows=int(len(sess.sku_status_lead_df)),
        daily_inventory_history_rows=int(len(sess.daily_inventory_history_df)),
        daily_inventory_history_skus=(
            int(sess.daily_inventory_history_df["OMS_SKU"].nunique())
            if not sess.daily_inventory_history_df.empty
            else 0
        ),
        po_raise_ledger_rows=int(len(_po_ledger)) if _po_ledger_ok else 0,
        return_sheet_skus=int(len(_ret_ov)) if _ret_ok else 0,
        pause_auto_data_restore=paused,
        sales_rebuild=getattr(sess, "sales_rebuild_status", "idle") or "idle",
        sales_rebuild_message=getattr(sess, "sales_rebuild_message", "") or "",
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
        daily_inventory_upload_status=getattr(sess, "daily_inventory_upload_status", "idle") or "idle",
        daily_inventory_upload_message=getattr(sess, "daily_inventory_upload_message", "") or "",
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

    sess = _sess(request)
    df0 = apply_upload_report_day_gate(sess.sales_df.copy())

    # Detect whether Meesho is loaded but has no per-SKU data (TCS ZIP format)
    meesho_note: str | None = None
    if not sess.meesho_df.empty:
        meesho_skus_in_sales = (
            sess.sales_df[sess.sales_df["Source"].astype(str) == "Meesho"]["Sku"]
            .dropna().unique().tolist()
            if not sess.sales_df.empty and "Source" in sess.sales_df.columns else []
        )
        if meesho_skus_in_sales == ["MEESHO_TOTAL"] or set(meesho_skus_in_sales) == {"MEESHO_TOTAL"}:
            meesho_total_units = int(
                sess.sales_df[
                    (sess.sales_df["Source"].astype(str) == "Meesho") &
                    (sess.sales_df["Transaction Type"].astype(str) == "Shipment")
                ]["Quantity"].sum()
            ) if not sess.sales_df.empty else 0
            meesho_note = (
                f"Meesho data loaded ({meesho_total_units:,} total units) but your uploaded "
                f"Meesho TCS ZIP reports don't include per-SKU data. "
                f"To see Meesho in SKU breakdown, upload the Meesho Order Report CSV "
                f"(Supplier Panel → Reports → Order Reports)."
            )

    if df0.empty:
        return {"loaded": False, "message": "No sales data loaded"}

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

    sku_variants = _sku_deepdive_aliases(sku)
    if all_sizes:
        parent_targets = {
            canonical_sales_sku(str(get_parent_sku(s)).strip()) for s in sku_variants
        }
        sub_skus = df0.loc[pre_mask, "Sku"].astype(str)
        uniq = sub_skus.unique()
        parent_map = {u: canonical_sales_sku(str(get_parent_sku(u)).strip()) for u in uniq}
        sub_parents = sub_skus.map(parent_map)
        sku_match_sub = sub_parents.isin(parent_targets)
        sku_mask = pd.Series(False, index=df0.index)
        sku_mask.loc[pre_mask] = sku_match_sub.values
    else:
        targets = {canonical_sales_sku(s) for s in sku_variants}
        sku_mask = canonical_sales_sku_series(df0["Sku"]).isin(targets)

    final_mask = pre_mask & sku_mask
    sku_df = df0.loc[final_mask].copy()
    sku_df["TxnDate"] = txn_dates.loc[final_mask]

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

    df = apply_upload_report_day_gate(sess.sales_df.copy())
    if df.empty:
        return df, (date or "").strip()

    if not date or not str(date).strip():
        d = df.copy()
        d["TxnDate"] = txn_reporting_naive_ist(d["TxnDate"])
        d = d.dropna(subset=["TxnDate"])
        if d.empty:
            return df, ""
        latest = d["TxnDate"].max()
        if pd.isna(latest):
            return df, ""
        date = str(latest.normalize().date())

    return df, str(date).strip()


@router.get("/daily-dsr")
def daily_dsr(request: Request, date: Optional[str] = None):
    """
    Daily DSR-style report for one calendar day: marketplace sections with optional
    segment rows (Flipkart Brand, Snapdeal Company, etc.) and an Others bucket.
    Query: ``date`` = ISO ``YYYY-MM-DD`` (defaults to latest day with sales if omitted).
    """
    sess = _sess(request)
    _restore_daily_if_needed(sess)
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


# ── MTR Analytics ─────────────────────────────────────────────

@router.get("/mtr-analytics")
def mtr_analytics(request: Request):
    sess = _sess(request)
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
    returned = float(df[df["Transaction_Type"] == "Refund"]["Quantity"].sum())
    net_units = int(shipped - returned)

    if "shipments" in monthly.columns and "refunds" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly["refunds"]
    elif "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"]

    return {
        "loaded":       True,
        "rows":         len(df),
        "date_range":   [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":      int(shipped),
        "returned":     int(returned),
        "net_units":    net_units,
        "return_rate":  round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":      monthly.to_dict("records"),
        "top_skus":     top.to_dict("records"),
    }


# ── Myntra Analytics ─────────────────────────────────────────

@router.get("/myntra-analytics")
def myntra_analytics(request: Request):
    sess = _sess(request)
    df = sess.myntra_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

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

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "top_skus":    top_skus.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Meesho Analytics ─────────────────────────────────────────

@router.get("/meesho-analytics")
def meesho_analytics(request: Request):
    sess = _sess(request)
    df = sess.meesho_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

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

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
        "by_state":    by_state.to_dict("records"),
    }


# ── Flipkart Analytics ────────────────────────────────────────

@router.get("/flipkart-analytics")
def flipkart_analytics(request: Request):
    sess = _sess(request)
    df = sess.flipkart_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

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

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
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

    import pandas as pd

    ensure_inventory_snapshot_metadata(sess)
    cols = [c for c in df.columns if c != "OMS_SKU"]

    totals = {}
    for c in cols:
        try:
            totals[c] = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        except Exception:
            totals[c] = 0

    rows, total_rows = inventory_rows_for_api(
        df, search=search or "", offset=offset, limit=limit
    )
    dbg = getattr(sess, "inventory_debug", {}) or {}
    return {
        "loaded": True,
        "rows": rows,
        "total_rows": total_rows,
        "offset": max(0, int(offset)),
        "limit": max(1, min(int(limit), 5000)),
        "columns": ["OMS_SKU"] + cols,
        "totals": totals,
        "debug": dbg,
        "marketplaces": inventory_marketplace_breakdown(df, dbg),
        "missing_marketplace_hints": inventory_missing_marketplace_warnings(dbg),
        **inventory_snapshot_meta_for_api(sess),
    }


# ── Snapdeal Analytics ────────────────────────────────────────

@router.get("/snapdeal-analytics")
def snapdeal_analytics(request: Request, company: Optional[str] = None):
    sess = _sess(request)
    df = sess.snapdeal_df
    if df.empty:
        return {"loaded": False}

    import pandas as pd

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Collect unique companies before filtering
    companies: list = []
    if "Company" in df.columns:
        companies = sorted(df["Company"].dropna().str.strip().unique().tolist())
        companies = [c for c in companies if c]

    # Apply company filter
    if company and "Company" in df.columns:
        df = df[df["Company"].str.strip() == company.strip()]

    shipped  = float(df[df["TxnType"] == "Shipment"]["Quantity"].sum())
    returned = float(df[df["TxnType"] == "Refund"]["Quantity"].sum())

    monthly = (
        df.groupby(["Month", "TxnType"])["Quantity"]
        .sum().reset_index()
        .pivot_table(index="Month", columns="TxnType", values="Quantity", fill_value=0)
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={"Shipment": "shipments", "Refund": "refunds"})
    if "shipments" not in monthly.columns:
        monthly["shipments"] = 0
    if "refunds" not in monthly.columns:
        monthly["refunds"] = 0

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

    if "shipments" in monthly.columns:
        monthly["net"] = monthly["shipments"] - monthly.get("refunds", 0)

    return {
        "loaded":      True,
        "rows":        len(df),
        "companies":   companies,
        "date_range":  [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "shipped":     int(shipped),
        "returned":    int(returned),
        "net_units":   int(shipped - returned),
        "return_rate": round(returned / shipped * 100, 1) if shipped > 0 else 0,
        "monthly":     monthly.to_dict("records"),
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
def delete_daily_upload(upload_id: int, _request: Request):
    ok = delete_upload(upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"ok": True, "message": f"Deleted upload {upload_id}"}


# ── Data Debug / Coverage ────────────────────────────────────

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

    from ..services.po_engine import calculate_quarterly_history
    # When sales_df exists it already includes Amazon & Myntra — never pass raw DFs (was doubling).
    # When sales_df is empty, pass platform frames so quarterly still works before first build-sales.
    _boot = sess.sales_df.empty or "Sku" not in sess.sales_df.columns
    pivot = calculate_quarterly_history(
        sales_df=sess.sales_df,
        mtr_df=sess.mtr_df if _boot and not sess.mtr_df.empty else None,
        myntra_df=sess.myntra_df if _boot and not sess.myntra_df.empty else None,
        sku_mapping=sess.sku_mapping or None,
        group_by_parent=group_by_parent,
        n_quarters=n_quarters,
    )
    if pivot.empty:
        return {"loaded": False, "rows": []}

    return {
        "loaded":   True,
        "columns":  list(pivot.columns),
        "rows":     pivot.fillna(0).to_dict("records"),
    }

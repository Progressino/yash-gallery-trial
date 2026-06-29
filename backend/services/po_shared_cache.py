"""
Shared PO calculate cache — one successful run per planning day + settings + data snapshot
can be reused by other browser sessions on the same server (seconds vs minutes).

Cache key = SHA-256 of (planning_date, calc params, data fingerprint).
Stored as parquet + JSON meta under PO_SHARED_CACHE_DIR (default: beside session spills).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Bump when PO engine merge/ADS semantics change (invalidates shared cache).
PO_MERGE_LOGIC_VERSION = 43


def po_merge_result_is_stale(meta: dict[str, Any] | None) -> bool:
    """True when a stored PO result was computed with an older engine merge version."""
    if not meta:
        return False
    stored = meta.get("po_merge_version")
    if stored is None:
        return True
    try:
        return int(stored) < PO_MERGE_LOGIC_VERSION
    except (TypeError, ValueError):
        return True


_CALC_PARAM_KEYS = (
    "period_days",
    "lead_time",
    "target_days",
    "demand_basis",
    "use_seasonality",
    "seasonal_weight",
    "group_by_parent",
    "min_denominator",
    "grace_days",
    "safety_pct",
    "enforce_two_size_minimum",
    "enforce_lead_time_release_gate",
    "raise_ledger_lookback_days",
    "auto_import_yesterday_ledger",
    "urgent_all_sizes_days",
    "use_ly_fallback",
)

# Mirror ``PORequest`` defaults so partial query bodies match POST fingerprints.
_CALC_PARAM_DEFAULTS: dict[str, Any] = {
    "period_days": 90,
    "lead_time": 30,
    "target_days": 135,
    "demand_basis": "Sold",
    "use_seasonality": False,
    "seasonal_weight": 0.5,
    "group_by_parent": False,
    "min_denominator": 7,
    "grace_days": 0,
    "safety_pct": 0.0,
    "enforce_two_size_minimum": False,
    "enforce_lead_time_release_gate": True,
    "raise_ledger_lookback_days": 14,
    "auto_import_yesterday_ledger": True,
    "urgent_all_sizes_days": 45,
    "use_ly_fallback": True,
}


def _calc_params_for_fingerprint(body: dict) -> dict[str, Any]:
    params = dict(_CALC_PARAM_DEFAULTS)
    for k in _CALC_PARAM_KEYS:
        if k in body and body[k] is not None and body[k] != "":
            params[k] = body[k]
    return params


def shared_cache_enabled() -> bool:
    raw = (os.environ.get("PO_SHARED_CACHE_ENABLED") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _shared_dir() -> Path:
    raw = (os.environ.get("PO_SHARED_CACHE_DIR") or "").strip()
    if raw:
        root = Path(raw)
    else:
        spill = (os.environ.get("PO_RESULT_SPILL_DIR") or "/data/po_results").strip()
        root = Path(spill) / "shared"
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError:
        dev = Path("./po_results_dev/shared")
        dev.mkdir(parents=True, exist_ok=True)
        return dev


def _meta_path(cache_key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)[:80]
    return _shared_dir() / f"{safe}.meta.json"


def _parquet_path(cache_key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)[:80]
    return _shared_dir() / f"{safe}.parquet"


def normalize_planning_date(body: dict) -> str:
    raw = body.get("planning_date")
    if raw and str(raw).strip():
        try:
            return str(pd.Timestamp(pd.to_datetime(str(raw).strip()).normalize()).date())
        except Exception:
            return str(raw).strip()[:10]
    return str(pd.Timestamp.now(tz=IST).normalize().date())


def _sales_through(sess) -> str:
    """Server-wide sales freshness for cache fingerprint (not per-session pointers)."""
    try:
        from .tier3_session_merge import tier3_sales_through

        t3 = tier3_sales_through()
        if t3:
            return str(t3)[:10]
    except Exception:
        pass
    from .tier3_session_merge import effective_sales_through

    return effective_sales_through(sess)


def _existing_po_fingerprint(sess) -> str:
    """Changes whenever the uploaded Existing PO sheet is replaced or its totals change."""
    ep = getattr(sess, "existing_po_df", None)
    gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    if ep is None or getattr(ep, "empty", True):
        return f"ep:{gen}:0"
    uploaded = str(getattr(sess, "existing_po_uploaded_at", "") or "")
    fn = str(getattr(sess, "existing_po_filename", "") or "")
    n = int(len(ep))
    sku_n = 0
    if "OMS_SKU" in ep.columns:
        sku_n = int(ep["OMS_SKU"].astype(str).nunique())
    sums: dict[str, int] = {}
    for c in ("PO_Pipeline_Total", "Pending_Cutting", "Balance_to_Dispatch", "PO_Qty_Ordered"):
        if c in ep.columns:
            sums[c] = int(pd.to_numeric(ep[c], errors="coerce").fillna(0).sum())
    return f"ep:{gen}:{n}:{sku_n}:{uploaded}:{fn}:{sums}"


def _sku_status_fingerprint(sess) -> str:
    """Changes when SKU status / lead sheet is uploaded or closed counts shift."""
    df = getattr(sess, "sku_status_lead_df", None)
    if df is None or getattr(df, "empty", True):
        return "ss:0"
    n = int(len(df))
    closed = 0
    if "SKU_Sheet_Closed" in df.columns:
        closed = int(pd.to_numeric(df["SKU_Sheet_Closed"], errors="coerce").fillna(0).sum())
    elif "SKU_Sheet_Status" in df.columns:
        from .sku_status_lead import is_closed_sku_status

        closed = int(df["SKU_Sheet_Status"].map(is_closed_sku_status).sum())
    return f"ss:{n}:{closed}"


def _sku_mapping_fingerprint(sess) -> str:
    """Changes when SKU mapping master is merged or replaced."""
    m = getattr(sess, "sku_mapping", None) or {}
    if not m:
        return "map:0"
    return f"map:{len(m)}"


def _raise_ledger_fingerprint(planning_date: str, lookback_days: int) -> str:
    """Stable signature for confirmed raises in the planning window (shared DB)."""
    try:
        from ..db.po_raised_db import ledger_rows_as_dataframe

        plan = pd.Timestamp(pd.to_datetime(planning_date).normalize())
        lb = max(1, int(lookback_days))
        start = plan - pd.Timedelta(days=lb)
        db_df = ledger_rows_as_dataframe(
            start_date=str(start.date()),
            end_date=str(plan.date()),
        )
        if db_df is None or db_df.empty:
            return "raises:0"
        qty = int(pd.to_numeric(db_df.get("Raised_Qty"), errors="coerce").fillna(0).sum())
        n = int(len(db_df))
        return f"raises:{n}:{qty}"
    except Exception:
        return "raises:?"


def _return_overlay_fingerprint(sess) -> str:
    """Changes when return overlay import/rebuild alters PO net demand."""
    df = getattr(sess, "po_return_overlay_df", None)
    if df is None or getattr(df, "empty", True):
        return "rov:0"
    n = int(len(df))
    units = 0
    if "Return_Overlay_Units" in df.columns:
        units = int(pd.to_numeric(df["Return_Overlay_Units"], errors="coerce").fillna(0).sum())
    as_of = str(getattr(sess, "return_overlay_as_of", "") or "")[:10]
    return f"rov:{n}:{units}:{as_of}"


def invalidate_po_after_sales_or_returns_change(sess) -> None:
    """Drop session PO results and shared cache after sales/returns data changes."""
    try:
        from .po_raise_remove import invalidate_po_calculate_result

        invalidate_po_calculate_result(sess)
    except Exception:
        _log.exception("invalidate_po_calculate_result after data change failed")
    try:
        invalidate_all_shared_caches()
    except Exception:
        _log.exception("invalidate_all_shared_caches after data change failed")
    try:
        from backend.routers.data import _invalidate_intelligence_bundle_cache

        if hasattr(sess, "_intelligence_bundle_cache"):
            sess._intelligence_bundle_cache.clear()
        _invalidate_intelligence_bundle_cache()
    except Exception:
        _log.exception("intelligence bundle cache invalidation after data change failed")


def build_data_fingerprint(sess, body: dict) -> dict[str, Any]:
    """Inputs that must match for a shared PO table to stay valid."""
    planning = normalize_planning_date(body)
    inv = getattr(sess, "inventory_df_variant", None)
    inv_rows = int(len(inv)) if inv is not None and hasattr(inv, "__len__") else 0
    inv_skus = 0
    if inv is not None and not getattr(inv, "empty", True) and "OMS_SKU" in inv.columns:
        inv_skus = int(inv["OMS_SKU"].astype(str).nunique())

    sales = getattr(sess, "sales_df", None)
    sales_rows = int(len(sales)) if sales is not None and hasattr(sales, "__len__") else 0

    hist = getattr(sess, "daily_inventory_history_df", None)
    hist_rows = int(len(hist)) if hist is not None and hasattr(hist, "__len__") else 0

    warm_gen = 0
    try:
        import backend.main as _main

        warm_gen = int(getattr(_main, "_warm_cache_generation", 0) or 0)
    except Exception:
        pass

    params = _calc_params_for_fingerprint(body)

    tier3_token: dict[str, str] = {}
    try:
        from .daily_store import get_tier3_sync_token

        tier3_token = get_tier3_sync_token() or {}
    except Exception:
        pass

    git_sha = ""
    try:
        from ..app_version import get_build_info

        git_sha = str(get_build_info().get("git_sha") or "")
    except Exception:
        pass

    try:
        from .po_pipeline import PO_PIPELINE_VERSION
    except Exception:
        PO_PIPELINE_VERSION = 0

    return {
        "planning_date": planning,
        "params": params,
        "po_merge_version": PO_MERGE_LOGIC_VERSION,
        "git_sha": git_sha,
        "sales_through": _sales_through(sess),
        "sales_rows": sales_rows,
        "inventory_rows": inv_rows,
        "inventory_skus": inv_skus,
        "inventory_snapshot": str(getattr(sess, "inventory_snapshot_date", "") or ""),
        "inventory_history_rows": hist_rows,
        "warm_cache_generation": warm_gen,
        "raise_ledger": _raise_ledger_fingerprint(
            planning, int(body.get("raise_ledger_lookback_days") or 14)
        ),
        "existing_po": _existing_po_fingerprint(sess),
        "sku_status": _sku_status_fingerprint(sess),
        "sku_mapping": _sku_mapping_fingerprint(sess),
        "return_overlay": _return_overlay_fingerprint(sess),
        "tier3_sync_token": tier3_token,
        "pipeline_snapshot_hash": str(getattr(sess, "po_pipeline_snapshot_id", "") or ""),
        "pipeline_version": PO_PIPELINE_VERSION,
    }


def cache_key_from_fingerprint(fp: dict[str, Any]) -> str:
    payload = json.dumps(fp, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    planning = str(fp.get("planning_date") or "unknown")
    return f"{planning}_{digest}"


def build_cache_key(sess, body: dict) -> tuple[str, dict[str, Any]]:
    fp = build_data_fingerprint(sess, body)
    return cache_key_from_fingerprint(fp), fp


def _meta_is_fresh(meta: dict[str, Any]) -> bool:
    planning = str(meta.get("planning_date") or "")
    if not planning:
        return False
    try:
        plan_day = pd.Timestamp(pd.to_datetime(planning).normalize()).date()
        today = datetime.now(IST).date()
        # Valid through the day after planning_date (covers overnight runs).
        if today > plan_day + timedelta(days=1):
            return False
    except Exception:
        pass
    created = float(meta.get("created_at_unix") or 0)
    if created and (time.time() - created) > 48 * 3600:
        return False
    return True


def _shared_cache_stale_vs_disk(meta: dict[str, Any]) -> bool:
    """Reject cache saved before a newer Existing PO sheet landed on disk."""
    try:
        from .existing_po import read_existing_po_disk_meta

        disk = read_existing_po_disk_meta() or {}
    except Exception:
        return False
    disk_gen = int(disk.get("existing_po_generation") or 0)
    cache_gen = int(meta.get("existing_po_generation") or 0)
    if disk_gen > cache_gen:
        return True
    disk_fn = str(disk.get("existing_po_filename") or "").strip()
    cache_fn = str(meta.get("existing_po_filename") or "").strip()
    return bool(disk_fn and cache_fn and disk_fn != cache_fn)


def lookup_shared_cache(sess, body: dict) -> Optional[dict[str, Any]]:
    """Return meta dict if a matching shared PO result exists."""
    if not shared_cache_enabled():
        return None
    key, fp = build_cache_key(sess, body)
    meta_path = _meta_path(key)
    parquet = _parquet_path(key)
    if not meta_path.is_file() or not parquet.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _meta_is_fresh(meta):
        return None
    if meta.get("fingerprint") != fp:
        return None
    if _shared_cache_stale_vs_disk(meta):
        return None
    meta["cache_key"] = key
    return meta


def save_shared_cache(
    sess,
    body: dict,
    po_df: pd.DataFrame,
    result: dict[str, Any],
) -> Optional[str]:
    """Persist successful PO result for other sessions. Returns cache_key or None."""
    if not shared_cache_enabled():
        return None
    if po_df is None or getattr(po_df, "empty", True):
        return None
    key, fp = build_cache_key(sess, body)
    try:
        tmp = _parquet_path(key).with_suffix(".parquet.tmp")
        final = _parquet_path(key)
        from .helpers import _coerce_df_for_parquet

        _coerce_df_for_parquet(po_df).to_parquet(tmp, index=False)
        tmp.replace(final)
        summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        meta = {
            "cache_key": key,
            "fingerprint": fp,
            "planning_date": fp.get("planning_date"),
            "created_at_unix": time.time(),
            "created_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
            "total_rows": int(len(po_df)),
            "columns": list(po_df.columns),
            "sales_through": result.get("sales_through"),
            "planning_date_out": result.get("planning_date"),
            "raise_ledger_rows": result.get("raise_ledger_rows"),
            "po_merge_version": fp.get("po_merge_version"),
            "existing_po_generation": int(getattr(sess, "existing_po_generation", 0) or 0),
            "existing_po_filename": str(getattr(sess, "existing_po_filename", "") or ""),
            "new_po_qty_sum": int(summ.get("new_po_qty_sum") or 0),
        }
        _meta_path(key).write_text(json.dumps(meta, default=str), encoding="utf-8")
        _log.info(
            "PO shared cache saved key=%s rows=%s planning=%s",
            key[:16],
            len(po_df),
            fp.get("planning_date"),
        )
        return key
    except Exception:
        _log.exception("save_shared_cache failed")
        return None


def apply_shared_cache_to_session(
    sess,
    session_id: str,
    body: dict,
    *,
    job_id: str | None = None,
) -> Optional[dict[str, Any]]:
    """
    Copy shared parquet into this session's spill and mark job done.
    Returns API-shaped dict for POST /po/calculate, or None if no hit.
    """
    meta = lookup_shared_cache(sess, body)
    if not meta or not session_id:
        return None
    key = str(meta["cache_key"])
    try:
        from .po_calculate_jobs import set_po_job
        from .po_result_spill import clear_spill, copy_spill_from_path

        clear_spill(session_id)
        if not copy_spill_from_path(_parquet_path(key), session_id):
            return None

        n = int(meta.get("total_rows") or 0)
        cols = list(meta.get("columns") or [])
        _fp = meta.get("fingerprint") if isinstance(meta.get("fingerprint"), dict) else {}
        _merge_ver = meta.get("po_merge_version") or _fp.get("po_merge_version")
        msg = (
            f"Loaded shared PO run from {meta.get('created_at_ist', 'earlier today')} "
            f"({n:,} rows) — same planning date and settings."
        )
        result = {
            "ok": True,
            "status": "done",
            "from_shared_cache": True,
            "shared_cache_at": meta.get("created_at_ist"),
            "message": msg,
            "total_rows": n,
            "columns": cols,
            "sales_through": meta.get("sales_through"),
            "planning_date": meta.get("planning_date_out") or meta.get("planning_date"),
            "raise_ledger_rows": meta.get("raise_ledger_rows"),
            "po_merge_version": _merge_ver,
        }
        try:
            from .tier3_session_merge import effective_sales_through

            fresh_through = effective_sales_through(sess)
            if fresh_through:
                result["sales_through"] = fresh_through
        except Exception:
            pass
        if int(meta.get("new_po_qty_sum") or 0) > 0:
            result["summary"] = {
                "new_po_qty_sum": int(meta.get("new_po_qty_sum") or 0),
                "existing_po_filename": meta.get("existing_po_filename"),
                "existing_po_generation": meta.get("existing_po_generation"),
            }
        sess.po_calculate_status = "done"
        sess.po_calculate_progress = 100
        sess.po_calculate_message = msg
        sess.po_calculate_result = {k: v for k, v in result.items() if k != "status"}
        sess.po_calculate_result_df = pd.DataFrame()
        target_job = job_id
        if not target_job:
            from .po_calculate_jobs import get_latest_job_id

            target_job = get_latest_job_id(session_id)
        if target_job:
            set_po_job(
                target_job,
                status="done",
                ok=True,
                progress=100,
                message=msg,
                total_rows=n,
                columns=cols,
                sales_through=result.get("sales_through"),
                planning_date=result.get("planning_date"),
                from_shared_cache=True,
            )
        return result
    except Exception:
        _log.exception("apply_shared_cache_to_session failed")
        return None


def shared_cache_availability(sess, body: dict) -> dict[str, Any]:
    """Lightweight check for UI — does not copy into session."""
    from .tier3_session_merge import effective_sales_through

    meta = lookup_shared_cache(sess, body)
    if not meta:
        return {"available": False}
    _fp = meta.get("fingerprint") if isinstance(meta.get("fingerprint"), dict) else {}
    return {
        "available": True,
        "planning_date": meta.get("planning_date"),
        "row_count": int(meta.get("total_rows") or 0),
        "computed_at": meta.get("created_at_ist"),
        "sales_through": effective_sales_through(sess) or meta.get("sales_through"),
        "po_merge_version": meta.get("po_merge_version") or _fp.get("po_merge_version"),
    }


def invalidate_all_shared_caches() -> int:
    """Drop every shared PO calculate result (e.g. after Existing PO re-upload)."""
    removed = 0
    for p in _shared_dir().glob("*.meta.json"):
        try:
            key = p.stem.replace(".meta", "")
            p.unlink(missing_ok=True)
            _parquet_path(key).unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return removed


def invalidate_planning_date(planning_date: str) -> int:
    """Remove shared entries for a planning day (e.g. after major ledger reset)."""
    day = str(planning_date).strip()[:10]
    if not day:
        return 0
    removed = 0
    for p in _shared_dir().glob(f"{day}_*.meta.json"):
        try:
            key = p.stem.replace(".meta", "")
            p.unlink(missing_ok=True)
            _parquet_path(key).unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return removed


def cleanup_old_shared(max_age_hours: int = 72) -> int:
    cutoff = time.time() - max(0, int(max_age_hours)) * 3600
    removed = 0
    for p in _shared_dir().glob("*.meta.json"):
        try:
            if p.stat().st_mtime < cutoff:
                key = p.name.replace(".meta.json", "")
                p.unlink()
                pq = _shared_dir() / f"{key}.parquet"
                if pq.is_file():
                    pq.unlink()
                removed += 1
        except Exception:
            continue
    return removed

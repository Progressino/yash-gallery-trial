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
    "raise_view_date",
)


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

    params = {k: body.get(k) for k in _CALC_PARAM_KEYS}
    params.setdefault("period_days", 90)
    params.setdefault("lead_time", 30)
    params.setdefault("target_days", 135)
    params.setdefault("demand_basis", "Sold")
    params.setdefault("group_by_parent", False)

    return {
        "planning_date": planning,
        "params": params,
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
        po_df.to_parquet(tmp, index=False)
        tmp.replace(final)
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


def apply_shared_cache_to_session(sess, session_id: str, body: dict) -> Optional[dict[str, Any]]:
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
        }
        sess.po_calculate_status = "done"
        sess.po_calculate_progress = 100
        sess.po_calculate_message = msg
        sess.po_calculate_result = {k: v for k, v in result.items() if k != "status"}
        sess.po_calculate_result_df = pd.DataFrame()
        set_po_job(
            session_id,
            status="done",
            ok=True,
            progress=100,
            message=msg,
            total_rows=n,
            columns=cols,
            sales_through=result.get("sales_through"),
            planning_date=result.get("planning_date"),
        )
        return result
    except Exception:
        _log.exception("apply_shared_cache_to_session failed")
        return None


def shared_cache_availability(sess, body: dict) -> dict[str, Any]:
    """Lightweight check for UI — does not copy into session."""
    meta = lookup_shared_cache(sess, body)
    if not meta:
        return {"available": False}
    return {
        "available": True,
        "planning_date": meta.get("planning_date"),
        "row_count": int(meta.get("total_rows") or 0),
        "computed_at": meta.get("created_at_ist"),
        "sales_through": meta.get("sales_through"),
    }


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

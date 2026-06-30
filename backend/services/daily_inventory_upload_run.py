"""Background daily inventory history upload — avoids 502 on large wide Excel files."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

# Auto-clear stuck wide-matrix uploads after this many seconds (default 6 min).
_DAILY_INV_UPLOAD_STUCK_SEC = int(os.environ.get("DAILY_INV_UPLOAD_STUCK_SEC", "360"))


def clear_stuck_daily_inventory_upload(sess, *, force: bool = False) -> bool:
    """Reset a session stuck in daily_inventory_upload_status=running."""
    if getattr(sess, "daily_inventory_upload_status", "idle") != "running":
        return False
    started = float(getattr(sess, "daily_inventory_upload_started", 0) or 0)
    age = time.time() - started if started > 0 else 999999
    if not force and age < _DAILY_INV_UPLOAD_STUCK_SEC:
        return False
    msg = (
        "Previous daily inventory upload did not finish (timed out or server was busy). "
        "Try uploading again."
    )
    sess.daily_inventory_upload_status = "error"
    sess.daily_inventory_upload_message = msg
    sess.daily_inventory_upload_started = 0.0
    sess.daily_inventory_upload_result = {"ok": False, "message": msg}
    return True

# PO engine only uses recent history for Eff_Days / roll-forward (default: last 30 calendar
# days in the sheet, anchored on the latest snapshot date). Older columns are dropped at
# ingest to keep memory and Calculate PO fast. Set DAILY_INV_MAX_DAYS to keep more (e.g. 90).
_MAX_HISTORY_DAYS = int(os.environ.get("DAILY_INV_MAX_DAYS", "30"))


def _series_as_dates(col: pd.Series) -> pd.Series:
    """Parse Date column once; reuse when already datetime64."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    return pd.to_datetime(col, errors="coerce")


def _trim_history_to_recent(df: pd.DataFrame, max_days: int) -> tuple[pd.DataFrame, str]:
    """
    Keep only the most-recent ``max_days`` calendar days in the history.

    Anchors on the latest date present in the upload so old baseline files
    are not accidentally discarded (the sheet may be weeks behind today).

    Returns (trimmed_df, trim_note) where trim_note is "" if nothing was removed.
    """
    if max_days <= 0 or df.empty or "Date" not in df.columns:
        return df, ""
    dates = _series_as_dates(df["Date"])
    max_date = dates.max()
    if pd.isna(max_date):
        return df, ""
    cutoff = pd.Timestamp(max_date).normalize() - pd.Timedelta(days=max_days)
    min_date = dates.min()
    if pd.notna(min_date) and min_date >= cutoff:
        return df, ""
    mask = dates >= cutoff
    kept = int(mask.sum())
    orig = int(len(df))
    if kept >= orig:
        return df, ""
    trimmed = df.loc[mask].reset_index(drop=True)
    note = (
        f"Trimmed to last {max_days} days ({cutoff.date()} → {pd.Timestamp(max_date).date()}): "
        f"{orig:,} → {kept:,} rows. "
        f"Set DAILY_INV_MAX_DAYS env-var to keep more than {max_days} days."
    )
    logger.info("Daily inventory history: %s", note)
    return trimmed, note


def execute_daily_inventory_upload(
    sess,
    raw: bytes,
    filename: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from .daily_inventory_history import (
        filter_inventory_history_window,
        inventory_history_max_date,
        inventory_history_min_date,
        inventory_sheet_end_date_from_filename,
        inventory_sheet_start_date_from_filename,
        is_full_matrix_inventory_reupload,
        merge_inventory_history,
        parse_daily_inventory_history_upload,
        promote_daily_inventory_matrix_max_date,
        recanonicalize_inventory_history_skus,
        drop_zero_derived_rows,
    )

    def _progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        if sess is not None:
            sess.daily_inventory_upload_message = msg

    fn_end = inventory_sheet_end_date_from_filename(filename or "")
    fn_start = inventory_sheet_start_date_from_filename(filename or "")
    history_days = _MAX_HISTORY_DAYS
    if fn_start and fn_end:
        try:
            span = int((pd.Timestamp(fn_end) - pd.Timestamp(fn_start)).days) + 1
            history_days = max(history_days, span)
        except Exception:
            pass

    _progress("Parsing daily inventory sheet…")
    df = parse_daily_inventory_history_upload(
        BytesIO(raw),
        filename,
        sku_mapping=sess.sku_mapping or None,
        max_days=history_days,
        on_progress=_progress,
    )
    if df.empty:
        return {
            "ok": False,
            "message": "No usable rows. Need a wide-format sheet: column 1 = SKU, "
            "column 2 = parent (optional), then daily snapshot columns whose first row is the date.",
        }

    if sess.sku_mapping:
        df = recanonicalize_inventory_history_skus(df, sess.sku_mapping)
    df = drop_zero_derived_rows(df)

    sheet_max = inventory_history_max_date(df)
    end_anchor = str(sheet_max.date()) if sheet_max is not None else None
    fn_end = inventory_sheet_end_date_from_filename(filename or "")
    if fn_end and (not end_anchor or fn_end > end_anchor):
        end_anchor = fn_end
    df = filter_inventory_history_window(df, days=history_days, end_date=end_anchor)
    trim_note = (
        f"Kept last {history_days} calendar days ending "
        f"{end_anchor or 'today (IST)'}."
    )

    existing = getattr(sess, "daily_inventory_history_df", None)
    full_matrix = is_full_matrix_inventory_reupload(existing, df, filename or "")
    if existing is not None and not existing.empty:
        incoming_dates = set(
            pd.to_datetime(df["Date"], errors="coerce").dt.normalize().dropna().unique()
        )
        ex_dates = pd.to_datetime(existing["Date"], errors="coerce").dt.normalize()
        ex_date_set = set(ex_dates.dropna().unique())
        in_max = inventory_history_max_date(df)
        ex_max = inventory_history_max_date(existing)
        in_skus = int(df["OMS_SKU"].astype(str).nunique())
        ex_skus = int(existing["OMS_SKU"].astype(str).nunique())
        full_reupload = full_matrix or (
            in_max is not None
            and ex_max is not None
            and in_max >= ex_max
            and in_skus >= max(500, int(ex_skus * 0.85))
        )
        if full_reupload:
            in_min = inventory_history_min_date(df)
            if in_min is not None and in_max is not None:
                # Drop every existing row inside (or before) the matrix — only keep post-matrix tail.
                kept = existing.loc[ex_dates > in_max]
                if len(kept) < len(existing):
                    _progress("Replacing wide-matrix date range…")
                    df = merge_inventory_history(kept, df) if not kept.empty else df
                snap_dates = list(getattr(sess, "daily_inventory_history_snapshot_dates", None) or [])
                in_lo = str(pd.Timestamp(in_min).date())
                in_hi = str(pd.Timestamp(in_max).date())
                sess.daily_inventory_history_snapshot_dates = [
                    d for d in snap_dates if d < in_lo or d > in_hi
                ]
        elif incoming_dates and incoming_dates >= ex_date_set:
            # Full wide-matrix re-upload — skip merge with stale derived/zero rows.
            pass
        elif incoming_dates:
            _progress("Merging with saved history…")
            kept = existing.loc[~ex_dates.isin(incoming_dates)]
            df = merge_inventory_history(kept, df)
        else:
            df = merge_inventory_history(existing, df)
        sheet_max = inventory_history_max_date(df)
        end_anchor = str(sheet_max.date()) if sheet_max is not None else end_anchor
        fn_end = inventory_sheet_end_date_from_filename(filename or "")
        if fn_end and (not end_anchor or fn_end > end_anchor):
            end_anchor = fn_end
        df = filter_inventory_history_window(df, days=history_days, end_date=end_anchor)
        trim_note = (
            f"Replaced overlapping dates; kept last {history_days} days ending "
            f"{end_anchor or 'today (IST)'}."
        )

    sess.daily_inventory_history_df = df
    sess.daily_inventory_history_uploaded_at = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S")
    sess.daily_inventory_history_filename = filename or ""
    fn_end = inventory_sheet_end_date_from_filename(filename or "")
    if fn_end:
        sess.daily_inventory_history_wide_end_date = fn_end
    if end_anchor:
        promote_daily_inventory_matrix_max_date(sess, end_anchor)
    sess._quarterly_cache.clear()
    skus = int(df["OMS_SKU"].nunique())
    dates_norm = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    days = int(dates_norm.nunique())
    min_d = dates_norm.min()
    max_d = dates_norm.max()
    msg = f"Loaded {len(df):,} SKU-day rows ({skus:,} SKUs × {days:,} days) for effective-days math."
    if pd.notna(min_d) and pd.notna(max_d):
        msg += f" Snapshot range: {pd.Timestamp(min_d).date()} → {pd.Timestamp(max_d).date()}."
    if trim_note:
        msg += f" Note: {trim_note}"
    return {
        "ok": True,
        "rows": int(len(df)),
        "skus": skus,
        "days": days,
        "min_date": str(pd.Timestamp(min_d).date()) if pd.notna(min_d) else "",
        "max_date": str(pd.Timestamp(max_d).date()) if pd.notna(max_d) else "",
        "message": msg,
    }


def _acquire_daily_inventory_memory_lock(sess, session_id: str, progress) -> bool:
    """Wait briefly for the global upload lock (same as snapshot inventory)."""
    from ..concurrency import _UPLOAD_MEMORY_LOCK

    wait_sec = int(os.environ.get("INVENTORY_MEMORY_LOCK_WAIT_SEC", "120"))
    if _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
        return True
    progress(f"Queued — waiting for server ({wait_sec}s max)…")
    logger.info("daily-inventory-history queued behind another heavy job (session=%s)", session_id[:8])
    if _UPLOAD_MEMORY_LOCK.acquire(timeout=wait_sec):
        return True
    logger.warning("daily-inventory-history proceeding without memory lock (session=%s)", session_id[:8])
    progress("Parsing sheet (server finishing cache load in background)…")
    return False


def background_daily_inventory_upload(
    session_id: str,
    raw: bytes,
    filename: str,
) -> None:
    from ..session import store

    sess = store.get(session_id)
    if sess is None:
        return

    def _progress(msg: str) -> None:
        sess.daily_inventory_upload_message = msg

    def _sync_disk_and_cache() -> None:
        try:
            import backend.main as _main

            _main.sync_daily_inventory_history_sidecar(sess)
        except Exception:
            logger.exception("sync_daily_inventory_history_sidecar failed")

    def _persist_pg_background() -> None:
        try:
            from ..db.forecast_session_pg import persist_session_bundle_thread_safe

            persist_session_bundle_thread_safe(session_id, sess)
        except Exception:
            logger.exception("PostgreSQL persist after daily inventory upload failed")

    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_message = "Starting parse…"
    sess.daily_inventory_upload_started = time.time()
    lock_held = _acquire_daily_inventory_memory_lock(sess, session_id, _progress)
    try:
        result = execute_daily_inventory_upload(sess, raw, filename, on_progress=_progress)
        sess.daily_inventory_upload_result = result
        if result.get("ok"):
            from .daily_inventory_history import persist_upload_pipeline_snapshot

            persist_upload_pipeline_snapshot(sess.daily_inventory_history_df)
            _progress("Saving to server…")
            _sync_disk_and_cache()
            threading.Thread(target=_persist_pg_background, daemon=True).start()
            sess.daily_inventory_upload_status = "done"
            sess.daily_inventory_upload_message = result.get("message") or "Daily inventory loaded."
        else:
            sess.daily_inventory_upload_status = "error"
            sess.daily_inventory_upload_message = result.get("message") or "Upload failed."
    except Exception as e:
        logger.exception("background_daily_inventory_upload failed")
        sess.daily_inventory_upload_status = "error"
        sess.daily_inventory_upload_message = str(e)
        sess.daily_inventory_upload_result = {"ok": False, "message": str(e)}
    finally:
        if lock_held:
            from ..concurrency import _UPLOAD_MEMORY_LOCK

            _UPLOAD_MEMORY_LOCK.release()

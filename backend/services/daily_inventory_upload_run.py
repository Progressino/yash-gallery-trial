"""Background daily inventory history upload — avoids 502 on large wide Excel files."""
from __future__ import annotations

import logging
import os
import threading
from io import BytesIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# PO engine only uses recent history for Eff_Days / roll-forward (default: last 30 calendar
# days in the sheet, anchored on the latest snapshot date). Older columns are dropped at
# ingest to keep memory and Calculate PO fast. Set DAILY_INV_MAX_DAYS to keep more (e.g. 90).
_MAX_HISTORY_DAYS = int(os.environ.get("DAILY_INV_MAX_DAYS", "30"))


def _trim_history_to_recent(df: pd.DataFrame, max_days: int) -> tuple[pd.DataFrame, str]:
    """
    Keep only the most-recent ``max_days`` calendar days in the history.

    Anchors on the latest date present in the upload so old baseline files
    are not accidentally discarded (the sheet may be weeks behind today).

    Returns (trimmed_df, trim_note) where trim_note is "" if nothing was removed.
    """
    if max_days <= 0 or df.empty or "Date" not in df.columns:
        return df, ""
    dates = pd.to_datetime(df["Date"], errors="coerce")
    max_date = dates.max()
    if pd.isna(max_date):
        return df, ""
    cutoff = pd.Timestamp(max_date).normalize() - pd.Timedelta(days=max_days)
    mask = dates >= cutoff
    kept = int(mask.sum())
    orig = int(len(df))
    if kept >= orig:
        return df, ""
    trimmed = df[mask].reset_index(drop=True)
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
) -> dict[str, Any]:
    from .daily_inventory_history import parse_daily_inventory_history_upload

    df = parse_daily_inventory_history_upload(
        BytesIO(raw),
        filename,
        sku_mapping=sess.sku_mapping or None,
    )
    if df.empty:
        return {
            "ok": False,
            "message": "No usable rows. Need a wide-format sheet: column 1 = SKU, "
            "column 2 = parent (optional), then daily snapshot columns whose first row is the date.",
        }

    # Trim to recent window before storing — prevents OOM during calculate_po_base
    # on multi-year baselines (e.g. 30M rows → ~3.5M for a 9,500-SKU catalog).
    df, trim_note = _trim_history_to_recent(df, _MAX_HISTORY_DAYS)

    from .daily_inventory_history import merge_inventory_history

    existing = getattr(sess, "daily_inventory_history_df", None)
    if existing is not None and not existing.empty:
        df = merge_inventory_history(existing, df)
        df, trim_note2 = _trim_history_to_recent(df, _MAX_HISTORY_DAYS)
        if trim_note2:
            trim_note = (trim_note + " " + trim_note2).strip() if trim_note else trim_note2

    sess.daily_inventory_history_df = df
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


def background_daily_inventory_upload(
    session_id: str,
    raw: bytes,
    filename: str,
) -> None:
    from ..session import store

    sess = store.get(session_id)
    if sess is None:
        return

    def _sync_sidecars() -> None:
        try:
            import backend.main as _main

            _main.merge_po_optional_sheets_into_warm_cache(sess)
        except Exception:
            logger.exception("merge_po_optional_sheets_into_warm_cache failed")
        try:
            from ..db.forecast_session_pg import persist_session_bundle_thread_safe

            persist_session_bundle_thread_safe(session_id, sess)
        except Exception:
            logger.exception("PostgreSQL persist after daily inventory upload failed")

    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_message = "Parsing daily inventory sheet…"
    try:
        result = execute_daily_inventory_upload(sess, raw, filename)
        sess.daily_inventory_upload_result = result
        if result.get("ok"):
            sess.daily_inventory_upload_status = "done"
            sess.daily_inventory_upload_message = result.get("message") or "Daily inventory loaded."
            # Respond to polls immediately; warm-cache + PG save can take minutes.
            threading.Thread(
                target=_sync_sidecars,
                daemon=True,
                name=f"daily-inv-save-{session_id[:8]}",
            ).start()
        else:
            sess.daily_inventory_upload_status = "error"
            sess.daily_inventory_upload_message = result.get("message") or "Upload failed."
    except Exception as e:
        logger.exception("background_daily_inventory_upload failed")
        sess.daily_inventory_upload_status = "error"
        sess.daily_inventory_upload_message = str(e)
        sess.daily_inventory_upload_result = {"ok": False, "message": str(e)}

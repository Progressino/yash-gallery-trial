"""Background daily inventory history upload — avoids 502 on large wide Excel files."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


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
    sess.daily_inventory_history_df = df
    sess._quarterly_cache.clear()
    skus = int(df["OMS_SKU"].nunique())
    days = int(pd.to_datetime(df["Date"], errors="coerce").dt.normalize().nunique())
    return {
        "ok": True,
        "rows": int(len(df)),
        "skus": skus,
        "days": days,
        "message": f"Loaded {len(df):,} SKU-day rows ({skus:,} SKUs × {days:,} days) for effective-days math.",
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
            _sync_sidecars()
        else:
            sess.daily_inventory_upload_status = "error"
            sess.daily_inventory_upload_message = result.get("message") or "Upload failed."
    except Exception as e:
        logger.exception("background_daily_inventory_upload failed")
        sess.daily_inventory_upload_status = "error"
        sess.daily_inventory_upload_message = str(e)
        sess.daily_inventory_upload_result = {"ok": False, "message": str(e)}

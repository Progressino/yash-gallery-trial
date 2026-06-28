#!/usr/bin/env python3
"""Roll inventory history to matrix end + snapshot, then persist (production)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)
_CACHE = Path("/data/warm_cache")


def _load_sales_window(start: str, end: str):
    import pandas as pd

    path = _CACHE / "sales_df.parquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return df
    date_col = "TxnDate" if "TxnDate" in df.columns else "Date"
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    lo = pd.Timestamp(start) - pd.Timedelta(days=1)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df[date_col] >= lo) & (df[date_col] <= hi)]


def main() -> int:
    import pandas as pd

    from backend.session import AppSession
    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        inventory_history_matrix_cap_date,
        inventory_sheet_end_date_from_filename,
        persist_daily_inventory_history_meta,
        promote_daily_inventory_matrix_max_date,
        read_daily_inventory_history_disk_meta,
        refresh_inventory_history_rollforward,
    )
    from backend.services.helpers import _coerce_df_for_parquet

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        _log.error("missing %s", hist_path)
        return 1

    meta_before = read_daily_inventory_history_disk_meta() or {}
    sess = AppSession()
    sess.daily_inventory_history_df = pd.read_parquet(hist_path)
    disk_meta = meta_before
    for key in (
        "daily_inventory_history_uploaded_at",
        "daily_inventory_history_filename",
        "daily_inventory_history_matrix_max_date",
    ):
        if disk_meta.get(key):
            setattr(sess, key, disk_meta[key])

    inv_meta_path = _CACHE / "inventory_session_meta.json"
    if inv_meta_path.is_file():
        inv_meta = json.loads(inv_meta_path.read_text(encoding="utf-8"))
        sess.inventory_snapshot_date = str(inv_meta.get("inventory_snapshot_date") or "")

    fn = str(getattr(sess, "daily_inventory_history_filename", "") or "")
    fn_end = inventory_sheet_end_date_from_filename(fn)
    if fn_end:
        promote_daily_inventory_matrix_max_date(sess, fn_end)
    cap = inventory_history_matrix_cap_date(sess)
    if cap is not None:
        promote_daily_inventory_matrix_max_date(sess, str(cap.date()))

    before = pd.to_datetime(sess.daily_inventory_history_df["Date"]).max()
    cap_s = str(cap.date()) if cap is not None else str(before.date())[:10]
    sales = _load_sales_window("2026-05-01", cap_s)
    if (_CACHE / "inventory_df_variant.parquet").is_file():
        sess.inventory_df_variant = pd.read_parquet(_CACHE / "inventory_df_variant.parquet")

    result = refresh_inventory_history_rollforward(sess, include_snapshot=True, sales_df=sales)
    _log.info("rollforward: %s", result)
    if not result.get("ok"):
        return 1

    after = pd.to_datetime(sess.daily_inventory_history_df["Date"]).max()
    if after.normalize() < pd.Timestamp("2026-06-20"):
        _log.error("history max still too old: %s", after.date())
        return 1

    _coerce_df_for_parquet(sess.daily_inventory_history_df).to_parquet(hist_path, index=False)
    persist_daily_inventory_history_meta(sess)
    meta = daily_inventory_history_meta_bundle(sess)
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, default=str, indent=2),
        encoding="utf-8",
    )
    try:
        import backend.main as main_mod

        if not main_mod._warm_cache:
            main_mod._warm_cache = {}
        main_mod._warm_cache["daily_inventory_history_df"] = sess.daily_inventory_history_df.copy()
        main_mod._warm_cache[main_mod._DAILY_INV_META_WARM_KEY] = meta
    except Exception:
        pass

    _log.info(
        "OK rows=%s max=%s matrix_max=%s",
        len(sess.daily_inventory_history_df),
        after.date(),
        meta.get("daily_inventory_history_matrix_max_date"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

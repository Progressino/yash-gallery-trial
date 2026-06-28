"""
Reconcile on-disk daily inventory history: promote matrix end date + roll forward.

Run on production:
  docker compose -p progressino -f docker-compose.prod.yml exec -T backend \
    python -m backend.scripts.reconcile_latest_inventory_history_disk
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> int:
    from pathlib import Path

    import pandas as pd

    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        inventory_history_matrix_cap_date,
        inventory_sheet_end_date_from_filename,
        persist_daily_inventory_history_meta,
        promote_daily_inventory_matrix_max_date,
        read_daily_inventory_history_disk_meta,
        refresh_inventory_history_rollforward,
    )
    from backend.session import AppSession

    cache = Path("/data/warm_cache")
    hist_path = cache / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        _log.warning("No daily_inventory_history_df.parquet at %s", hist_path)
        return 0

    meta_before = read_daily_inventory_history_disk_meta() or {}
    sess = AppSession()
    sess.daily_inventory_history_df = pd.read_parquet(hist_path)
    for key in (
        "daily_inventory_history_uploaded_at",
        "daily_inventory_history_filename",
        "daily_inventory_history_matrix_max_date",
    ):
        if meta_before.get(key):
            setattr(sess, key, meta_before[key])

    inv_meta_path = cache / "inventory_session_meta.json"
    if inv_meta_path.is_file():
        inv_meta = json.loads(inv_meta_path.read_text(encoding="utf-8"))
        sess.inventory_snapshot_date = str(inv_meta.get("inventory_snapshot_date") or "")

    sales_path = cache / "sales_df.parquet"
    if sales_path.is_file():
        sess.sales_df = pd.read_parquet(sales_path)
    inv_path = cache / "inventory_df_variant.parquet"
    if inv_path.is_file():
        sess.inventory_df_variant = pd.read_parquet(inv_path)

    fn = str(getattr(sess, "daily_inventory_history_filename", "") or "")
    fn_end = inventory_sheet_end_date_from_filename(fn)
    cap = inventory_history_matrix_cap_date(sess)
    before_max = pd.to_datetime(sess.daily_inventory_history_df["Date"]).max()
    _log.info(
        "Before: df_max=%s cap=%s filename_end=%s snapshot=%s",
        before_max.date() if pd.notna(before_max) else "?",
        cap.date() if cap is not None else "?",
        fn_end or "?",
        getattr(sess, "inventory_snapshot_date", ""),
    )

    if cap is not None:
        promote_daily_inventory_matrix_max_date(sess, str(cap.date()))

    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    _log.info("Rollforward: %s", result)
    if not result.get("ok"):
        return 1

    end_anchor = str(result.get("max_date") or "")
    if end_anchor:
        promote_daily_inventory_matrix_max_date(sess, end_anchor)
    cap = inventory_history_matrix_cap_date(sess)
    if cap is not None:
        promote_daily_inventory_matrix_max_date(sess, str(cap.date()))

    persist_daily_inventory_history_meta(sess)
    from backend.services.helpers import _coerce_df_for_parquet

    _coerce_df_for_parquet(sess.daily_inventory_history_df).to_parquet(hist_path, index=False)
    meta = daily_inventory_history_meta_bundle(sess)
    (cache / "daily_inventory_history_meta.json").write_text(
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
        _log.exception("warm cache seed failed")

    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        invalidate_all_shared_caches()
    except Exception:
        pass

    after_max = pd.to_datetime(sess.daily_inventory_history_df["Date"]).max()
    _log.info(
        "After: rows=%s df_max=%s matrix_max=%s",
        len(sess.daily_inventory_history_df),
        after_max.date() if pd.notna(after_max) else "?",
        meta.get("daily_inventory_history_matrix_max_date"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

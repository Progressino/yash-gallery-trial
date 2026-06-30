#!/usr/bin/env python3
"""Upload wide daily inventory history on production (disk + warm cache + meta)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def _hydrate_sess_from_disk(sess) -> None:
    """Load existing history so matrix re-upload can replace overlapping dates."""
    from backend.services.daily_inventory_history import (
        apply_daily_inventory_history_meta,
        read_daily_inventory_history_disk_meta,
    )

    cache = Path("/data/warm_cache")
    hist_path = cache / "daily_inventory_history_df.parquet"
    if hist_path.is_file():
        import pandas as pd

        sess.daily_inventory_history_df = pd.read_parquet(hist_path)
        meta = read_daily_inventory_history_disk_meta() or {}
        apply_daily_inventory_history_meta(sess, meta)
        _log.info(
            "Loaded existing history: %s rows, %s..%s",
            len(sess.daily_inventory_history_df),
            meta.get("daily_inventory_history_min_date"),
            meta.get("daily_inventory_history_max_date"),
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: upload_daily_inventory_production.py /path/to/matrix.xlsx [original_filename.xlsx]",
            file=sys.stderr,
        )
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        _log.error("File not found: %s", path)
        return 1

    raw = path.read_bytes()
    filename = sys.argv[2] if len(sys.argv) > 2 else path.name
    _log.info("Uploading %s (%s bytes)", filename, len(raw))

    from backend.session import AppSession
    from backend.services.daily_inventory_upload_run import execute_daily_inventory_upload

    sess = AppSession()
    try:
        import backend.main as _main

        _main.bootstrap_warm_cache_if_empty()
        wc = _main._warm_cache or {}
        if isinstance(wc.get("sku_mapping"), dict):
            sess.sku_mapping = wc["sku_mapping"]
    except Exception:
        _log.debug("warm cache sku_mapping unavailable", exc_info=True)
    _hydrate_sess_from_disk(sess)
    result = execute_daily_inventory_upload(sess, raw, filename)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("ok"):
        return 1

    try:
        import backend.main as _main

        _main.sync_daily_inventory_history_sidecar(sess)
    except Exception:
        _log.exception("sync_daily_inventory_history_sidecar failed")
        return 1

    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        invalidate_all_shared_caches()
    except Exception:
        pass

    df = sess.daily_inventory_history_df
    max_d = str(df["Date"].max())[:10] if df is not None and not df.empty else "?"
    _log.info("Done: %s rows, max_date=%s, matrix_max=%s", len(df), max_d, getattr(sess, "daily_inventory_history_matrix_max_date", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

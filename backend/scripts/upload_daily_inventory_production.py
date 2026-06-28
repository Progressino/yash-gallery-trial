#!/usr/bin/env python3
"""Upload wide daily inventory history on production (disk + warm cache + meta)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: upload_daily_inventory_production.py /path/to/matrix.xlsx", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        _log.error("File not found: %s", path)
        return 1

    raw = path.read_bytes()
    filename = path.name
    _log.info("Uploading %s (%s bytes)", filename, len(raw))

    from backend.session import AppSession
    from backend.services.daily_inventory_upload_run import execute_daily_inventory_upload

    sess = AppSession()
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

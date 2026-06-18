#!/usr/bin/env python3
"""Backfill indexed PostgreSQL tables from warm-cache blob or disk parquets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.db.forecast_ops_pg import (
        init_db,
        load_shared_snapshot,
        ops_pg_enabled,
        persist_shared_snapshot,
        warm_cache_dict_from_bundle,
    )
    from backend.db.forecast_ops_tables import (
        normalized_tables_enabled,
        persist_warm_cache_tables,
        tables_status,
    )
    import backend.main as main_mod

    init_db()
    if not ops_pg_enabled():
        print("ERROR: set FORECAST_SESSION_DATABASE_URL")
        return 1
    if not normalized_tables_enabled():
        print("ERROR: FORECAST_OPS_NORMALIZED is disabled")
        return 1

    cache = load_shared_snapshot()
    if not cache:
        main_mod.bootstrap_warm_cache_if_empty()
        cache = dict(main_mod._warm_cache or {})
    if not cache:
        disk_ok, disk_data = main_mod._load_warm_cache_from_disk(ignore_age=True)
        if disk_ok:
            cache = disk_data
    if not cache:
        print("No warm-cache data found on disk or PostgreSQL blob")
        return 1

    stats = persist_warm_cache_tables(cache)
    persist_shared_snapshot(cache)
    print("Backfill complete:", stats)
    print("Tables status:", tables_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

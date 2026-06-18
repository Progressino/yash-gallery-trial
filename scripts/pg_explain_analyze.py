#!/usr/bin/env python3
"""EXPLAIN ANALYZE hot forecast normalized-table queries; exit 1 on seq scans."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.db.explain_analyze import (
        NORMALIZED_HOT_QUERIES,
        plan_uses_index_scan,
        run_explain_analyze,
        seq_scans_in_plan,
    )
    from backend.db.forecast_ops_pg import _require_conn, init_db
    from backend.db.forecast_ops_tables import ensure_tables

    init_db()
    conn = _require_conn()
    if conn is None:
        print("ERROR: set FORECAST_SESSION_DATABASE_URL")
        return 1

    with conn:
        ensure_tables(conn)

    issues = 0
    for label, (sql, params) in NORMALIZED_HOT_QUERIES:
        print("=" * 72)
        print(label)
        print("-" * 72)
        try:
            plan = run_explain_analyze(conn, sql, params or None)
        except Exception as exc:
            print(f"ERROR: {exc}")
            issues += 1
            continue
        print(plan)
        seq = seq_scans_in_plan(plan)
        if seq:
            issues += 1
            for table, rows in seq:
                extra = f" rows≈{rows:,}" if rows is not None else ""
                print(f"WARN Seq Scan on {table}{extra} — missing or unused index")
        elif plan_uses_index_scan(plan):
            print("OK Index Scan / Bitmap Index Scan")
        else:
            print("NOTE verify plan above (no seq scan on forecast sales/inventory)")
        print()

    if issues:
        print(f"DONE: {issues} query plan(s) need index attention")
        return 1
    print("DONE: hot queries use indexes (no seq scan on forecast sales/inventory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

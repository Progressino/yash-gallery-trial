#!/usr/bin/env python3
"""EXPLAIN ANALYZE on forecast sales/inventory queries; audit indexes before CREATE."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_sales_audit(conn) -> int:
    from backend.db.sales_index_audit import (
        PROPOSED_SALES_INDEXES,
        SALES_PROBE_QUERIES,
        analyze_sales_query_plans,
        apply_recommended_indexes,
        find_equivalent_index,
        index_names_in_plan,
        list_table_indexes,
        recommend_sales_indexes,
    )
    from backend.db.explain_analyze import seq_scans_in_plan

    print("=" * 72)
    print("forecast_sales_transactions — existing indexes")
    print("-" * 72)
    existing = list_table_indexes(conn)
    if not existing:
        print("(none — table empty or not migrated)")
    for name, indexdef in existing:
        print(f"  {name}")
        print(f"    {indexdef}")

    print()
    print("=" * 72)
    print("Proposed ERP indexes vs existing coverage")
    print("-" * 72)
    for spec in PROPOSED_SALES_INDEXES:
        equiv = find_equivalent_index(existing, spec)
        cols = ", ".join(spec.columns)
        if equiv:
            print(f"  {spec.name} ({cols}) — covered by {equiv}")
        else:
            print(f"  {spec.name} ({cols}) — NOT present")

    print()
    issues = 0
    plans = analyze_sales_query_plans(conn)
    for result in plans:
        print("=" * 72)
        print(result.label)
        print(f"  expected index: {result.expected_index or '—'}")
        print("-" * 72)
        print(result.plan)
        seq = seq_scans_in_plan(result.plan, tables=("forecast_sales_transactions",))
        if seq:
            issues += 1
            for table, rows in seq:
                extra = f" rows≈{rows:,}" if rows is not None else ""
                print(f"WARN Seq Scan on {table}{extra}")
        elif result.index_names:
            print(f"OK uses: {', '.join(result.index_names)}")
            if result.rows_filtered >= 5000:
                print(
                    f"NOTE {result.rows_filtered:,} rows removed by Filter — "
                    "check sku+platform composite below"
                )
        else:
            print("NOTE verify plan (no seq scan on sales table)")
        print()

    print("=" * 72)
    print("Recommendations (CREATE only when EXPLAIN shows a gap)")
    print("-" * 72)
    recs = recommend_sales_indexes(conn, plans)
    create_recs = [r for r in recs if not r.satisfied_by]
    ops_notes = [r for r in recs if r.satisfied_by]
    if not recs:
        print("  None — hot queries use indexes; no new btree needed.")
    for rec in ops_notes:
        print(f"  OPS: {rec.reason}")
    for rec in create_recs:
        print(f"  CREATE {rec.spec.name}: {rec.reason}")
        print(f"        {rec.spec.ddl}")
        issues += 1
    print()
    return issues, create_recs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="CREATE indexes recommended by EXPLAIN ANALYZE audit (skips when equivalent exists)",
    )
    parser.add_argument(
        "--sales-only",
        action="store_true",
        help="Run sales index audit only (skip inventory hot queries)",
    )
    args = parser.parse_args()

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

    if not args.sales_only:
        for label, (sql, params) in NORMALIZED_HOT_QUERIES:
            if "forecast_sales_transactions" not in sql and args.sales_only:
                continue
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
                print("NOTE verify plan above")
            print()

    sales_issues, create_recs = _print_sales_audit(conn)
    issues += sales_issues

    if args.apply and create_recs:
        from backend.db.sales_index_audit import apply_recommended_indexes

        ddls = apply_recommended_indexes(conn, create_recs, dry_run=False)
        for ddl in ddls:
            print(f"APPLIED: {ddl}")
    elif create_recs:
        print("Run with --apply to CREATE recommended indexes after reviewing plans above.")

    if issues:
        print(f"DONE: {issues} item(s) need attention")
        return 1
    print("DONE: hot queries use indexes; no new sales btree required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

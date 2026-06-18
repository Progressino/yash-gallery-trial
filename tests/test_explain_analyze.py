"""EXPLAIN ANALYZE plan parsing."""
from __future__ import annotations

from backend.db.explain_analyze import (
    plan_uses_index_scan,
    seq_scans_in_plan,
)


def test_seq_scans_in_plan_detects_large_scan():
    plan = """
Index Scan using idx_fst_platform_txn_date on forecast_sales_transactions
  Index Cond: (platform = 'amazon'::text)
Seq Scan on forecast_inventory_lines
  Filter: (snapshot_id = 1)
  Rows Removed by Filter: 0
  rows=1300000
"""
    seq = seq_scans_in_plan(plan)
    assert len(seq) == 1
    assert seq[0][0] == "forecast_inventory_lines"
    assert seq[0][1] == 1300000


def test_plan_uses_index_scan():
    plan = """
Bitmap Index Scan on idx_fst_platform_txn_date
  Index Cond: (platform = 'amazon'::text)
"""
    assert plan_uses_index_scan(plan) is True


def test_seq_scans_ignores_other_tables():
    plan = "Seq Scan on forecast_app_sessions\n  rows=10"
    seq = seq_scans_in_plan(plan)
    assert seq == []

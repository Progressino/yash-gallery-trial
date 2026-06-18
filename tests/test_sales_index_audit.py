"""Tests for EXPLAIN-driven sales index audit."""
from __future__ import annotations

from backend.db.sales_index_audit import (
    PROPOSED_SALES_INDEXES,
    IndexRecommendation,
    QueryPlanResult,
    find_equivalent_index,
    index_names_in_plan,
    plan_has_sku_filter,
    recommend_sales_indexes,
    rows_removed_by_filter,
)


def test_index_names_in_plan():
    plan = """
Bitmap Index Scan on idx_fst_platform_txn_date
  Index Cond: (platform = 'amazon'::text)
"""
    assert index_names_in_plan(plan) == ["idx_fst_platform_txn_date"]


def test_rows_removed_by_filter():
    plan = """
Index Scan using idx_fst_platform_txn_date on forecast_sales_transactions
  Filter: (sku = 'ABC'::text)
  Rows Removed by Filter: 12000
"""
    assert rows_removed_by_filter(plan) == 12000
    assert plan_has_sku_filter(plan) is True


def test_find_equivalent_platform_index():
    existing = [
        (
            "idx_fst_platform_txn_date_asc",
            "CREATE INDEX idx_fst_platform_txn_date_asc ON forecast_sales_transactions "
            "(platform, txn_date)",
        ),
    ]
    spec = next(s for s in PROPOSED_SALES_INDEXES if s.name == "idx_sales_platform_date")
    assert find_equivalent_index(existing, spec) == "idx_fst_platform_txn_date_asc"


def test_find_equivalent_sku_index_desc_covers_asc():
    existing = [
        (
            "idx_fst_sku_txn_date",
            "CREATE INDEX idx_fst_sku_txn_date ON forecast_sales_transactions "
            "(sku, txn_date DESC)",
        ),
    ]
    spec = next(s for s in PROPOSED_SALES_INDEXES if s.name == "idx_sales_sku_date")
    assert find_equivalent_index(existing, spec) == "idx_fst_sku_txn_date"


def test_recommend_skips_when_platform_index_covers():
    plans = [
        QueryPlanResult(
            label="platform + date window (load_platform_sales_dataframe)",
            plan="Index Scan using idx_fst_platform_txn_date_asc on forecast_sales_transactions",
            seq_scan=False,
            index_names=["idx_fst_platform_txn_date_asc"],
            rows_filtered=0,
            expected_index="idx_sales_platform_date",
        ),
    ]
    existing = [
        (
            "idx_fst_platform_txn_date_asc",
            "CREATE INDEX idx_fst_platform_txn_date_asc ON forecast_sales_transactions "
            "(platform, txn_date)",
        ),
    ]

    class _Conn:
        def execute(self, *a, **k):
            class _R:
                def fetchall(_self):
                    return existing

            return _R()

    assert recommend_sales_indexes(_Conn(), plans) == []


def test_recommend_triple_index_on_heavy_sku_filter():
    plans = [
        QueryPlanResult(
            label="sku + platform + date window (PO per marketplace)",
            plan="""
Index Scan using idx_fst_platform_txn_date on forecast_sales_transactions
  Index Cond: (platform = 'amazon'::text)
  Filter: (sku = 'X'::text)
  Rows Removed by Filter: 80000
""",
            seq_scan=False,
            index_names=["idx_fst_platform_txn_date"],
            rows_filtered=80000,
            expected_index="idx_sales_sku_platform_date",
        ),
    ]

    class _Conn:
        def execute(self, *a, **k):
            class _R:
                def fetchall(_self):
                    return [
                        (
                            "idx_fst_platform_txn_date",
                            "CREATE INDEX idx_fst_platform_txn_date ON forecast_sales_transactions "
                            "(platform, txn_date DESC)",
                        ),
                    ]

            return _R()

    recs = recommend_sales_indexes(_Conn(), plans)
    assert len(recs) == 1
    assert recs[0].spec.name == "idx_sales_sku_platform_date"
    assert "sku Filter" in recs[0].reason


def test_recommend_seq_scan_when_no_equivalent():
    plans = [
        QueryPlanResult(
            label="sku + date window (PO / single-SKU ADS)",
            plan="Seq Scan on forecast_sales_transactions\n  rows=500000",
            seq_scan=True,
            index_names=[],
            rows_filtered=0,
            expected_index="idx_sales_sku_date",
        ),
    ]

    class _Conn:
        def execute(self, *a, **k):
            class _R:
                def fetchall(_self):
                    return []

            return _R()

    recs = recommend_sales_indexes(_Conn(), plans)
    assert len(recs) == 1
    assert recs[0].spec.name == "idx_sales_sku_date"
    assert "sequential scan" in recs[0].reason

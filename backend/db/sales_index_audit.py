"""EXPLAIN ANALYZE–driven index audit for ``forecast_sales_transactions``.

Proposed ERP indexes (audit before create — do not add blindly):

  idx_sales_platform_date      (platform, txn_date)
  idx_sales_sku_date           (sku, txn_date)
  idx_sales_sku_platform_date  (sku, platform, txn_date)

Existing equivalents are detected via ``pg_indexes`` so duplicate btree indexes
are not created under a second name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .explain_analyze import (
    run_explain_analyze,
    seq_scans_in_plan,
)

_TABLE = "forecast_sales_transactions"

_INDEX_SCAN_RE = re.compile(
    r"(?:Index Scan using|Bitmap Index Scan on|Index Only Scan using)\s+(\w+)",
    re.IGNORECASE,
)
_ROWS_REMOVED_FILTER_RE = re.compile(
    r"Rows Removed by Filter:\s*(\d+)",
    re.IGNORECASE,
)
_SKU_FILTER_RE = re.compile(r"\bsku\s*=", re.IGNORECASE)

# Probe queries mirror ``load_platform_sales_dataframe`` and likely PO/ADS lookups.
SALES_PROBE_QUERIES: tuple[tuple[str, str, tuple], ...] = (
    (
        "platform + date window (load_platform_sales_dataframe)",
        """
        SELECT sku, txn_date, quantity, transaction_type, order_id,
               line_key, dsr_segment, source_file, units_effective, extra
        FROM forecast_sales_transactions
        WHERE platform = %s AND txn_date >= NOW() - INTERVAL '180 days'
        ORDER BY txn_date
        """,
        ("amazon",),
    ),
    (
        "platform all rows (load_platform_sales_dataframe)",
        """
        SELECT sku, txn_date, quantity, transaction_type, order_id,
               line_key, dsr_segment, source_file, units_effective, extra
        FROM forecast_sales_transactions
        WHERE platform = %s
        ORDER BY txn_date
        """,
        ("amazon",),
    ),
    (
        "sku + date window (PO / single-SKU ADS)",
        """
        SELECT sku, txn_date, quantity, platform
        FROM forecast_sales_transactions
        WHERE sku = %s AND txn_date >= NOW() - INTERVAL '180 days'
        ORDER BY txn_date
        """,
        ("1001YKBEIGE-3XL",),
    ),
    (
        "sku + platform + date window (PO per marketplace)",
        """
        SELECT sku, txn_date, quantity
        FROM forecast_sales_transactions
        WHERE sku = %s AND platform = %s AND txn_date >= NOW() - INTERVAL '180 days'
        ORDER BY txn_date
        """,
        ("1001YKBEIGE-3XL", "amazon"),
    ),
)


@dataclass(frozen=True)
class IndexSpec:
    name: str
    columns: tuple[str, ...]
    ddl: str
    # Existing names that satisfy the same leading column prefix (from ensure_tables).
    legacy_names: tuple[str, ...] = ()


PROPOSED_SALES_INDEXES: tuple[IndexSpec, ...] = (
    IndexSpec(
        name="idx_sales_platform_date",
        columns=("platform", "txn_date"),
        ddl=(
            f"CREATE INDEX IF NOT EXISTS idx_sales_platform_date "
            f"ON {_TABLE} (platform, txn_date)"
        ),
        legacy_names=("idx_fst_platform_txn_date_asc", "idx_fst_platform_txn_date"),
    ),
    IndexSpec(
        name="idx_sales_sku_date",
        columns=("sku", "txn_date"),
        ddl=(
            f"CREATE INDEX IF NOT EXISTS idx_sales_sku_date "
            f"ON {_TABLE} (sku, txn_date)"
        ),
        legacy_names=("idx_fst_sku_txn_date",),
    ),
    IndexSpec(
        name="idx_sales_sku_platform_date",
        columns=("sku", "platform", "txn_date"),
        ddl=(
            f"CREATE INDEX IF NOT EXISTS idx_sales_sku_platform_date "
            f"ON {_TABLE} (sku, platform, txn_date)"
        ),
        legacy_names=(),
    ),
)

# Map probe queries to the index spec that should serve them.
_PROBE_INDEX_HINT: dict[str, str] = {
    "platform + date window (load_platform_sales_dataframe)": "idx_sales_platform_date",
    "platform all rows (load_platform_sales_dataframe)": "idx_sales_platform_date",
    "sku + date window (PO / single-SKU ADS)": "idx_sales_sku_date",
    "sku + platform + date window (PO per marketplace)": "idx_sales_sku_platform_date",
}

_FILTER_ROWS_THRESHOLD = int(__import__("os").environ.get("DB_INDEX_FILTER_ROWS_THRESHOLD", "5000"))


def index_names_in_plan(plan: str) -> list[str]:
    return list(dict.fromkeys(_INDEX_SCAN_RE.findall(plan or "")))


def rows_removed_by_filter(plan: str) -> int:
    return sum(int(m.group(1)) for m in _ROWS_REMOVED_FILTER_RE.finditer(plan or ""))


def plan_has_sku_filter(plan: str) -> bool:
    return bool(_SKU_FILTER_RE.search(plan or ""))


@dataclass
class QueryPlanResult:
    label: str
    plan: str
    seq_scan: bool
    index_names: list[str]
    rows_filtered: int
    expected_index: str


@dataclass
class IndexRecommendation:
    spec: IndexSpec
    reason: str
    satisfied_by: str | None = None  # existing index name when equivalent already present


def list_table_indexes(conn: Any) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = %s
        ORDER BY indexname
        """,
        (_TABLE,),
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def _normalize_indexdef(indexdef: str) -> tuple[str, ...]:
    """Extract ordered column list from pg_indexes.indexdef."""
    m = re.search(r"\(([^)]+)\)", indexdef or "", re.IGNORECASE)
    if not m:
        return ()
    cols: list[str] = []
    for part in m.group(1).split(","):
        token = part.strip().split()[0].strip('"').lower()
        if token:
            cols.append(token)
    return tuple(cols)


def find_equivalent_index(existing: list[tuple[str, str]], spec: IndexSpec) -> str | None:
    want = tuple(c.lower() for c in spec.columns)
    for name, indexdef in existing:
        if name in spec.legacy_names:
            return name
        cols = _normalize_indexdef(indexdef)
        if not cols:
            continue
        if cols[: len(want)] == want:
            return name
        # DESC variant covers ASC probes (btree backward scan).
        if len(want) == 2 and cols[:2] == want[:2]:
            return name
    return None


def analyze_sales_query_plans(conn: Any) -> list[QueryPlanResult]:
    out: list[QueryPlanResult] = []
    for label, sql, params in SALES_PROBE_QUERIES:
        plan = run_explain_analyze(conn, sql, params)
        seq = bool(seq_scans_in_plan(plan, tables=(_TABLE,)))
        out.append(
            QueryPlanResult(
                label=label,
                plan=plan,
                seq_scan=seq,
                index_names=index_names_in_plan(plan),
                rows_filtered=rows_removed_by_filter(plan),
                expected_index=_PROBE_INDEX_HINT.get(label, ""),
            )
        )
    return out


def recommend_sales_indexes(
    conn: Any,
    plans: list[QueryPlanResult] | None = None,
) -> list[IndexRecommendation]:
    """Return indexes to CREATE — empty when EXPLAIN shows existing coverage."""
    existing = list_table_indexes(conn)
    if plans is None:
        plans = analyze_sales_query_plans(conn)

    by_name = {s.name: s for s in PROPOSED_SALES_INDEXES}
    out: dict[str, IndexRecommendation] = {}

    for result in plans:
        spec = by_name.get(result.expected_index)
        if spec is None:
            continue
        equiv = find_equivalent_index(existing, spec)

        if result.seq_scan:
            out[spec.name] = IndexRecommendation(
                spec=spec,
                reason=f"{result.label}: sequential scan on {_TABLE}",
                satisfied_by=equiv,
            )
            continue

        if (
            spec.name == "idx_sales_sku_platform_date"
            and not equiv
            and result.rows_filtered >= _FILTER_ROWS_THRESHOLD
            and plan_has_sku_filter(result.plan)
            and spec.name not in result.index_names
        ):
            out[spec.name] = IndexRecommendation(
                spec=spec,
                reason=(
                    f"{result.label}: {result.rows_filtered:,} rows removed by sku Filter "
                    "after platform index scan"
                ),
            )

    return list(out.values())


def apply_recommended_indexes(
    conn: Any,
    recommendations: list[IndexRecommendation],
    *,
    dry_run: bool = True,
) -> list[str]:
    """Execute CREATE INDEX for recommendations that are not already satisfied."""
    existing = list_table_indexes(conn)
    applied: list[str] = []
    for rec in recommendations:
        if rec.satisfied_by or find_equivalent_index(existing, rec.spec):
            continue
        applied.append(rec.spec.ddl)
        if not dry_run:
            conn.execute(rec.spec.ddl)
    return applied


def index_audit_apply_enabled() -> bool:
    raw = __import__("os").environ.get("FORECAST_OPS_INDEX_AUDIT_APPLY", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ensure_sales_indexes_from_audit(conn: Any) -> list[str]:
    """Run EXPLAIN ANALYZE audit and CREATE only missing indexes (opt-in via env)."""
    if not index_audit_apply_enabled():
        return []
    plans = analyze_sales_query_plans(conn)
    recs = recommend_sales_indexes(conn, plans)
    create_recs = [r for r in recs if not r.satisfied_by]
    return apply_recommended_indexes(conn, create_recs, dry_run=False)

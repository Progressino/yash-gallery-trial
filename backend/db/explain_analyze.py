"""EXPLAIN ANALYZE helpers for PostgreSQL query tuning."""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger("db.perf")

_FORECAST_TABLES = (
    "forecast_sales_transactions",
    "forecast_inventory_lines",
    "forecast_inventory_snapshots",
    "forecast_daily_uploads",
    "forecast_shared_snapshot",
)

_SEQ_SCAN_RE = re.compile(
    r"Seq Scan on (\w+)",
    re.IGNORECASE,
)


def explain_analyze_enabled() -> bool:
    raw = (os.environ.get("DB_EXPLAIN_ANALYZE") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _is_select_statement(statement: Any) -> bool:
    text = str(statement or "").lstrip().upper()
    return text.startswith("SELECT") or text.startswith("WITH")


def _touches_forecast_table(statement: Any) -> bool:
    text = str(statement or "").lower()
    return any(t in text for t in _FORECAST_TABLES)


def seq_scans_in_plan(plan: str, *, tables: Iterable[str] | None = None) -> list[tuple[str, int | None]]:
    """Return [(table_name, row_estimate), ...] for sequential scans in a plan."""
    want = {t.lower() for t in (tables or _FORECAST_TABLES)}
    lines = plan.splitlines()
    out: list[tuple[str, int | None]] = []
    for i, line in enumerate(lines):
        m = _SEQ_SCAN_RE.search(line)
        if not m:
            continue
        table = m.group(1)
        if table.lower() not in want:
            continue
        rows_m = re.search(r"rows=(\d+)", line, re.I)
        if not rows_m:
            for j in range(i + 1, min(i + 4, len(lines))):
                rows_m = re.search(r"rows=(\d+)", lines[j], re.I)
                if rows_m:
                    break
        rows = int(rows_m.group(1)) if rows_m else None
        out.append((table, rows))
    return out


def plan_uses_index_scan(plan: str) -> bool:
    upper = plan.upper()
    return "INDEX SCAN" in upper or "BITMAP INDEX SCAN" in upper or "INDEX ONLY SCAN" in upper


def run_explain_analyze(conn: Any, statement: Any, params: Any = None) -> str:
    """Run EXPLAIN (ANALYZE, BUFFERS) for a SELECT (executes the query)."""
    sql = str(statement or "").strip()
    if not _is_select_statement(sql):
        return ""
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"
    if params is None:
        cur = conn.execute(explain_sql)
    else:
        cur = conn.execute(explain_sql, params)
    rows = cur.fetchall()
    return "\n".join(str(r[0]) for r in rows)


def maybe_explain_slow_query(
    conn: Any,
    statement: Any,
    duration: float,
    params: Any = None,
    *,
    threshold_sec: float | None = None,
) -> None:
    """Log EXPLAIN ANALYZE for slow SELECTs on forecast tables; warn on seq scans."""
    if conn is None or not explain_analyze_enabled():
        return
    if not _is_select_statement(statement) or not _touches_forecast_table(statement):
        return
    if threshold_sec is None:
        from .query_logging import slow_query_threshold_sec

        threshold_sec = slow_query_threshold_sec()
    if duration < threshold_sec:
        return
    try:
        plan = run_explain_analyze(conn, statement, params)
    except Exception:
        logger.exception("EXPLAIN ANALYZE failed")
        return
    if not plan:
        return
    seq = seq_scans_in_plan(plan)
    if seq:
        parts = [f"{t} ({rows:,} rows)" if rows is not None else t for t, rows in seq]
        logger.warning(
            "Seq Scan on %s — consider adding or fixing an index.\n%s",
            ", ".join(parts),
            plan,
        )
    else:
        logger.info("EXPLAIN ANALYZE (%.2fs):\n%s", duration, plan)


# Canonical hot-path queries for scripts/pg_explain_analyze.sh
NORMALIZED_HOT_QUERIES: tuple[tuple[str, tuple], ...] = (
    (
        "sales by platform (6mo window)",
        (
            """
            SELECT sku, txn_date, quantity, transaction_type
            FROM forecast_sales_transactions
            WHERE platform = %s AND txn_date >= NOW() - INTERVAL '180 days'
            ORDER BY txn_date
            """,
            ("amazon",),
        ),
    ),
    (
        "sales by platform (all rows)",
        (
            """
            SELECT sku, txn_date, quantity
            FROM forecast_sales_transactions
            WHERE platform = %s
            ORDER BY txn_date
            """,
            ("amazon",),
        ),
    ),
    (
        "inventory current snapshot lines",
        (
            """
            SELECT l.oms_sku, l.total_inventory
            FROM forecast_inventory_lines l
            JOIN forecast_inventory_snapshots s ON s.id = l.snapshot_id
            WHERE s.is_current = TRUE
            """,
            (),
        ),
    ),
    (
        "inventory lines by snapshot_id",
        (
            """
            SELECT oms_sku, total_inventory
            FROM forecast_inventory_lines
            WHERE snapshot_id = (
                SELECT id FROM forecast_inventory_snapshots
                WHERE is_current = TRUE
                ORDER BY uploaded_at DESC
                LIMIT 1
            )
            """,
            (),
        ),
    ),
    (
        "sales count by platform",
        (
            """
            SELECT platform, COUNT(*)
            FROM forecast_sales_transactions
            GROUP BY platform
            """,
            (),
        ),
    ),
)

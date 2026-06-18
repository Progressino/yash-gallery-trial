"""Compact dashboard aggregates from Tier-3 / PostgreSQL (no session frame copy)."""
from __future__ import annotations

from typing import Any, Optional


def _compact_platforms(platform_summary: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in platform_summary or []:
        name = str(row.get("platform") or "").strip().lower()
        key = {
            "amazon": "amazon",
            "myntra": "myntra",
            "meesho": "meesho",
            "flipkart": "flipkart",
            "snapdeal": "snapdeal",
        }.get(name, name.replace(" ", "_"))
        if not key:
            continue
        out[key] = {
            "sales": int(row.get("total_units") or 0),
            "returns": int(row.get("total_returns") or 0),
            "net": int(row.get("net_units") or 0),
            "loaded": bool(row.get("loaded")),
        }
    return out


def _summary_from_pg(start_date: str, end_date: str, limit: int) -> dict[str, Any] | None:
    try:
        from ..db.forecast_ops_pg import _require_conn
        from ..db.forecast_ops_tables import normalized_tables_enabled

        if not normalized_tables_enabled():
            return None
        conn = _require_conn()
        if conn is None:
            return None
        s, e = start_date[:10], end_date[:10]
        with conn:
            plat_rows = conn.execute(
                """
                SELECT platform,
                       SUM(CASE WHEN LOWER(transaction_type) LIKE 'ship%%' THEN quantity ELSE 0 END),
                       SUM(CASE WHEN LOWER(transaction_type) LIKE '%%refund%%'
                                     OR LOWER(transaction_type) LIKE '%%return%%'
                                THEN ABS(quantity) ELSE 0 END)
                FROM forecast_sales_transactions
                WHERE platform IN ('amazon', 'myntra', 'meesho', 'flipkart', 'snapdeal')
                  AND txn_date::date >= %s::date
                  AND txn_date::date <= %s::date
                GROUP BY platform
                """,
                (s, e),
            ).fetchall()
            top_rows = conn.execute(
                """
                SELECT sku,
                       SUM(CASE WHEN LOWER(transaction_type) LIKE 'ship%%' THEN quantity ELSE 0 END) AS units
                FROM forecast_sales_transactions
                WHERE platform IN ('amazon', 'myntra', 'meesho', 'flipkart', 'snapdeal', 'unified')
                  AND txn_date::date >= %s::date
                  AND txn_date::date <= %s::date
                GROUP BY sku
                HAVING SUM(CASE WHEN LOWER(transaction_type) LIKE 'ship%%' THEN quantity ELSE 0 END) > 0
                ORDER BY units DESC
                LIMIT %s
                """,
                (s, e, int(limit)),
            ).fetchall()
        if not plat_rows and not top_rows:
            return None
        platforms = {
            str(r[0]): {
                "sales": int(r[1] or 0),
                "returns": int(r[2] or 0),
                "net": int((r[1] or 0) - (r[2] or 0)),
                "loaded": int(r[1] or 0) > 0,
            }
            for r in plat_rows
        }
        return {
            "source": "postgres_normalized",
            "platforms": platforms,
            "top_skus": [{"sku": str(r[0]), "units": float(r[1] or 0)} for r in top_rows],
            "sales_summary": {
                "total_units": sum(p["sales"] for p in platforms.values()),
                "total_returns": sum(p["returns"] for p in platforms.values()),
            },
        }
    except Exception:
        return None


def build_dashboard_summary(
    sess,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Lightweight dashboard aggregates — Tier-3 / PG first, never requires session platform copies.
    """
    s = str(start_date or "")[:10]
    e = str(end_date or "")[:10]
    has_window = len(s) == 10 and len(e) == 10

    if has_window:
        try:
            from ..routers.data import _build_intelligence_bundle_payload_from_tier3

            tier3 = _build_intelligence_bundle_payload_from_tier3(
                sess, s, e, int(limit), "gross", include_extras=False
            )
            if tier3 and tier3.get("platform_summary"):
                return {
                    "source": "tier3_sqlite",
                    "platforms": _compact_platforms(tier3.get("platform_summary") or []),
                    "top_skus": tier3.get("top_skus") or [],
                    "sales_summary": tier3.get("sales_summary") or {},
                }
        except Exception:
            pass

        pg = _summary_from_pg(s, e, limit)
        if pg:
            return pg

    pg_counts = {}
    try:
        from .intelligence_readiness import _pg_platform_sales_counts

        pg_counts = _pg_platform_sales_counts()
    except Exception:
        pass
    if pg_counts:
        return {
            "source": "postgres_normalized",
            "platforms": {
                k: {"sales": int(v), "loaded": int(v) > 0}
                for k, v in pg_counts.items()
                if k in ("amazon", "myntra", "meesho", "flipkart", "snapdeal", "unified")
            },
            "top_skus": [],
            "sales_summary": {"total_units": int(pg_counts.get("unified", 0) or 0)},
        }

    return {
        "source": "none",
        "platforms": {},
        "top_skus": [],
        "sales_summary": {},
        "message": "No dashboard aggregates available for this window yet.",
    }

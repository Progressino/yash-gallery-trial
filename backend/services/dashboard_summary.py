"""Compact dashboard aggregates — prebuilt artifacts first, Tier-3 fallback only."""
from __future__ import annotations

import time
from typing import Any, Optional

_SUMMARY_CACHE: dict[tuple, dict[str, Any]] = {}
_SUMMARY_CACHE_TTL_SEC = int(__import__("os").environ.get("DASHBOARD_SUMMARY_CACHE_TTL", "180"))


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


def _summary_cache_key(s: str, e: str, limit: int) -> tuple:
    try:
        from .intelligence_artifacts import intelligence_version_for_window

        ver = intelligence_version_for_window(s, e)
    except Exception:
        ver = ""
    return (s, e, int(limit), ver)


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


def _attach_version_meta(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["version"] = str(meta.get("current_version") or meta.get("version") or "")
    out["stale"] = bool(meta.get("stale"))
    out["refresh_queued"] = bool(meta.get("stale"))
    if meta.get("source"):
        out["source"] = f"artifact_{meta['source']}"
    return out


def _artifact_covers_tier3_window(payload: dict[str, Any], start_date: str, end_date: str) -> bool:
    """Reject hot artifacts that only show a subset of platforms Tier-3 has for this window."""
    try:
        from ..services.daily_store import platforms_with_uploads_in_range

        expected = {
            str(p).strip().lower()
            for p in (platforms_with_uploads_in_range(start_date, end_date) or [])
            if str(p).strip()
        }
    except Exception:
        return True
    if not expected:
        return True
    loaded: set[str] = set()
    for row in payload.get("platform_summary") or []:
        if int(row.get("total_units") or 0) > 0:
            loaded.add(str(row.get("platform") or "").strip().lower())
    for key, row in (payload.get("platforms") or {}).items():
        if int((row or {}).get("sales") or 0) > 0:
            loaded.add(str(key).strip().lower())
    core_expected = expected & {"amazon", "flipkart", "meesho", "myntra"}
    if not core_expected:
        return True
    return core_expected.issubset(loaded)


def build_dashboard_summary(
    sess,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Hot path: prebuilt artifact → in-process cache → Tier-3 fallback (last resort).
    """
    s = str(start_date or "")[:10]
    e = str(end_date or "")[:10]
    has_window = len(s) == 10 and len(e) == 10

    if has_window:
        cache_key = _summary_cache_key(s, e, limit)
        try:
            from ..services.platform_session_window import session_platform_shorter_than_tier3
            from ..routers.data import _tier3_token_mismatch

            prefer_tier3 = session_platform_shorter_than_tier3(sess) or _tier3_token_mismatch(sess)
        except Exception:
            prefer_tier3 = False

        if not prefer_tier3:
            hit = _SUMMARY_CACHE.get(cache_key)
            if hit and (time.time() - float(hit.get("_ts", 0))) < _SUMMARY_CACHE_TTL_SEC:
                cached = {k: v for k, v in hit.items() if k != "_ts"}
                if _artifact_covers_tier3_window(cached, s, e):
                    return cached

        if not prefer_tier3:
            try:
                from .intelligence_artifacts import (
                    load_hot_summary_for_request,
                    save_artifact,
                    schedule_artifact_build,
                    KIND_HOT,
                )

                artifact, meta = load_hot_summary_for_request(sess, s, e, limit)
                if artifact and _artifact_covers_tier3_window(artifact, s, e):
                    payload = _attach_version_meta(artifact, meta)
                    _SUMMARY_CACHE[cache_key] = {**payload, "_ts": time.time()}
                    return payload
            except Exception:
                pass

        # Tier-3 fallback when no artifact exists or session lags SQLite dailies.
        try:
            from ..routers.data import _build_intelligence_bundle_payload_from_tier3
            from .intelligence_artifacts import save_artifact, schedule_artifact_build, KIND_HOT

            tier3 = _build_intelligence_bundle_payload_from_tier3(
                sess,
                s,
                e,
                int(limit),
                "gross",
                include_extras=False,
                headline_only=True,
            )
            if tier3 and tier3.get("platform_summary"):
                payload = {
                    "source": "tier3_sqlite_fallback",
                    "platforms": _compact_platforms(tier3.get("platform_summary") or []),
                    "platform_summary": tier3.get("platform_summary") or [],
                    "top_skus": tier3.get("top_skus") or [],
                    "sales_summary": tier3.get("sales_summary") or {},
                    "data_completeness": "partial",
                }
                try:
                    from .intelligence_artifacts import intelligence_version_for_window

                    ver = save_artifact(s, e, KIND_HOT, payload)
                    payload["version"] = ver
                except Exception:
                    schedule_artifact_build(s, e, KIND_HOT, limit=limit)
                _SUMMARY_CACHE[cache_key] = {**payload, "_ts": time.time()}
                return payload
        except Exception:
            pass

        pg = _summary_from_pg(s, e, limit)
        if pg:
            _SUMMARY_CACHE[cache_key] = {**pg, "_ts": time.time()}
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

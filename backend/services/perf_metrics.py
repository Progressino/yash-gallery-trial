"""In-process + PostgreSQL performance metrics for the admin dashboard."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
_MEM_MAX = int(os.environ.get("PERF_METRICS_MEM_MAX", "8000"))
_RETAIN_DAYS = int(os.environ.get("PERF_METRICS_RETAIN_DAYS", "30"))

_lock = threading.Lock()
_events: deque[dict[str, Any]] = deque(maxlen=_MEM_MAX)
_cache_hits = 0
_cache_misses = 0
_table_ready = False


def perf_metrics_enabled() -> bool:
    raw = (os.environ.get("PERF_METRICS") or os.environ.get("PERF_TIMING") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def day_ist(when: datetime | None = None) -> date:
    dt = when or datetime.now(IST)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)
    return dt.date()


def init_db() -> None:
    global _table_ready
    if not perf_metrics_enabled():
        _table_ready = False
        return
    try:
        from ..db.forecast_ops_pg import _require_conn, ops_pg_enabled

        if not ops_pg_enabled():
            _table_ready = False
            return
        conn = _require_conn()
        if conn is None:
            _table_ready = False
            return
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_perf_events (
                    id           BIGSERIAL PRIMARY KEY,
                    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    day_ist      DATE NOT NULL,
                    kind         TEXT NOT NULL,
                    name         TEXT NOT NULL,
                    duration_ms  DOUBLE PRECISION NOT NULL,
                    meta         JSONB NOT NULL DEFAULT '{}'::jsonb
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fpe_kind_day "
                "ON forecast_perf_events (kind, day_ist, recorded_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fpe_name_day "
                "ON forecast_perf_events (name, day_ist)"
            )
        _table_ready = True
    except Exception:
        _log.exception("forecast_perf_events init failed")
        _table_ready = False


def _persist_async(event: dict[str, Any]) -> None:
    if not _table_ready:
        return

    def _write() -> None:
        try:
            from ..db.forecast_ops_pg import _require_conn

            conn = _require_conn()
            if conn is None:
                return
            with conn:
                conn.execute(
                    """
                    INSERT INTO forecast_perf_events
                        (recorded_at, day_ist, kind, name, duration_ms, meta)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        event["recorded_at"],
                        event["day_ist"],
                        event["kind"],
                        event["name"],
                        event["duration_ms"],
                        json.dumps(event.get("meta") or {}),
                    ),
                )
        except Exception:
            _log.debug("perf event persist failed", exc_info=True)

    threading.Thread(target=_write, daemon=True, name="perf-persist").start()


def record(
    kind: str,
    name: str,
    duration_ms: float,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    if not perf_metrics_enabled():
        return
    now = datetime.now(timezone.utc)
    event = {
        "recorded_at": now,
        "day_ist": day_ist(now.astimezone(IST)),
        "kind": kind,
        "name": (name or "")[:500],
        "duration_ms": float(max(0.0, duration_ms)),
        "meta": meta or {},
    }
    with _lock:
        _events.append(event)
    _persist_async(event)


def record_http(method: str, path: str, status: int, duration_sec: float) -> None:
    record(
        "http",
        f"{method} {path}",
        duration_sec * 1000.0,
        meta={"status": int(status), "path": path, "method": method},
    )


def record_db_query(backend: str, statement: Any, duration_sec: float) -> None:
    preview = " ".join(str(statement or "").split())[:280]
    record("db_query", preview, duration_sec * 1000.0, meta={"backend": backend})


def record_po_calculate(
    duration_sec: float,
    *,
    ok: bool,
    total_rows: int = 0,
    stage_timings: dict[str, Any] | None = None,
    ads_source: str | None = None,
) -> None:
    record(
        "po_calculate",
        "PO calculate",
        duration_sec * 1000.0,
        meta={
            "ok": ok,
            "total_rows": int(total_rows),
            "stage_timings": stage_timings or {},
            "ads_source": ads_source or "",
        },
    )


def record_cache(*, hit: bool, source: str, name: str = "hydrate") -> None:
    global _cache_hits, _cache_misses
    with _lock:
        if hit:
            _cache_hits += 1
        else:
            _cache_misses += 1
    record(
        "cache",
        name,
        0.0,
        meta={"hit": bool(hit), "source": source},
    )


def record_session_restore(source: str, duration_sec: float, *, ok: bool = True) -> None:
    record(
        "session_restore",
        source,
        duration_sec * 1000.0,
        meta={"ok": ok, "source": source},
    )


def _events_since(hours: float = 24.0) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with _lock:
        mem = [e for e in _events if e["recorded_at"] >= cutoff]
    if _table_ready:
        try:
            from ..db.forecast_ops_pg import _require_conn

            conn = _require_conn()
            if conn is not None:
                with conn:
                    rows = conn.execute(
                        """
                        SELECT recorded_at, day_ist, kind, name, duration_ms, meta
                        FROM forecast_perf_events
                        WHERE recorded_at >= %s
                        ORDER BY recorded_at DESC
                        LIMIT 5000
                        """,
                        (cutoff,),
                    ).fetchall()
                pg = [
                    {
                        "recorded_at": r[0],
                        "day_ist": r[1],
                        "kind": k if isinstance(k := r[2], str) else str(k),
                        "name": r[3],
                        "duration_ms": float(r[4]),
                        "meta": r[5] if isinstance(r[5], dict) else json.loads(r[5] or "{}"),
                    }
                    for r in rows
                ]
                if len(pg) > len(mem):
                    return pg
        except Exception:
            _log.debug("perf events pg read failed", exc_info=True)
    return mem


def _day_summary(events: list[dict[str, Any]], kind: str, target_day: date) -> dict[str, Any]:
    rows = [e for e in events if e["kind"] == kind and e["day_ist"] == target_day and e["duration_ms"] > 0]
    if not rows:
        return {"count": 0, "avg_sec": None, "p95_sec": None, "max_sec": None}
    secs = sorted(e["duration_ms"] / 1000.0 for e in rows)
    n = len(secs)
    p95_i = min(n - 1, int(n * 0.95))
    return {
        "count": n,
        "avg_sec": round(sum(secs) / n, 2),
        "p95_sec": round(secs[p95_i], 2),
        "max_sec": round(secs[-1], 2),
    }


def _slowest(events: list[dict[str, Any]], kind: str, *, limit: int = 15) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for e in events:
        if e["kind"] != kind or e["duration_ms"] <= 0:
            continue
        buckets.setdefault(e["name"], []).append(e["duration_ms"] / 1000.0)
    out: list[dict[str, Any]] = []
    for name, secs in buckets.items():
        secs.sort()
        n = len(secs)
        out.append(
            {
                "name": name,
                "count": n,
                "avg_sec": round(sum(secs) / n, 3),
                "max_sec": round(secs[-1], 3),
                "p95_sec": round(secs[min(n - 1, int(n * 0.95))], 3),
            }
        )
    out.sort(key=lambda x: (x["max_sec"], x["avg_sec"]), reverse=True)
    return out[:limit]


def _cache_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    hits = misses = 0
    by_source: Counter[str] = Counter()
    for e in events:
        if e["kind"] != "cache":
            continue
        meta = e.get("meta") or {}
        src = str(meta.get("source") or "unknown")
        if meta.get("hit"):
            hits += 1
            by_source[f"{src}:hit"] += 1
        else:
            misses += 1
            by_source[f"{src}:miss"] += 1
    with _lock:
        hits += _cache_hits
        misses += _cache_misses
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total else None,
        "by_source": dict(by_source.most_common(20)),
    }


def _recent_po(events: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    rows = [e for e in events if e["kind"] == "po_calculate"]
    rows.sort(key=lambda e: e["recorded_at"], reverse=True)
    out = []
    for e in rows[:limit]:
        meta = e.get("meta") or {}
        out.append(
            {
                "at": e["recorded_at"].astimezone(IST).isoformat(),
                "duration_sec": round(e["duration_ms"] / 1000.0, 2),
                "ok": meta.get("ok"),
                "total_rows": meta.get("total_rows"),
                "ads_source": meta.get("ads_source"),
                "stage_timings": meta.get("stage_timings") or {},
            }
        )
    return out


def postgres_stats() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        from ..db.forecast_ops_pg import _require_conn, ops_pg_enabled

        if not ops_pg_enabled():
            return out
        conn = _require_conn()
        if conn is None:
            return out
        with conn:
            settings = conn.execute(
                """
                SELECT name, setting, unit FROM pg_settings
                WHERE name IN (
                    'shared_buffers', 'effective_cache_size', 'work_mem',
                    'maintenance_work_mem', 'max_connections'
                )
                """
            ).fetchall()
            activity = conn.execute(
                """
                SELECT COUNT(*) FILTER (WHERE state = 'active'),
                       COUNT(*) FILTER (WHERE state = 'idle'),
                       COUNT(*)
                FROM pg_stat_activity
                WHERE datname = current_database()
                """
            ).fetchone()
            dbstat = conn.execute(
                """
                SELECT blks_hit, blks_read, xact_commit, xact_rollback
                FROM pg_stat_database
                WHERE datname = current_database()
                """
            ).fetchone()
            size = conn.execute(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            ).fetchone()
            try:
                mem_ctx = conn.execute(
                    "SELECT COALESCE(SUM(num_bytes), 0) FROM pg_backend_memory_contexts"
                ).fetchone()
            except Exception:
                mem_ctx = None
        out["available"] = True
        out["settings"] = {
            str(r[0]): f"{r[1]}{r[2] or ''}" if r[2] else str(r[1]) for r in settings
        }
        if activity:
            out["connections"] = {
                "active": int(activity[0] or 0),
                "idle": int(activity[1] or 0),
                "total": int(activity[2] or 0),
            }
        if dbstat:
            hit = int(dbstat[0] or 0)
            read = int(dbstat[1] or 0)
            total = hit + read
            out["buffer_cache_hit_ratio"] = round(hit / total, 4) if total else None
            out["transactions"] = {
                "commit": int(dbstat[2] or 0),
                "rollback": int(dbstat[3] or 0),
            }
        if size:
            out["database_size"] = str(size[0])
        if mem_ctx:
            out["backend_memory_bytes"] = int(mem_ctx[0] or 0)
            out["backend_memory_pretty"] = _pretty_bytes(int(mem_ctx[0] or 0))
    except Exception as exc:
        out["error"] = str(exc)[:200]
    try:
        for path in (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ):
            try:
                with open(path, encoding="utf-8") as fh:
                    out["container_memory_bytes"] = int(fh.read().strip())
                    out["container_memory_pretty"] = _pretty_bytes(out["container_memory_bytes"])
                    break
            except OSError:
                continue
    except Exception:
        pass
    return out


def _pretty_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def build_dashboard(*, hours: float = 48.0) -> dict[str, Any]:
    events = _events_since(hours=hours)
    today = day_ist()
    yesterday = today - timedelta(days=1)
    return {
        "generated_at": datetime.now(IST).isoformat(),
        "window_hours": hours,
        "po_calculate": {
            "today": _day_summary(events, "po_calculate", today),
            "yesterday": _day_summary(events, "po_calculate", yesterday),
            "recent": _recent_po(events),
        },
        "slowest_endpoints": _slowest(events, "http"),
        "slowest_queries": _slowest(events, "db_query"),
        "session_restore": {
            "summary": _day_summary(events, "session_restore", today),
            "slowest": _slowest(events, "session_restore", limit=10),
        },
        "cache": _cache_stats(events),
        "postgres": postgres_stats(),
        "samples_in_window": len(events),
    }

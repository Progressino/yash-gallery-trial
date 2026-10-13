"""User usage analytics — who uses the app, which screens / tabs / actions, and when.

Events are aggregated in memory (per IST day × user × kind × key) and flushed to
users.db every ``FLUSH_INTERVAL_SEC``. Only counts and timestamps are stored —
never request bodies or document contents.

kinds:
  page   — frontend route view (key = normalised path)
  tab    — tab switch inside a page (key = "path › tab")
  action — successful POST/PUT/PATCH/DELETE (key = "METHOD /module/{id}/…")
  ping   — "user is active" heartbeat (visible tab + recent input)
Active time = distinct 5-minute slots with any activity (``usage_active_slots``).
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
SLOT_MINUTES = 5
FLUSH_INTERVAL_SEC = 30
RETENTION_DAYS = 400
KINDS = ("page", "tab", "action", "ping")
REPORT_ADMIN_ROLES = frozenset({"Super Admin", "Admin"})

API_MODULES = {
    "purchase": "purchase",
    "gate": "purchase",
    "production": "production",
    "grey": "grey",
    "items": "items",
    "sales": "sales",
    "tna": "tna",
    "stitching": "stitching",
    "hrm": "hrm",
    "finance": "finance",
    "po": "po",
    "shipment": "po",
    "upload": "upload",
    "cache": "upload",
    "data": "intelligence",
    "erp-admin": "admin",
    "admin": "admin",
    "marketplace": "marketplace",
    "myntra": "marketplace",
    "document-audit": "document_audit",
}

_SKIP_ACTION_PREFIXES = ("/api/usage/", "/api/auth/", "/api/health")
# Fired automatically by the frontend (session restore / stuck-job recovery), not by a user click.
_SKIP_ACTION_PATHS = frozenset({
    "/api/cache/hydrate-warm",
    "/api/cache/load",
    "/api/cache/sync-tier3",
})
_SKIP_ACTION_SUFFIXES = ("/reset-stuck",)
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_lock = threading.Lock()
# (day, username, kind, key) -> [hits, module, first_at, last_at]
_counts: dict[tuple[str, str, str, str], list] = {}
# (day, username, slot) -> role
_slots: dict[tuple[str, str, int], str] = {}
_flusher_started = False
_schema_ready_for: str | None = None
_last_prune_day: str | None = None


def _now() -> datetime:
    return datetime.now(tz=IST)


def _connect():
    from ..db import users_db

    return users_db._connect()


def _db_path() -> str:
    from ..db import users_db

    return os.path.abspath(users_db._DB)


def ensure_schema(conn) -> None:
    global _schema_ready_for
    path = _db_path()
    if _schema_ready_for == path:
        return
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS usage_daily (
            day       TEXT NOT NULL,
            username  TEXT NOT NULL,
            kind      TEXT NOT NULL,
            key       TEXT NOT NULL,
            module    TEXT DEFAULT '',
            hits      INTEGER DEFAULT 0,
            first_at  TEXT,
            last_at   TEXT,
            PRIMARY KEY (day, username, kind, key)
        );
        CREATE INDEX IF NOT EXISTS idx_usage_daily_user ON usage_daily(username, day);
        CREATE INDEX IF NOT EXISTS idx_usage_daily_kind ON usage_daily(kind, day);
        CREATE TABLE IF NOT EXISTS usage_active_slots (
            day       TEXT NOT NULL,
            username  TEXT NOT NULL,
            slot      INTEGER NOT NULL,
            role      TEXT DEFAULT '',
            PRIMARY KEY (day, username, slot)
        );
        CREATE TABLE IF NOT EXISTS usage_logins (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT NOT NULL,
            role        TEXT DEFAULT '',
            method      TEXT DEFAULT '',
            ip          TEXT DEFAULT '',
            user_agent  TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_usage_logins_user ON usage_logins(username, created_at);
        """
    )
    _schema_ready_for = path


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm_segment(seg: str) -> str:
    if not seg:
        return seg
    if re.search(r"\d", seg) or len(seg) > 40 or "%" in seg:
        return "{id}"
    return seg


def normalize_api_path(path: str) -> tuple[str, str]:
    """``/api/purchase/min/12/lines/3/revise`` → (``/purchase/min/{id}/lines/{id}/revise``, ``purchase``)."""
    p = (path or "").split("?", 1)[0]
    if p.startswith("/api/"):
        p = p[4:]
    parts = [_norm_segment(s) for s in p.strip("/").split("/") if s]
    module = API_MODULES.get(parts[0], parts[0]) if parts else ""
    return "/" + "/".join(parts), module


def normalize_page_path(path: str) -> str:
    p = (path or "/").split("?", 1)[0].split("#", 1)[0].strip()
    if not p.startswith("/"):
        p = "/" + p
    parts = [_norm_segment(s) for s in p.strip("/").split("/") if s]
    return ("/" + "/".join(parts))[:120] if parts else "/"


def _clean(v: object, limit: int) -> str:
    return re.sub(r"[\x00-\x1f]", "", str(v or "")).strip()[:limit]


# ── Recording ─────────────────────────────────────────────────────────────────

def _bump(username: str, kind: str, key: str, module: str, role: str, now: datetime) -> None:
    day = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    slot = (now.hour * 60 + now.minute) // SLOT_MINUTES
    with _lock:
        cur = _counts.get((day, username, kind, key))
        if cur is None:
            _counts[(day, username, kind, key)] = [1, module, ts, ts]
        else:
            cur[0] += 1
            cur[1] = module or cur[1]
            cur[3] = ts
        _slots[(day, username, slot)] = role
    _ensure_flusher()


def record_api_action(username: str, role: str, method: str, path: str, status: int) -> None:
    """Called from auth middleware after a successful mutating request."""
    try:
        if not username or method.upper() not in _MUTATING or status >= 400:
            return
        if (
            path.startswith(_SKIP_ACTION_PREFIXES)
            or path in _SKIP_ACTION_PATHS
            or path.endswith(_SKIP_ACTION_SUFFIXES)
        ):
            return
        norm, module = normalize_api_path(path)
        _bump(username, "action", f"{method.upper()} {norm}", module, role or "", _now())
    except Exception:
        log.debug("usage action record failed", exc_info=True)


def record_client_events(username: str, role: str, events: list[dict], active: bool = False) -> int:
    """Page views / tab switches / heartbeat from the frontend tracker."""
    if not username:
        return 0
    now = _now()
    n = 0
    for ev in (events or [])[:50]:
        if not isinstance(ev, dict):
            continue
        kind = _clean(ev.get("kind"), 10)
        if kind not in ("page", "tab"):
            continue
        path = normalize_page_path(_clean(ev.get("path"), 200))
        module = _clean(ev.get("module"), 40)
        if kind == "tab":
            tab = _clean(ev.get("tab"), 60)
            if not tab:
                continue
            key = f"{path} › {tab}"
        else:
            key = path
        _bump(username, kind, key, module, role or "", now)
        n += 1
    if active and not n:
        _bump(username, "ping", "", "", role or "", now)
    return n


def record_login(username: str, role: str, *, method: str = "password", ip: str = "", user_agent: str = "") -> None:
    if not username:
        return
    try:
        now = _now()
        conn = _connect()
        try:
            ensure_schema(conn)
            conn.execute(
                "INSERT INTO usage_logins(username, role, method, ip, user_agent, created_at) VALUES(?,?,?,?,?,?)",
                (username, role or "", method, _clean(ip, 64), _clean(user_agent, 300), now.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        finally:
            conn.close()
        _bump(username, "ping", "", "", role or "", now)
    except Exception:
        log.warning("usage login record failed", exc_info=True)


# ── Flush ─────────────────────────────────────────────────────────────────────

def flush() -> int:
    """Write buffered counts to SQLite. Returns number of buffered keys written."""
    with _lock:
        counts = dict(_counts)
        slots = dict(_slots)
        _counts.clear()
        _slots.clear()
    if not counts and not slots:
        return 0
    try:
        conn = _connect()
        try:
            ensure_schema(conn)
            conn.executemany(
                """INSERT INTO usage_daily(day, username, kind, key, module, hits, first_at, last_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(day, username, kind, key) DO UPDATE SET
                     hits = hits + excluded.hits,
                     module = CASE WHEN excluded.module <> '' THEN excluded.module ELSE module END,
                     last_at = MAX(COALESCE(last_at, ''), excluded.last_at)""",
                [(d, u, k, key, v[1], v[0], v[2], v[3]) for (d, u, k, key), v in counts.items()],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO usage_active_slots(day, username, slot, role) VALUES(?,?,?,?)",
                [(d, u, s, r) for (d, u, s), r in slots.items()],
            )
            conn.commit()
        finally:
            conn.close()
        return len(counts)
    except Exception:
        log.warning("usage flush failed — keeping buffer", exc_info=True)
        with _lock:
            for k, v in counts.items():
                cur = _counts.get(k)
                if cur is None:
                    _counts[k] = v
                else:
                    cur[0] += v[0]
                    cur[2] = min(cur[2], v[2])
            for k, r in slots.items():
                _slots.setdefault(k, r)
        return 0


def prune(retention_days: int = RETENTION_DAYS) -> None:
    cutoff = (_now().date() - timedelta(days=retention_days)).isoformat()
    conn = _connect()
    try:
        ensure_schema(conn)
        conn.execute("DELETE FROM usage_daily WHERE day < ?", (cutoff,))
        conn.execute("DELETE FROM usage_active_slots WHERE day < ?", (cutoff,))
        conn.execute("DELETE FROM usage_logins WHERE created_at < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


def _flusher_loop() -> None:
    global _last_prune_day
    while True:
        time.sleep(FLUSH_INTERVAL_SEC)
        flush()
        today = _now().strftime("%Y-%m-%d")
        if _last_prune_day != today:
            try:
                prune()
                _last_prune_day = today
            except Exception:
                log.debug("usage prune failed", exc_info=True)


def _ensure_flusher() -> None:
    global _flusher_started
    if _flusher_started or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    with _lock:
        if _flusher_started:
            return
        _flusher_started = True
    threading.Thread(target=_flusher_loop, name="usage-flush", daemon=True).start()
    atexit.register(flush)


# ── Report ────────────────────────────────────────────────────────────────────

def _load_users(conn) -> dict[str, dict]:
    rows = conn.execute(
        """SELECT u.username, u.full_name, u.department, u.active, r.role_name
           FROM erp_users u LEFT JOIN roles r ON r.id = u.role_id"""
    ).fetchall()
    return {
        r["username"]: {
            "username": r["username"],
            "full_name": r["full_name"] or "",
            "department": r["department"] or "",
            "role": r["role_name"] or "",
            "active": bool(r["active"]),
        }
        for r in rows
    }


def _in_clause(names: list[str] | None) -> tuple[str, list]:
    if names is None:
        return "", []
    if not names:
        return " AND 1=0", []
    return f" AND username IN ({','.join('?' * len(names))})", list(names)


def usage_report(
    days: int = 30,
    *,
    role: str | None = None,
    department: str | None = None,
    username: str | None = None,
) -> dict:
    flush()
    days = max(1, min(int(days or 30), 365))
    today = _now().date()
    start = (today - timedelta(days=days - 1)).isoformat()
    d7 = (today - timedelta(days=6)).isoformat()
    d30 = (today - timedelta(days=29)).isoformat()
    today_s = today.isoformat()

    conn = _connect()
    try:
        ensure_schema(conn)
        users = _load_users(conn)
        seen = {
            r["username"]: r["role"]
            for r in conn.execute("SELECT username, MAX(role) AS role FROM usage_active_slots GROUP BY username")
        }
        for u, r in seen.items():
            if u not in users:
                users[u] = {"username": u, "full_name": "", "department": "", "role": r or "", "active": True}

        scope: list[str] | None = None
        if role or department or username:
            scope = [
                u for u, info in users.items()
                if (not role or info["role"] == role)
                and (not department or info["department"] == department)
                and (not username or u == username)
            ]
        flt, args = _in_clause(scope)

        def distinct_users(since: str) -> int:
            return conn.execute(
                f"SELECT COUNT(DISTINCT username) FROM usage_active_slots WHERE day >= ?{flt}",
                [since, *args],
            ).fetchone()[0]

        # Per-user rollups (window)
        per_user: dict[str, dict] = {}
        for r in conn.execute(
            f"""SELECT username, COUNT(DISTINCT day) AS days_active, COUNT(*) AS slots
                FROM usage_active_slots WHERE day >= ?{flt} GROUP BY username""",
            [start, *args],
        ):
            per_user.setdefault(r["username"], {})["days_active"] = r["days_active"]
            per_user[r["username"]]["active_minutes"] = r["slots"] * SLOT_MINUTES
        for r in conn.execute(
            f"""SELECT username, kind, SUM(hits) AS hits FROM usage_daily
                WHERE day >= ?{flt} GROUP BY username, kind""",
            [start, *args],
        ):
            per_user.setdefault(r["username"], {})[f"{r['kind']}_hits"] = r["hits"]
        top_mod: dict[str, tuple[str, int]] = {}
        for r in conn.execute(
            f"""SELECT username, module, SUM(hits) AS hits FROM usage_daily
                WHERE day >= ? AND kind IN ('page','tab','action') AND module <> ''{flt}
                GROUP BY username, module""",
            [start, *args],
        ):
            if r["hits"] > top_mod.get(r["username"], ("", 0))[1]:
                top_mod[r["username"]] = (r["module"], r["hits"])
        last_seen = {
            r["username"]: r["ls"]
            for r in conn.execute(f"SELECT username, MAX(last_at) AS ls FROM usage_daily WHERE 1=1{flt} GROUP BY username", args)
        }
        last_login = {}
        logins_in_window = {}
        for r in conn.execute(
            f"""SELECT username, MAX(created_at) AS ll,
                       SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS n
                FROM usage_logins WHERE 1=1{flt} GROUP BY username""",
            [start, *args],
        ):
            last_login[r["username"]] = r["ll"]
            logins_in_window[r["username"]] = r["n"]

        names = scope if scope is not None else list(users.keys())
        user_rows = []
        for u in names:
            info = users.get(u) or {"username": u, "full_name": "", "department": "", "role": "", "active": True}
            if not info["active"] and u not in per_user:
                continue
            pu = per_user.get(u, {})
            user_rows.append({
                **info,
                "last_seen": last_seen.get(u),
                "last_login": last_login.get(u),
                "logins": logins_in_window.get(u, 0),
                "days_active": pu.get("days_active", 0),
                "active_minutes": pu.get("active_minutes", 0),
                "page_views": pu.get("page_hits", 0) + pu.get("tab_hits", 0),
                "actions": pu.get("action_hits", 0),
                "top_module": top_mod.get(u, ("", 0))[0],
            })
        user_rows.sort(key=lambda r: (-(r["active_minutes"] or 0), -(r["actions"] or 0), r["username"]))

        trend_map: dict[str, dict] = {}
        for i in range(days):
            d = (today - timedelta(days=days - 1 - i)).isoformat()
            trend_map[d] = {"day": d, "active_users": 0, "active_minutes": 0, "page_views": 0, "actions": 0}
        for r in conn.execute(
            f"""SELECT day, COUNT(DISTINCT username) AS u, COUNT(*) AS s FROM usage_active_slots
                WHERE day >= ?{flt} GROUP BY day""",
            [start, *args],
        ):
            if r["day"] in trend_map:
                trend_map[r["day"]]["active_users"] = r["u"]
                trend_map[r["day"]]["active_minutes"] = r["s"] * SLOT_MINUTES
        for r in conn.execute(
            f"""SELECT day, kind, SUM(hits) AS h FROM usage_daily
                WHERE day >= ? AND kind IN ('page','tab','action'){flt} GROUP BY day, kind""",
            [start, *args],
        ):
            if r["day"] in trend_map:
                bucket = "actions" if r["kind"] == "action" else "page_views"
                trend_map[r["day"]][bucket] += r["h"]

        modules = [
            {
                "module": r["module"],
                "views": r["views"],
                "actions": r["actions"],
                "users": r["users"],
            }
            for r in conn.execute(
                f"""SELECT module,
                           SUM(CASE WHEN kind IN ('page','tab') THEN hits ELSE 0 END) AS views,
                           SUM(CASE WHEN kind = 'action' THEN hits ELSE 0 END) AS actions,
                           COUNT(DISTINCT username) AS users
                    FROM usage_daily
                    WHERE day >= ? AND kind IN ('page','tab','action') AND module <> ''{flt}
                    GROUP BY module ORDER BY views + actions DESC""",
                [start, *args],
            )
        ]

        def top(kind: str, limit: int = 100) -> list[dict]:
            return [
                {"key": r["key"], "module": r["module"], "hits": r["hits"], "users": r["users"], "last_at": r["last_at"]}
                for r in conn.execute(
                    f"""SELECT key, MAX(module) AS module, SUM(hits) AS hits,
                               COUNT(DISTINCT username) AS users, MAX(last_at) AS last_at
                        FROM usage_daily WHERE day >= ? AND kind = ?{flt}
                        GROUP BY key ORDER BY hits DESC LIMIT ?""",
                    [start, kind, *args, limit],
                )
            ]

        by_hour = [{"hour": h, "active_users": 0, "active_minutes": 0} for h in range(24)]
        per_slot = 60 // SLOT_MINUTES
        for r in conn.execute(
            f"""SELECT slot / ? AS hr, COUNT(DISTINCT username) AS u, COUNT(*) AS s
                FROM usage_active_slots WHERE day >= ?{flt} GROUP BY hr""",
            [per_slot, start, *args],
        ):
            h = int(r["hr"])
            if 0 <= h < 24:
                by_hour[h] = {"hour": h, "active_users": r["u"], "active_minutes": r["s"] * SLOT_MINUTES}

        recent_logins = [
            dict(r)
            for r in conn.execute(
                f"""SELECT username, role, method, ip, user_agent, created_at FROM usage_logins
                    WHERE created_at >= ?{flt} ORDER BY created_at DESC LIMIT 100""",
                [start, *args],
            )
        ]

        tracking_since = conn.execute(
            "SELECT MIN(day) FROM usage_active_slots"
        ).fetchone()[0]

        total_users = sum(1 for u in names if (users.get(u) or {}).get("active", True))
        summary = {
            "total_users": total_users,
            "active_today": distinct_users(today_s),
            "active_7d": distinct_users(d7),
            "active_30d": distinct_users(d30),
            "active_in_window": sum(1 for r in user_rows if r["days_active"] > 0),
            "page_views": sum(r["page_views"] for r in user_rows),
            "actions": sum(r["actions"] for r in user_rows),
            "active_hours": round(sum(r["active_minutes"] for r in user_rows) / 60.0, 1),
            "logins": sum(r["logins"] for r in user_rows),
            "never_seen": sum(1 for r in user_rows if not r["last_seen"]),
        }
        top_pages, top_tabs, top_actions = top("page"), top("tab"), top("action")
        roles = sorted({(i.get("role") or "") for i in users.values() if i.get("role")})
        departments = sorted({(i.get("department") or "") for i in users.values() if i.get("department")})
    finally:
        conn.close()

    return {
        "generated_at": _now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "IST",
        "days": days,
        "from": start,
        "to": today_s,
        "tracking_since": tracking_since,
        "filters": {"role": role or "", "department": department or "", "username": username or ""},
        "summary": summary,
        "trend": list(trend_map.values()),
        "by_hour": by_hour,
        "modules": modules,
        "pages": top_pages,
        "tabs": top_tabs,
        "actions": top_actions,
        "users": user_rows,
        "recent_logins": recent_logins,
        "roles": roles,
        "departments": departments,
    }


def reset_buffers_for_tests() -> None:
    global _schema_ready_for
    with _lock:
        _counts.clear()
        _slots.clear()
    _schema_ready_for = None


__all__ = [
    "KINDS",
    "REPORT_ADMIN_ROLES",
    "flush",
    "normalize_api_path",
    "normalize_page_path",
    "record_api_action",
    "record_client_events",
    "record_login",
    "usage_report",
]

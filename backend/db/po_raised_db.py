"""
SQLite persistence for the daily "Raised PO" ledger.

When the ops team clicks "Export & Confirm PO" in the PO Engine, the same
payload is also recorded here. Tomorrow's PO calculation reads this ledger
and adds the recorded qty to the SKU's PO_Pipeline_Total, so the same SKU
isn't recommended again until the raised PO is either:
  - older than its lead time (auto-expired), or
  - the user explicitly clears the ledger, or
  - the user uploads a fresh "Existing PO" sheet that already includes it
    (operator can clear the internal ledger to avoid double-counting).

DB path: ``PO_RAISED_DB_PATH`` env var, default ``/data/po_raised.db`` with
a local-dev fallback to ``./po_raised_dev.db``.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

DB_PATH = os.environ.get("PO_RAISED_DB_PATH", "/data/po_raised.db")


def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        conn = sqlite3.connect("./po_raised_dev.db")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def _ensure_suppressed_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS po_raise_suppressed (
            raised_date   TEXT PRIMARY KEY,
            suppressed_at TEXT NOT NULL DEFAULT (datetime('now')),
            note          TEXT DEFAULT ''
        )
        """
    )


def _ensure_raise_day_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS po_raise_day_meta (
            raised_date   TEXT PRIMARY KEY,
            lead_time     INTEGER NOT NULL,
            period_days   INTEGER,
            target_days   INTEGER,
            source        TEXT DEFAULT '',
            saved_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def init_db() -> None:
    """Create the ledger table on app startup."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS po_raised (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                oms_sku     TEXT    NOT NULL,
                qty         REAL    NOT NULL,
                raised_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                raised_date TEXT    NOT NULL DEFAULT (date('now')),
                lead_time   INTEGER,
                note        TEXT    DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_po_raised_sku_date "
            "ON po_raised(oms_sku, raised_date)"
        )
        _ensure_suppressed_table(conn)
        _ensure_raise_day_meta_table(conn)
        conn.commit()
    finally:
        conn.close()


def replace_raises_for_date(raised_date: str, items: Iterable[dict]) -> int:
    """Replace all ledger rows for ``raised_date`` (YYYY-MM-DD) with ``items``."""
    day = str(raised_date).strip()[:10]
    rows: list[tuple[str, float, Optional[int], str, str]] = []
    for it in items:
        sku = str(it.get("oms_sku") or it.get("OMS_SKU") or "").strip()
        try:
            qty = float(it.get("qty") or it.get("Raised_Qty") or it.get("Final_PO_Qty") or 0)
        except Exception:
            qty = 0.0
        if not sku or qty <= 0:
            continue
        try:
            lt = it.get("lead_time")
            lt_int: Optional[int] = int(lt) if lt is not None else None
        except Exception:
            lt_int = None
        note = str(it.get("note") or "")[:512]
        rows.append((sku, qty, lt_int, note, day))

    conn = _connect()
    try:
        conn.execute("DELETE FROM po_raised WHERE raised_date = ?", (day,))
        if rows:
            conn.executemany(
                "INSERT INTO po_raised (oms_sku, qty, lead_time, note, raised_date) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def ledger_rows_as_dataframe(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> "pd.DataFrame":
    """SKU-day ledger suitable for ``sess.po_raise_ledger_df``."""
    import pandas as pd

    rows = list_raises(start_date=start_date, end_date=end_date, limit=500_000)
    if not rows:
        return pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    df = pd.DataFrame(
        {
            "OMS_SKU": [str(r["oms_sku"]) for r in rows],
            "Raised_Qty": [int(float(r["qty"] or 0)) for r in rows],
            "Raised_Date": pd.to_datetime([str(r["raised_date"]) for r in rows], errors="coerce"),
        }
    )
    df = df[df["Raised_Qty"] > 0].dropna(subset=["Raised_Date"])
    if df.empty:
        return pd.DataFrame(columns=["OMS_SKU", "Raised_Qty", "Raised_Date"])
    from ..services.po_raise_ledger import normalize_raise_ledger_df

    return normalize_raise_ledger_df(df)


def record_raises(items: Iterable[dict]) -> int:
    """Persist a batch of raised POs. Each item must have ``oms_sku`` and
    ``qty``; ``lead_time`` (days) and ``note`` are optional. Returns count.
    """
    rows: list[tuple[str, float, Optional[int], str]] = []
    for it in items:
        sku = str(it.get("oms_sku") or it.get("OMS_SKU") or "").strip()
        try:
            qty = float(it.get("qty") or it.get("Final_PO_Qty") or 0)
        except Exception:
            qty = 0.0
        if not sku or qty <= 0:
            continue
        try:
            lt = it.get("lead_time")
            lt_int: Optional[int] = int(lt) if lt is not None else None
        except Exception:
            lt_int = None
        note = str(it.get("note") or "")[:512]
        rows.append((sku, qty, lt_int, note))

    if not rows:
        return 0

    conn = _connect()
    try:
        conn.executemany(
            "INSERT INTO po_raised (oms_sku, qty, lead_time, note) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def list_raises(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sku: Optional[str] = None,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Return raised PO rows ordered newest-first."""
    where: list[str] = []
    args: list[Any] = []
    if start_date:
        where.append("raised_date >= ?")
        args.append(start_date)
    if end_date:
        where.append("raised_date <= ?")
        args.append(end_date)
    if sku:
        where.append("oms_sku = ?")
        args.append(sku)
    sql = (
        "SELECT id, oms_sku, qty, raised_at, raised_date, lead_time, note "
        "FROM po_raised"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY raised_at DESC LIMIT ?"
    args.append(int(max(1, min(50_000, limit))))

    conn = _connect()
    try:
        cur = conn.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def summary_by_sku(
    only_active: bool = True,
    default_lead_time: int = 60,
) -> list[dict[str, Any]]:
    """Aggregate active raised qty per SKU.

    ``only_active`` filters out rows whose age exceeds the row's
    ``lead_time`` (or ``default_lead_time`` if none stored). These are
    treated as already received and no longer part of pipeline.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT oms_sku, qty, raised_date, lead_time FROM po_raised"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    today = datetime.utcnow().date()
    tally: dict[str, dict[str, Any]] = {}
    for r in rows:
        sku = str(r["oms_sku"])
        qty = float(r["qty"] or 0)
        if qty <= 0:
            continue
        raised_date_str = str(r["raised_date"])
        try:
            rd = datetime.fromisoformat(raised_date_str).date()
        except Exception:
            rd = today
        lt = int(r["lead_time"]) if r["lead_time"] is not None else int(default_lead_time)
        active = (today - rd).days <= max(1, lt)
        if only_active and not active:
            continue
        slot = tally.setdefault(
            sku,
            {
                "oms_sku": sku,
                "qty": 0.0,
                "last_raised_date": raised_date_str,
                "first_raised_date": raised_date_str,
            },
        )
        slot["qty"] += qty
        if raised_date_str > slot["last_raised_date"]:
            slot["last_raised_date"] = raised_date_str
        if raised_date_str < slot["first_raised_date"]:
            slot["first_raised_date"] = raised_date_str

    return sorted(tally.values(), key=lambda x: x["last_raised_date"], reverse=True)


def clear_all() -> int:
    """Wipe the ledger. Returns number of rows deleted."""
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM po_raised")
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def clear_older_than_days(days: int) -> int:
    """Delete entries older than ``days``. Returns rowcount."""
    cutoff = (datetime.utcnow().date() - timedelta(days=max(0, int(days)))).isoformat()
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM po_raised WHERE raised_date < ?", (cutoff,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def suppress_raise_date(raised_date: str, note: str = "") -> None:
    """Block auto-import from archives for this calendar day (user removed bad raise)."""
    day = str(raised_date).strip()[:10]
    if not day:
        return
    conn = _connect()
    try:
        _ensure_suppressed_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO po_raise_suppressed (raised_date, note) VALUES (?, ?)",
            (day, (note or "user deleted")[:512]),
        )
        conn.commit()
    finally:
        conn.close()


def clear_raise_date_suppression(raised_date: str) -> None:
    """Allow auto-import again after an explicit manual import for this day."""
    day = str(raised_date).strip()[:10]
    if not day:
        return
    conn = _connect()
    try:
        conn.execute("DELETE FROM po_raise_suppressed WHERE raised_date = ?", (day,))
        conn.commit()
    finally:
        conn.close()


def is_raise_date_suppressed(raised_date: str) -> bool:
    day = str(raised_date).strip()[:10]
    if not day:
        return False
    conn = _connect()
    try:
        _ensure_suppressed_table(conn)
        cur = conn.execute(
            "SELECT 1 FROM po_raise_suppressed WHERE raised_date = ? LIMIT 1",
            (day,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def list_suppressed_raise_dates() -> list[str]:
    """All calendar days blocked from archive auto-import."""
    conn = _connect()
    try:
        _ensure_suppressed_table(conn)
        rows = conn.execute(
            "SELECT raised_date FROM po_raise_suppressed ORDER BY raised_date"
        ).fetchall()
        return [str(r["raised_date"]).strip()[:10] for r in rows if r["raised_date"]]
    finally:
        conn.close()


def delete_raises_for_date(raised_date: str) -> int:
    """Delete all ledger rows for one calendar day (YYYY-MM-DD)."""
    day = str(raised_date).strip()[:10]
    if not day:
        return 0
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM po_raised WHERE raised_date = ?", (day,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def delete_raises_for_date_skus(raised_date: str, oms_skus: Iterable[str]) -> int:
    """Delete specific SKU lines for one calendar day."""
    day = str(raised_date).strip()[:10]
    skus = [str(s).strip() for s in oms_skus if str(s).strip()]
    if not day or not skus:
        return 0
    conn = _connect()
    try:
        total = 0
        for sku in skus:
            cur = conn.execute(
                "DELETE FROM po_raised WHERE raised_date = ? AND oms_sku = ?",
                (day, sku),
            )
            total += cur.rowcount or 0
        conn.commit()
        return total
    finally:
        conn.close()


def delete_by_ids(ids: Iterable[int]) -> int:
    id_list = [int(i) for i in ids if i is not None]
    if not id_list:
        return 0
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(id_list))
        cur = conn.execute(
            f"DELETE FROM po_raised WHERE id IN ({placeholders})", id_list
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def save_raise_day_meta(
    raised_date: str,
    lead_time: int,
    *,
    period_days: Optional[int] = None,
    target_days: Optional[int] = None,
    source: str = "",
) -> None:
    day = str(raised_date).strip()[:10]
    if not day or int(lead_time) <= 0:
        return
    conn = _connect()
    try:
        _ensure_raise_day_meta_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO po_raise_day_meta
                (raised_date, lead_time, period_days, target_days, source, saved_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                day,
                int(lead_time),
                int(period_days) if period_days is not None else None,
                int(target_days) if target_days is not None else None,
                str(source or "")[:128],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_raise_day_meta(raised_date: str) -> None:
    day = str(raised_date).strip()[:10]
    if not day:
        return
    conn = _connect()
    try:
        _ensure_raise_day_meta_table(conn)
        conn.execute("DELETE FROM po_raise_day_meta WHERE raised_date = ?", (day,))
        conn.commit()
    finally:
        conn.close()


def get_raise_day_meta(raised_date: str) -> Optional[dict[str, Any]]:
    day = str(raised_date).strip()[:10]
    if not day:
        return None
    conn = _connect()
    try:
        _ensure_raise_day_meta_table(conn)
        cur = conn.execute(
            "SELECT raised_date, lead_time, period_days, target_days, source, saved_at "
            "FROM po_raise_day_meta WHERE raised_date = ? LIMIT 1",
            (day,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_raises_lead_time_for_date(raised_date: str, lead_time: int) -> int:
    day = str(raised_date).strip()[:10]
    if not day or int(lead_time) <= 0:
        return 0
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE po_raised SET lead_time = ? WHERE raised_date = ?",
            (int(lead_time), day),
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def latest_raise_lead_time_before(
    planning_date: Optional[str],
    *,
    lookback_days: int = 14,
    default_lead_time: int = 45,
) -> Optional[dict[str, Any]]:
    """
    Most recent raise day on or before ``planning_date`` within lookback.

    Prefers ``po_raise_day_meta``; falls back to rows in ``po_raised`` with a
    stored per-line lead time, else ``default_lead_time`` when raises exist.
    """
    try:
        if planning_date and str(planning_date).strip():
            plan = datetime.fromisoformat(str(planning_date).strip()[:10]).date()
        else:
            plan = datetime.utcnow().date()
    except Exception:
        plan = datetime.utcnow().date()
    lb = max(1, int(lookback_days))
    start = (plan - timedelta(days=lb - 1)).isoformat()
    end = plan.isoformat()
    suppressed = set(list_suppressed_raise_dates())

    conn = _connect()
    try:
        _ensure_raise_day_meta_table(conn)
        _ensure_suppressed_table(conn)
        cur = conn.execute(
            """
            SELECT raised_date, lead_time, period_days, target_days, source
            FROM po_raise_day_meta
            WHERE raised_date >= ? AND raised_date <= ?
            ORDER BY raised_date DESC
            """,
            (start, end),
        )
        for row in cur.fetchall():
            day = str(row["raised_date"]).strip()[:10]
            if day in suppressed:
                continue
            lt = int(row["lead_time"] or 0)
            if lt > 0:
                return {
                    "raised_date": day,
                    "lead_time": lt,
                    "period_days": row["period_days"],
                    "target_days": row["target_days"],
                    "source": row["source"] or "meta",
                }

        cur = conn.execute(
            """
            SELECT raised_date, lead_time, SUM(qty) AS units
            FROM po_raised
            WHERE raised_date >= ? AND raised_date <= ?
            GROUP BY raised_date
            HAVING SUM(qty) > 0
            ORDER BY raised_date DESC
            """,
            (start, end),
        )
        for row in cur.fetchall():
            day = str(row["raised_date"]).strip()[:10]
            if day in suppressed:
                continue
            units = float(row["units"] or 0)
            if units <= 0:
                continue
            if row["lead_time"] is not None and int(row["lead_time"]) > 0:
                lt = int(row["lead_time"])
            else:
                lt = int(default_lead_time)
            return {
                "raised_date": day,
                "lead_time": lt,
                "period_days": None,
                "target_days": None,
                "source": "ledger_rows",
            }
    finally:
        conn.close()
    return None

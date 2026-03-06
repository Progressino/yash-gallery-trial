"""
SQLite persistence layer for Finance module.
DB path: /data/finance.db (env FINANCE_DB_PATH), fallback ./finance_dev.db for local dev.
"""
import os
import sqlite3
from typing import Optional

DB_PATH = os.environ.get("FINANCE_DB_PATH", "/data/finance.db")


def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        # Fallback for local dev when /data doesn't exist
        conn = sqlite3.connect("./finance_dev.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Called on app startup."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            description TEXT    NOT NULL DEFAULT '',
            amount      REAL    NOT NULL,
            gst_amount  REAL    NOT NULL DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def add_expense(
    date: str,
    category: str,
    description: str,
    amount: float,
    gst_amount: float = 0.0,
) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO expenses (date, category, description, amount, gst_amount) VALUES (?,?,?,?,?)",
        (date, category, description, amount, gst_amount),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_expenses(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM expenses WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_expense(expense_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted

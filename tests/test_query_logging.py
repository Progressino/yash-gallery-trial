"""Slow database query logging (psycopg + sqlite3)."""
from __future__ import annotations

import logging
import sqlite3

from backend.db import query_logging as ql


def test_log_slow_query_respects_threshold(monkeypatch, caplog):
    monkeypatch.setenv("DB_SLOW_QUERY_LOG", "1")
    monkeypatch.setenv("DB_SLOW_QUERY_SEC", "0.5")

    with caplog.at_level(logging.WARNING, logger="db.perf"):
        ql.log_slow_query(backend="postgresql", statement="SELECT 1", duration=0.1)
        ql.log_slow_query(
            backend="postgresql",
            statement="SELECT * FROM forecast_daily_uploads WHERE platform = %s",
            duration=0.75,
        )

    messages = [r.getMessage() for r in caplog.records if r.name == "db.perf"]
    assert len(messages) == 1
    assert "Slow Query 0.75s" in messages[0]
    assert "forecast_daily_uploads" in messages[0]


def test_timed_sqlite_connection_calls_logger(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_SLOW_QUERY_LOG", "1")
    calls: list[tuple] = []

    def _capture(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(ql, "log_slow_query", _capture)

    db = tmp_path / "slow.db"
    conn = ql.connect_sqlite(str(db), backend="sqlite-test")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("SELECT 1")

    assert len(calls) >= 2
    assert calls[-1]["backend"] == "sqlite-test"
    assert "SELECT 1" in str(calls[-1]["statement"])


def test_query_logging_disabled_skips_wrap(monkeypatch):
    monkeypatch.setenv("DB_SLOW_QUERY_LOG", "0")

    conn = ql.connect_sqlite(":memory:")
    assert isinstance(conn, sqlite3.Connection)

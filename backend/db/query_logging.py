"""Slow query logging for psycopg (PostgreSQL) and sqlite3 connections."""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any

logger = logging.getLogger("db.perf")


def query_logging_enabled() -> bool:
    raw = (os.environ.get("DB_SLOW_QUERY_LOG") or os.environ.get("PERF_DB_QUERIES") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def slow_query_threshold_sec() -> float:
    raw = (os.environ.get("DB_SLOW_QUERY_SEC") or "0.5").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.5


def _statement_preview(statement: Any) -> str:
    text = str(statement or "").strip()
    text = " ".join(text.split())
    return text[:300]


def log_slow_query(*, backend: str, statement: Any, duration: float) -> None:
    if not query_logging_enabled():
        return
    if duration < slow_query_threshold_sec():
        return
    logger.warning(
        "Slow Query %.2fs [%s]\n%s",
        duration,
        backend,
        _statement_preview(statement),
    )


class _TimedPsycopgConnection:
    __slots__ = ("_conn", "_backend")

    def __init__(self, conn: Any, *, backend: str = "postgresql") -> None:
        self._conn = conn
        self._backend = backend

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            if params is None:
                return self._conn.execute(query, **kwargs)
            return self._conn.execute(query, params, **kwargs)
        finally:
            log_slow_query(
                backend=self._backend,
                statement=query,
                duration=time.perf_counter() - start,
            )

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _TimedPsycopgCursor(self._conn.cursor(*args, **kwargs), self._backend)

    def __enter__(self) -> _TimedPsycopgConnection:
        self._conn.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._conn.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _TimedPsycopgCursor:
    __slots__ = ("_cur", "_backend", "_block_start", "_block_label")

    def __init__(self, cur: Any, backend: str) -> None:
        self._cur = cur
        self._backend = backend
        self._block_start: float | None = None
        self._block_label = ""

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            if params is None:
                return self._cur.execute(query, **kwargs)
            return self._cur.execute(query, params, **kwargs)
        finally:
            log_slow_query(
                backend=self._backend,
                statement=query,
                duration=time.perf_counter() - start,
            )

    def copy(self, statement: Any) -> Any:
        self._block_label = _statement_preview(statement)
        self._block_start = time.perf_counter()
        return _TimedCopy(self._cur.copy(statement), self)

    def __enter__(self) -> _TimedPsycopgCursor:
        self._cur.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            return self._cur.__exit__(*args)
        finally:
            if self._block_start is not None:
                log_slow_query(
                    backend=self._backend,
                    statement=self._block_label or "CURSOR BLOCK",
                    duration=time.perf_counter() - self._block_start,
                )
                self._block_start = None
                self._block_label = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)


class _TimedCopy:
    __slots__ = ("_copy", "_cursor")

    def __init__(self, copy: Any, cursor: _TimedPsycopgCursor) -> None:
        self._copy = copy
        self._cursor = cursor

    def __enter__(self) -> Any:
        return self._copy.__enter__()

    def __exit__(self, *args: Any) -> Any:
        try:
            return self._copy.__exit__(*args)
        finally:
            if self._cursor._block_start is not None:
                log_slow_query(
                    backend=self._cursor._backend,
                    statement=self._cursor._block_label or "COPY",
                    duration=time.perf_counter() - self._cursor._block_start,
                )
                self._cursor._block_start = None
                self._cursor._block_label = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._copy, name)


class _TimedSqliteConnection:
    __slots__ = ("_conn", "_backend")

    def __init__(self, conn: sqlite3.Connection, *, backend: str = "sqlite") -> None:
        self._conn = conn
        self._backend = backend

    def _timed(self, statement: Any, fn: Any) -> Any:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            log_slow_query(
                backend=self._backend,
                statement=statement,
                duration=time.perf_counter() - start,
            )

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        return self._timed(sql, lambda: self._conn.execute(sql, parameters))

    def executemany(self, sql: str, parameters: Any) -> sqlite3.Cursor:
        return self._timed(sql, lambda: self._conn.executemany(sql, parameters))

    def executescript(self, sql_script: str) -> sqlite3.Connection:
        return self._timed(sql_script, lambda: self._conn.executescript(sql_script))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def connect_psycopg(url: str, **kwargs: Any) -> Any:
    import psycopg

    conn = psycopg.connect(url, **kwargs)
    if not query_logging_enabled():
        return conn
    backend = "postgresql"
    if "forecast_session" in url or "session" in (kwargs.get("name") or ""):
        backend = "postgresql-session"
    return _TimedPsycopgConnection(conn, backend=backend)


def connect_sqlite(path: str, *, backend: str = "sqlite", **kwargs: Any) -> Any:
    conn = sqlite3.connect(path, **kwargs)
    if not query_logging_enabled():
        return conn
    return _TimedSqliteConnection(conn, backend=backend)

"""PostgreSQL connection pool (psycopg_pool — equivalent to SQLAlchemy pool settings).

Defaults: ``pool_size=2``, ``max_overflow=3`` (5 connections total on the VPS).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

_log = logging.getLogger(__name__)

_pools: dict[str, Any] = {}
_pools_lock = threading.Lock()


def pg_pool_enabled() -> bool:
    raw = (os.environ.get("DB_POOL_ENABLED") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def pool_size() -> int:
    return max(1, _int_env("DB_POOL_SIZE", 2))


def pool_max_overflow() -> int:
    return max(0, _int_env("DB_POOL_MAX_OVERFLOW", 3))


def pool_max_size() -> int:
    return pool_size() + pool_max_overflow()


def pool_recycle_sec() -> int:
    return max(60, _int_env("DB_POOL_RECYCLE_SEC", 1800))


def pool_pre_ping() -> bool:
    raw = (os.environ.get("DB_POOL_PRE_PING") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _pool_kwargs(connect_kwargs: dict[str, Any]) -> dict[str, Any]:
    from psycopg_pool import ConnectionPool

    kw: dict[str, Any] = {
        "min_size": pool_size(),
        "max_size": pool_max_size(),
        "max_lifetime": float(pool_recycle_sec()),
        "kwargs": dict(connect_kwargs),
        "open": True,
    }
    if pool_pre_ping():
        kw["check"] = ConnectionPool.check_connection
    return kw


def get_pool(url: str, **connect_kwargs: Any) -> Any | None:
    """Return a shared ``ConnectionPool`` for ``url`` (lazy open)."""
    if not url or not pg_pool_enabled():
        return None
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        _log.warning("psycopg_pool not installed — PostgreSQL pooling disabled")
        return None

    key = url
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = ConnectionPool(url, **_pool_kwargs(connect_kwargs))
            _pools[key] = pool
        return pool


def close_all_pools() -> None:
    with _pools_lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:
                _log.exception("closing PostgreSQL pool failed")
        _pools.clear()


class PooledConnectionLease:
    """Deferred checkout — supports ``conn = lease(); with conn: ...`` and ``lease.execute()``."""

    __slots__ = ("_pool", "_backend", "_cm", "_timed_factory", "_active")

    def __init__(self, pool: Any, *, backend: str, timed_factory: Any) -> None:
        self._pool = pool
        self._backend = backend
        self._cm = None
        self._timed_factory = timed_factory
        self._active = None

    def __enter__(self) -> Any:
        self._cm = self._pool.connection()
        raw = self._cm.__enter__()
        self._active = self._timed_factory(raw, backend=self._backend)
        return self._active

    def __exit__(self, *exc: Any) -> Any:
        try:
            if self._cm is not None:
                return self._cm.__exit__(*exc)
            return False
        finally:
            self._active = None
            self._cm = None

    def _use_conn(self):
        if self._active is not None:
            return self._active
        return self.__enter__()

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        nested = self._active is not None
        conn = self._use_conn()
        try:
            if params is None:
                return conn.execute(query, **kwargs)
            return conn.execute(query, params, **kwargs)
        finally:
            if not nested:
                self.__exit__(None, None, None)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        nested = self._active is not None
        conn = self._use_conn()
        try:
            return conn.cursor(*args, **kwargs)
        finally:
            if not nested:
                self.__exit__(None, None, None)


class DirectConnectionLease:
    """Single connection (tests / DB_POOL_ENABLED=0)."""

    __slots__ = ("_url", "_kwargs", "_backend", "_timed_factory", "_conn", "_wrapped", "_active")

    def __init__(self, url: str, *, backend: str, timed_factory: Any, **kwargs: Any) -> None:
        self._url = url
        self._kwargs = kwargs
        self._backend = backend
        self._timed_factory = timed_factory
        self._conn = None
        self._wrapped = None
        self._active = None

    def __enter__(self) -> Any:
        import psycopg

        self._conn = psycopg.connect(self._url, **self._kwargs)
        self._wrapped = self._timed_factory(self._conn, backend=self._backend)
        self._active = self._wrapped
        return self._wrapped

    def __exit__(self, *exc: Any) -> Any:
        try:
            if self._conn is not None:
                return self._conn.__exit__(*exc)
            return False
        finally:
            self._active = None
            self._conn = None
            self._wrapped = None

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        nested = self._active is not None
        if not nested:
            self.__enter__()
        try:
            if params is None:
                return self._active.execute(query, **kwargs)
            return self._active.execute(query, params, **kwargs)
        finally:
            if not nested:
                self.__exit__(None, None, None)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        nested = self._active is not None
        if not nested:
            self.__enter__()
        try:
            return self._active.cursor(*args, **kwargs)
        finally:
            if not nested:
                self.__exit__(None, None, None)

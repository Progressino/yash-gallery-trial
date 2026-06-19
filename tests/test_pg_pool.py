"""PostgreSQL connection pool configuration."""
from __future__ import annotations

from backend.db import pg_pool


def test_pool_defaults_match_sqlalchemy_equivalent():
    assert pg_pool.pool_size() == 20
    assert pg_pool.pool_max_overflow() == 40
    assert pg_pool.pool_max_size() == 60
    assert pg_pool.pool_recycle_sec() == 1800
    assert pg_pool.pool_pre_ping() is True
    assert pg_pool.pg_pool_enabled() is True


def test_pool_env_overrides(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "10")
    monkeypatch.setenv("DB_POOL_MAX_OVERFLOW", "20")
    monkeypatch.setenv("DB_POOL_RECYCLE_SEC", "900")
    monkeypatch.setenv("DB_POOL_PRE_PING", "0")
    monkeypatch.setenv("DB_POOL_ENABLED", "0")

    assert pg_pool.pool_size() == 10
    assert pg_pool.pool_max_size() == 30
    assert pg_pool.pool_recycle_sec() == 900
    assert pg_pool.pool_pre_ping() is False
    assert pg_pool.pg_pool_enabled() is False


def test_connect_psycopg_uses_direct_lease_when_pool_disabled(monkeypatch):
    monkeypatch.setenv("DB_POOL_ENABLED", "0")
    from backend.db.query_logging import connect_psycopg
    from backend.db.pg_pool import DirectConnectionLease

    lease = connect_psycopg("postgresql://u:p@127.0.0.1:5432/forecast", autocommit=True)
    assert isinstance(lease, DirectConnectionLease)


def test_pool_kwargs_open_on_create(monkeypatch):
    """Regression: pool must open on create — open=False without pool.open() broke ops_pg."""
    import types

    monkeypatch.setitem(
        __import__("sys").modules,
        "psycopg_pool",
        types.SimpleNamespace(ConnectionPool=type("CP", (), {"check_connection": None})),
    )
    assert pg_pool._pool_kwargs({})["open"] is True

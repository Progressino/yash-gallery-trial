"""Tier-3 SQLite path and list_uploads API."""

from pathlib import Path


def test_resolve_db_path_env_override(monkeypatch, tmp_path):
    from backend.services import daily_store

    p = tmp_path / "custom.db"
    p.write_bytes(b"x")
    monkeypatch.setenv("DAILY_SALES_DB", str(p))
    assert daily_store._resolve_db_path() == p


def test_list_uploads_reads_configured_db(monkeypatch, tmp_path):
    from backend.services import daily_store

    db = tmp_path / "tier3_test.db"
    monkeypatch.setattr(daily_store, "_DB_PATH", db)
    daily_store._get_conn()
    conn = daily_store._get_conn()
    conn.execute(
        "INSERT INTO daily_uploads (platform, file_date, filename, rows, data_parquet, date_from, date_to) "
        "VALUES ('amazon', '2026-06-16', 'test.csv', 10, X'00', '2026-06-16', '2026-06-16')"
    )
    conn.commit()
    conn.close()
    daily_store.invalidate_upload_coverage_cache()
    rows = daily_store.list_uploads()
    assert len(rows) == 1
    assert rows[0]["platform"] == "amazon"

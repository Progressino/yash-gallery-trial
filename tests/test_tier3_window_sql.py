"""Tier-3 SQLite window queries must find uploads when date_from has time suffixes."""

from __future__ import annotations

import pandas as pd

from backend.services import daily_store


def test_platforms_with_uploads_in_range_finds_datetime_date_from(tmp_path, monkeypatch):
    db = tmp_path / "tier3_win.db"
    monkeypatch.setattr(daily_store, "_DB_PATH", db)
    daily_store.invalidate_upload_coverage_cache()

    conn = daily_store._get_conn()
    blob = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [10],
            "Order_Id": ["O1"],
        }
    )
    pq = daily_store._df_to_parquet(blob)
    conn.execute(
        """
        INSERT INTO daily_uploads
        (platform, file_date, filename, rows, data_parquet, date_from, date_to)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "amazon",
            "2026-06-01",
            "test.csv",
            1,
            pq,
            "2026-06-01 00:00:00",
            "2026-06-02 11:00:00",
        ),
    )
    conn.commit()
    conn.close()
    daily_store.invalidate_upload_coverage_cache()

    found = daily_store.platforms_with_uploads_in_range("2026-06-01", "2026-06-04")
    assert "amazon" in found

    df = daily_store.load_platform_data_for_report_range(
        "amazon", "2026-06-01", "2026-06-04", dedup=False
    )
    assert len(df) == 1
    assert int(df["Quantity"].iloc[0]) == 10

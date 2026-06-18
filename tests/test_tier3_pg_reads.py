"""Tier-3 daily_store reads PostgreSQL when DAILY_SALES_BACKEND=postgres."""
from __future__ import annotations

import pandas as pd

from backend.services import daily_store as ds


def test_load_platform_data_for_report_range_prefers_pg(monkeypatch):
    daily = pd.DataFrame(
        {
            "Date": ["2026-06-17"],
            "SKU": ["SKU1"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [7],
        }
    )
    blob = daily.to_parquet(index=False)

    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.daily_uploads_pg_read",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.pg_load_platform_rows_for_range",
        lambda plat, s, e: [("june17.parquet", blob)] if plat == "amazon" else [],
    )

    def _fail_sqlite(*_a, **_k):
        raise AssertionError("SQLite should not be queried when PG returns rows")

    monkeypatch.setattr(ds, "_get_conn", _fail_sqlite)

    out = ds.load_platform_data_for_report_range(
        "amazon", "2026-06-17", "2026-06-17", dedup=False
    )
    assert len(out) == 1
    assert int(out["Quantity"].sum()) == 7


def test_platforms_with_uploads_in_range_prefers_pg(monkeypatch):
    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.daily_uploads_pg_read",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.pg_platforms_with_uploads_in_range",
        lambda s, e: ["amazon", "myntra"],
    )
    monkeypatch.setattr(ds, "_get_conn", lambda: (_ for _ in ()).throw(AssertionError("no sqlite")))

    assert ds.platforms_with_uploads_in_range("2026-06-17", "2026-06-18") == [
        "amazon",
        "myntra",
    ]


def test_get_tier3_sync_token_prefers_pg(monkeypatch):
    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.daily_uploads_pg_read",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.db.forecast_ops_pg.pg_get_tier3_sync_token",
        lambda: {"amazon": "10:5000:2026-06-18"},
    )
    monkeypatch.setattr(ds, "_get_conn", lambda: (_ for _ in ()).throw(AssertionError("no sqlite")))

    assert ds.get_tier3_sync_token() == {"amazon": "10:5000:2026-06-18"}

"""HTTP tests for hour-wise production entry save (Stitching Costing)."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, save_sheet_df


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_pe_http.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()
    save_sheet_df(
        "style_master",
        pd.DataFrame(
            [
                {"Style": "SKU-HOUR", "Operation": "Ghari Pipeing", "Target": 30, "Rate_Rs": 2.0},
            ]
        ),
    )


def test_production_entry_http_save_hour_wise(client):
    """POST /api/stitching/production-entry persists hour-wise rows."""
    r = client.post(
        "/api/stitching/production-entry",
        json={
            "date": "2026-06-04",
            "karigar_id": "K001",
            "karigar_name": "Test Karigar",
            "challan_no": "CH-99",
            "style": "SKU-HOUR",
            "hour_entries": [
                {
                    "hour_col": "H_09_10",
                    "operation": "Ghari Pipeing",
                    "pieces": 25,
                    "sticker_in": 0,
                    "sticker_out": 0,
                    "manual_pieces": True,
                },
                {
                    "hour_col": "H_10_11",
                    "operation": "Ghari Pipeing",
                    "pieces": 25,
                    "sticker_in": 0,
                    "sticker_out": 0,
                    "manual_pieces": True,
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("rows_added", 0) >= 1

    load = client.get(
        "/api/stitching/production-entry/load",
        params={
            "date": "2026-06-04",
            "karigar_id": "K001",
            "challan_no": "CH-99",
            "style": "SKU-HOUR",
        },
    )
    assert load.status_code == 200
    hours = load.json().get("hours") or {}
    assert int(hours.get("H_09_10", {}).get("pieces") or 0) == 25

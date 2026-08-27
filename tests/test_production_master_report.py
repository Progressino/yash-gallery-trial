"""Master Production Status Report — multi-stage qty for same SKU."""
from __future__ import annotations

import pytest

from backend.db import production_db
from backend.services.production_master_report import (
    query_master_production_status,
    stage_report_config,
)


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    yield


def _credit_stock(so: str, sku: str, process: str, qty: int):
    conn = production_db._connect()
    conn.execute(
        """
        INSERT INTO process_stock(so_number, sku, process, available_qty, total_in, total_out, updated_at)
        VALUES(?,?,?,?,?,?,datetime('now'))
        ON CONFLICT(so_number, sku, process) DO UPDATE SET
            available_qty = excluded.available_qty,
            total_in = excluded.total_in,
            total_out = excluded.total_out,
            updated_at = datetime('now')
        """,
        (so, sku, process, qty, qty, 0),
    )
    conn.commit()
    conn.close()


def test_same_sku_multiple_stages_not_collapsed(iso):
    so = "SO-MST-1"
    sku_s = "1001YKBEIGE-S"
    sku_m = "1001YKBEIGE-M"
    _credit_stock(so, sku_s, "Stitching", 50)
    _credit_stock(so, sku_m, "Embroidery", 30)
    _credit_stock(so, sku_m, "Finishing", 20)
    _credit_stock(so, "1001YKBEIGE-L-TOP", "Packing", 10)

    out = query_master_production_status(so_number=so, limit=500)
    assert out["ok"] is True
    lines = out["lines"]
    assert len(lines) >= 4

    # Same size SKU at two stages remains two lines
    m_lines = [r for r in lines if r["sku"] == sku_m]
    procs = {r["process"] for r in m_lines}
    assert procs == {"Embroidery", "Finishing"}
    assert sum(r["available_qty"] for r in m_lines) == 50

    overview = { (r["sku"],): r for r in out["overview"] }
    assert overview[(sku_m,)]["stage_qty"]["Embroidery"]["available"] == 30
    assert overview[(sku_m,)]["stage_qty"]["Finishing"]["available"] == 20
    assert overview[(sku_m,)]["total_available"] == 50


def test_filter_main_sku_and_component(iso):
    so = "SO-MST-2"
    _credit_stock(so, "STYLEA-M-TOP", "Cutting", 8)
    _credit_stock(so, "STYLEA-M-PANT", "Cutting", 5)
    _credit_stock(so, "OTHER-M", "Stitching", 99)

    out = query_master_production_status(main_sku="STYLEA", component="TOP")
    skus = {r["sku"] for r in out["lines"]}
    assert "STYLEA-M-TOP" in skus
    assert "STYLEA-M-PANT" not in skus
    assert "OTHER-M" not in skus


def test_open_jo_balance_without_stock_row(iso):
    production_db.create_jo(
        {
            "so_number": "SO-MST-3",
            "so_source": "manual",
            "sku": "JOONLY-M",
            "process": "Stitching",
            "planned_qty": 40,
            "main_sku": "JOONLY",
        }
    )
    out = query_master_production_status(so_number="SO-MST-3", sku="JOONLY-M")
    assert len(out["lines"]) >= 1
    row = out["lines"][0]
    assert row["process"] == "Stitching"
    assert row["available_qty"] == 0
    assert row["jo_planned"] >= 40
    assert row["jo_balance"] >= 40


def test_stage_report_config_uses_live_stages(iso):
    cfg = stage_report_config()
    assert cfg["ok"] is True
    assert isinstance(cfg["stages"], list)
    assert len(cfg["columns"]) >= 5
    ids = {c["id"] for c in cfg["columns"]}
    assert "available_qty" in ids
    assert "jo_balance" in ids

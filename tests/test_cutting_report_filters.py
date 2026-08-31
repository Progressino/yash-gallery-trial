"""Cutting report — production mode, component multi-filter, balance level."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.db import production_db
from backend.services.cutting_reports import build_cutting_report


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    yield


def _cutting_jo(so: str, sku: str, comp: str, qty: int, mode: str = "inhouse"):
    import sqlite3

    num = production_db.create_jo(
        {
            "so_number": so,
            "so_source": "manual",
            "sku": sku,
            "process": "Cutting",
            "planned_qty": qty,
            "component_code": comp,
            "main_sku": "1001-M",
            "production_mode": mode,
            "create_component_jos": False,
            "jo_date": date.today().isoformat(),
            "lines": [
                {
                    "sku": sku,
                    "style": "M",
                    "planned_qty": qty,
                    "component_code": comp,
                    "parent_sku": "1001-M",
                }
            ],
        }
    )
    conn = sqlite3.connect(production_db._DB)
    conn.execute(
        "UPDATE job_orders SET production_mode=? WHERE jo_number=?",
        (mode, num),
    )
    conn.commit()
    conn.close()
    return num


def _insert_component_cutting_jo(so: str, sku: str, comp: str, qty: int, mode: str = "inhouse"):
    """Insert a component-level Cutting JO row (mirrors exploded set components)."""
    import sqlite3

    conn = sqlite3.connect(production_db._DB)
    num = f"JO-{so}-{comp}"
    conn.execute(
        """INSERT INTO job_orders(
            jo_number, jo_date, so_number, sku, process, status, planned_qty, balance_qty,
            component_code, main_sku, production_mode, updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (num, date.today().isoformat(), so, sku, "Cutting", "Created", qty, qty, comp, "1001-M", mode),
    )
    jid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        """INSERT INTO jo_lines(jo_id, sku, style, planned_qty, balance_qty, component_code, parent_sku)
           VALUES(?,?,?,?,?,?,?)""",
        (jid, sku, "M", qty, qty, comp, "1001-M"),
    )
    conn.commit()
    conn.close()
    return num


def test_cutting_report_production_mode_filter(iso):
    _cutting_jo("SO-A", "1001-M", "TOP", 100, "inhouse")
    _insert_component_cutting_jo("SO-A", "1001-M-TOP-CTP", "TOP", 50, "cut_to_pack")
    all_out = build_cutting_report(export=True, balance_level="component")
    assert all_out["kpis"]["planned_qty"] == 150
    inh = build_cutting_report(export=True, production_mode="inhouse", balance_level="component")
    assert inh["kpis"]["planned_qty"] == 100
    ctp = build_cutting_report(export=True, production_mode="cut_to_pack", balance_level="component")
    assert ctp["kpis"]["planned_qty"] == 50


def test_cutting_report_set_balance_level_rollup(iso):
    _insert_component_cutting_jo("SO-B", "1001-M-TOP", "TOP", 100)
    _insert_component_cutting_jo("SO-B", "1001-M-PANT", "PANT", 100)
    _insert_component_cutting_jo("SO-B", "1001-M-DUPATTA", "DUPATTA", 100)
    comp = build_cutting_report(so_number="SO-B", export=True, balance_level="component")
    assert comp["kpis"]["planned_qty"] == 300
    sett = build_cutting_report(so_number="SO-B", export=True, balance_level="set")
    assert sett["kpis"]["planned_qty"] == 100
    assert len(sett["rows"]) == 1
    assert sett["rows"][0]["row_type"] == "set"


def test_cutting_report_components_multi_filter(iso):
    _insert_component_cutting_jo("SO-C", "1001-M-TOP", "TOP", 80)
    _insert_component_cutting_jo("SO-C", "1001-M-PANT", "PANT", 90)
    top_only = build_cutting_report(so_number="SO-C", components="TOP", export=True, balance_level="component")
    assert top_only["kpis"]["planned_qty"] == 80

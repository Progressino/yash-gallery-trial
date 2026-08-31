"""Finishing migration reset — cancel JOs, reverse receipts, preserve Ready-to-WIP."""
from __future__ import annotations

import pytest

from backend.db import production_db


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    yield


def test_reset_finishing_migration_dry_run_and_apply(iso):
    production_db.import_ready_to_wip(
        [
            {
                "Ready_To_Stage": "Finishing",
                "SO_Number": "SO-R1",
                "OMS_SKU": "SKU-A-M",
                "Quantity": 50,
            }
        ],
        default_stage="Finishing",
    )
    n1 = production_db.create_jo(
        {
            "so_number": "SO-R1",
            "sku": "SKU-A-M",
            "process": "Finishing",
            "planned_qty": 100,
            "create_component_jos": False,
        }
    )
    n2 = production_db.create_jo(
        {
            "so_number": "SO-R1",
            "sku": "SKU-A-M",
            "process": "Finishing",
            "planned_qty": 100,
            "create_component_jos": False,
        }
    )
    j1 = next(j for j in production_db.list_jos(process="Finishing") if j["jo_number"] == n1)
    production_db.receive_pieces(j1["id"], {"received_qty": 10, "process": "Finishing"})

    dry = production_db.reset_finishing_migration(dry_run=True)
    assert dry["jos_found"] == 2
    assert dry["duplicate_keys_before"] == 1
    assert dry["receipts_removed"] == 1
    assert dry["ready_to_wip_preserved"] >= 1

    applied = production_db.reset_finishing_migration(dry_run=False, actor="test")
    assert applied["jos_cancelled"] == 2

    open_jos = production_db.list_jos(process="Finishing", status="Created")
    in_progress = production_db.list_jos(process="Finishing", status="In Progress")
    assert len(open_jos) + len(in_progress) == 0

    ready = production_db._connect()
    try:
        n = ready.execute("SELECT COUNT(*) FROM ready_to_wip_imports").fetchone()[0]
    finally:
        ready.close()
    assert int(n) >= 1

    # Re-import should create fresh JO without doubling
    from backend.services.jo_import import aggregate_jo_import_payloads

    payloads = aggregate_jo_import_payloads(
        [
            {"so_number": "SO-R1", "sku": "SKU-A-M", "process": "Finishing", "planned_qty": 40},
            {"so_number": "SO-R1", "sku": "SKU-A-M", "process": "Finishing", "planned_qty": 60},
        ]
    )
    assert len(payloads) == 1
    assert payloads[0]["planned_qty"] == 100
    r = production_db.upsert_jo_from_import(payloads[0])
    assert r["action"] == "created"
    active = [
        j
        for j in production_db.list_jos(process="Finishing")
        if j.get("status") not in ("Cancelled", "Closed")
    ]
    assert len(active) == 1
    assert int(active[0]["planned_qty"]) == 100

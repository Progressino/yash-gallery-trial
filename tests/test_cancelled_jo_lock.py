"""Cancelled/Closed JOs must be immutable and excluded from active lists."""
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


def test_cancelled_jo_blocks_production_transactions(iso):
    num = production_db.create_jo(
        {
            "so_number": "SO-LOCK",
            "sku": "SKU-LOCK-M",
            "process": "Finishing",
            "planned_qty": 100,
            "create_component_jos": False,
        }
    )
    jo = next(j for j in production_db.list_jos(process="Finishing") if j["jo_number"] == num)
    production_db.update_jo(jo["id"], {"status": "Cancelled"})

    with pytest.raises(ValueError, match="cancelled"):
        production_db.receive_pieces(jo["id"], {"received_qty": 5, "process": "Finishing"})
    with pytest.raises(ValueError, match="cancelled"):
        production_db.issue_pieces(
            jo["id"],
            {"issued_qty": 5, "from_process": "Finishing", "to_process": "Packing"},
        )
    with pytest.raises(ValueError, match="cancelled"):
        production_db.update_jo(jo["id"], {"status": "In Progress"})
    with pytest.raises(ValueError, match="cancelled"):
        production_db.add_cost(jo["id"], {"amount": 10})


def test_list_jos_hides_cancelled_by_default(iso):
    n1 = production_db.create_jo(
        {
            "so_number": "SO-HIDE",
            "sku": "SKU-H-M",
            "process": "Finishing",
            "planned_qty": 10,
            "create_component_jos": False,
        }
    )
    n2 = production_db.create_jo(
        {
            "so_number": "SO-HIDE",
            "sku": "SKU-H-L",
            "process": "Finishing",
            "planned_qty": 20,
            "create_component_jos": False,
        }
    )
    j2 = next(j for j in production_db.list_jos(process="Finishing") if j["jo_number"] == n2)
    production_db.update_jo(j2["id"], {"status": "Cancelled"})

    active = production_db.list_jos(process="Finishing")
    assert len(active) == 1
    assert active[0]["jo_number"] == n1

    all_rows = production_db.list_jos(process="Finishing", include_inactive=True)
    assert len(all_rows) == 2

    cancelled_only = production_db.list_jos(process="Finishing", status="Cancelled")
    assert len(cancelled_only) == 1
    assert cancelled_only[0]["jo_number"] == n2


def test_import_after_cancel_creates_new_then_updates_active(iso):
    production_db.create_jo(
        {
            "so_number": "SO-REIMP",
            "sku": "SKU-RE-M",
            "process": "Finishing",
            "planned_qty": 80,
            "create_component_jos": False,
        }
    )
    jo = next(j for j in production_db.list_jos(process="Finishing") if j["so_number"] == "SO-REIMP")
    production_db.update_jo(jo["id"], {"status": "Cancelled"})

    payload = {
        "so_number": "SO-REIMP",
        "sku": "SKU-RE-M",
        "process": "Finishing",
        "planned_qty": 100,
    }
    r1 = production_db.upsert_jo_from_import(payload)
    assert r1["action"] == "created"

    active = [
        j
        for j in production_db.list_jos(process="Finishing")
        if j.get("status") not in ("Cancelled", "Closed")
    ]
    assert len(active) == 1
    assert int(active[0]["planned_qty"]) == 100

    r2 = production_db.upsert_jo_from_import({**payload, "planned_qty": 120})
    assert r2["action"] == "updated"
    active2 = production_db.list_jos(process="Finishing")
    assert len(active2) == 1
    assert int(active2[0]["planned_qty"]) == 120

    cancelled = production_db.list_jos(process="Finishing", status="Cancelled")
    assert len(cancelled) == 1
    assert int(cancelled[0]["planned_qty"]) == 80

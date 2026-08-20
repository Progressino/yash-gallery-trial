"""Performance-oriented correctness tests for list_jos batching."""
from __future__ import annotations

import pytest

from backend.db import production_db


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setattr(production_db, "_DB", prod)
    production_db.init_db()
    monkeypatch.setattr(
        production_db,
        "get_component_routing",
        lambda sku: ["Cutting", "Stitching", "Finishing"],
    )
    yield


def test_list_jos_batches_lines_and_light_mode(iso):
    a = production_db.create_jo(
        {
            "so_number": "SO-A",
            "sku": "SKU-A-M",
            "process": "Cutting",
            "planned_qty": 10,
            "production_mode": "inhouse",
            "lines": [{"so_number": "SO-A", "sku": "SKU-A-M", "planned_qty": 10}],
        }
    )
    b = production_db.create_jo(
        {
            "so_number": "SO-B",
            "sku": "SKU-B-M",
            "process": "Cutting",
            "planned_qty": 20,
            "production_mode": "cut_to_pack",
            "lines": [{"so_number": "SO-B", "sku": "SKU-B-M", "planned_qty": 20}],
        }
    )
    assert isinstance(a, str) or isinstance(a, list)
    assert isinstance(b, str) or isinstance(b, list)

    full = production_db.list_jos(process="Cutting")
    assert len(full) >= 2
    for jo in full:
        assert "lines" in jo
        assert "routing" in jo
        assert "next_process" in jo
        assert "fabric_issues" in jo
        assert isinstance(jo["fabric_issues"], list)

    light = production_db.list_jos(process="Cutting", light=True)
    assert len(light) == len(full)
    for jo in light:
        assert jo["fabric_issues"] == []
        assert jo["cost_entries"] == []
        assert jo["lines"]
        assert jo.get("routing")

    c2p = next(j for j in light if (j.get("production_mode") or "") == "cut_to_pack")
    assert c2p["next_process"] == "Finishing"
    assert c2p["routing"] == ["Cutting", "Finishing"]


def test_list_jos_limit_offset(iso):
    for i in range(5):
        production_db.create_jo(
            {
                "so_number": f"SO-L{i}",
                "sku": f"STYLE{i}-M",
                "process": "Cutting",
                "planned_qty": 1,
                "create_component_jos": False,
                "lines": [{"so_number": f"SO-L{i}", "sku": f"STYLE{i}-M", "planned_qty": 1}],
            }
        )
    page = production_db.list_jos(process="Cutting", light=True, limit=2, offset=0)
    assert len(page) == 2
    page2 = production_db.list_jos(process="Cutting", light=True, limit=2, offset=2)
    assert len(page2) == 2
    ids = {j["id"] for j in page} | {j["id"] for j in page2}
    assert len(ids) == 4

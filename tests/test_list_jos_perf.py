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


def test_list_jos_q_searches_beyond_first_page(iso):
    """Search must hit JO / line SKUs even when the match is past the default page."""
    for i in range(5):
        production_db.create_jo(
            {
                "so_number": f"SO-P{i}",
                "sku": f"COMMON{i}-M",
                "process": "Cutting",
                "planned_qty": 1,
                "create_component_jos": False,
                "lines": [{"sku": f"COMMON{i}-M", "planned_qty": 1}],
            }
        )
    production_db.create_jo(
        {
            "so_number": "SO-NEEDLE",
            "sku": "NEEDLESKU-XXL",
            "process": "Cutting",
            "planned_qty": 7,
            "create_component_jos": False,
            "lines": [
                {"sku": "NEEDLESKU-XXL", "planned_qty": 4},
                {"sku": "NEEDLESKU-L", "planned_qty": 3},
            ],
        }
    )
    # Newest first — needle is first; bury it by creating more after wouldn't work.
    # Create older needle by inserting then creating many newer rows.
    buried = production_db.create_jo(
        {
            "so_number": "SO-BURIED",
            "sku": "ZZZ-HIDDEN-M",
            "process": "Cutting",
            "planned_qty": 9,
            "create_component_jos": False,
            "lines": [{"sku": "ZZZ-HIDDEN-M", "planned_qty": 9}],
        }
    )
    for i in range(8):
        production_db.create_jo(
            {
                "so_number": f"SO-NEW{i}",
                "sku": f"NEWER{i}-M",
                "process": "Cutting",
                "planned_qty": 1,
                "create_component_jos": False,
                "lines": [{"sku": f"NEWER{i}-M", "planned_qty": 1}],
            }
        )
    page = production_db.list_jos(process="Cutting", light=True, limit=5)
    assert all("ZZZ-HIDDEN" not in (j.get("sku") or "") for j in page)
    found = production_db.list_jos(process="Cutting", light=True, q="ZZZ-HIDDEN", limit=5)
    assert len(found) == 1
    assert found[0]["jo_number"] == buried or found[0]["sku"] == "ZZZ-HIDDEN-M"
    by_sku = production_db.list_jos(process="Cutting", light=True, sku="NEEDLESKU-L", limit=5)
    assert len(by_sku) == 1
    assert any(l["sku"] == "NEEDLESKU-L" for l in by_sku[0]["lines"])

    page = production_db.list_jos(process="Cutting", light=True, limit=2, offset=0)
    assert len(page) == 2
    page2 = production_db.list_jos(process="Cutting", light=True, limit=2, offset=2)
    assert len(page2) == 2
    ids = {j["id"] for j in page} | {j["id"] for j in page2}
    assert len(ids) == 4


def test_list_jos_production_mode_filter(iso):
    production_db.create_jo(
        {
            "so_number": "SO-C2P",
            "sku": "C2P-M",
            "process": "Cutting",
            "planned_qty": 5,
            "production_mode": "cut_to_pack",
            "create_component_jos": False,
            "lines": [{"sku": "C2P-M", "planned_qty": 5}],
        }
    )
    production_db.create_jo(
        {
            "so_number": "SO-IH",
            "sku": "IH-M",
            "process": "Cutting",
            "planned_qty": 5,
            "production_mode": "inhouse",
            "create_component_jos": False,
            "lines": [{"sku": "IH-M", "planned_qty": 5}],
        }
    )
    c2p = production_db.list_jos(process="Cutting", light=True, production_mode="cut_to_pack")
    assert len(c2p) >= 1
    assert all("cut" in str(j.get("production_mode") or "").lower() for j in c2p)
    by_q = production_db.list_jos(process="Cutting", light=True, q="cut_to_pack")
    assert any(j.get("sku") == "C2P-M" for j in by_q)

"""Set-component JOs (…-TOP / …-PANT) continue from Ready-To to the next process."""
from __future__ import annotations

import pytest

from backend.db import grey_db, production_db, sales_db


def _fake_bom(main):
    return {
        "style_key": main,
        "lines": [
            {
                "component_code": "TOP",
                "component_name": "Top",
                "component_role": "SET_COMPONENT",
                "qty_per_set": 1,
                "materials": [],
            },
            {
                "component_code": "PANT",
                "component_name": "Pant",
                "component_role": "SET_COMPONENT",
                "qty_per_set": 1,
                "materials": [],
            },
        ],
    }


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    prod = str(tmp_path / "production.db")
    grey = str(tmp_path / "grey.db")
    sales = str(tmp_path / "sales.db")
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setenv("GREY_DB_PATH", grey)
    monkeypatch.setenv("SALES_DB_PATH", sales)
    monkeypatch.setattr(production_db, "_DB", prod)
    monkeypatch.setattr(grey_db, "_DB", grey)
    monkeypatch.setattr(sales_db, "_DB", sales)
    production_db.init_db()
    grey_db.init_db()
    sales_db.init_db()
    monkeypatch.setattr("backend.services.component_bom.effective_set_bom_for_cutting", _fake_bom)
    monkeypatch.setattr(
        "backend.services.component_bom.set_component_lines",
        lambda bom: bom["lines"] if bom else [],
    )
    yield


def _cut_and_receive(so: str, main: str, qty: int) -> dict[str, dict]:
    nums = production_db.create_jo(
        {"so_number": so, "so_source": "manual", "sku": main, "process": "Cutting", "planned_qty": qty}
    )
    assert isinstance(nums, list) and len(nums) == 2
    jos = {}
    for num in nums:
        ref = production_db.get_jo_by_number(num)
        jo = production_db.get_jo(ref["id"])
        production_db.receive_pieces(
            jo["id"], {"received_qty": qty, "process": "Cutting", "sku": jo["sku"], "split_components": False}
        )
        jos[jo["component_code"]] = jo
    return jos


def test_component_cutting_jo_continues_to_stitching(iso):
    so, main = "SO-SET-1", "1742YKBLUE-8XL"
    jos = _cut_and_receive(so, main, 10)
    pant = f"{main}-PANT"
    assert production_db.get_process_stock(so, pant, "Cutting") == 10

    num = production_db.create_jo(
        {
            "so_number": so,
            "so_source": "manual",
            "sku": pant,
            "sku_name": "Blue 8XL Pant",
            "process": "Stitching",
            "planned_qty": 10,
            "lines": [{"so_number": so, "sku": pant, "sku_name": "Pant", "planned_qty": 10}],
        }
    )
    assert isinstance(num, str)
    jo = production_db.get_jo(production_db.get_jo_by_number(num)["id"])
    assert jo["sku"] == pant
    assert jo["main_sku"] == main
    assert jo["component_code"] == "PANT"
    assert jo["sku_role"] == "COMPONENT"
    ln = jo["lines"][0]
    assert ln["sku"] == pant and ln["parent_sku"] == main and ln["component_code"] == "PANT"
    assert jos["PANT"]["sku"] == pant


def test_component_jo_for_every_ready_to_process(iso):
    so, main = "SO-SET-2", "1742YKBLUE-L"
    _cut_and_receive(so, main, 6)
    top = f"{main}-TOP"
    for process in ("Stitching", "Embroidery", "Printing", "Kaj Button", "Handwork", "Finishing", "Packing"):
        conn = production_db._connect()
        production_db._update_process_stock(conn, so, top, production_db.get_previous_process(top, process, so_number=so) or "Cutting", qty_in=6)
        conn.commit()
        conn.close()
        num = production_db.create_jo(
            {"so_number": so, "so_source": "manual", "sku": top, "process": process, "planned_qty": 1,
             "lines": [{"so_number": so, "sku": top, "planned_qty": 1}]}
        )
        jo = production_db.get_jo(production_db.get_jo_by_number(num)["id"])
        assert jo["process"] == process
        assert (jo["sku"], jo["main_sku"], jo["component_code"]) == (top, main, "TOP")


def test_mixed_component_lines_keep_per_line_traceability(iso):
    so, main = "SO-SET-3", "1742YKBLUE-M"
    _cut_and_receive(so, main, 5)
    num = production_db.create_jo(
        {
            "so_number": so,
            "so_source": "manual",
            "sku": f"{main}-TOP",
            "process": "Stitching",
            "planned_qty": 8,
            "lines": [
                {"so_number": so, "sku": f"{main}-TOP", "planned_qty": 5},
                {"so_number": so, "sku": f"{main}-PANT", "planned_qty": 3},
            ],
        }
    )
    jo = production_db.get_jo(production_db.get_jo_by_number(num)["id"])
    assert int(jo["planned_qty"]) == 8
    by_code = {ln["component_code"]: ln for ln in jo["lines"]}
    assert set(by_code) == {"TOP", "PANT"}
    assert all(ln["parent_sku"] == main for ln in jo["lines"])


def test_component_next_process_rejects_over_ready_qty(iso):
    so, main = "SO-SET-4", "1742YKBLUE-S"
    _cut_and_receive(so, main, 4)
    with pytest.raises(ValueError, match="available"):
        production_db.create_jo(
            {
                "so_number": so,
                "so_source": "manual",
                "sku": f"{main}-TOP",
                "process": "Stitching",
                "planned_qty": 9,
                "lines": [
                    {"so_number": so, "sku": f"{main}-TOP", "planned_qty": 4},
                    {"so_number": so, "sku": f"{main}-PANT", "planned_qty": 5},
                ],
            }
        )


def test_create_next_process_jo_keeps_component_fields(iso):
    so, main = "SO-SET-5", "1742YKBLUE-XL"
    jos = _cut_and_receive(so, main, 7)
    res = production_db.create_next_process_jo(jos["PANT"]["id"])
    assert res["ok"], res
    child = production_db.get_jo(production_db.get_jo_by_number(res["jo_number"])["id"])
    assert (child["sku"], child["main_sku"], child["component_code"], child["sku_role"]) == (
        f"{main}-PANT", main, "PANT", "COMPONENT"
    )


def test_ready_to_create_jo_http_component_sku(iso, client):
    so, main = "SO-SET-7", "1742YKBLUE-8XL"
    _cut_and_receive(so, main, 12)
    pant = f"{main}-PANT"
    r = client.post(
        "/api/production/orders",
        json={
            "so_number": so,
            "so_source": "manual",
            "sku": pant,
            "process": "Stitching",
            "planned_qty": 12,
            "lines": [{"so_number": so, "sku": pant, "planned_qty": 12}],
        },
    )
    assert r.status_code == 200, r.text
    over = client.post(
        "/api/production/orders",
        json={
            "so_number": so,
            "so_source": "manual",
            "sku": f"{main}-TOP",
            "process": "Stitching",
            "planned_qty": 13,
            "lines": [{"so_number": so, "sku": f"{main}-TOP", "planned_qty": 13}],
        },
    )
    assert over.status_code == 400 and "Cannot create a Cutting JO" not in over.text


def test_cutting_on_component_sku_still_rejected(iso):
    with pytest.raises(ValueError, match="Cannot create a Cutting JO on component SKU"):
        production_db.create_jo(
            {
                "so_number": "SO-SET-6",
                "so_source": "manual",
                "process": "Cutting",
                "sku": "1742YKBLUE-8XL",
                "lines": [
                    {"sku": "1742YKBLUE-8XL-PANT", "planned_qty": 2},
                    {"sku": "1742YKBLUE-7XL", "planned_qty": 2},
                ],
                "create_component_jos": False,
            }
        )

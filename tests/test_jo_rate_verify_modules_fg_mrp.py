"""JO rate save, verify lock, module access update, direct FG MRP purchase."""
from __future__ import annotations

import json

import pytest

from backend.db import document_audit_db as audit
from backend.db import item_db, production_db, purchase_db, sales_db, users_db
from backend.routers import production as prod_router
from backend.routers.production import JOLineQtyUpdate, JOUpdate, calculate_mrp


@pytest.fixture()
def erp_env(tmp_path, monkeypatch):
    pdb = str(tmp_path / "purchase.db")
    idb = str(tmp_path / "items.db")
    sdb = str(tmp_path / "sales.db")
    prdb = str(tmp_path / "production.db")
    udb = str(tmp_path / "users.db")
    monkeypatch.setenv("PURCHASE_DB_PATH", pdb)
    monkeypatch.setenv("DOCUMENT_AUDIT_DB_PATH", pdb)
    monkeypatch.setenv("ITEM_DB_PATH", idb)
    monkeypatch.setenv("SALES_DB_PATH", sdb)
    monkeypatch.setenv("PRODUCTION_DB_PATH", prdb)
    monkeypatch.setenv("USERS_DB_PATH", udb)
    monkeypatch.setattr(purchase_db, "_DB", pdb)
    monkeypatch.setattr(audit, "DB_PATH", pdb)
    monkeypatch.setattr(item_db, "DB_PATH", idb)
    monkeypatch.setattr(sales_db, "_DB", sdb)
    monkeypatch.setattr(production_db, "_DB", prdb)
    monkeypatch.setattr(production_db, "_ITEM_DB", idb)
    monkeypatch.setattr(prod_router, "_ITEM_DB_PATH", idb, raising=False)
    monkeypatch.setattr(users_db, "_DB", udb)
    purchase_db.init_db()
    item_db.init_db()
    sales_db.init_db()
    production_db.init_db()
    audit.init_db()
    users_db.init_db()
    return True


def test_jo_line_qty_update_model_keeps_vendor_rate():
    row = JOLineQtyUpdate(id=1, planned_qty=10, vendor_rate=12.5)
    assert row.model_dump()["vendor_rate"] == 12.5
    body = JOUpdate(lines=[row], qty_change_remarks="test")
    assert body.lines[0].vendor_rate == 12.5


def test_jo_rate_saves_and_rolls_amount(erp_env):
    jo_num = production_db.create_jo(
        {
            "so_number": "SO-R1",
            "sku": "STY-R1",
            "process": "Cutting",
            "planned_qty": 10,
            "vendor_rate": 0,
            "exec_type": "Outsource",
            "vendor_name": "V1",
            "lines": [
                {"sku": "STY-R1-S", "sku_name": "S", "planned_qty": 4, "vendor_rate": 0},
                {"sku": "STY-R1-M", "sku_name": "M", "planned_qty": 6, "vendor_rate": 0},
            ],
            "create_component_jos": False,
        }
    )
    if isinstance(jo_num, list):
        jo_num = jo_num[0]
    jo = production_db.get_jo_by_number(jo_num)
    assert jo and jo["lines"]
    lids = [int(l["id"]) for l in jo["lines"]]
    production_db.update_jo(
        int(jo["id"]),
        {
            "lines": [
                {"id": lids[0], "planned_qty": 4, "vendor_rate": 25},
                {"id": lids[1], "planned_qty": 6, "vendor_rate": 25},
            ],
            "qty_change_remarks": "rate fix",
        },
    )
    jo2 = production_db.get_jo(int(jo["id"]))
    assert all(float(l["vendor_rate"]) == 25 for l in jo2["lines"])
    assert float(jo2.get("total_cost") or 0) == pytest.approx(10 * 25)


def test_verified_jo_blocks_edit(erp_env):
    jo_num = production_db.create_jo(
        {
            "so_number": "SO-V1",
            "sku": "STYLEVERIFY",
            "process": "Cutting",
            "planned_qty": 5,
            "lines": [{"sku": "STYLEVERIFY", "planned_qty": 5, "vendor_rate": 1}],
            "create_component_jos": False,
        }
    )
    if isinstance(jo_num, list):
        jo_num = jo_num[0]
    jo = production_db.get_jo_by_number(jo_num)
    joid = int(jo["id"])
    audit.enroll_document("JO", joid, doc_number=jo_num)
    audit.verify_document("JO", joid, actor="accounts")
    with pytest.raises(ValueError, match="Verified"):
        production_db.update_jo(
            joid, {"vendor_rate": 99, "vendor_name": "X", "exec_type": "Outsource"}
        )
    with pytest.raises(ValueError, match="Verified"):
        production_db.add_cost(joid, {"amount": 10, "cost_type": "Labour"})
    with pytest.raises(ValueError, match="Verified"):
        production_db.create_next_process_jo(joid)
    audit.unverify_document("JO", joid, actor="accounts", reason="fix typo ok", force=True)
    production_db.update_jo(
        joid, {"vendor_rate": 9, "vendor_name": "X", "exec_type": "Outsource"}
    )
    jo2 = production_db.get_jo(joid)
    assert float(jo2["vendor_rate"]) == 9
    assert jo2.get("accounts_verified") is False


def test_update_user_module_access(erp_env):
    roles = users_db.list_roles()
    emp_role = next(r for r in roles if r["role_name"] == "Employee")
    users_db.create_user(
        {
            "username": "hrm_only_user",
            "password": "changeme123",
            "full_name": "HRM Only",
            "role_id": emp_role["id"],
            "department": "Production",
            "module_access": json.dumps(["hrm"]),
        }
    )
    users = users_db.list_users(active_only=False)
    u0 = next(x for x in users if x["username"] == "hrm_only_user")
    users_db.update_user(u0["id"], {"module_access": json.dumps(["hrm", "sales", "purchase"])})
    u = users_db.get_user_by_id(u0["id"])
    mods = json.loads(u["module_access"] or "[]")
    assert "sales" in mods and "purchase" in mods and "hrm" in mods


def test_mrp_direct_fg_purchase_only_when_procurement_purchase(erp_env, monkeypatch):
    monkeypatch.setattr(prod_router, "_item_connect", item_db._connect)
    types = item_db.list_item_types()
    fg_type = types[0]
    item_db.create_item(
        item_code="BUY-FG-1",
        item_name="Bought FG",
        item_type_id=fg_type["id"],
        procurement_type="Purchase",
        uom="PCS",
    )
    make_id = item_db.create_item(
        item_code="MAKE-FG-1",
        item_name="Made FG",
        item_type_id=fg_type["id"],
        procurement_type="Make",
        uom="PCS",
    )
    bom_id = item_db.create_bom(make_id, "Default", is_default=1)
    rm_id = item_db.create_item(
        item_code="RM-X1",
        item_name="RM",
        item_type_id=fg_type["id"],
        procurement_type="Purchase",
        uom="MTR",
    )
    item_db.add_bom_line(
        bom_id,
        component_name="RM-X1",
        component_type="RM",
        quantity=1,
        unit="MTR",
        component_item_id=rm_id,
    )

    so_buy = sales_db.create_order(
        {
            "buyer": "B",
            "status": "Confirmed",
            "lines": [{"sku": "BUY-FG-1", "sku_name": "Bought FG", "qty": 8, "unit": "PCS"}],
        }
    )
    so_make = sales_db.create_order(
        {
            "buyer": "B",
            "status": "Confirmed",
            "lines": [{"sku": "MAKE-FG-1", "sku_name": "Made FG", "qty": 5, "unit": "PCS"}],
        }
    )
    payload = calculate_mrp([so_buy, so_make])
    mats = payload["materials"]
    assert "BUY-FG-1" in mats
    assert mats["BUY-FG-1"].get("direct_fg_purchase") is True
    assert mats["BUY-FG-1"]["total_req"] == pytest.approx(8)
    assert mats["BUY-FG-1"].get("type") == "FG"
    assert not mats.get("MAKE-FG-1", {}).get("direct_fg_purchase")
    assert "RM-X1" in mats


def test_so_line_procurement_override_to_purchase(erp_env, monkeypatch):
    monkeypatch.setattr(prod_router, "_item_connect", item_db._connect)
    types = item_db.list_item_types()
    item_db.create_item(
        item_code="FLEX-FG",
        item_name="Flex",
        item_type_id=types[0]["id"],
        procurement_type="Make",
        uom="PCS",
    )
    so = sales_db.create_order(
        {
            "buyer": "B",
            "status": "Confirmed",
            "lines": [
                {
                    "sku": "FLEX-FG",
                    "sku_name": "Flex",
                    "qty": 3,
                    "unit": "PCS",
                    "procurement_override": "Purchase",
                }
            ],
        }
    )
    mats = calculate_mrp([so])["materials"]
    assert "FLEX-FG" in mats
    assert mats["FLEX-FG"].get("direct_fg_purchase") is True

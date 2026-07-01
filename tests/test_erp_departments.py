"""ERP user departments — dynamic Admin dropdown."""
from __future__ import annotations

import pytest

from backend.db import users_db


def test_list_and_create_erp_department(tmp_path, monkeypatch):
    db = tmp_path / "users_test.db"
    monkeypatch.setenv("USERS_DB_PATH", str(db))
    users_db.init_db()

    names = [d["name"] for d in users_db.list_erp_departments()]
    assert "Sales" in names
    assert "Production" in names

    row = users_db.create_erp_department("E-commerce")
    assert row["name"] == "E-commerce"

    names2 = [d["name"] for d in users_db.list_erp_departments()]
    assert "E-commerce" in names2

    with pytest.raises(ValueError, match="already exists"):
        users_db.create_erp_department("E-commerce")


def test_may_manage_erp_departments_for_harsh():
    from backend.services.permissions import may_manage_erp_departments

    assert may_manage_erp_departments("Employee", "harsh") is True
    assert may_manage_erp_departments("Manager", "anyone") is True
    assert may_manage_erp_departments("Employee", "bob") is False

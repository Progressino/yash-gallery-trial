"""Auth and Karigar role access tests."""
from __future__ import annotations

import bcrypt
import pytest

from backend.db import users_db
from backend.db.users_db import init_db, create_user, list_roles, verify_erp_user
from backend.routers.auth import create_token, decode_token
from backend.services.permissions import karigar_may_access_api, permissions_for_role


@pytest.fixture(autouse=True)
def isolated_users_db(tmp_path, monkeypatch):
    path = str(tmp_path / "users.db")
    monkeypatch.setenv("USERS_DB_PATH", path)
    monkeypatch.setattr(users_db, "_DB", path)
    init_db()


def test_karigar_role_seeded():
    roles = {r["role_name"] for r in list_roles()}
    assert "Karigar" in roles


def test_karigar_login_and_api_scope():
    roles = list_roles()
    karigar_role = next(r for r in roles if r["role_name"] == "Karigar")
    create_user(
        {
            "username": "k1065",
            "password": "test1234",
            "full_name": "Ramesh Kumar",
            "role_id": karigar_role["id"],
            "department": "Production",
            "karigar_id": "K001",
        }
    )
    user = verify_erp_user("k1065", "test1234")
    assert user is not None
    assert user["role_name"] == "Karigar"
    assert user["karigar_id"] == "K001"

    token = create_token(
        "k1065",
        role="Karigar",
        user_id=user["id"],
        karigar_id="K001",
    )
    payload = decode_token(token)
    assert payload["role"] == "Karigar"
    assert "stitching.production_entry" in permissions_for_role("Karigar")

    assert karigar_may_access_api("/api/stitching/production-entry", "POST")
    assert karigar_may_access_api("/api/stitching/sheets/style_master", "GET")
    assert not karigar_may_access_api("/api/stitching/dashboard", "GET")
    assert not karigar_may_access_api("/api/purchase/orders", "GET")

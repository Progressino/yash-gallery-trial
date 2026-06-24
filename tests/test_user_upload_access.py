"""Per-user upload module grants and Irfan account bootstrap."""
import bcrypt
import pytest

from backend.db import users_db
from backend.services.rbac import resolve_module_access
from backend.services.upload_policy import may_upload_historical, upload_policy_for_role


@pytest.fixture(autouse=True)
def isolated_users_db(tmp_path, monkeypatch):
    path = str(tmp_path / "users.db")
    monkeypatch.setenv("USERS_DB_PATH", path)
    monkeypatch.setattr(users_db, "_DB", path)
    users_db.init_db()


def test_ensure_user_upload_access_grants_upload_modules():
    hod_id = users_db.get_role_id("HOD")
    users_db.create_user(
        {
            "username": "irfan_test",
            "password": "secret",
            "role_id": hod_id,
            "full_name": "Irfan Test",
        }
    )

    conn = users_db._connect()
    users_db.ensure_user_upload_access(conn, "irfan_test")
    conn.commit()
    conn.close()

    profile = users_db.get_user_auth_profile("irfan_test")
    mods = resolve_module_access(profile["role_name"], profile.get("module_access"))
    assert "upload" in mods
    assert "intelligence" in mods
    assert "meesho" in mods


def test_login_username_is_case_insensitive():
    exec_id = users_db.get_role_id("Executive")
    users_db.create_user(
        {"username": "irfan", "password": "secret", "role_id": exec_id, "full_name": "Irfan"}
    )

    assert users_db.verify_erp_user("IRFAN", "secret") is not None
    assert users_db.verify_erp_user("Irfan", "secret") is not None
    assert users_db.get_user_auth_profile("IRFAN")["username"] == "irfan"


def test_create_user_rejects_case_insensitive_duplicate():
    exec_id = users_db.get_role_id("Executive")
    users_db.create_user(
        {"username": "irfan", "password": "secret", "role_id": exec_id, "full_name": "Irfan"}
    )

    with pytest.raises(ValueError, match="case-insensitive"):
        users_db.create_user(
            {"username": "Irfan", "password": "x", "role_id": exec_id, "full_name": "Dup"}
        )


def test_ensure_irfan_upload_account_consolidates_and_grants_historical():
    exec_id = users_db.get_role_id("Executive")
    users_db.create_user(
        {"username": "Irfan", "password": "old", "role_id": exec_id, "full_name": ""}
    )
    # Legacy duplicate (allowed before case-insensitive create guard).
    hashed = bcrypt.hashpw(b"keep", bcrypt.gensalt()).decode()
    conn = users_db._connect()
    conn.execute(
        """INSERT INTO erp_users(username, password_hash, full_name, role_id, active)
           VALUES (?,?,?,?,1)""",
        ("irfan", hashed, "Irfan", exec_id),
    )
    conn.commit()
    conn.close()

    conn = users_db._connect()
    users_db.ensure_irfan_upload_account(conn)
    conn.commit()
    conn.close()

    active = [u for u in users_db.list_users(active_only=True) if u["username"].lower() == "irfan"]
    assert len(active) == 1
    user = active[0]
    assert user["username"] == "irfan"
    assert user["role_name"] == "Admin"

    profile = users_db.get_user_auth_profile("IRFAN")
    assert profile["username"] == "irfan"
    assert users_db.verify_erp_user("Irfan", "keep") is not None
    assert users_db.verify_erp_user("IRFAN", "old") is None

    pol = upload_policy_for_role(user["role_name"], user["username"])
    assert may_upload_historical(user["role_name"]) is True
    assert pol["may_upload_historical"] is True
    assert pol["may_upload_po_baseline"] is True

    mods = resolve_module_access(user["role_name"], user.get("module_access"))
    assert "upload" in mods

"""Inventory RAR upload — parse worker and chunk finalize routing."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.routers import upload as upload_router
from backend.session import AppSession, store


RAR_FIXTURES = [
    Path("/Users/samraisinghani/Downloads/Inventory 1-Aug-26.rar"),
    Path("/Users/samraisinghani/Downloads/Inventory 2-Aug-26 1.rar"),
    Path("/Users/samraisinghani/Downloads/Inventory 3-Aug-26.rar"),
]


def test_finalize_chunk_upload_routes_inventory_to_inventory_executor(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class _Pool:
        def __init__(self, label: str):
            self.label = label

        def submit(self, fn, *args):
            calls.append((self.label, args))

    monkeypatch.setattr(upload_router, "INVENTORY_EXECUTOR", _Pool("inventory"))
    monkeypatch.setattr(upload_router, "DAILY_UPLOAD_EXECUTOR", _Pool("daily"))
    monkeypatch.setattr(
        upload_router.chunk_store,
        "get_target",
        lambda _sid, _uid: "inventory-auto",
    )

    upload_router._finalize_chunk_upload("sess-test", "upload-test")
    assert calls and calls[0][0] == "inventory"


def test_finalize_chunk_upload_routes_daily_to_daily_executor(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class _Pool:
        def __init__(self, label: str):
            self.label = label

        def submit(self, fn, *args):
            calls.append((self.label, args))

    monkeypatch.setattr(upload_router, "INVENTORY_EXECUTOR", _Pool("inventory"))
    monkeypatch.setattr(upload_router, "DAILY_UPLOAD_EXECUTOR", _Pool("daily"))
    monkeypatch.setattr(
        upload_router.chunk_store,
        "get_target",
        lambda _sid, _uid: "daily-auto",
    )

    upload_router._finalize_chunk_upload("sess-test", "upload-test")
    assert calls and calls[0][0] == "daily"


@pytest.mark.parametrize(
    "rar_path,expected_date",
    [
        (RAR_FIXTURES[0], "2026-08-01"),
        (RAR_FIXTURES[1], "2026-08-02"),
        (RAR_FIXTURES[2], "2026-08-03"),
    ],
)
def test_inventory_rar_daily_bundle_parses(session_for_client, rar_path: Path, expected_date: str, monkeypatch):
    if not rar_path.is_file():
        pytest.skip(f"RAR fixture missing: {rar_path}")

    monkeypatch.setattr(upload_router, "_persist_inventory_upload_status", lambda *a, **k: None)
    monkeypatch.setattr(upload_router, "_finish_inventory_server_save", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.append_snapshot_inventory_to_history",
        lambda _sess: {"appended": False, "reason": "test"},
    )
    monkeypatch.setattr(upload_router, "_finalize_inventory_data_refresh", lambda _sess: None)
    monkeypatch.setattr(upload_router, "_session_data_changed", lambda _sess: None)

    sid, sess = session_for_client
    sess.sku_mapping = {"TEST-SKU": "TEST-SKU"}
    store._sessions[sid] = sess

    upload_router._run_inventory_auto_from_parts(sid, [(rar_path.name, rar_path.read_bytes())])

    assert sess.inventory_upload_status == "done", sess.inventory_upload_message
    assert sess.inventory_df_variant is not None
    assert not sess.inventory_df_variant.empty
    snap = str(getattr(sess, "inventory_snapshot_date", "") or "")
    assert expected_date in snap or expected_date in str(getattr(sess, "inventory_snapshot_date_label", "") or "")


def test_po_gate_blocks_during_sales_rebuild():
    from backend.services.po_pipeline import check_calculate_gate
    from backend.session import AppSession

    sess = AppSession()
    sess.sales_rebuild_status = "running"
    gate = check_calculate_gate(sess)
    assert gate["calculate_allowed"] is False
    assert any("sales rebuild" in b.lower() for b in gate["blockers"])

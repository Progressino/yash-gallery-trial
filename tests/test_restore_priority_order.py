"""Restore-full must load GitHub before disk/Tier-3 (full MTR priority)."""
import pandas as pd
import pytest

from backend.routers import data as data_router
from backend.session import AppSession


@pytest.fixture
def restore_sess():
    return AppSession()


def test_full_restore_calls_github_before_disk_and_tier3(monkeypatch, restore_sess):
    order: list[str] = []
    sess = restore_sess
    sess.sku_mapping = {"A": "A"}

    monkeypatch.setattr(
        data_router,
        "_set_restore_step",
        lambda s, step, detail=None: order.append(step),
    )
    monkeypatch.setattr(
        "backend.session.resume_auto_data_restore",
        lambda s: None,
    )
    import backend.main as main

    monkeypatch.setattr(main, "restore_po_sidecars_from_warm", lambda s: None)
    monkeypatch.setattr(main, "force_restore_session_from_server_cache", lambda s, g: True)
    monkeypatch.setattr(main, "publish_warm_cache_from_session", lambda s: None)
    monkeypatch.setattr(
        data_router,
        "_merge_github_bulk_into_session",
        lambda s: order.append("github_fn") or True,
    )
    monkeypatch.setattr(
        data_router,
        "_merge_disk_warm_into_session",
        lambda s: order.append("disk_fn") or "",
    )
    monkeypatch.setattr(data_router, "_restore_inventory_from_warm", lambda s: order.append("inv_fn"))
    monkeypatch.setattr(
        data_router,
        "_restore_daily_if_needed",
        lambda *a, **k: order.append("tier3_fn"),
    )
    monkeypatch.setattr(
        "backend.routers.cache._merge_daily_store_into_session",
        lambda s: "",
    )
    monkeypatch.setattr(
        "backend.services.sku_mapping.restore_sku_mapping_to_session",
        lambda s: None,
    )

    data_router.full_restore_session(sess, defer_sales_rebuild=True)

    assert "github_fn" in order
    assert "disk_fn" in order
    assert "tier3_fn" in order
    assert order.index("github_fn") < order.index("disk_fn")
    assert order.index("disk_fn") < order.index("tier3_fn")


def test_restore_daily_if_needed_accepts_restore_full_mode_flag():
    import inspect

    sig = inspect.signature(data_router._restore_daily_if_needed)
    assert "restore_full_mode" in sig.parameters
    assert "skip_sales_rebuild" in sig.parameters

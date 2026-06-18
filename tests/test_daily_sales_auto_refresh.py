"""Daily upload must bust PO/Intelligence caches and bump sales_data_revision."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.routers import upload as upload_router
from backend.session import AppSession


def test_finalize_sales_data_refresh_bumps_revision_and_sync_token(monkeypatch, tmp_path):
    from backend.services import po_shared_cache as psc

    shared = tmp_path / "shared_po"
    shared.mkdir()
    monkeypatch.setattr(psc, "_shared_dir", lambda: shared)

    invalidate_calls: list[str] = []

    def _fake_invalidate(sess):
        invalidate_calls.append("po")

    monkeypatch.setattr(psc, "invalidate_po_after_sales_or_returns_change", _fake_invalidate)

    applied: list[dict] = []

    def _fake_mark(sess):
        applied.append({"amazon": "1:10:now"})
        sess._tier3_sync_token_applied = {"amazon": "1:10:now"}

    monkeypatch.setattr(
        "backend.services.tier3_session_merge.mark_tier3_sync_applied",
        _fake_mark,
    )

    sess = AppSession()
    sess.sales_data_revision = 2
    sess.daily_restored = False

    upload_router._finalize_sales_data_refresh(sess)

    assert sess.sales_data_revision == 3
    assert sess.daily_restored is True
    assert invalidate_calls == ["po"]
    assert applied


def test_incremental_sales_rebuild_calls_finalize(monkeypatch):
    """Fast ingest path must invalidate caches (not only the full rebuild path)."""
    calls: list[str] = []

    monkeypatch.setattr(
        upload_router,
        "_finalize_sales_data_refresh",
        lambda sess: calls.append("finalize"),
    )
    monkeypatch.setattr(
        upload_router,
        "_session_data_changed",
        lambda sess: calls.append("session"),
    )
    monkeypatch.setattr(
        upload_router,
        "_combine_buffered_platform_df",
        lambda sess, plat: pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-17"]),
                "SKU": ["SKU-A"],
                "Transaction_Type": ["Shipment"],
                "Quantity": [5],
                "OrderId": ["O1"],
            }
        ),
    )
    monkeypatch.setattr(
        "backend.services.sales.build_sales_df",
        lambda **kw: pd.DataFrame(
            {
                "TxnDate": pd.to_datetime(["2026-06-17"]),
                "Sku": ["SKU-A"],
                "Transaction Type": ["Shipment"],
                "Quantity": [5],
                "Source": ["Amazon"],
                "OrderId": ["O1"],
            }
        ),
    )
    monkeypatch.setattr(
        "backend.services.sales.patch_sales_df_after_daily_upload",
        lambda existing, fresh, plats, d0, d1: fresh,
    )
    monkeypatch.setattr(
        upload_router,
        "_merge_platform_data",
        lambda cur, combined, plat, **kw: combined,
    )

    sess = AppSession()
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    sess._daily_auto_parsed_buffers = {"amazon": [pd.DataFrame()]}

    ok, msg = upload_router._incremental_sales_rebuild_from_buffers(sess, {"amazon"})

    assert ok is True
    assert "Sales updated" in msg
    assert calls == ["session", "finalize"]


def test_warm_cache_skipped_when_tier3_mismatch(monkeypatch):
    """Coverage polls must not overwrite a session that has newer Tier-3 uploads."""
    import backend.main as main_mod

    sess = AppSession()
    sess.mtr_df = pd.DataFrame({"Date": pd.to_datetime(["2026-06-17"]), "SKU": ["A"]})
    sess.sales_df = pd.DataFrame({"TxnDate": pd.to_datetime(["2026-06-17"]), "Sku": ["A"]})

    main_mod._warm_cache = {
        "mtr_df": pd.DataFrame({"Date": pd.to_datetime(["2026-06-01"] * 5), "SKU": ["X"] * 5}),
        "sales_df": pd.DataFrame({"TxnDate": pd.to_datetime(["2026-06-01"] * 5), "Sku": ["X"] * 5}),
    }

    monkeypatch.setattr(
        "backend.routers.data._tier3_token_mismatch",
        lambda s: True,
    )

    assert main_mod._apply_warm_cache_if_needed(sess, 2) is False
    assert len(sess.mtr_df) == 1
    assert len(sess.sales_df) == 1

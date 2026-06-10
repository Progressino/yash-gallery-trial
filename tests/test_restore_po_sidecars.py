"""PO optional sheets must restore from warm cache without full daily-restore lock."""
import pandas as pd
import pytest

from backend.main import merge_po_optional_sheets_into_warm_cache, restore_po_sidecars_from_warm
from backend.session import AppSession


@pytest.fixture
def warm_with_po_sidecars(monkeypatch):
    import backend.main as main

    main._warm_cache = {
        "sku_status_lead_df": pd.DataFrame(
            {"SKU": ["A1"], "Status": ["Active"], "Lead time": [14]}
        ),
        "daily_inventory_history_df": pd.DataFrame(
            {
                "OMS_SKU": ["A1", "A1"],
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "Qty": [10, 12],
            }
        ),
    }
    yield
    main._warm_cache = {}


def test_restore_po_sidecars_from_warm_fills_empty_session(warm_with_po_sidecars):
    sess = AppSession()
    assert sess.sku_status_lead_df.empty
    assert sess.daily_inventory_history_df.empty

    changed = restore_po_sidecars_from_warm(sess)
    assert changed is True
    assert not sess.sku_status_lead_df.empty
    assert not sess.daily_inventory_history_df.empty
    assert len(sess.sku_status_lead_df) == 1
    assert len(sess.daily_inventory_history_df) == 2


def test_restore_po_sidecars_skips_non_inventory_when_session_already_has_data(warm_with_po_sidecars):
    """sku_status_lead_df must NOT be overwritten when session already has it.
    daily_inventory_history_df IS merged (not skipped) — warm cache rows for new
    SKU/dates are combined with session rows even when the session is non-empty."""
    sess = AppSession()
    sess.sku_status_lead_df = pd.DataFrame({"SKU": ["X"], "Lead time": [7]})
    sess.daily_inventory_history_df = pd.DataFrame(
        {"OMS_SKU": ["X"], "Date": pd.to_datetime(["2026-02-01"]), "Qty": [1]}
    )
    changed = restore_po_sidecars_from_warm(sess)
    # Inventory history was merged (warm cache has A1 rows, session had X rows → combined)
    assert changed is True
    # Non-inventory sidecar (sku_status_lead_df) must NOT be replaced
    assert sess.sku_status_lead_df.iloc[0]["SKU"] == "X"
    # Merged inventory must contain both the original session row and warm-cache row(s)
    assert len(sess.daily_inventory_history_df) >= 2


def test_restore_po_sidecars_merges_inventory_when_both_have_rows(warm_with_po_sidecars):
    """Session keeps its rows and gains any SKU-days present only in warm cache."""
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        {"OMS_SKU": ["X"], "Date": pd.to_datetime(["2026-02-01"]), "Qty": [1]}
    )
    changed = restore_po_sidecars_from_warm(sess)
    assert changed is True
    assert len(sess.daily_inventory_history_df) >= 2


def test_merge_po_optional_sheets_updates_warm_cache():
    import backend.main as main

    main._warm_cache = {}
    sess = AppSession()
    sess.sku_status_lead_df = pd.DataFrame({"SKU": ["B2"], "Lead time": [21]})
    merge_po_optional_sheets_into_warm_cache(sess)
    assert not main._warm_cache["sku_status_lead_df"].empty
    assert main._warm_cache["sku_status_lead_df"].iloc[0]["SKU"] == "B2"


def test_clearing_raise_ledger_removes_stale_disk_parquet(tmp_path, monkeypatch):
    """A deleted raise ledger must not be resurrected from a stale disk parquet.

    Reproduces: admin deletes a raise ledger day (ledger becomes empty), then a
    later /api/po/calculate hydrate reads ``po_raise_ledger_df.parquet`` from disk
    and restores the deleted row because the stale parquet was never removed.
    """
    import backend.main as main

    monkeypatch.setattr(main, "_DISK_CACHE_DIR", str(tmp_path))
    main._warm_cache = {}

    sess = AppSession()
    sess.po_raise_ledger_df = pd.DataFrame(
        {
            "OMS_SKU": ["1234YKBLACK-XL"],
            "Raised_Qty": [59466],
            "Raised_Date": pd.to_datetime(["2026-05-14"]),
        }
    )
    merge_po_optional_sheets_into_warm_cache(sess)
    parquet_path = tmp_path / "po_raise_ledger_df.parquet"
    assert parquet_path.is_file()

    # Admin deletes the raise — ledger becomes empty.
    sess.po_raise_ledger_df = pd.DataFrame()
    merge_po_optional_sheets_into_warm_cache(sess)

    assert not parquet_path.is_file()
    assert main._warm_cache["po_raise_ledger_df"].empty

    # Disk hydrate must not bring the deleted row back.
    from backend.services.po_session_hydrate import load_po_calc_essentials_from_disk

    monkeypatch.setattr(
        "backend.services.po_session_hydrate._warm_cache_dir", lambda: tmp_path
    )
    disk = load_po_calc_essentials_from_disk()
    assert "po_raise_ledger_df" not in disk

    main._warm_cache = {}

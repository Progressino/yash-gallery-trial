"""Fast Existing PO attach for light coverage polls."""
import pandas as pd


def test_light_coverage_attaches_existing_po_from_warm_cache(monkeypatch):
    from backend.session import AppSession
    from backend.services.existing_po import (
        ensure_existing_po_coverage_light,
        session_has_existing_po,
    )

    ep = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "PO_Qty_Ordered": [10, 0],
            "PO_Pipeline_Total": [10, 5],
        }
    )
    meta = {
        "existing_po_rows": 2,
        "existing_po_pipeline_skus": 2,
        "existing_po_per_size_skus": 2,
        "existing_po_new_order_skus": 1,
        "existing_po_generation": 3,
        "existing_po_uploaded_at": "2026-06-28T10:00:00",
        "existing_po_filename": "Po 27-Jun-26.xlsx",
    }
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "_warm_cache", {"existing_po_df": ep, main_mod._EXISTING_PO_META_WARM_KEY: meta})
    monkeypatch.setattr(main_mod, "bootstrap_warm_cache_if_empty", lambda: True)
    import backend.services.existing_po as ep_mod

    monkeypatch.setattr(ep_mod, "ensure_latest_existing_po_authoritative", lambda _s: False)

    sess = AppSession()
    assert sess.existing_po_df.empty
    assert ensure_existing_po_coverage_light(sess) is True
    assert session_has_existing_po(sess) is True
    assert len(sess.existing_po_df) == 2
    assert sess.existing_po_filename == "Po 27-Jun-26.xlsx"


def test_session_has_existing_po_from_meta_without_parquet(monkeypatch):
    from backend.session import AppSession
    from backend.services.existing_po import session_has_existing_po
    import backend.services.existing_po as ep_mod

    sess = AppSession()
    monkeypatch.setattr(
        ep_mod,
        "read_existing_po_disk_meta",
        lambda: {"existing_po_rows": 1200, "existing_po_filename": "Po.xlsx"},
    )
    monkeypatch.setattr(ep_mod, "_existing_po_meta_for_coverage", lambda _s: {"existing_po_rows": 1200})
    assert session_has_existing_po(sess) is True


def test_ensure_existing_po_hydrated_prefers_warm_cache(monkeypatch):
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    import backend.services.existing_po as ep_mod

    calls: list[str] = []

    def _warm(_s):
        calls.append("warm")
        return True

    def _disk(_s):
        calls.append("disk")
        return False

    monkeypatch.setattr(ep_mod, "ensure_latest_existing_po_authoritative", lambda _s: False)
    monkeypatch.setattr(ep_mod, "restore_existing_po_from_warm_cache", _warm)
    monkeypatch.setattr(ep_mod, "restore_existing_po_from_disk", _disk)
    monkeypatch.setattr(ep_mod, "existing_po_looks_aggregated_bundled_only", lambda _e: False)

    sess = AppSession()
    ensure_existing_po_hydrated(sess)
    assert calls == ["warm", "disk"]


def test_inventory_staleness_ignores_placeholder_disk_meta(monkeypatch):
    from backend.session import AppSession
    from backend.services.po_readiness import _attach_inventory_staleness
    import backend.services.daily_inventory_history as dih

    dates = pd.date_range("2026-06-01", "2026-06-27", freq="D")
    rows = [{"OMS_SKU": "SKU-1", "Date": d, "Qty": 4} for d in dates]
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"

    monkeypatch.setattr(
        dih,
        "read_daily_inventory_history_disk_meta",
        lambda: {
            "daily_inventory_history_max_date": "2026-05-30",
            "daily_inventory_history_rows": 1,
            "daily_inventory_history_skus": 1,
        },
    )

    data: dict = {"inventory": True, "daily_inventory_history": True}
    _attach_inventory_staleness(sess, data)
    assert data.get("daily_inventory_history_max_date") == "2026-06-27"
    assert data.get("daily_inventory_history_stale") is False

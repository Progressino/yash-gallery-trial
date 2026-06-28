"""Daily inventory history must never be downgraded on disk/warm-cache by stale PO sessions."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_is_newer_than,
    inventory_history_view_end_date,
    overlay_inventory_variant_from_history,
)


def _hist(sku: str, end: str, days: int = 30) -> pd.DataFrame:
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    return pd.DataFrame({"OMS_SKU": [sku] * days, "Date": dates, "Qty": [5.0] * days})


def test_inventory_history_is_newer_than_prefers_later_max_date():
    old = _hist("A", "2026-05-30", 3)
    new = _hist("A", "2026-06-26", 30)
    assert inventory_history_is_newer_than(new, old)
    assert not inventory_history_is_newer_than(old, new)


def test_inventory_history_is_newer_than_wins_with_fewer_rows():
    """June upload trimmed to 30d must beat May blob with more rows."""
    may = _hist("A", "2026-05-30", 30)
    june = _hist("A", "2026-06-26", 10)
    assert len(may) > len(june)
    assert inventory_history_is_newer_than(june, may)
    assert not inventory_history_is_newer_than(may, june)


def test_recanonicalize_inventory_history_skus_maps_aliases():
    from backend.services.daily_inventory_history import recanonicalize_inventory_history_skus

    df = pd.DataFrame(
        {
            "OMS_SKU": ["OLD-SKU", "OLD-SKU"],
            "Date": pd.to_datetime(["2026-06-25", "2026-06-26"]),
            "Qty": [3.0, 5.0],
        }
    )
    mapping = {"OLD-SKU": "NEW-SKU"}
    out = recanonicalize_inventory_history_skus(df, mapping)
    assert set(out["OMS_SKU"].astype(str)) == {"NEW-SKU"}
    assert len(out) == 2


def test_existing_po_frame_is_newer_than_prefers_generation():
    from backend.services.existing_po import existing_po_frame_is_newer_than

    old = pd.DataFrame({"OMS_SKU": ["A"] * 500, "PO_Pipeline_Total": [1.0] * 500})
    new = pd.DataFrame({"OMS_SKU": ["A"], "PO_Pipeline_Total": [1.0]})
    assert existing_po_frame_is_newer_than(
        new,
        old,
        incoming_meta={"existing_po_generation": 3, "existing_po_uploaded_at": "2026-06-26T10:00:00"},
        existing_meta={"existing_po_generation": 2, "existing_po_uploaded_at": "2026-05-01T10:00:00"},
    )
    assert not existing_po_frame_is_newer_than(
        old,
        new,
        incoming_meta={"existing_po_generation": 2},
        existing_meta={"existing_po_generation": 3},
    )


def test_view_end_date_anchors_on_matrix_when_behind_today():
    df = _hist("A", "2026-05-30", 3)
    assert inventory_history_view_end_date(df) == "2026-05-30"


def test_overlay_updates_total_inventory_only():
    hist = _hist("SKU-A", "2026-06-26", 5)
    inv = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "OMS_Inventory": [10.0], "Total_Inventory": [25.0]},
    )
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is True
    assert float(out.loc[0, "Total_Inventory"]) == 5.0
    assert float(out.loc[0, "OMS_Inventory"]) == 10.0


def test_authoritative_cap_date_never_exceeds_upload_end():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import inventory_history_authoritative_cap_date

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-26", 5)
    sess.inventory_snapshot_date = "2026-06-26"
    cap = inventory_history_authoritative_cap_date(sess)
    assert str(cap.date()) == "2026-06-26"


def test_authoritative_cap_ignores_newer_snapshot_than_matrix():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-27", 5)
    sess.inventory_snapshot_date = "2026-06-27"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-06-26",
        "daily_inventory_history_max_date": "2026-06-26",
        "daily_inventory_history_rows": 5000,
        "daily_inventory_history_skus": 2000,
    }
    try:
        cap = dih.inventory_history_authoritative_cap_date(sess)
        assert str(cap.date()) == "2026-06-26"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_matrix_cap_prefers_session_over_stale_disk_placeholder():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-05-30",
        "daily_inventory_history_max_date": "2026-05-30",
        "daily_inventory_history_rows": 1,
        "daily_inventory_history_skus": 1,
    }
    try:
        cap = dih.inventory_history_matrix_cap_date(sess)
        assert str(cap.date()) == "2026-06-27"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_authoritative_read_keeps_full_matrix_when_disk_meta_stale(monkeypatch):
    from backend.session import AppSession
    from backend.services.po_session_hydrate import ensure_inventory_history_authoritative_for_read

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("1488YKWHITE-XS-S", "2026-06-27", 30)
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"
    sess.sku_mapping = {"1488YKWHITE-XS-S": "1488YKWHITE-XS-S"}

    monkeypatch.setattr(
        "backend.services.po_session_hydrate.ensure_po_sidecars_hydrated",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.read_daily_inventory_history_disk_meta",
        lambda: {
            "daily_inventory_history_matrix_max_date": "2026-05-30",
            "daily_inventory_history_max_date": "2026-05-30",
            "daily_inventory_history_rows": 1,
            "daily_inventory_history_skus": 1,
        },
    )

    out = ensure_inventory_history_authoritative_for_read(sess)
    sub = out[out["OMS_SKU"] == "1488YKWHITE-XS-S"]
    assert len(sub) == 30
    assert str(sub["Date"].max().date()) == "2026-06-27"


def test_snapshot_append_does_not_inflate_sku_count():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("MATRIX-SKU", "2026-06-26", 10)
    sess.daily_inventory_history_matrix_max_date = "2026-06-26"
    sess.inventory_snapshot_date = "2026-06-26"
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": [f"EXTRA-{i}" for i in range(500)] + ["MATRIX-SKU"],
            "Total_Inventory": [1] * 500 + [12],
        }
    )
    out = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert out.get("ok") is True
    skus = sess.daily_inventory_history_df["OMS_SKU"].astype(str).nunique()
    assert skus == 1


def test_ensure_inventory_history_trims_beyond_cap(monkeypatch):
    from backend.session import AppSession
    from backend.services.po_session_hydrate import ensure_inventory_history_authoritative_for_read

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-27", 3)
    sess.inventory_snapshot_date = "2026-06-26"

    monkeypatch.setattr(
        "backend.services.po_session_hydrate.ensure_po_sidecars_hydrated",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.read_daily_inventory_history_disk_meta",
        lambda: {
            "daily_inventory_history_max_date": "2026-06-26",
            "daily_inventory_history_matrix_max_date": "2026-06-26",
            "daily_inventory_history_rows": 5000,
            "daily_inventory_history_skus": 2000,
        },
    )

    out = ensure_inventory_history_authoritative_for_read(sess)
    mx = out["Date"].max().normalize()
    assert str(mx.date()) == "2026-06-26"

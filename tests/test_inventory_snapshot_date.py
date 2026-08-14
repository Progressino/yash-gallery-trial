"""Inventory snapshot as-of date inference from upload filenames."""

from backend.services.inventory import infer_inventory_snapshot_date


def test_infer_date_from_oms_filename():
    meta = infer_inventory_snapshot_date(
        [("OMS 25-05-2026.csv", b"")],
        {},
    )
    assert meta["snapshot_date"] == "2026-05-25"
    assert meta["snapshot_date_label"] == "25 May 2026"
    assert "OMS 25-05-2026.csv" in meta["snapshot_date_sources"]


def test_infer_date_from_inventory_rar_prefers_bundle_name():
    meta = infer_inventory_snapshot_date(
        [("Inventory 25-May-26.rar", b"")],
        {"amz_disclaimer": {"latest_report_date": "2026-05-24"}},
    )
    assert meta["snapshot_date"] == "2026-05-25"
    assert any("Inventory" in s for s in meta["snapshot_date_sources"])


def test_infer_date_from_amz_only():
    meta = infer_inventory_snapshot_date(
        [],
        {"amz_disclaimer": {"latest_report_date": "2026-05-24"}},
    )
    assert meta["snapshot_date"] == "2026-05-24"
    assert any("Amazon" in s for s in meta["snapshot_date_sources"])


def test_infer_prefers_latest_oms_date_not_earliest():
    meta = infer_inventory_snapshot_date(
        [
            ("OMS 08-08-2026.csv", b""),
            ("OMS 13-08-2026.csv", b""),
        ],
        {},
    )
    assert meta["snapshot_date"] == "2026-08-13"


def test_infer_empty_filename_uses_today_ist():
    from backend.services.inventory import _inventory_asof_today_ist

    meta = infer_inventory_snapshot_date([("OMS.rar", b"")], {})
    assert meta["snapshot_date"] == _inventory_asof_today_ist().isoformat()

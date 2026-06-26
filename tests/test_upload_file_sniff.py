"""Upload routing: snapshot vs daily inventory history matrix."""

from io import BytesIO

from backend.services.upload_file_sniff import (
    check_file_for_daily_inventory_history,
    check_files_for_snapshot_upload,
    sniff_upload_document,
)


def test_sniff_wide_history_matrix_csv():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
"""
    sniff = sniff_upload_document(csv.encode(), "Daily Inventory History 1-May To 25-Jun-26.csv")
    assert sniff["kind"] == "daily_inventory_history_matrix"
    assert sniff["date_columns"] >= 3


def test_snapshot_upload_rejects_history_matrix():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
"""
    parts = [("inventory-matrix.csv", csv.encode())]
    msg = check_files_for_snapshot_upload(parts)
    assert msg is not None
    assert "Daily Inventory History" in msg


def test_history_upload_rejects_oms_snapshot_headers():
    csv = "Item SkuCode,Buffer Stock,Total Inv.,Marketplace\nSKU-A,1,2,3\n"
    msg = check_file_for_daily_inventory_history(csv.encode(), "OMS-24-06-2026.csv")
    assert msg is not None
    assert "Snapshot inventory" in msg


def test_history_upload_accepts_matrix():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
"""
    msg = check_file_for_daily_inventory_history(csv.encode(), "inventory-matrix.csv")
    assert msg is None

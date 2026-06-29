"""Upload routing: detect misplaced files across upload sections."""

from io import BytesIO

from backend.services.upload_file_sniff import (
    check_file_for_daily_inventory_history,
    check_files_for_snapshot_upload,
    check_files_for_upload_target,
    check_upload_target,
    classify_upload_document,
    partition_files_by_upload_target,
)


def test_sniff_wide_history_matrix_csv():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
"""
    sniff = classify_upload_document(csv.encode(), "Daily Inventory History 1-May To 25-Jun-26.csv")
    assert sniff["category"] == "daily_inventory_history_matrix"
    assert sniff["date_columns"] >= 3


def test_snapshot_upload_rejects_history_matrix():
    csv = """SKU,28-5-26,29-5-26,1-6-26,5-6-26,25-6-26,,Days
SKU-A,10,10,0,5,8,,5
"""
    parts = [("inventory-matrix.csv", csv.encode())]
    msg = check_files_for_snapshot_upload(parts)
    assert msg is not None
    assert "Daily inventory history matrix" in msg


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


def test_daily_sales_rejects_return_filename():
    csv = "SKU,Return_Qty\nSTYLE-1-XL,5\n"
    msg = check_upload_target("daily_sales", csv.encode(), "LAST 30 DAYS RETURN.xlsx")
    assert msg is not None
    assert "Returns" in msg


def test_returns_rejects_shipment_sales_export():
    csv = (
        "amazon-order-id,purchase-date,merchant-sku,quantity,transaction type\n"
        "123,2026-06-01,STYLE-1-XL,2,Shipment\n"
    )
    msg = check_upload_target("returns", csv.encode(), "amazon-shipment-jun.csv")
    assert msg is not None
    assert "Daily order upload" in msg or "daily sales" in msg.lower()


def test_existing_po_rejects_sku_status_sheet():
    csv = "SKU,Lead time (days),Status\nSTYLE-1-XL,45,Active\n"
    msg = check_upload_target("existing_po", csv.encode(), "sku_status_lead.xlsx")
    assert msg is not None
    assert "SKU status" in msg


def test_sku_status_rejects_existing_po_headers():
    csv = "OMS SKU,PO_Pipeline_Total,Pending Cutting,Balance to Dispatch\nSTYLE-1-XL,10,2,3\n"
    msg = check_upload_target("sku_status_lead", csv.encode(), "existing_po_pipeline.xlsx")
    assert msg is not None
    assert "Existing PO" in msg


def test_sales_rar_classified_as_daily_sales():
    cls = classify_upload_document(b"Rar!\x1a\x07\x00", "Sales 27-28-6-26.rar")
    assert cls["category"] == "daily_sales"
    assert cls["confidence"] == "high"
    assert check_upload_target("daily_sales", b"Rar!\x1a\x07\x00", "Sales 27-28-6-26.rar") is None


def test_partition_routes_inventory_from_daily_sales_batch():
    sales_rar = ("Sales 27-28-6-26.rar", b"Rar!\x1a\x07\x00")
    oms_csv = ("OMS-inventory.csv", b"Item SkuCode,Buffer Stock\nSKU-A,1\n")
    buckets, notes = partition_files_by_upload_target([sales_rar, oms_csv], "daily_sales")
    assert sales_rar in buckets.get("daily_sales", [])
    assert oms_csv in buckets.get("snapshot_inventory", [])
    assert notes


def test_daily_sales_batch_rejects_return_file():
    ret = b"SKU,units refunded\nSTYLE-1,2\n"
    msg = check_files_for_upload_target("daily_sales", [("myntra_seller_returns.csv", ret)])
    assert msg is not None

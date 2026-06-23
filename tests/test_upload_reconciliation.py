"""Tests for daily vs monthly upload reconciliation."""
import pandas as pd

from backend.services.upload_reconciliation import (
    _classify_upload_kind,
    build_upload_reconciliation_report,
)


def test_classify_daily_vs_monthly():
    assert _classify_upload_kind("Orders_2026-06-01_meesho.csv", "2026-06-01", "2026-06-01") == "daily"
    assert _classify_upload_kind("Akiko Amazon Jan To May 2026 1.rar", "2026-01-01", "2026-05-31") == "monthly"


def test_reconciliation_flags_monthly_daily_gap(monkeypatch, tmp_path):
    from backend.session import AppSession

    monkeypatch.setenv("DAILY_SALES_DB", str(tmp_path / "daily_sales.db"))
    from backend.services import daily_store

    daily_store._DB_PATH = None

    daily_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [10, 5],
            "Invoice_Amount": [100.0, 50.0],
            "OrderId": ["o1", "o2"],
            "OMS_SKU": ["S1", "S2"],
            "DSR_Segment": ["YG", "YG"],
        }
    )
    monthly_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "Transaction_Type": ["Shipment", "Shipment", "Shipment"],
            "Quantity": [10, 5, 3],
            "Invoice_Amount": [100.0, 50.0, 30.0],
            "Order_Id": ["o1", "o2", "o3"],
            "SKU": ["S1", "S2", "S3"],
            "DSR_Segment": ["YG", "YG", "YG"],
        }
    )
    daily_store.save_daily_file("amazon", "Orders_2026-06-01.csv", daily_df)
    daily_store.save_daily_file("amazon", "YG Amazon Jan To Jun 2026.rar", monthly_df)

    report = build_upload_reconciliation_report(AppSession())
    assert report["file_count"] == 2
    assert report["mismatch_count"] >= 1
    hit = [m for m in report["mismatches"] if m["month"] == "2026-06" and m["segment"] == "YG"]
    assert hit
    assert hit[0]["unit_diff"] == 3

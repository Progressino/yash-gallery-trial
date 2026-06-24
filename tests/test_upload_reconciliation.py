"""Tests for daily vs monthly upload reconciliation."""
import time

import pandas as pd

from backend.services.upload_reconciliation import (
    _classify_upload_kind,
    _uploads_needing_parquet,
    build_upload_reconciliation_report,
    invalidate_upload_reconciliation_cache,
)


def test_classify_daily_vs_monthly():
    assert _classify_upload_kind("Orders_2026-06-01_meesho.csv", "2026-06-01", "2026-06-01") == "daily"
    assert _classify_upload_kind("Akiko Amazon Jan To May 2026 1.rar", "2026-01-01", "2026-05-31") == "monthly"


def test_reconciliation_flags_monthly_daily_gap(monkeypatch, tmp_path):
    from backend.session import AppSession
    from backend.services import daily_store

    db = tmp_path / "daily_sales.db"
    monkeypatch.setattr(daily_store, "_DB_PATH", db)
    invalidate_upload_reconciliation_cache()

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
    assert report["parquet_files_loaded"] == 2
    assert report["mismatch_count"] >= 1
    hit = [m for m in report["mismatches"] if m["month"] == "2026-06" and m["segment"] == "YG"]
    assert hit
    assert hit[0]["unit_diff"] == 3


def test_reconciliation_skips_unrelated_daily_parquets(monkeypatch, tmp_path):
    """Many daily files on other months/platforms should not be loaded."""
    from backend.session import AppSession
    from backend.services import daily_store

    db = tmp_path / "daily_sales.db"
    monkeypatch.setattr(daily_store, "_DB_PATH", db)
    invalidate_upload_reconciliation_cache()

    row = {
        "Date": pd.to_datetime(["2026-06-01"]),
        "TxnType": ["Shipment"],
        "Quantity": [1],
        "Invoice_Amount": [10.0],
        "OrderId": ["o1"],
        "OMS_SKU": ["S1"],
        "DSR_Segment": ["YG"],
    }
    for day in range(1, 31):
        daily_store.save_daily_file(
            "amazon",
            f"Orders_2026-04-{day:02d}.csv",
            pd.DataFrame({**row, "Date": pd.to_datetime([f"2026-04-{day:02d}"])}),
        )
    daily_store.save_daily_file(
        "amazon",
        "Orders_2026-06-01.csv",
        pd.DataFrame(row),
    )
    daily_store.save_daily_file(
        "amazon",
        "YG Amazon Jan To Jun 2026.rar",
        pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
                "Transaction_Type": ["Shipment", "Shipment"],
                "Quantity": [1, 2],
                "Invoice_Amount": [10.0, 20.0],
                "Order_Id": ["o1", "o2"],
                "SKU": ["S1", "S2"],
                "DSR_Segment": ["YG", "YG"],
            }
        ),
    )

    t0 = time.perf_counter()
    report = build_upload_reconciliation_report(AppSession())
    elapsed = time.perf_counter() - t0

    assert report["file_count"] == 32
    assert report["parquet_files_loaded"] == 2
    assert report["parquet_files_skipped"] == 30
    assert elapsed < 5.0
    assert report["elapsed_ms"] < 5000


def test_uploads_needing_parquet_only_overlapping_daily():
    enriched = [
        {
            "platform": "amazon",
            "filename": "mtr.rar",
            "upload_kind": "monthly",
            "months": {"2026-06"},
        },
        {
            "platform": "amazon",
            "filename": "Orders_2026-06-01.csv",
            "upload_kind": "daily",
            "months": {"2026-06"},
        },
        {
            "platform": "amazon",
            "filename": "Orders_2026-04-01.csv",
            "upload_kind": "daily",
            "months": {"2026-04"},
        },
        {
            "platform": "flipkart",
            "filename": "Orders_2026-06-01.csv",
            "upload_kind": "daily",
            "months": {"2026-06"},
        },
    ]
    need = _uploads_needing_parquet(enriched, None, None)
    names = {e["filename"] for e in need}
    assert names == {"mtr.rar", "Orders_2026-06-01.csv"}

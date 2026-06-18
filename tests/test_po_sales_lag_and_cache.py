"""Sales lag + shared cache staleness tests."""
from __future__ import annotations

from backend.services.tier3_session_merge import (
    sales_data_gap_needs_warning,
    sales_data_lag_days,
)


def test_sales_data_lag_one_day_is_expected():
    assert sales_data_lag_days("2026-06-18", "2026-06-17") == 1
    assert sales_data_gap_needs_warning("2026-06-18", "2026-06-17") is False


def test_sales_data_lag_multi_day_warns():
    assert sales_data_lag_days("2026-06-18", "2026-06-15") == 3
    assert sales_data_gap_needs_warning("2026-06-18", "2026-06-15") is True


def test_effective_sales_through_prefers_tier3_over_stale_session(monkeypatch):
    import pandas as pd

    from backend.session import AppSession
    from backend.services.tier3_session_merge import (
        effective_sales_through,
        sales_data_gap_needs_warning,
    )

    sess = AppSession()
    sess.sales_df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2026-06-14"]),
            "Sku": ["A"],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Source": ["Flipkart"],
        }
    )
    monkeypatch.setattr(
        "backend.services.tier3_session_merge.tier3_sales_through",
        lambda: "2026-06-17",
    )
    assert effective_sales_through(sess) == "2026-06-17"
    assert sales_data_gap_needs_warning("2026-06-18", effective_sales_through(sess)) is False


def test_tier3_sales_through_max_from_summary(monkeypatch):
    from backend.services.tier3_session_merge import tier3_sales_through

    monkeypatch.setattr(
        "backend.services.daily_store.get_summary",
        lambda: {
            "amazon": {"file_count": 10, "max_date": "2026-06-16"},
            "meesho": {"file_count": 5, "max_date": "2026-06-17"},
        },
    )
    assert tier3_sales_through() == "2026-06-17"


def test_shared_cache_rejected_when_disk_existing_po_newer(tmp_path, monkeypatch):
    from backend.services import po_shared_cache as psc
    from backend.session import AppSession

    shared = tmp_path / "shared_po"
    shared.mkdir()
    monkeypatch.setattr(psc, "_shared_dir", lambda: shared)
    monkeypatch.setattr(
        "backend.services.existing_po.read_existing_po_disk_meta",
        lambda: None,
    )

    meta = {
        "fingerprint": {"planning_date": "2026-06-18"},
        "existing_po_generation": 1,
        "existing_po_filename": "old.xlsx",
    }
    assert psc._shared_cache_stale_vs_disk(meta) is False

    monkeypatch.setattr(
        "backend.services.existing_po.read_existing_po_disk_meta",
        lambda: {"existing_po_generation": 6, "existing_po_filename": "Po 17-Jun-26.xlsx"},
    )
    assert psc._shared_cache_stale_vs_disk(meta) is True

    sess = AppSession()
    body = {"planning_date": "2026-06-18", "period_days": 30}
    key, fp = psc.build_cache_key(sess, body)
    meta_path = psc._meta_path(key)
    meta_path.write_text(
        __import__("json").dumps(
            {
                **meta,
                "fingerprint": fp,
                "planning_date": "2026-06-18",
                "created_at_unix": __import__("time").time(),
            }
        ),
        encoding="utf-8",
    )
    psc._parquet_path(key).write_bytes(b"")
    assert psc.lookup_shared_cache(sess, body) is None

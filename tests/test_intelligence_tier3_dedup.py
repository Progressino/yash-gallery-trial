"""Tier-3 intelligence metrics must dedupe overlapping daily uploads (no 2× gross)."""

from __future__ import annotations

import pandas as pd

from backend.routers import data as data_router
from backend.session import AppSession


def test_tier3_direct_metrics_dedup_overlapping_upload_rows(monkeypatch):
    """Same shipment in two overlapping Tier-3 blobs counts once on the dashboard."""
    from backend.services import daily_store

    # Two blobs that repeat the same Amazon line (typical re-upload / overlap).
    dup_rows = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-20"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [100],
            "Order_Id": ["O1"],
            "Invoice_Number": ["INV1"],
        }
    )
    combined = pd.concat([dup_rows, dup_rows], ignore_index=True)

    def _load(plat, start, end, dedup=True, **kwargs):
        if plat != "amazon":
            return pd.DataFrame()
        return daily_store._dedup_platform_df(combined, plat) if dedup else combined.copy()

    monkeypatch.setattr(daily_store, "platforms_with_uploads_in_range", lambda s, e: ["amazon"])
    monkeypatch.setattr(daily_store, "load_platform_data_for_report_range", _load)

    sess = AppSession()
    out = data_router._intelligence_payload_from_tier3_direct(
        sess, "2026-06-20", "2026-06-20", 10, "gross"
    )
    assert out is not None
    sales_summary, platform_summary, *_rest = out
    amazon = next(p for p in platform_summary if p["platform"] == "Amazon")
    assert amazon["total_units"] == 100
    assert sales_summary["total_units"] == 100
    daily = {r["date"]: r["shipments"] for r in amazon.get("daily") or []}
    assert daily.get("2026-06-20") == 100

"""Tier-3 window checks — ensures uploaded dailies are queryable for dashboard/PO."""
from __future__ import annotations

import pytest

from backend.services.daily_store import _resolve_db_path, get_summary, platforms_with_uploads_in_range
from backend.services.sales import _compute_platform_metrics
from backend.services.daily_store import load_platform_data_for_report_range


def test_tier3_db_has_recent_upload_window():
    """On VPS/local with daily uploads, Jun 17–18 window must be discoverable."""
    if not _resolve_db_path().is_file():
        pytest.skip("no daily_sales.db")
    summary = get_summary() or {}
    total_files = sum(int((summary.get(p) or {}).get("file_count") or 0) for p in summary)
    if total_files < 10:
        pytest.skip("tier3 store nearly empty")
    plats = platforms_with_uploads_in_range("2026-06-17", "2026-06-18")
    assert plats, f"expected uploads for 2026-06-17..18, summary max={ {k: v.get('max_date') for k,v in summary.items()} }"


def test_tier3_june_17_has_shipment_units_when_db_present():
    if not _resolve_db_path().is_file():
        pytest.skip("no daily_sales.db")
    plats = platforms_with_uploads_in_range("2026-06-17", "2026-06-17")
    if not plats:
        pytest.skip("no tier3 overlap for 2026-06-17")
    specs = {
        "amazon": ("Amazon", "Date", "SKU", "Transaction_Type"),
        "myntra": ("Myntra", "Date", "OMS_SKU", "TxnType"),
        "meesho": ("Meesho", "Date", "OMS_SKU", "TxnType"),
    }
    total = 0
    for plat in plats:
        if plat not in specs:
            continue
        name, dc, sku, txn = specs[plat]
        df = load_platform_data_for_report_range(plat, "2026-06-17", "2026-06-17", dedup=False)
        m = _compute_platform_metrics(df, name, sku, txn, start_date="2026-06-17", end_date="2026-06-17")
        total += int(m.get("total_units") or 0)
    assert total > 0, f"platforms {plats} returned 0 units for 2026-06-17"

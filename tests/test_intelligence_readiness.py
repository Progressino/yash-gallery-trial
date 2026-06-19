"""Intelligence dashboard readiness — stronger than operational 8/8 alone."""
from __future__ import annotations

import pandas as pd

from backend.models.schemas import CoverageResponse
from backend.services.intelligence_readiness import (
    build_intelligence_readiness,
    dashboard_gate_ready,
    intelligence_ready,
    platform_frames_available,
)
from backend.services.po_readiness import PO_MIN_INVENTORY_ROWS, PO_MIN_SALES_ROWS, augment_coverage


def _cov(**kwargs) -> CoverageResponse:
    base = {
        "sku_mapping": True,
        "mtr": True,
        "sales": True,
        "myntra": True,
        "meesho": True,
        "flipkart": True,
        "snapdeal": True,
        "inventory": True,
        "daily_orders": False,
        "existing_po": False,
        "mtr_rows": 1000,
        "sales_rows": PO_MIN_SALES_ROWS + 1,
        "myntra_rows": 1000,
        "meesho_rows": 1000,
        "flipkart_rows": 1000,
        "snapdeal_rows": 1000,
        "inventory_rows": PO_MIN_INVENTORY_ROWS + 1,
    }
    base.update(kwargs)
    return CoverageResponse(**base)


class _Sess:
    _session_id = "test-sid"
    sales_rebuild_status = "running"
    session_restore_status = "idle"
    inventory_upload_status = "idle"
    daily_auto_ingest_status = "idle"
    daily_inventory_upload_status = "idle"
    tier1_bulk_status = "idle"
    returns_import_status = "idle"
    mtr_df = None
    myntra_df = None
    meesho_df = None
    flipkart_df = None
    snapdeal_df = None
    sales_df = None
    inventory_df_variant = None


def test_dashboard_not_ready_when_platform_rows_zero(monkeypatch):
    sess = _Sess()
    cov = _cov(
        mtr=False, myntra=False, meesho=False, flipkart=False, snapdeal=False,
        mtr_rows=0, myntra_rows=0, meesho_rows=0, flipkart_rows=0, snapdeal_rows=0,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness._pg_platform_sales_counts",
        lambda: {},
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness._tier3_all_platforms_have_uploads",
        lambda: False,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_complete",
        lambda _s, _sid="": True,
    )
    assert platform_frames_available(sess, cov) is False
    assert dashboard_gate_ready(sess, cov) is False


def test_dashboard_ready_with_pg_platform_counts(monkeypatch):
    sess = _Sess()
    cov = _cov(mtr_rows=0, myntra_rows=0, meesho_rows=0, flipkart_rows=0, snapdeal_rows=0)
    monkeypatch.setattr(
        "backend.services.intelligence_readiness._pg_platform_sales_counts",
        lambda: {
            "amazon": 100,
            "myntra": 100,
            "meesho": 100,
            "flipkart": 100,
            "snapdeal": 100,
            "unified": 1_568_085,
        },
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_complete",
        lambda _s, _sid="": True,
    )
    assert platform_frames_available(sess, cov) is True
    assert dashboard_gate_ready(sess, cov) is True


def test_intelligence_ready_while_sales_rebuild_running(monkeypatch):
    from backend.session import AppSession

    sess = AppSession()
    sess.sales_rebuild_status = "running"
    days = pd.date_range("2025-12-01", periods=5, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["X"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [1] * len(days),
            "Units_Effective": [1] * len(days),
            "Source": ["Amazon"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["X"], "Total_Inventory": [1]})
    for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
        setattr(sess, attr, pd.DataFrame({"x": [1]}))
    cov = augment_coverage(sess, _cov())
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_complete",
        lambda _s, _sid="": True,
    )
    assert intelligence_ready(sess, cov) is True
    payload = build_intelligence_readiness(sess, cov)
    assert payload["dashboard_ready"] is True
    assert "sales_rebuild" in payload["background_jobs"]


def test_intelligence_readiness_endpoint(client, session_for_client, monkeypatch):
    _, sess = session_for_client
    days = pd.date_range("2025-12-01", periods=5, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["X"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [1] * len(days),
            "Units_Effective": [1] * len(days),
            "Source": ["Amazon"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["X"], "Total_Inventory": [1]})
    for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
        setattr(sess, attr, pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(
        "backend.services.po_readiness.PO_MIN_SALES_ROWS",
        1,
    )
    monkeypatch.setattr(
        "backend.services.po_readiness.PO_MIN_INVENTORY_ROWS",
        1,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_complete",
        lambda _s, _sid="": True,
    )

    r = client.get("/api/data/intelligence/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "dashboard_ready" in body
    assert "platforms_loaded" in body
    assert "hydration_complete" in body
    assert body["dashboard_ready"] is True


def test_dashboard_summary_endpoint(client):
    r = client.get(
        "/api/data/dashboard/summary",
        params={"start_date": "2025-01-01", "end_date": "2025-12-31", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert "platforms" in body
    assert "top_skus" in body
    assert "source" in body

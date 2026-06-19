"""PO readiness — decoupled from non-critical background jobs."""
from __future__ import annotations

import pandas as pd

from backend.models.schemas import CoverageResponse
from backend.services.po_readiness import (
    PO_MIN_INVENTORY_ROWS,
    PO_MIN_SALES_ROWS,
    augment_coverage,
    compute_data_ready,
    compute_po_ready,
    critical_restore_running,
)


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
        "sales_rows": PO_MIN_SALES_ROWS + 1,
        "inventory_rows": PO_MIN_INVENTORY_ROWS + 1,
    }
    base.update(kwargs)
    return CoverageResponse(**base)


class _Sess:
    sales_rebuild_status = "running"
    session_restore_status = "idle"
    inventory_upload_status = "idle"
    daily_auto_ingest_status = "idle"
    daily_inventory_upload_status = "idle"
    tier1_bulk_status = "idle"
    returns_import_status = "idle"


def test_po_ready_while_sales_rebuild_running():
    cov = _cov()
    sess = _Sess()
    assert compute_data_ready(cov) is True
    assert compute_po_ready(sess, cov) is True


def test_po_not_ready_during_session_restore():
    cov = _cov()
    sess = _Sess()
    sess.session_restore_status = "running"
    assert critical_restore_running(sess) is True
    assert compute_po_ready(sess, cov) is False


def test_augment_coverage_light_dashboard_ready_partial_rows(monkeypatch):
    """Light coverage marks dashboard ready when flags + Tier-3 exist (no 1M sales rows)."""
    from backend.services.po_readiness import augment_coverage

    sess = _Sess()
    cov = _cov(
        sales_rows=87_981,
        inventory_rows=6_000,
        mtr_rows=1000,
        myntra_rows=1000,
        meesho_rows=1000,
        flipkart_rows=1000,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness._tier3_all_platforms_have_uploads",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_inflight",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "backend.services.intelligence_readiness.hydration_complete",
        lambda _s, _sid="": True,
    )
    out = augment_coverage(sess, cov, light=True)
    assert out.dashboard_ready is True
    assert out.intelligence_ready is True
    assert out.po_ready is False


def test_augment_coverage_adds_po_ready():
    from backend.session import AppSession

    sess = AppSession()
    sess.sales_rebuild_status = "running"
    cov = augment_coverage(sess, _cov())
    assert cov.data_ready is True
    assert cov.po_ready is True
    assert cov.background_tasks.get("sales_rebuild") is True
    assert hasattr(cov, "platforms_loaded")
    assert hasattr(cov, "dashboard_ready")


def test_po_readiness_endpoint(client, session_for_client, monkeypatch):
    import pandas as pd

    _, sess = session_for_client
    days = pd.date_range("2025-12-01", periods=5, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["X"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [1] * len(days),
            "Units_Effective": [1] * len(days),
            "Source": ["Meesho"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["X"], "Total_Inventory": [1]})
    sess.sales_rebuild_status = "running"

    monkeypatch.setattr(
        "backend.services.po_readiness.PO_MIN_SALES_ROWS",
        1,
    )
    monkeypatch.setattr(
        "backend.services.po_readiness.PO_MIN_INVENTORY_ROWS",
        1,
    )

    r = client.get("/api/po/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "po_ready" in body
    assert "sales_rows" in body
    assert "background_jobs" in body
    assert "sales_rebuild" in body["background_jobs"]
    assert body["po_ready"] is True

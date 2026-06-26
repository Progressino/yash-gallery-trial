"""PO pipeline — snapshot gate, versioning, materialized tables."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.models.schemas import CoverageResponse
from backend.services.po_pipeline import (
    PO_PIPELINE_VERSION,
    PoInputSnapshot,
    check_calculate_gate,
    collect_dataset_versions,
    prepare_po_snapshot,
)
from backend.services.po_readiness import PO_MIN_INVENTORY_ROWS, PO_MIN_SALES_ROWS, build_po_readiness
from backend.session import AppSession


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
        "sales_rows": PO_MIN_SALES_ROWS + 100,
        "inventory_rows": PO_MIN_INVENTORY_ROWS + 10,
    }
    base.update(kwargs)
    return CoverageResponse(**base)


def _seed_sess(sess: AppSession) -> None:
    days = pd.date_range("2025-12-01", periods=30, freq="D")
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["PIPE-SKU"] * len(days),
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * len(days),
            "Quantity": [2] * len(days),
            "Units_Effective": [2] * len(days),
            "Source": ["Meesho"] * len(days),
        }
    )
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["PIPE-SKU"], "Total_Inventory": [50], "OMS_Inventory": [50]}
    )
    sess.inventory_snapshot_date = "2025-12-30"


def test_collect_dataset_versions_has_pipeline_fields():
    sess = AppSession()
    _seed_sess(sess)
    v = collect_dataset_versions(sess, {"planning_date": "2025-12-30"})
    d = v.as_dict()
    assert d["pipeline_version"] == PO_PIPELINE_VERSION
    assert "sales" in d and "inventory" in d
    assert len(v.composite_hash()) == 24


def test_gate_blocks_during_inventory_upload():
    sess = AppSession()
    _seed_sess(sess)
    sess.inventory_upload_status = "running"
    gate = check_calculate_gate(sess)
    assert gate["calculate_allowed"] is False
    assert any("inventory upload" in b.lower() for b in gate["blockers"])


def test_gate_allows_ready_session():
    sess = AppSession()
    _seed_sess(sess)
    # inflate row counts for gate floors
    sess.sales_df = pd.concat([sess.sales_df] * 40, ignore_index=True)
    sess.inventory_df_variant = pd.concat(
        [
            sess.inventory_df_variant,
            pd.DataFrame(
                {
                    "OMS_SKU": [f"SKU-{i}" for i in range(PO_MIN_INVENTORY_ROWS)],
                    "Total_Inventory": [1] * PO_MIN_INVENTORY_ROWS,
                }
            ),
        ],
        ignore_index=True,
    )
    gate = check_calculate_gate(sess)
    assert gate["calculate_allowed"] is True


def test_prepare_po_snapshot_returns_locked_bundle(monkeypatch):
    sess = AppSession()
    _seed_sess(sess)
    sess.sales_df = pd.concat([sess.sales_df] * 40, ignore_index=True)
    sess.inventory_df_variant = pd.concat(
        [
            sess.inventory_df_variant,
            pd.DataFrame(
                {
                    "OMS_SKU": [f"SKU-{i}" for i in range(PO_MIN_INVENTORY_ROWS)],
                    "Total_Inventory": [1] * PO_MIN_INVENTORY_ROWS,
                }
            ),
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(
        "backend.services.po_pipeline._resolve_ads_sales_source",
        lambda *a, **k: (sess.sales_df, "test"),
    )
    monkeypatch.setattr(
        "backend.services.po_pipeline._prepare_inventory_history",
        lambda *a, **k: (sess.inventory_df_variant, sess.inventory_df_variant, None),
    )
    monkeypatch.setattr(
        "backend.services.po_pipeline.materialize_intermediate_tables",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.po_session_hydrate.hydrate_po_session_for_calculate",
        lambda _s: None,
    )

    out = prepare_po_snapshot(
        sess,
        {"period_days": 30, "planning_date": "2025-12-30"},
        skip_hydrate=True,
        enforce_gate=True,
    )
    assert isinstance(out, PoInputSnapshot)
    assert out.ready is True
    assert out.snapshot_id.startswith("PO_")
    assert not out.sales_df.empty
    assert not out.inv_df.empty


def test_readiness_includes_pipeline_fields():
    sess = AppSession()
    _seed_sess(sess)
    sess.sales_df = pd.concat([sess.sales_df] * 40, ignore_index=True)
    sess.inventory_df_variant = pd.concat(
        [
            sess.inventory_df_variant,
            pd.DataFrame(
                {
                    "OMS_SKU": [f"SKU-{i}" for i in range(PO_MIN_INVENTORY_ROWS)],
                    "Total_Inventory": [1] * PO_MIN_INVENTORY_ROWS,
                }
            ),
        ],
        ignore_index=True,
    )
    body = build_po_readiness(sess, _cov())
    assert "calculate_allowed" in body
    assert "pipeline_version" in body
    assert body["pipeline_version"] == PO_PIPELINE_VERSION

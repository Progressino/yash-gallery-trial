"""Fail-closed reconciliation against real Inventory/Sales RAR uploads when present."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.scripts.source_upload_reconciliation import run_reconciliation

_INV = Path("/Users/samraisinghani/Downloads/Inventory 6-Aug-26.rar")
_SALES = Path("/Users/samraisinghani/Downloads/Sales 5-Aug-26.rar")


@pytest.mark.skipif(not _INV.is_file(), reason="Inventory 6-Aug-26.rar not on this machine")
def test_inventory_and_sales_rar_source_reconciliation():
    report = run_reconciliation(
        _INV,
        _SALES if _SALES.is_file() else None,
        oms_tolerance=50.0,
    )
    assert report["ok"], report.get("failures")


def test_powder_map_terminal_does_not_split_power_inventory():
    """Regression: master map POWDER terminals must fold onto POWER warehouse codes."""
    import pandas as pd

    from backend.services.po_engine import calculate_po_base, canonical_oms_key
    from backend.services.sku_mapping import resolve_sku_replacement_map

    mp = resolve_sku_replacement_map(
        {
            "5038YKPOWDERBLUE-3XL": "5038YKPOWDERBLUE-3XL",
            "5038PLYKPOWDERBLUE-3XL": "5038YKPOWDERBLUE-3XL",
        }
    )
    assert mp["5038YKPOWDERBLUE-3XL"] == "5038YKPOWERBLUE-3XL"
    assert canonical_oms_key("5038PLYKPOWDERBLUE-3XL", mp) == "5038YKPOWERBLUE-3XL"

    inv = pd.DataFrame(
        {
            "OMS_SKU": ["5038YKPOWERBLUE-3XL"],
            "OMS_Inventory": [47.0],
            "Amazon_Inventory": [0.0],
            "Total_Inventory": [47.0],
            "Marketplace_Total": [0.0],
            "Buffer_Stock": [0.0],
            "Manual_InTransit": [0.0],
            "Not_In_Inventory_Qty": [0.0],
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": ["5038YKPOWDERBLUE-3XL"] * 6,
            "TxnDate": pd.date_range("2026-07-01", periods=6, freq="D"),
            "Transaction Type": ["Shipment"] * 6,
            "Quantity": [1] * 6,
            "Units_Effective": [1] * 6,
        }
    )
    po = calculate_po_base(
        sales, inv, period_days=30, lead_time=45, target_days=45, sku_mapping=mp
    )
    rows = po[po["OMS_SKU"].astype(str).str.contains("5038YK")]
    assert list(rows["OMS_SKU"]) == ["5038YKPOWERBLUE-3XL"]
    assert float(rows.iloc[0]["Total_Inventory"]) == 47.0
    assert float(rows.iloc[0]["Sold_Units"]) == 6.0

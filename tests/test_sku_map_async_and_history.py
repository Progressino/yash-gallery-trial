"""Async SKU mapping upload + inventory history remapping fixes."""
from __future__ import annotations

import io

import pandas as pd
import pytest


def test_existing_po_merge_key_yeal_to_teal():
    from backend.services.existing_po import existing_po_merge_key

    assert existing_po_merge_key("7100YKYEAL-L-XL") == "7100YKTEAL-L-XL"
    assert existing_po_merge_key("1415YKBALCK-6XL") == "1415YKBLACK-6XL"


def test_recanonicalize_sums_twin_qty():
    from backend.services.daily_inventory_history import recanonicalize_inventory_history_skus

    df = pd.DataFrame(
        {
            "OMS_SKU": ["7100YKYEAL-M", "7100YKTEAL-M", "7100YKTEAL-M"],
            "Date": ["2026-07-30", "2026-07-30", "2026-07-30"],
            "Qty": [10.0, 20.0, 5.0],
            "Source": ["uploaded", "uploaded", "uploaded"],
            "Channel": ["oms", "oms", "oms"],
        }
    )
    # Empty mapping still coalesces YEAL→TEAL via alias path now
    out = recanonicalize_inventory_history_skus(df, {})
    teal = out[out["OMS_SKU"].astype(str) == "7100YKTEAL-M"]
    assert len(teal) == 1
    assert float(teal["Qty"].iloc[0]) == 35.0
    assert not out["OMS_SKU"].astype(str).str.contains("YEAL", case=False).any()


def test_recanonicalize_with_seller_map_sums(tmp_path, monkeypatch):
    from backend.services.daily_inventory_history import recanonicalize_inventory_history_skus

    df = pd.DataFrame(
        {
            "OMS_SKU": ["SELLER-A", "CANON-A"],
            "Date": ["2026-07-30", "2026-07-30"],
            "Qty": [4.0, 6.0],
            "Source": ["uploaded", "uploaded"],
        }
    )
    out = recanonicalize_inventory_history_skus(df, {"SELLER-A": "CANON-A"})
    assert set(out["OMS_SKU"]) == {"CANON-A"}
    assert float(out["Qty"].iloc[0]) == 10.0

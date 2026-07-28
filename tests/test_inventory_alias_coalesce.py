"""BOTTEL/1180 inventory twins must sum onto the canonical OMS SKU."""

import pandas as pd

from backend.services.daily_inventory_history import (
    _is_parent_style_inventory_sku,
    scrub_absurd_inventory_history_rows,
    _parse_one_sheet,
)
from backend.services.inventory import coalesce_inventory_by_sku_mapping


def test_parent_style_yr_sku_detected():
    assert _is_parent_style_inventory_sku("YR021")
    assert _is_parent_style_inventory_sku("yr027")
    assert not _is_parent_style_inventory_sku("YR021-XL")
    assert not _is_parent_style_inventory_sku("DPT21/MULTI")


def test_scrub_drops_yr_day_total_leak():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["YR021", "DPT21MULTI", "YR022"],
            "Date": ["2026-07-16"] * 3,
            "Qty": [171653.0, 1500.0, 171653.0],
            "Source": ["uploaded"] * 3,
        }
    )
    out = scrub_absurd_inventory_history_rows(df)
    assert list(out["OMS_SKU"]) == ["DPT21MULTI"]
    assert float(out["Qty"].iloc[0]) == 1500.0


def test_parse_rejects_yr_parent_and_absurd_qty():
    df = pd.DataFrame(
        [
            ["Item SkuCode", "Item", "2026-07-15", "2026-07-16"],
            ["YR021", "YR021", 0, 171653],
            ["DPT21MULTI", "DPT", 1400, 1410],
        ]
    )
    tall = _parse_one_sheet(df, {})
    assert "YR021" not in set(tall["OMS_SKU"])
    assert set(tall["OMS_SKU"]) == {"DPT21MULTI"}


def test_coalesce_bottel_and_1180_inventory():
    inv = pd.DataFrame(
        {
            "OMS_SKU": [
                "5041YKBOTTELGREEN-XL",
                "5041YKBOTTLEGREEN-XL",
                "289YK345YELLOW-M",
                "1180YKYELLOW-M",
            ],
            "OMS_Inventory": [7.0, 0.0, 12.0, 0.0],
            "Amazon_Inventory": [0.0, 51.0, 14.0, 25.0],
            "Manual_InTransit": [0.0, 3.0, 0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0, 0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert float(by.loc["5041YKBOTTLEGREEN-XL", "Total_Inventory"]) == 61.0
    assert float(by.loc["289YK345YELLOW-M", "Total_Inventory"]) == 51.0
    assert "5041YKBOTTELGREEN-XL" not in by.index
    assert "1180YKYELLOW-M" not in by.index

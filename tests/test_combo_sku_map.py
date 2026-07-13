"""Combo / DPT SKU BOM map — parse + PO demand explode."""

import pandas as pd
import pytest

from backend.services.combo_sku_map import (
    explode_sku_qty_dataframe,
    merge_combo_sku_map,
    parse_combo_sku_map,
    resolve_demand_components,
    sheet_looks_like_combo_bom,
)
from backend.services.po_engine import calculate_po_base, calculate_quarterly_history
from backend.services.sku_mapping import parse_sku_mapping


def _combo_xlsx(tmp_path, rows):
    p = tmp_path / "combo.xlsx"
    pd.DataFrame(rows).to_excel(p, sheet_name="Sheet2", index=False)
    return p.read_bytes()


def test_parse_combo_sku_map_multi_component(tmp_path):
    raw = _combo_xlsx(
        tmp_path,
        [
            {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "DPT21MULTI"},
            {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "1003YKMUSTARD-3XL"},
            {"DPT Sku": "1003YK5027DPT21-L", "Sku": "1003YKMUSTARD-L"},
            {"DPT Sku": "1003YK5027DPT21-L", "Sku": "5027YKMULTI-L"},
            {"DPT Sku": "1003YK5027DPT21-L", "Sku": "DPT21MULTI"},
        ],
    )
    bom = parse_combo_sku_map(raw)
    assert set(c for c, _ in bom["1003DPT21MULTI-3XL"]) == {
        "1003YKMUSTARD-3XL",
        "DPT21MULTI",
    }
    assert len(bom["1003YK5027DPT21-L"]) == 3


def test_parse_sku_mapping_skips_combo_sheet(tmp_path):
    """Combo BOM must not collapse to last-wins 1:1 in the master map."""
    raw = _combo_xlsx(
        tmp_path,
        [
            {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "DPT21MULTI"},
            {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "1003YKMUSTARD-3XL"},
        ],
    )
    assert sheet_looks_like_combo_bom(
        pd.DataFrame(
            [
                {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "DPT21MULTI"},
                {"DPT Sku": "1003DPT21MULTI-3XL", "Sku": "1003YKMUSTARD-3XL"},
            ]
        )
    )
    m = parse_sku_mapping(raw)
    assert "1003DPT21MULTI-3XL" not in m


def test_resolve_demand_prefers_combo_over_one_to_one():
    bom = {
        "1003DPT21MULTI-3XL": [
            ("1003YKMUSTARD-3XL", 1.0),
            ("DPT21MULTI", 1.0),
        ]
    }
    # A bad 1:1 master would only keep one component — combo BOM wins.
    mapping = {"1003DPT21MULTI-3XL": "1003YKMUSTARD-3XL"}
    comps = resolve_demand_components("1003DPT21MULTI-3XL", mapping, bom)
    assert {c for c, _ in comps} == {"1003YKMUSTARD-3XL", "DPT21MULTI"}


def test_explode_scales_qty_per_component():
    bom = {
        "COMBO-L": [("A-L", 1.0), ("DPT-X", 2.0)],
    }
    df = pd.DataFrame({"SKU": ["COMBO-L", "PLAIN-L"], "Qty": [3, 4]})
    out = explode_sku_qty_dataframe(
        df, sku_col="SKU", qty_col="Qty", sku_mapping={}, combo_map=bom
    )
    assert len(out) == 3
    assert float(out.loc[out["SKU"] == "A-L", "Qty"].iloc[0]) == 3.0
    assert float(out.loc[out["SKU"] == "DPT-X", "Qty"].iloc[0]) == 6.0
    assert float(out.loc[out["SKU"] == "PLAIN-L", "Qty"].iloc[0]) == 4.0


def test_explode_retain_combo_listings_keeps_listing_and_components():
    bom = {
        "COMBO-L": [("A-L", 1.0), ("DPT-X", 2.0)],
    }
    df = pd.DataFrame({"SKU": ["COMBO-L"], "Qty": [3], "Units_Effective": [3]})
    out = explode_sku_qty_dataframe(
        df,
        sku_col="SKU",
        qty_col="Qty",
        sku_mapping={},
        combo_map=bom,
        retain_combo_listings=True,
    )
    assert set(out["SKU"]) == {"COMBO-L", "A-L", "DPT-X"}
    assert float(out.loc[out["SKU"] == "COMBO-L", "Qty"].iloc[0]) == 3.0
    assert float(out.loc[out["SKU"] == "A-L", "Qty"].iloc[0]) == 3.0
    assert float(out.loc[out["SKU"] == "DPT-X", "Qty"].iloc[0]) == 6.0
    assert float(out.loc[out["SKU"] == "COMBO-L", "Units_Effective"].iloc[0]) == 3.0
    assert float(out.loc[out["SKU"] == "DPT-X", "Units_Effective"].iloc[0]) == 6.0


def test_calculate_po_base_attributes_combo_demand_to_components():
    today = pd.Timestamp.today().normalize()
    sales = pd.DataFrame(
        {
            "Sku": ["1003DPT21MULTI-3XL"] * 5,
            "TxnDate": [today - pd.Timedelta(days=i) for i in range(5)],
            "Quantity": [1, 1, 1, 1, 1],
            "Units_Effective": [1, 1, 1, 1, 1],
            "Transaction Type": ["Shipment"] * 5,
            "Source": ["Meesho"] * 5,
        }
    )
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["1003YKMUSTARD-3XL", "DPT21MULTI", "OTHER-L"],
            "Total_Inventory": [10, 20, 5],
            "OMS_Inventory": [10, 20, 5],
        }
    )
    bom = {
        "1003DPT21MULTI-3XL": [
            ("1003YKMUSTARD-3XL", 1.0),
            ("DPT21MULTI", 1.0),
        ]
    }
    po = calculate_po_base(
        sales,
        inv,
        period_days=30,
        lead_time=14,
        target_days=30,
        combo_sku_map=bom,
    )
    assert not po.empty
    mustard = po.loc[po["OMS_SKU"] == "1003YKMUSTARD-3XL"].iloc[0]
    dpt = po.loc[po["OMS_SKU"] == "DPT21MULTI"].iloc[0]
    assert int(mustard["Sold_Units"]) == 5
    assert int(dpt["Sold_Units"]) == 5
    # Listing retained for visibility; PO_Qty stays on components only.
    assert "1003DPT21MULTI-3XL" in set(po["OMS_SKU"].astype(str))
    listing = po.loc[po["OMS_SKU"] == "1003DPT21MULTI-3XL"].iloc[0]
    assert int(listing["Sold_Units"]) == 5
    assert int(listing["PO_Qty"]) == 0
    assert "1003YKMUSTARD-3XL" in str(listing.get("Bundle_Size") or "")


def test_quarterly_history_keeps_combo_listing_without_component_fan():
    """Quarterly matches File: combo listing units stay on the listing SKU only."""
    today = pd.Timestamp.today().normalize()
    sales = pd.DataFrame(
        {
            "Sku": ["1003DPT21MULTI-L"],
            "TxnDate": [today - pd.Timedelta(days=3)],
            "Quantity": [2],
            "Transaction Type": ["Shipment"],
            "Source": ["Meesho"],
        }
    )
    bom = {
        "1003DPT21MULTI-L": [
            ("1003YKMUSTARD-L", 1.0),
            ("DPT21MULTI", 1.0),
        ]
    }
    pivot = calculate_quarterly_history(
        sales_df=sales, sku_mapping={}, n_quarters=4, combo_sku_map=bom
    )

    assert not pivot.empty
    skus = set(pivot["OMS_SKU"].astype(str))
    assert "1003DPT21MULTI-L" in skus
    assert "1003YKMUSTARD-L" not in skus
    assert "DPT21MULTI" not in skus
    listing = pivot.loc[pivot["OMS_SKU"] == "1003DPT21MULTI-L"].iloc[0]
    assert int(listing["Units_90d"]) == 2


def test_merge_combo_sku_map_overlay():
    a = {"X": [("A", 1.0)]}
    b = {"X": [("A", 1.0), ("B", 1.0)], "Y": [("C", 1.0)]}
    m = merge_combo_sku_map(a, b)
    assert len(m["X"]) == 2
    assert m["Y"] == [("C", 1.0)]

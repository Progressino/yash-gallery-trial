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
    # Ops / Replace-SKU terminal is BOTTEL (not warehouse BOTTLE typo).
    assert float(by.loc["5041YKBOTTELGREEN-XL", "Total_Inventory"]) == 61.0
    assert float(by.loc["289YK345YELLOW-M", "Total_Inventory"]) == 51.0
    assert "5041YKBOTTLEGREEN-XL" not in by.index
    assert "1180YKYELLOW-M" not in by.index


def test_coalesce_bottle_twins_follow_replace_map():
    """Explicit replace map BOTTLE→BOTTEL consolidates stock onto Right SKU."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["5041YKBOTTLEGREEN-3XL", "5041YKBOTTELGREEN-3XL"],
            "OMS_Inventory": [32.0, 0.0],
            "Amazon_Inventory": [0.0, 0.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    mp = {"5041YKBOTTLEGREEN-3XL": "5041YKBOTTELGREEN-3XL"}
    out = coalesce_inventory_by_sku_mapping(inv, mp)
    by = out.set_index("OMS_SKU")
    assert list(by.index) == ["5041YKBOTTELGREEN-3XL"]
    assert float(by.loc["5041YKBOTTELGREEN-3XL", "Total_Inventory"]) == 32.0


def test_coalesce_balck_typo_inventory():
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["1415YKBALCK-6XL", "1415YKBLACK-6XL"],
            "OMS_Inventory": [12.0, 3.0],
            "Amazon_Inventory": [0.0, 5.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert "1415YKBALCK-6XL" not in by.index
    assert float(by.loc["1415YKBLACK-6XL", "Total_Inventory"]) == 20.0


def test_balck_typo_in_canonical_oms_key():
    from backend.services.po_engine import canonical_oms_key, inventory_oms_key

    assert canonical_oms_key("1415YKBALCK-7XL", {}) == "1415YKBLACK-7XL"
    assert inventory_oms_key("1415YKBALCK-8XL") == "1415YKBLACK-8XL"


def test_coalesce_yeal_typo_inventory():
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["7100YKYEAL-L-XL", "7100YKTEAL-L-XL"],
            "OMS_Inventory": [120.0, 0.0],
            "Amazon_Inventory": [0.0, 15.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert "7100YKYEAL-L-XL" not in by.index
    assert float(by.loc["7100YKTEAL-L-XL", "Total_Inventory"]) == 135.0


def test_yeal_typo_in_canonical_oms_key():
    from backend.services.po_engine import canonical_oms_key, inventory_oms_key

    assert canonical_oms_key("7100YKYEAL-L-XL", {}) == "7100YKTEAL-L-XL"
    assert inventory_oms_key("7100YKYEAL-XXL-3XL") == "7100YKTEAL-XXL-3XL"


def test_history_alias_coalesce_yeal():
    from backend.services.daily_inventory_history import coalesce_inventory_history_sku_aliases

    hist = pd.DataFrame(
        {
            "OMS_SKU": ["7100YKYEAL-L-XL", "7100YKTEAL-L-XL"],
            "Date": ["2026-07-20", "2026-07-20"],
            "Qty": [120.0, 0.0],
            "Source": ["uploaded", "uploaded"],
            "Channel": ["oms", "oms"],
        }
    )
    out = coalesce_inventory_history_sku_aliases(hist, {})
    assert len(out) == 1
    assert str(out.iloc[0]["OMS_SKU"]) == "7100YKTEAL-L-XL"
    assert float(out.iloc[0]["Qty"]) == 120.0


def test_coalesce_powder_to_power_inventory():
    """Warehouse POWDERBLUE stock must land on ops POWERBLUE Status SKU."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["7114YKPOWDERBLUE-F", "7114YKPOWERBLUE-F"],
            "OMS_Inventory": [175.0, 0.0],
            "Amazon_Inventory": [0.0, 0.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert "7114YKPOWDERBLUE-F" not in by.index
    assert float(by.loc["7114YKPOWERBLUE-F", "Total_Inventory"]) == 175.0


def test_po_joins_power_stock_with_powder_sales_under_master_map():
    """5038 master map targets POWDERBLUE; inventory holds POWERBLUE — one PO row."""
    from backend.services.po_engine import calculate_po_base
    from backend.services.sku_mapping import (
        clear_bundled_sku_mapping_cache,
        load_bundled_sku_mapping,
        resolve_sku_replacement_map,
    )

    clear_bundled_sku_mapping_cache()
    m = load_bundled_sku_mapping()
    # Destinations in the shipped map are rewritten to POWERBLUE via resolve.
    assert m.get("5038YKPOWDERBLUE-3XL") == "5038YKPOWERBLUE-3XL"
    assert m.get("5038PLYKPOWDERBLUE-3XL") == "5038YKPOWERBLUE-3XL"

    # Also prove raw (unresolved) POWDER terminals no longer split after resolve.
    raw = {
        "5038YKPOWDERBLUE-3XL": "5038YKPOWDERBLUE-3XL",
        "5038PLYKPOWDERBLUE-3XL": "5038YKPOWDERBLUE-3XL",
        "5038YKPOWDERBLUE-L": "5038YKPOWDERBLUE-L",
        "5038PLYKPOWDERBLUE-L": "5038YKPOWDERBLUE-L",
    }
    assert resolve_sku_replacement_map(raw)["5038PLYKPOWDERBLUE-3XL"] == "5038YKPOWERBLUE-3XL"

    inv = pd.DataFrame(
        {
            "OMS_SKU": ["5038YKPOWERBLUE-3XL", "5038YKPOWERBLUE-L"],
            "OMS_Inventory": [47.0, 60.0],
            "Amazon_Inventory": [0.0, 0.0],
            "Total_Inventory": [47.0, 60.0],
            "Marketplace_Total": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": (["5038YKPOWDERBLUE-3XL"] * 6) + (["5038YKPOWDERBLUE-L"] * 11),
            "TxnDate": pd.date_range("2026-07-01", periods=17, freq="D"),
            "Transaction Type": ["Shipment"] * 17,
            "Quantity": [1] * 17,
            "Units_Effective": [1] * 17,
        }
    )
    po = calculate_po_base(
        sales, inv, period_days=30, lead_time=45, target_days=45, sku_mapping=m
    )
    sub = po[po["OMS_SKU"].astype(str).str.contains("5038YK.*BLUE", regex=True)]
    skus = set(sub["OMS_SKU"].astype(str))
    assert "5038YKPOWDERBLUE-3XL" not in skus
    assert "5038YKPOWDERBLUE-L" not in skus
    by = sub.set_index("OMS_SKU")
    assert float(by.loc["5038YKPOWERBLUE-3XL", "Total_Inventory"]) == 47.0
    assert float(by.loc["5038YKPOWERBLUE-3XL", "Sold_Units"]) == 6.0
    assert float(by.loc["5038YKPOWERBLUE-L", "Total_Inventory"]) == 60.0
    assert float(by.loc["5038YKPOWERBLUE-L", "Sold_Units"]) == 11.0


def test_coalesce_case_twins_max_not_double():
    """6xl vs 6XL re-upload must not sum to 2× stock."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["1130YKPURPLE-6XL", "1130YKPURPLE-6xl"],
            "OMS_Inventory": [81.0, 81.0],
            "Amazon_Inventory": [0.0, 0.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert list(by.index) == ["1130YKPURPLE-6XL"]
    assert float(by.loc["1130YKPURPLE-6XL", "Total_Inventory"]) == 81.0


def test_history_case_and_powder_coalesce():
    from backend.services.daily_inventory_history import coalesce_inventory_history_sku_aliases
    from backend.services.po_engine import inventory_oms_key, canonical_oms_key

    assert inventory_oms_key("7114YKPOWDERBLUE-F") == "7114YKPOWERBLUE-F"
    assert canonical_oms_key("7114YKPOWDERBLUE-F", {}) == "7114YKPOWERBLUE-F"

    hist = pd.DataFrame(
        {
            "OMS_SKU": [
                "1130YKPURPLE-6XL",
                "1130YKPURPLE-6xl",
                "7114YKPOWDERBLUE-F",
                "7114YKPOWERBLUE-F",
            ],
            "Date": ["2026-08-05"] * 4,
            "Qty": [81.0, 81.0, 175.0, 0.0],
            "Source": ["uploaded"] * 4,
            "Channel": ["oms"] * 4,
        }
    )
    out = coalesce_inventory_history_sku_aliases(hist, {})
    by = out.set_index("OMS_SKU")["Qty"].to_dict()
    assert float(by["1130YKPURPLE-6XL"]) == 81.0
    assert float(by["7114YKPOWERBLUE-F"]) == 175.0
    assert "7114YKPOWDERBLUE-F" not in by


def test_scrub_drops_total_inv_label():
    from backend.services.daily_inventory_history import scrub_absurd_inventory_history_rows

    df = pd.DataFrame(
        {
            "OMS_SKU": ["Total inv.", "1219YKBLACK-M"],
            "Date": ["2026-08-05", "2026-08-05"],
            "Qty": [198827.0, 120.0],
            "Source": ["uploaded", "uploaded"],
        }
    )
    out = scrub_absurd_inventory_history_rows(df)
    assert list(out["OMS_SKU"]) == ["1219YKBLACK-M"]


def test_yeal_teal_still_sums_after_case_collapse():
    """True spelling twins must still sum (not max)."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["7100YKYEAL-L-XL", "7100YKTEAL-L-XL"],
            "OMS_Inventory": [81.0, 3.0],
            "Amazon_Inventory": [0.0, 3.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert float(by.loc["7100YKTEAL-L-XL", "OMS_Inventory"]) == 84.0
    assert float(by.loc["7100YKTEAL-L-XL", "Total_Inventory"]) == 87.0


def test_bottle_dual_census_max_not_double():
    """When BOTTLE and BOTTEL both hold the full FBA/OMS census, do not 2×."""
    from backend.services.inventory import smart_coalesce_qty

    assert smart_coalesce_qty([27, 27]) == 27.0
    assert smart_coalesce_qty([81, 3]) == 84.0
    assert smart_coalesce_qty([41, 41, 0]) == 41.0

    inv = pd.DataFrame(
        {
            "OMS_SKU": ["5041YKBOTTLEGREEN-XL", "5041YKBOTTELGREEN-XL"],
            "OMS_Inventory": [27.0, 27.0],
            "Amazon_Inventory": [41.0, 41.0],
            "Manual_InTransit": [0.0, 0.0],
            "Not_In_Inventory_Qty": [0.0, 0.0],
            "Buffer_Stock": [0.0, 0.0],
        }
    )
    out = coalesce_inventory_by_sku_mapping(inv, {})
    by = out.set_index("OMS_SKU")
    assert float(by.loc["5041YKBOTTELGREEN-XL", "OMS_Inventory"]) == 27.0
    assert float(by.loc["5041YKBOTTELGREEN-XL", "Amazon_Inventory"]) == 41.0
    assert float(by.loc["5041YKBOTTELGREEN-XL", "Total_Inventory"]) == 68.0


def test_history_bottle_dual_census_not_double():
    from backend.services.daily_inventory_history import coalesce_inventory_history_sku_aliases

    hist = pd.DataFrame(
        {
            "OMS_SKU": [
                "5041YKBOTTLEGREEN-XL",
                "5041YKBOTTELGREEN-XL",
                "5041YKBOTTLEGREEN-XL",
                "5041YKBOTTELGREEN-XL",
            ],
            "Date": ["2026-07-01"] * 4,
            "Qty": [27.0, 27.0, 41.0, 41.0],
            "Source": ["snapshot"] * 4,
            "Channel": ["oms", "oms", "amazon", "amazon"],
        }
    )
    out = coalesce_inventory_history_sku_aliases(hist, {})
    by = out.set_index(["OMS_SKU", "Channel"])["Qty"]
    assert float(by.loc[("5041YKBOTTELGREEN-XL", "oms")]) == 27.0
    assert float(by.loc[("5041YKBOTTELGREEN-XL", "amazon")]) == 41.0

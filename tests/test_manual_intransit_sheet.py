"""Tests for admin Intrasit / Not In Inventory workbook upload."""
from pathlib import Path

import pandas as pd
import pytest

from backend.services.inventory import recompute_inventory_totals
from backend.services.manual_intransit_sheet import (
    _combine_manual_overlay,
    apply_manual_intransit_import,
    apply_manual_intransit_overlay_to_inventory,
    parse_manual_intransit_workbook,
)


def test_parse_user_intransit_workbook():
    path = Path("/Users/samraisinghani/Downloads/Intrasit And Not In Inventory 1.xlsx")
    if not path.is_file():
        pytest.skip("fixture not on disk")
    raw = path.read_bytes()
    it_df, ni_df, report = parse_manual_intransit_workbook(raw, path.name)
    assert not report.get("error"), report.get("error")
    assert int(report["intransit_units"]) >= 500
    assert int(report["not_in_inventory_units"]) >= 100
    assert int(report["intransit_skus"]) >= 500
    assert int(report["not_in_inventory_skus"]) >= 100


def test_reupload_replaces_not_duplicates():
    path = Path("/Users/samraisinghani/Downloads/Intrasit And Not In Inventory 1.xlsx")
    if not path.is_file():
        pytest.skip("fixture not on disk")
    from backend.session import AppSession

    sess = AppSession()
    raw = path.read_bytes()
    it1, ni1, r1 = parse_manual_intransit_workbook(raw, path.name)
    out1 = apply_manual_intransit_import(sess, it1, ni1, r1, filename=path.name)
    units1 = int(out1["intransit_units"])
    out2 = apply_manual_intransit_import(sess, it1, ni1, r1, filename=path.name)
    units2 = int(out2["intransit_units"])
    assert units1 == units2
    assert len(sess.manual_intransit_overlay_df) == int(out1["skus"])


def test_duplicate_sku_rows_merged_in_sheet():
    import openpyxl
    from io import BytesIO

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Intrasit Inventory"
    ws.append(["Sku", "Qty"])
    ws.append(["SKU-A", 2])
    ws.append(["SKU-A", 3])
    buf = BytesIO()
    wb.save(buf)
    it_df, ni_df, report = parse_manual_intransit_workbook(buf.getvalue(), "test.xlsx")
    assert int(it_df["Manual_InTransit"].sum()) == 5
    assert any("Duplicate SKU" in (d.get("reason") or "") for d in report.get("skip_details") or [])


def test_combine_overlay_columns():
    it = pd.DataFrame({"OMS_SKU": ["A"], "Manual_InTransit": [4]})
    ni = pd.DataFrame({"OMS_SKU": ["B"], "Not_In_Inventory_Qty": [9]})
    comb = _combine_manual_overlay(it, ni)
    assert set(comb.columns) == {"OMS_SKU", "Manual_InTransit", "Not_In_Inventory_Qty"}
    assert len(comb) == 2


def test_recompute_inventory_totals_excludes_total_inventory_column():
    """Total_Inventory must not be summed into Marketplace_Total (double-count bug)."""
    df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "OMS_Inventory": [100],
            "Buffer_Stock": [20],
            "Amazon_Inventory": [50],
            "Marketplace_Total": [999],
            "Total_Inventory": [150],
        }
    )
    out = recompute_inventory_totals(df)
    assert int(out["Marketplace_Total"].iloc[0]) == 50
    assert int(out["Total_Inventory"].iloc[0]) == 150


def test_manual_intransit_overlay_total_matches_user_expected_formula():
    """OMS + marketplaces + manual in-transit + not-in-inventory (buffer excluded)."""
    inv = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B"],
            "OMS_Inventory": [100_000, 30_729],
            "Buffer_Stock": [24_070, 0],
            "Amazon_Inventory": [30_907, 0],
            "Myntra_Other_Inventory": [2_301, 0],
            "Flipkart_Inventory": [545, 0],
            "Marketplace_Total": [33_753, 0],
            "Total_Inventory": [133_753, 30_729],
        }
    )
    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["C", "D"],
            "Manual_InTransit": [4_648, 0],
            "Not_In_Inventory_Qty": [0, 7_986],
        }
    )

    class Sess:
        inventory_df_variant = inv.copy()
        inventory_df_parent = pd.DataFrame()
        manual_intransit_overlay_df = overlay

    sess = Sess()
    apply_manual_intransit_overlay_to_inventory(sess)
    totals = sess.inventory_df_variant[
        [
            "OMS_Inventory",
            "Amazon_Inventory",
            "Myntra_Other_Inventory",
            "Flipkart_Inventory",
            "Manual_InTransit",
            "Not_In_Inventory_Qty",
            "Marketplace_Total",
            "Total_Inventory",
        ]
    ].sum(numeric_only=True)

    expected_marketplace = (
        int(totals["Amazon_Inventory"])
        + int(totals["Myntra_Other_Inventory"])
        + int(totals["Flipkart_Inventory"])
        + int(totals["Manual_InTransit"])
        + int(totals["Not_In_Inventory_Qty"])
    )
    expected_total = int(totals["OMS_Inventory"]) + expected_marketplace

    assert int(totals["Marketplace_Total"]) == expected_marketplace
    assert int(totals["Total_Inventory"]) == expected_total
    assert int(totals["Total_Inventory"]) == 177_116

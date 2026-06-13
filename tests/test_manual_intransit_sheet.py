"""Tests for admin Intrasit / Not In Inventory workbook upload."""
from pathlib import Path

import pandas as pd
import pytest

from backend.services.manual_intransit_sheet import (
    _combine_manual_overlay,
    apply_manual_intransit_import,
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

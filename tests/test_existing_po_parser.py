"""Tests for existing PO sheet ingestion."""

from io import BytesIO

import pandas as pd
from openpyxl import Workbook

from backend.services.existing_po import parse_existing_po


def test_parse_existing_po_csv_minimal():
    csv = "OMS SKU,Total Balance\nABC-RED-L,5\n"
    out = parse_existing_po(csv.encode("utf-8"), "po.csv")
    assert len(out) == 1
    assert out["OMS_SKU"].iloc[0] == "ABC-RED-L"
    assert int(out["PO_Pipeline_Total"].iloc[0]) == 5


def test_parse_existing_po_excel_title_row_before_header():
    """Workbooks with a title line above the real header must still parse."""
    wb = Workbook()
    ws = wb.active
    ws.append(["PO status — April"])
    ws.append(["OMS SKU", "Total Balance", "Note"])
    ws.append(["SKU99-XS", "3", ""])
    buf = BytesIO()
    wb.save(buf)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert len(out) == 1
    assert out["OMS_SKU"].iloc[0] == "SKU99-XS"
    assert int(out["PO_Pipeline_Total"].iloc[0]) == 3


def test_parse_existing_po_headerless_yash_export():
    """Operator exports without a header row (first row is 1917YKBLUE-3XL …)."""
    wb = Workbook()
    ws = wb.active
    for row in [
        ["1917YKBLUE-3XL", "New SKU", "1917YKBLUE", "Monika", 0, 130, 130, 0],
        ["1917YKBLUE-4XL", "New SKU", "1917YKBLUE", "Monika", 0, 170, 170, 0],
    ]:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    out = parse_existing_po(buf.getvalue(), "existing_po.xlsx")
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-3XL", "PO_Pipeline_Total"].iloc[0]) == 130
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-4XL", "PO_Pipeline_Total"].iloc[0]) == 170


def test_parse_existing_po_generic_numeric_columns():
    df = pd.DataFrame(
        {
            "SKU": ["1917YKBLUE-3XL", "1917YKBLUE-L"],
            "Status": ["New SKU", "New SKU"],
            "Style": ["1917YKBLUE", "1917YKBLUE"],
            "Name": ["Monika", "Monika"],
            "Col5": [0, 0],
            "Col6": [130, 120],
            "Col7": [130, 120],
            "Col8": [0, 0],
        }
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert int(out.loc[out["OMS_SKU"] == "1917YKBLUE-3XL", "PO_Pipeline_Total"].iloc[0]) == 130


def test_expand_bundled_po_skus_splits_size_ranges():
    from backend.services.existing_po import expand_bundled_po_skus

    ep = pd.DataFrame(
        {
            "OMS_SKU": ["1917YKBLUE-XXL-3XL"],
            "PO_Pipeline_Total": [194],
            "Pending_Cutting": [190],
            "Balance_to_Dispatch": [4],
        }
    )
    out = expand_bundled_po_skus(ep)
    assert set(out["OMS_SKU"]) == {"1917YKBLUE-XXL", "1917YKBLUE-3XL"}
    assert int(out["PO_Pipeline_Total"].sum()) == 194


def test_parse_existing_po_vendor_article_column():
    df = pd.DataFrame(
        {
            "Vendor Article Number": ["V1-A", "V1-B"],
            "Open Qty": [2, 4],
        }
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
    out = parse_existing_po(buf.getvalue(), "po.xlsx")
    assert len(out) == 2
    assert set(out["OMS_SKU"]) == {"V1-A", "V1-B"}

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

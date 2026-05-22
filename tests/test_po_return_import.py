"""Tests for marketplace return RAR/CSV import parsers."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from backend.services.po_return_import import (
    _parse_amazon_business_return,
    _parse_flipkart_returns_xlsx,
    _parse_single_return_file,
    parse_return_upload_bytes,
)


def test_amazon_business_return_units_refunded():
    df = pd.DataFrame(
        {
            "(Child) ASIN": ["B07Z5XTPH4", "B07Z5WYPQ8"],
            "Units Refunded": ["29", "1,234"],
            "Title": ["a", "b"],
        }
    )
    out, err = _parse_amazon_business_return(df)
    assert err is None
    assert int(out.loc[out["OMS_SKU"] == "B07Z5WYPQ8", "Return_Units"].iloc[0]) == 1234


def test_flipkart_sku_prefix_stripped():
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Returns"
    ws.append(["return_id", "sku", "quantity"])
    ws.append(["RI:1", "SKU:TEST-SKU-M", 2])
    ws.append(["RI:2", "SKU:OTHER-L", 1])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    out, err = _parse_flipkart_returns_xlsx(raw_xlsx.read())
    assert err is None
    skus = set(out["OMS_SKU"].tolist())
    assert "TEST-SKU-M" in skus
    assert int(out["Return_Units"].sum()) == 3


def test_meesho_csv_header_skip():
    csv_body = (
        '"Meesho Supplier Panel"\n'
        '"Supplier ID","1662411"\n'
        '\n'
        '"S No","Product Name","SKU","Qty"\n'
        '"1","Product","206PLYKN324MUSTARD","2"\n'
        '"2","Product2","165YKN251MUSTRAD","1"\n'
    ).encode()
    out, err = _parse_single_return_file(csv_body, "meesho_return.csv")
    assert err is None
    assert int(out["Return_Units"].sum()) == 3


@pytest.mark.skipif(
    not Path("/Users/samraisinghani/Downloads/Return Data 2.rar").is_file(),
    reason="Return Data 2.rar not on disk",
)
def test_parse_return_data_rar_bundle():
    rar = Path("/Users/samraisinghani/Downloads/Return Data 2.rar")
    raw = rar.read_bytes()
    df, err = parse_return_upload_bytes(raw, rar.name)
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) > 100

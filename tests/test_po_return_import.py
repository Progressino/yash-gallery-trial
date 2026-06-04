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


def test_parse_flipkart_xlsx_not_treated_as_zip_archive():
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Returns"
    ws.append(["return_id", "sku", "quantity"])
    ws.append(["RI:1", "SKU:TEST-SKU-M", 4])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    body = raw_xlsx.read()
    df, err = parse_return_upload_bytes(body, "Akiko Flipkart Return.xlsx")
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) == 4
    assert df.iloc[0]["Return_Platform"] == "flipkart"


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


def test_infer_return_platform_from_filename():
    from backend.services.po_return_import import _infer_return_platform_from_filename

    assert _infer_return_platform_from_filename("BusinessReport-Amazon Return.csv") == "amazon"
    assert _infer_return_platform_from_filename("Myntra Return.csv") == "myntra"
    assert _infer_return_platform_from_filename("Flipkart Return.xlsx") == "flipkart"


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


@pytest.mark.parametrize(
    "rar_name",
    ["Return Data 2.rar", "Return Data 3.rar"],
)
def test_parse_return_data_rar_bundle(rar_name):
    rar = Path("/Users/samraisinghani/Downloads") / rar_name
    if not rar.is_file():
        pytest.skip(f"{rar_name} not on disk")
    raw = rar.read_bytes()
    df, err = parse_return_upload_bytes(raw, rar.name)
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) > 100


def test_meesho_lost_skipped_in_rar(monkeypatch):
    from backend.services.po_return_import import _expand_upload_to_member_files

    members = [
        ("Return Data/meesho_return.csv", b"x"),
        ("Return Data/meesho_lost.csv", b"y"),
    ]
    monkeypatch.setattr(
        "backend.services.po_return_import._expand_upload_to_member_files",
        lambda raw, filename: members,
    )
    import pandas as pd
    from backend.services import po_return_import as pri

    def fake_single(raw, name, **k):
        if "lost" in name.lower():
            return pd.DataFrame({"OMS_SKU": ["BAD"], "Return_Units": [99]}), None
        return pd.DataFrame({"OMS_SKU": ["GOOD"], "Return_Units": [1]}), None

    monkeypatch.setattr(pri, "_parse_single_return_file", fake_single)
    df, err = parse_return_upload_bytes(b"fake", "bundle.rar")
    assert err is None
    assert "BAD" not in df["OMS_SKU"].tolist()
    assert int(df["Return_Units"].sum()) == 1

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
    ws.append(["return_id", "sku", "quantity", "return_approval_date"])
    ws.append(["RI:1", "SKU:TEST-SKU-M", 4, "2026-03-05"])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    body = raw_xlsx.read()
    df, err, _warn = parse_return_upload_bytes(body, "Akiko Flipkart Return.xlsx")
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) == 4
    assert df.iloc[0]["Return_Platform"] == "flipkart"
    assert "Return_Date" in df.columns
    assert str(df.iloc[0]["Return_Date"]) == "2026-03-05"


def test_parse_flipkart_sales_report_returns_event_type():
    """Monthly Flipkart B2C Sales Report — Return rows only (not Sale)."""
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales Report"
    ws.append(
        [
            "Event Type",
            "Event Sub Type",
            "SKU",
            "Item Quantity",
            "Order Date",
            "Buyer Invoice Date",
        ]
    )
    ws.append(["Return", "Return", '"""SKU:1899YKOFFWHITE-XXL"""', 1, "2026-04-30", "2026-05-12"])
    ws.append(["Sale", "Sale", '"""SKU:190YK214MEHROON-XXL"""', 1, "2026-05-12", "2026-05-12"])
    ws.append(["Return", "Return", '"""SKU:1057YKBLUE-M"""', 2, "2026-05-09", "2026-05-12"])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    out, err = _parse_flipkart_returns_xlsx(raw_xlsx.read())
    assert err is None, err
    assert int(out["Return_Units"].sum()) == 3
    assert "1899YKOFFWHITE-XXL" in set(out["OMS_SKU"])
    assert "190YK214MEHROON-XXL" not in set(out["OMS_SKU"])
    assert (out["Return_Platform"] == "flipkart").all()
    assert "2026-04-30" in set(out["Return_Date"].astype(str))


def test_parse_flipkart_sales_report_uuid_filename():
    """Seller Hub download names are UUIDs — detect via Sales Report sheet."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "flipkart_sales_report_returns_sample.xlsx"
    if not fixture.is_file():
        pytest.skip("flipkart sales report fixture missing")
    raw = fixture.read_bytes()
    df, err, _warn = parse_return_upload_bytes(raw, "0bedba5d-12da-4468-80bc-1368872b2087_1780814425000.xlsx")
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) == 657
    assert df["Return_Platform"].eq("flipkart").all()
    months = sorted({str(d)[:7] for d in df["Return_Date"] if str(d)[:4] >= "2020"})
    assert any(m >= "2026-04" for m in months), months


def test_flipkart_sku_prefix_stripped():
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Returns"
    ws.append(["return_id", "sku", "quantity", "return_completion_date"])
    ws.append(["RI:1", "SKU:TEST-SKU-M", 2, "10/03/2026"])
    ws.append(["RI:2", "SKU:OTHER-L", 1, "16/03/2026"])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    out, err = _parse_flipkart_returns_xlsx(raw_xlsx.read())
    assert err is None
    skus = set(out["OMS_SKU"].tolist())
    assert "TEST-SKU-M" in skus
    assert int(out["Return_Units"].sum()) == 3
    assert "Return_Date" in out.columns


def test_myntra_return_date_falls_back_from_rto_when_created_is_epoch():
    from backend.services.po_return_import import _parse_myntra_seller_returns_csv

    csv_body = (
        "seller_sku_code,quantity,return_created_date,order_rto_date\n"
        "SKU-A,2,1970-01-01,2026-01-20\n"
        "SKU-B,1,2026-02-10,\n"
    ).encode()
    out, err = _parse_myntra_seller_returns_csv(csv_body)
    assert err is None
    assert int(out["Return_Units"].sum()) == 3
    dates = set(out["Return_Date"].astype(str))
    assert "2026-01-20" in dates
    assert "2026-02-10" in dates
    assert "1970-01-01" not in dates


@pytest.mark.parametrize(
    "rar_name",
    ["Akiko Flipkart Return 1.rar", "YG Myntra Return 1.rar"],
)
def test_user_return_rars_have_calendar_dates(rar_name):
    rar = Path("/Users/samraisinghani/Downloads") / rar_name
    if not rar.is_file():
        pytest.skip(f"{rar_name} not on disk")
    df, err, _warn = parse_return_upload_bytes(rar.read_bytes(), rar.name)
    assert err is None, err
    assert "Return_Date" in df.columns
    months = sorted({str(d)[:7] for d in df["Return_Date"] if str(d)[:4] >= "2020"})
    assert any(m >= "2026-01" for m in months), months


def test_infer_return_platform_from_filename():
    from backend.services.po_return_import import _infer_return_platform_from_filename

    assert _infer_return_platform_from_filename("BusinessReport-Amazon Return.csv") == "amazon"
    assert _infer_return_platform_from_filename("Myntra Return.csv") == "myntra"
    assert _infer_return_platform_from_filename("Flipkart Return.xlsx") == "flipkart"
    assert (
        _infer_return_platform_from_filename(
            "Akiko Flipkart Return/431fe84d-60b4-491a-8dc6-429aee17f3a3.xlsx"
        )
        == "flipkart"
    )


def test_meesho_csv_header_skip():
    csv_body = (
        '"Meesho Supplier Panel"\n'
        '"Supplier ID","1662411"\n'
        '\n'
        '"S No","Product Name","SKU","Qty"\n'
        '"1","Product","206PLYKN324MUSTARD","2"\n'
        '"2","Product2","165YKN251MUSTRAD","1"\n'
    ).encode()
    out, err, _stats = _parse_single_return_file(csv_body, "meesho_return.csv")
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
    df, err, _warn = parse_return_upload_bytes(raw, rar.name)
    assert err is None, err
    assert not df.empty
    assert int(df["Return_Units"].sum()) > 100


def test_amazon_mtr_refund_rows_only_not_shipments():
    """MTR_B2C/B2B GST reports list every transaction — only Refund rows are returns."""
    csv_body = (
        "Transaction Type,Order Date,Invoice Date,Sku,Quantity\n"
        "Shipment,2026-04-20,2026-04-21,SKU-A,5\n"
        "Refund,2026-04-24,2026-04-25,SKU-A,1\n"
        "Cancel,2026-04-22,2026-04-22,SKU-B,3\n"
        "Refund,2026-04-26,2026-04-27,SKU-B,2\n"
    ).encode()
    out, err, _stats = _parse_single_return_file(csv_body, "MTR_B2C-APRIL-2026-XYZ.csv")
    assert err is None
    assert int(out["Return_Units"].sum()) == 3
    assert set(out["OMS_SKU"]) == {"SKU-A", "SKU-B"}
    assert (out["Return_Platform"] == "amazon").all()
    assert "2026-04-25" in set(out["Return_Date"].astype(str))
    assert "2026-04-27" in set(out["Return_Date"].astype(str))


def test_amazon_fba_returns_per_row_return_date():
    """Amazon FBA "Manage Returns" export — per-row return-date, not filename date."""
    csv_body = (
        "return-date,order-id,sku,asin,quantity\n"
        "2026-02-25T09:13:07+00:00,406-1,1065PLYKBLUE-4XL,B08S3VXM36,1\n"
        "2026-03-10T06:36:18+00:00,405-2,1300YKNORANGE-XXL,B092R1HPFJ,2\n"
    ).encode()
    out, err, _stats = _parse_single_return_file(csv_body, "Feb 4.csv")
    assert err is None
    assert (out["Return_Platform"] == "amazon").all()
    dates = set(out["Return_Date"].astype(str))
    assert dates == {"2026-02-25", "2026-03-10"}
    assert int(out["Return_Units"].sum()) == 3


def test_meesho_uses_return_created_date_not_filename_date():
    """Meesho panel CSV rows are dated by "Return Created Date", not the export filename."""
    csv_body = (
        '"Meesho Supplier Panel"\n'
        '"Supplier ID","1662411"\n'
        '\n'
        '"S No","Product Name","SKU","Qty","Dispatch Date","Return Created Date"\n'
        '"1","Product","206PLYKN324MUSTARD","2","2026-05-25","2026-06-09"\n'
        '"2","Product2","165YKN251MUSTRAD","1","2026-05-27","2026-06-08"\n'
    ).encode()
    out, err, _stats = _parse_single_return_file(
        csv_body, "YG Meesho Jan to jun-26/Yash Gallery__2026-06-11_13_0-13_59_1662411__.csv"
    )
    assert err is None
    assert int(out["Return_Units"].sum()) == 3
    dates = set(out["Return_Date"].astype(str))
    assert dates == {"2026-06-09", "2026-06-08"}
    assert (out["Return_Platform"] == "meesho").all()


def test_attach_return_platform_preserves_parser_assigned_platform():
    from backend.services.po_return_import import _attach_return_platform

    df = pd.DataFrame(
        {"OMS_SKU": ["A"], "Return_Units": [1], "Return_Platform": ["amazon"]}
    )
    out = _attach_return_platform(df, "Feb 4.csv")
    assert out.iloc[0]["Return_Platform"] == "amazon"


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
            return pd.DataFrame({"OMS_SKU": ["BAD"], "Return_Units": [99]}), None, {}
        return pd.DataFrame({"OMS_SKU": ["GOOD"], "Return_Units": [1]}), None, {}

    monkeypatch.setattr(pri, "_parse_single_return_file", fake_single)
    df, err, _warn = parse_return_upload_bytes(b"fake", "bundle.rar")
    assert err is None
    assert "BAD" not in df["OMS_SKU"].tolist()
    assert int(df["Return_Units"].sum()) == 1


def test_meesho_tcs_sales_return_maps_suborder_to_sku():
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(
        [
            "identifier",
            "sub_order_num",
            "order_date",
            "quantity",
            "cancel_return_date",
            "end_customer_state_new",
        ]
    )
    ws.append(["vvxpj", "246113537743954370_1", "2026-01-23", 2, "2026-01-29", "MH"])
    ws.append(["vvxpj", "243005505897928640_1", "2026-01-14", 1, "2026-01-14", "KA"])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    meesho_df = pd.DataFrame(
        {
            "OrderId": ["246113537743954370_1", "243005505897928640_1"],
            "SKU": ["1158YKGREEN-XL", "999YKRED-M"],
            "Date": pd.to_datetime(["2026-01-23", "2026-01-14"]),
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [2, 1],
        }
    )
    out, err, warns = parse_return_upload_bytes(
        raw_xlsx.read(),
        "gst_1662411_1_2026.zip/tcs_sales_return.xlsx",
        meesho_df=meesho_df,
    )
    assert err is None, err
    assert int(out["Return_Units"].sum()) == 3
    assert set(out["OMS_SKU"]) == {"1158YKGREEN-XL", "999YKRED-M"}
    assert (out["Return_Platform"] == "meesho").all()


def test_meesho_tcs_sales_return_warns_without_sales_lookup():
    openpyxl = pytest.importorskip("openpyxl")
    raw_xlsx = BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["identifier", "sub_order_num", "quantity", "cancel_return_date"])
    ws.append(["vvxpj", "SO-UNMAPPED-1", 5, "2026-02-01"])
    wb.save(raw_xlsx)
    raw_xlsx.seek(0)
    out, err, warns = parse_return_upload_bytes(
        raw_xlsx.read(),
        "folder/tcs_sales_return.xlsx",
    )
    assert out.empty
    assert err is not None
    assert any("mapped to SKU" in w or "mapped to a listing SKU" in w for w in warns + [err])


def _synthetic_meesho_df_from_tcs_rar(rar_path: Path) -> pd.DataFrame:
    from backend.services.po_return_import import _expand_upload_to_member_files

    raw = rar_path.read_bytes()
    members = _expand_upload_to_member_files(raw, rar_path.name)
    rows = []
    for name, data in members:
        if "tcs_sales_return" not in name.lower():
            continue
        tdf = pd.read_excel(BytesIO(data), sheet_name=0)
        for _, r in tdf.iterrows():
            sub = str(r.get("sub_order_num") or "").strip()
            if not sub:
                continue
            rows.append(
                {
                    "OrderId": sub,
                    "SKU": "1158YKGREEN-XL",
                    "Date": pd.to_datetime(r.get("cancel_return_date") or "2026-01-01"),
                    "TxnType": "Shipment",
                    "Quantity": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "rar_name,min_units",
    [
        ("YG Meesho Jan to jun-26 1.rar", 9400),
        ("Akiko Meesho Jan to jun-26 1.rar", 1800),
        ("Ashirwad Meesho Jan to jun-26 1.rar", 4500),
    ],
)
def test_meesho_jan_jun_rar_tcs_units_with_sales_lookup(rar_name, min_units):
    rar = Path("/Users/samraisinghani/Downloads") / rar_name
    if not rar.is_file():
        pytest.skip(f"{rar_name} not on disk")
    meesho_df = _synthetic_meesho_df_from_tcs_rar(rar)
    df, err, warns = parse_return_upload_bytes(
        rar.read_bytes(),
        rar.name,
        meesho_df=meesho_df,
    )
    assert err is None, err
    assert int(df["Return_Units"].sum()) >= min_units
    assert (df["Return_Platform"] == "meesho").all()

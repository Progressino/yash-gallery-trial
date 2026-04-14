"""Unit tests: sales aggregation, Amazon MTR dedup, platform summary."""

import pandas as pd

from backend.services.mtr import dedup_amazon_mtr_dataframe
from backend.services.helpers import map_to_oms_sku, sku_recognized_in_master
from backend.services.sales import (
    build_sales_df,
    filter_sales_for_export,
    get_platform_summary,
    get_sales_summary,
    list_sku_mapping_gaps,
)


def test_get_sales_summary_shipments_and_refunds():
    df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
            "Transaction Type": ["Shipment", "Shipment", "Refund"],
            "Quantity": [10, 5, 2],
            "Sku": ["A", "B", "A"],
            "Source": ["Amazon", "Amazon", "Amazon"],
            "Units_Effective": [10, 5, -2],
            "OrderId": [None, None, None],
        }
    )
    s = get_sales_summary(df, months=0)
    assert s["total_units"] == 15
    assert s["total_returns"] == 2
    assert s["net_units"] == 13
    assert abs(s["return_rate"] - (2 / 15 * 100)) < 0.15


def test_dedup_keeps_refund_sharing_order_with_invoice_shipment():
    d = pd.DataFrame(
        {
            "Invoice_Number": ["INV1", ""],
            "Order_Id": ["O1", "O1"],
            "SKU": ["SKU1", "SKU1"],
            "Transaction_Type": ["Shipment", "Refund"],
            "Quantity": [3.0, 1.0],
            "Date": pd.to_datetime(["2025-01-15", "2025-01-20"]),
        }
    )
    out = dedup_amazon_mtr_dataframe(d)
    assert len(out) == 2
    assert int(out[out["Transaction_Type"] == "Refund"]["Quantity"].sum()) == 1


def test_build_sales_df_amazon_with_mapping():
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-04-01"]),
            "SKU": ["1001YK-RED-L"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [4.0],
            "Order_Id": ["OID-1"],
            "Invoice_Number": [""],
        }
    )
    mapping = {"1001YK-RED-L": "1001YK-RED-L"}
    merged = build_sales_df(
        mtr_df=mtr,
        myntra_df=pd.DataFrame(),
        meesho_df=pd.DataFrame(),
        flipkart_df=pd.DataFrame(),
        sku_mapping=mapping,
    )
    assert not merged.empty
    assert (merged["Source"] == "Amazon").all()
    assert merged["Quantity"].sum() == 4


def test_build_sales_df_amazon_with_empty_mapping_includes_rows():
    """Without a SKU map file, Amazon MTR must still land in unified sales (PL-normalised SKU)."""
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-04-01"]),
            "SKU": ["1023PLYKBLUE-3XL"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [2.0],
            "Order_Id": ["O-99"],
            "Invoice_Number": [""],
        }
    )
    merged = build_sales_df(
        mtr_df=mtr,
        myntra_df=pd.DataFrame(),
        meesho_df=pd.DataFrame(),
        flipkart_df=pd.DataFrame(),
        sku_mapping={},
    )
    assert not merged.empty
    assert (merged["Source"] == "Amazon").all()
    assert merged["Quantity"].sum() == 2
    assert "1023YKBLUE-3XL" in merged["Sku"].astype(str).values or merged["Sku"].astype(str).str.contains("1023").any()


def test_filter_sales_for_export_date_and_source():
    df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2025-06-01", "2025-06-15", "2025-07-01"]),
            "Transaction Type": ["Shipment", "Refund", "Shipment"],
            "Quantity": [10, 2, 5],
            "Sku": ["A", "A", "B"],
            "Source": ["Amazon", "Amazon", "Myntra"],
            "Units_Effective": [10, -2, 5],
        }
    )
    out = filter_sales_for_export(
        df, start_date="2025-06-01", end_date="2025-06-30", sources=["Amazon"]
    )
    assert len(out) == 2
    assert set(out["Source"].unique()) == {"Amazon"}


def test_sku_recognized_in_master_keys_and_values():
    m = {"AMZ1": "OMSVAL", "FK": "OTHER"}
    assert sku_recognized_in_master("AMZ1", m)
    assert sku_recognized_in_master("OMSVAL", m)
    assert not sku_recognized_in_master("STRANGER", m)


def test_sku_recognized_oms_value_pl_yk_mirror():
    """Master OMS cell 1012PLYK* must match sales token 1012YK* in gap detection."""
    m = {"MKEY": "1012PLYKCPINK-3XL"}
    assert sku_recognized_in_master("1012YKCPINK-3XL", m)


def test_sku_recognized_hyphen_spacing_and_dot_before_size():
    m = {"1057KDBLUE-7-8": "OMS-A", "1556YKNGREEN-3XL": "OMS-B"}
    assert sku_recognized_in_master("1057KDBLUE -7-8", m)
    assert sku_recognized_in_master("1556YKNGREEN.-3XL", m)


def test_map_glued_myntra_size_hyphens():
    m = {"1061YKBLUE-4XL-BLUE": "OMS-G"}
    assert map_to_oms_sku("1061YKBLUE4XLBLUE", m) == "OMS-G"
    assert sku_recognized_in_master("1061YKBLUE4XLBLUE", m)


def test_map_glued_multi_trimmed_variant():
    m = {"1378YKMULTI-3XL": "OMS-M"}
    assert map_to_oms_sku("1378YKMULTI3XLMULTI", m) == "OMS-M"
    assert sku_recognized_in_master("1378YKMULTI3XLMULTI", m)


def test_mustrad_maps_to_mustard_master():
    m = {"165YK-251MUSTARD-3XL": "OMS-T"}
    assert map_to_oms_sku("165YK-251MUSTRAD-3XL", m) == "OMS-T"


def test_map_hyphen_between_style_id_and_yk_block():
    """Unified / channel exports sometimes insert 165-YK251… vs 165YK251…"""
    m = {"165YK251MUSTRAD": "OMS-165"}
    assert map_to_oms_sku("165-YK251MUSTRAD", m) == "OMS-165"
    assert map_to_oms_sku("165YK251MUSTRAD", m) == "OMS-165"


def test_map_y_hyphen_k_digit_listing_typo():
    m = {"165YK251MUSTRAD": "OMS-165"}
    assert map_to_oms_sku("165Y-K251MUSTRAD", m) == "OMS-165"


def test_map_plak_to_plyk_then_pl_strip():
    m = {"165YK251MUSTRAD": "OMS-165"}
    assert map_to_oms_sku("165PLAK251MUSTRAD", m) == "OMS-165"


def test_bare_digits_match_embedded_yrn_key():
    m = {"YARYKASS100506552": "OMS-YRN"}
    assert map_to_oms_sku("100506552", m) == "OMS-YRN"
    assert sku_recognized_in_master("100506552", m)


def test_yrn_decimal_form_matches_integer_map_key():
    """Myntra YRN in Excel is often 100672680.0 in sales while the map key is 100672680."""
    m = {"100672680": "1001YK-XL"}
    assert sku_recognized_in_master("100672680.0", m)
    assert sku_recognized_in_master("100672680", m)
    gaps = list_sku_mapping_gaps(
        pd.DataFrame({"Sku": ["100672680.0", "999999999"]}),
        m,
        limit=20,
    )
    assert "100672680.0" not in gaps and "100672680" not in gaps
    assert "999999999" in gaps


def test_list_sku_mapping_gaps():
    sales = pd.DataFrame({"Sku": ["OMSVAL", "STRANGER", "MEESHO_TOTAL", ""]})
    m = {"AMZ1": "OMSVAL", "FK99": "OTHER"}
    gaps = list_sku_mapping_gaps(sales, m)
    assert "STRANGER" in gaps
    assert "OMSVAL" not in gaps
    assert "OTHER" not in gaps
    assert "MEESHO_TOTAL" not in gaps


def test_meesho_detects_listing_sku_header_and_combines_size():
    from backend.services.meesho import _combine_meesho_sku_size, _meesho_sku_base_series

    df = pd.DataFrame({"listing sku": ["1158YKGREEN"], "size": ["XL"]})
    base = _meesho_sku_base_series(df)
    assert base.iloc[0] == "1158YKGREEN"
    assert _combine_meesho_sku_size(base, df["size"]).iloc[0] == "1158YKGREEN-XL"


def test_meesho_kids_size_band_hyphen():
    from backend.services.meesho import _combine_meesho_sku_size

    base = pd.Series(["1158YKMUSTARD"])
    size = pd.Series(["7-8"])
    assert _combine_meesho_sku_size(base, size).iloc[0] == "1158YKMUSTARD-7-8"


def test_meesho_order_export_xlsx_coalesces_listing_sku_column():
    from io import BytesIO

    import pandas as pd

    from backend.services.meesho import parse_meesho_order_export_xlsx

    df = pd.DataFrame(
        {
            "TxnDate": [pd.Timestamp("2024-12-16")],
            "Sku": ["MEESHO_TOTAL"],
            "Listing Sku": ["1592YKBLUE-5XL"],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Source": ["Meesho"],
            "OrderId": ["100056595506869312_1"],
        }
    )
    buf = BytesIO()
    df.to_excel(buf, index=False)
    out, msg = parse_meesho_order_export_xlsx(buf.getvalue())
    assert msg == "OK"
    assert out["SKU"].iloc[0] == "1592YKBLUE-5XL"


def test_meesho_order_export_xlsx_coalesces_sku1():
    from io import BytesIO

    import pandas as pd

    from backend.services.meesho import parse_meesho_order_export_xlsx

    df = pd.DataFrame(
        {
            "TxnDate": [pd.Timestamp("2024-12-16")],
            "Sku": ["MEESHO_TOTAL"],
            "OMS_Sku": [None],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Units_Effective": [1],
            "Source": ["Meesho"],
            "OrderId": ["100056595506869312_1"],
            "Sku.1": ["1592YKBLUE-5XL"],
        }
    )
    buf = BytesIO()
    df.to_excel(buf, index=False)
    out, msg = parse_meesho_order_export_xlsx(buf.getvalue())
    assert msg == "OK"
    assert len(out) == 1
    assert out["SKU"].iloc[0] == "1592YKBLUE-5XL"


def test_meesho_csv_skips_aggregate_sku_column_for_listing_sku():
    from backend.services.meesho import parse_meesho_csv

    csv = (
        "order date,sku,listing sku,quantity,customer state,sub order no\n"
        "2024-01-01,MEESHO_TOTAL,1158YKGREEN-XL,1,MH,SO1\n"
    )
    out, msg = parse_meesho_csv(csv.encode("utf-8"))
    assert msg == "OK"
    assert out["SKU"].iloc[0] == "1158YKGREEN-XL"


def test_meesho_nested_zip_finds_tcs_by_basename():
    """TCS files in subfolders (SomeFolder/tcs_sales_return.xlsx) must be detected."""
    import zipfile
    from io import BytesIO

    from backend.services.meesho import load_meesho_from_zip, meesho_to_sales_rows

    inner_df = pd.DataFrame(
        {
            "order_date": [pd.Timestamp("2025-12-31")],
            "sku": ["1158YKGREEN"],
            "size": ["XL"],
            "quantity": [1],
            "total_invoice_value": [100.0],
            "end_customer_state_new": ["MH"],
            "sub_order_num": ["SO1"],
        }
    )
    xb = BytesIO()
    inner_df.to_excel(xb, index=False)
    xlsx_bytes = xb.getvalue()

    inner_mem = BytesIO()
    with zipfile.ZipFile(inner_mem, "w", zipfile.ZIP_DEFLATED) as zin:
        zin.writestr("Reports/Dec/tcs_sales_return.xlsx", xlsx_bytes)

    outer_mem = BytesIO()
    with zipfile.ZipFile(outer_mem, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("dec.zip", inner_mem.getvalue())

    combined, _n, _skip = load_meesho_from_zip(outer_mem.getvalue())
    assert not combined.empty
    assert combined["SKU"].iloc[0] == "1158YKGREEN-XL"
    sales = meesho_to_sales_rows(combined, None)
    assert sales["Sku"].iloc[0] != "MEESHO_TOTAL"


def test_meesho_flat_outer_zip_without_nested_zip():
    """Outer archive may contain TCS xlsx directly (no inner .zip)."""
    import zipfile
    from io import BytesIO

    from backend.services.meesho import load_meesho_from_zip, meesho_to_sales_rows

    inner_df = pd.DataFrame(
        {
            "order_date": [pd.Timestamp("2025-11-15")],
            "sku": ["999YKRED"],
            "size": ["M"],
            "quantity": [2],
            "total_invoice_value": [50.0],
            "end_customer_state_new": ["KA"],
            "sub_order_num": ["SO2"],
        }
    )
    xb = BytesIO()
    inner_df.to_excel(xb, index=False)
    outer_mem = BytesIO()
    with zipfile.ZipFile(outer_mem, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("whatever/tcs_sales.xlsx", xb.getvalue())

    combined, _n, _sk = load_meesho_from_zip(outer_mem.getvalue())
    assert not combined.empty
    assert combined["SKU"].iloc[0] == "999YKRED-M"
    assert meesho_to_sales_rows(combined, None)["Sku"].iloc[0] != "MEESHO_TOTAL"


def test_meesho_to_sales_maps_and_oms_refresh_on_build():
    from backend.services.meesho import meesho_to_sales_rows

    mdf = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-12-31"]),
            "TxnType": ["Refund"],
            "Quantity": [1.0],
            "SKU": ["1158YKGREEN-XL"],
            "OrderId": ["237653394323428672_1"],
        }
    )
    assert meesho_to_sales_rows(mdf, {"1158YKGREEN-XL": "1001OMS-XL"})["Sku"].iloc[0] == "1001OMS-XL"
    mdf2 = mdf.copy()
    m = {"1158YKGREEN-XL": "1001OMS-XL"}
    build_sales_df(
        mtr_df=pd.DataFrame(),
        myntra_df=pd.DataFrame(),
        meesho_df=mdf2,
        flipkart_df=pd.DataFrame(),
        sku_mapping=m,
    )
    assert mdf2["OMS_SKU"].iloc[0] == "1001OMS-XL"


def test_get_platform_summary_amazon_refunds():
    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "SKU": ["X", "X"],
            "Transaction_Type": ["Shipment", "Refund"],
            "Quantity": [100.0, 5.0],
        }
    )
    plat = get_platform_summary(mtr, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)
    amz = next(p for p in plat if p["platform"] == "Amazon")
    assert amz["loaded"] is True
    assert amz["total_units"] == 100
    assert amz["total_returns"] == 5
    assert amz["return_rate"] == 5.0


def test_map_to_oms_normalizes_yrn_float_string():
    m = {"47061570": "OMS-YRN"}
    assert map_to_oms_sku("47061570.0", m) == "OMS-YRN"
    assert map_to_oms_sku("4.706157e+07", m) == "OMS-YRN"


def test_flipkart_sales_report_maps_sku_id_when_sku_column_is_dash():
    from io import BytesIO

    from backend.services.flipkart import _parse_flipkart_xlsx

    m = {"FKLISTING99": "OMS-FK"}
    df = pd.DataFrame(
        {
            "Buyer Invoice Date": [pd.Timestamp("2025-01-15")],
            "Event Sub Type": ["Sale"],
            "Item Quantity": [1],
            "Buyer Invoice Amount": [100.0],
            "SKU": ["-"],
            "SKU ID": ["FKLISTING99"],
            "Order ID": ["OD334028117901509100"],
        }
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Sales Report", index=False)
    out = _parse_flipkart_xlsx(buf.getvalue(), "fk-Jan-2025.xlsx", m)
    assert not out.empty
    assert out["OMS_SKU"].iloc[0] == "OMS-FK"

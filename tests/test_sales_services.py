"""Unit tests: sales aggregation, Amazon MTR dedup, platform summary."""

import pandas as pd

from backend.services.mtr import dedup_amazon_mtr_dataframe
from backend.services.helpers import sku_recognized_in_master
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


def test_list_sku_mapping_gaps():
    sales = pd.DataFrame({"Sku": ["OMSVAL", "STRANGER", "MEESHO_TOTAL", ""]})
    m = {"AMZ1": "OMSVAL", "FK99": "OTHER"}
    gaps = list_sku_mapping_gaps(sales, m)
    assert "STRANGER" in gaps
    assert "OMSVAL" not in gaps
    assert "OTHER" not in gaps
    assert "MEESHO_TOTAL" not in gaps


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

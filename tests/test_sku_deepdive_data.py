"""Tests for SKU Deepdive platform + sales merge."""
import pandas as pd

from backend.services.sku_deepdive_data import (
    build_deepdive_sales_frame,
    deepdive_parent_tokens,
    deepdive_sku_alias_tokens,
)


class _FakeSess:
    sku_mapping = {}

    def __init__(self):
        self.mtr_df = pd.DataFrame()
        self.myntra_df = pd.DataFrame()
        self.meesho_df = pd.DataFrame()
        self.flipkart_df = pd.DataFrame()
        self.snapdeal_df = pd.DataFrame()
        self.sales_df = pd.DataFrame()


def test_deepdive_aliases_bridge_hyphen_style_id():
    forms = deepdive_sku_alias_tokens("165YK-251MUSTRAD")
    assert "165YK251MUSTRAD" in forms
    assert "165YK-251MUSTRAD" in forms


def test_deepdive_parent_tokens_include_glued_and_split():
    parents = deepdive_parent_tokens("165YK-251MUSTRAD-M")
    assert "165YK251MUSTRAD" in parents or "165YK-251MUSTRAD" in parents


def test_build_deepdive_merges_amazon_platform_not_only_unified_sales(monkeypatch):
    """Unified sales may only have Meesho; Amazon bulk MTR must still appear."""
    sess = _FakeSess()
    sess.meesho_df = pd.DataFrame(
        {
            "SKU": ["165YK-251MUSTRAD-M"],
            "Date": ["2026-03-01"],
            "Quantity": [10],
            "TxnType": ["Shipment"],
            "OrderId": ["M1"],
            "LineKey": ["M1"],
        }
    )
    sess.mtr_df = pd.DataFrame(
        {
            "SKU": ["165YK251MUSTRAD-M", "165YK251MUSTRAD-M"],
            "Date": ["2025-01-15", "2025-02-10"],
            "Quantity": [50, 30],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Order_Id": ["A1", "A2"],
        }
    )
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["165YK-251MUSTRAD-M"],
            "TxnDate": ["2026-03-01"],
            "Quantity": [10],
            "Transaction Type": ["Shipment"],
            "Units_Effective": [10],
            "Source": ["Meesho"],
            "OrderId": ["M1"],
            "LineKey": ["M1"],
        }
    )

    out = build_deepdive_sales_frame(sess, "165YK-251MUSTRAD", all_sizes=True)
    assert not out.empty
    sources = set(out["Source"].astype(str))
    assert "Amazon" in sources
    assert "Meesho" in sources
    assert int(out.loc[out["Source"] == "Amazon", "Quantity"].sum()) == 80


def test_build_deepdive_exact_sku_without_all_sizes():
    sess = _FakeSess()
    sess.mtr_df = pd.DataFrame(
        {
            "SKU": ["165YK251MUSTRAD-L"],
            "Date": ["2025-06-01"],
            "Quantity": [25],
            "Transaction_Type": ["Shipment"],
            "Order_Id": ["A9"],
        }
    )
    sess.sales_df = pd.DataFrame()

    out = build_deepdive_sales_frame(sess, "165YK251MUSTRAD-L", all_sizes=False)
    assert len(out) == 1
    assert int(out.iloc[0]["Quantity"]) == 25


def test_build_deepdive_no_double_count_when_mtr_in_sales_df():
    """MTR data present in both mtr_df and sales_df must not be double-counted.

    In production, build_sales_df includes Amazon MTR rows in sess.sales_df.
    The deepdive must deduplicate so the same shipments are not counted twice.
    """
    sess = _FakeSess()
    # Amazon MTR: 19 units for 1001YKBEIGE-3XL on Jan 15, 2025
    sess.mtr_df = pd.DataFrame(
        {
            "SKU": ["1001YKBEIGE-3XL"],
            "Date": ["2025-01-15"],
            "Quantity": [19],
            "Transaction_Type": ["Shipment"],
            "Order_Id": ["A1"],
            "Invoice_Number": ["INV-001"],
        }
    )
    # sess.sales_df has the same Amazon data (as built by build_sales_df in real app)
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["1001YKBEIGE-3XL"],
            "TxnDate": ["2025-01-15"],
            "Quantity": [19],
            "Transaction Type": ["Shipment"],
            "Units_Effective": [19],
            "Source": ["Amazon"],
            "OrderId": ["A1"],
            "LineKey": ["A1-1001YKBEIGE-3XL"],
        }
    )

    out = build_deepdive_sales_frame(sess, "1001YKBEIGE-3XL", all_sizes=False)
    assert not out.empty
    amazon_units = int(out.loc[out["Source"].astype(str) == "Amazon", "Quantity"].sum())
    # Must be 19, not 38 (double-counted)
    assert amazon_units == 19, f"Expected 19 units, got {amazon_units} — double-counting!"


def test_build_deepdive_all_sizes_only_counts_matching_parent():
    """all_sizes=True should only include size variants of the searched SKU, not unrelated styles."""
    sess = _FakeSess()
    sess.mtr_df = pd.DataFrame(
        {
            "SKU": [
                "1001YKBEIGE-3XL",
                "1001YKBEIGE-XL",
                "1001YKBEIGE-M",
                "9999YKRED-3XL",  # different style — must NOT be included
            ],
            "Date": ["2025-01-05", "2025-01-05", "2025-01-05", "2025-01-05"],
            "Quantity": [19, 10, 8, 50],
            "Transaction_Type": ["Shipment", "Shipment", "Shipment", "Shipment"],
            "Order_Id": ["A1", "A2", "A3", "A4"],
        }
    )

    out = build_deepdive_sales_frame(sess, "1001YKBEIGE-3XL", all_sizes=True)
    assert not out.empty
    total = int(out["Quantity"].sum())
    # 19 + 10 + 8 = 37 for 1001YKBEIGE sizes; 9999YKRED-3XL should be excluded
    assert total == 37, f"Expected 37 units (3 sizes of 1001YKBEIGE), got {total}"

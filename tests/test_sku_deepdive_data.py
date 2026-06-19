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

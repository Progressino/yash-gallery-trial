"""Dashboard must merge Tier-3 daily uploads and rebuild sales when warm cache is stale."""
import pandas as pd

from backend.routers import data as data_router
from backend.services.sales import build_sales_df, get_platform_summary
from backend.session import AppSession


def test_session_sales_stale_when_meesho_loaded_but_not_in_sales_df():
    sess = AppSession()
    sess.sku_mapping = {"M1": "OMS-M1"}
    sess.meesho_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-23"]),
            "OMS_SKU": ["OMS-M1"],
            "TxnType": ["Shipment"],
            "Quantity": [12],
            "OrderId": ["O1"],
            "LineKey": [""],
        }
    )
    sess.sales_df = build_sales_df(
        mtr_df=pd.DataFrame(),
        myntra_df=pd.DataFrame(),
        meesho_df=pd.DataFrame(),
        flipkart_df=pd.DataFrame(),
        sku_mapping=sess.sku_mapping,
    )
    assert data_router._session_sales_stale_vs_platforms(sess) is True


def test_platform_summary_falls_back_to_raw_when_unified_missing_window():
    meesho = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-23", "2026-05-24"]),
            "OMS_SKU": ["A", "A"],
            "TxnType": ["Shipment", "Shipment"],
            "Quantity": [40, 56],
            "OrderId": ["1", "2"],
            "LineKey": ["", ""],
        }
    )
    sales = build_sales_df(
        mtr_df=pd.DataFrame(),
        myntra_df=pd.DataFrame(),
        meesho_df=pd.DataFrame(),
        flipkart_df=pd.DataFrame(),
        sku_mapping={},
    )
    plat = get_platform_summary(
        pd.DataFrame(),
        pd.DataFrame(),
        meesho,
        pd.DataFrame(),
        None,
        start_date="2026-05-23",
        end_date="2026-05-24",
        sales_df=sales,
    )
    row = next(p for p in plat if p["platform"] == "Meesho")
    assert row["loaded"] is True
    assert row["total_units"] == 96


def test_tier3_topup_needed_when_session_platform_empty(monkeypatch):
    from backend.services import daily_store

    sess = AppSession()
    sess.meesho_df = pd.DataFrame()
    monkeypatch.setattr(
        daily_store,
        "get_summary",
        lambda: {"meesho": {"file_count": 3, "max_date": "2026-05-24", "total_rows": 96}},
    )
    assert data_router._tier3_session_needs_topup(sess) is True

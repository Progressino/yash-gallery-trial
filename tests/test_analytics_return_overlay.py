"""Platform analytics monthly refunds include PO return overlay from 2026 onward."""

import pandas as pd

from backend.routers.data import _patch_analytics_monthly_returns
from backend.services.sales import merge_return_data_into_platform_summaries


class _Sess:
    po_return_overlay_df: pd.DataFrame
    sales_df: pd.DataFrame
    return_overlay_as_of: str = ""


def test_patch_analytics_monthly_adds_jan_2026_overlay():
    sess = _Sess()
    sess.po_return_overlay_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU1"],
            "Return_Platform": ["flipkart"],
            "Return_Date": ["2026-01-15"],
            "Return_Units": [40],
        }
    )
    sess.sales_df = pd.DataFrame()
    monthly = [{"Month": "2026-01", "shipments": 1000, "refunds": 0, "net": 1000}]
    _patch_analytics_monthly_returns(
        monthly, sess, "flipkart", display_name="Flipkart"
    )
    assert monthly[0]["refunds"] == 40
    assert monthly[0]["net"] == 960


def test_merge_return_data_into_platform_summaries_jan_2026():
    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["SKU1"],
            "Return_Platform": ["myntra"],
            "Return_Date": ["2026-02-01"],
            "Return_Units": [12],
        }
    )
    cards = [
        {
            "platform": "Myntra",
            "total_units": 500,
            "total_returns": 0,
            "net_units": 500,
            "return_rate": 0.0,
            "monthly": [{"month": "2026-02", "shipments": 500, "refunds": 0, "net": 500}],
            "daily": [],
        }
    ]
    out = merge_return_data_into_platform_summaries(
        cards, overlay, None, "2026-02-01", "2026-02-28"
    )
    assert out[0]["total_returns"] == 12
    assert out[0]["return_rate"] == 2.4
    assert out[0]["monthly"][0]["refunds"] == 12


def test_merge_return_data_does_not_double_count_overlay_in_sales():
    """Unified sales already has RETURN_SHEET refunds — merge must not add overlay again."""
    import pandas as pd

    from backend.services.sales import (
        RETURN_SHEET_ORDER_PLACEHOLDER,
        merge_return_data_into_platform_summaries,
    )

    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["FK-SKU"],
            "Return_Platform": ["flipkart"],
            "Return_Date": ["2026-03-15"],
            "Return_Units": [50],
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": ["FK-SKU"],
            "TxnDate": pd.to_datetime(["2026-03-15"]),
            "Transaction Type": ["Refund"],
            "Quantity": [50],
            "Units_Effective": [-50],
            "Source": ["Flipkart"],
            "OrderId": [RETURN_SHEET_ORDER_PLACEHOLDER],
            "LineKey": ["RETURN_SHEET|FK-SKU|Flipkart|2026-03-15"],
        }
    )
    cards = [
        {
            "platform": "Flipkart",
            "total_units": 1000,
            "total_returns": 50,
            "net_units": 950,
            "return_rate": 5.0,
            "monthly": [{"month": "2026-03", "shipments": 1000, "refunds": 50, "net": 950}],
            "daily": [],
        }
    ]
    out = merge_return_data_into_platform_summaries(
        cards, overlay, sales, "2026-03-01", "2026-03-31"
    )
    assert out[0]["total_returns"] == 50
    assert out[0]["monthly"][0]["refunds"] == 50


def test_patch_analytics_does_not_add_sales_refunds_on_top_of_raw_frame():
    """Platform analytics uses raw flipkart_df refunds; sales_df must not be added again."""
    import pandas as pd

    from backend.routers.data import _patch_analytics_monthly_returns
    from backend.services.sales import RETURN_SHEET_ORDER_PLACEHOLDER

    sess = _Sess()
    sess.po_return_overlay_df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU1"],
            "Return_Platform": ["flipkart"],
            "Return_Date": ["2026-01-15"],
            "Return_Units": [40],
        }
    )
    sess.sales_df = pd.DataFrame(
        {
            "Sku": ["SKU1", "SKU1"],
            "TxnDate": pd.to_datetime(["2026-01-10", "2026-01-15"]),
            "Transaction Type": ["Refund", "Refund"],
            "Quantity": [30, 40],
            "Units_Effective": [-30, -40],
            "Source": ["Flipkart", "Flipkart"],
            "OrderId": ["123", RETURN_SHEET_ORDER_PLACEHOLDER],
            "LineKey": ["", "RETURN_SHEET|SKU1|Flipkart|2026-01-15"],
        }
    )
    monthly = [{"Month": "2026-01", "shipments": 1000, "refunds": 30, "net": 970}]
    _patch_analytics_monthly_returns(monthly, sess, "flipkart", display_name="Flipkart")
    assert monthly[0]["refunds"] == 70
    assert monthly[0]["net"] == 930


def test_intelligence_hydrate_loads_overlay_from_disk(monkeypatch, tmp_path):
    from backend.routers.data import _hydrate_session_for_intelligence
    from backend.session import AppSession

    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "_DISK_CACHE_DIR", str(warm))
    main_mod._warm_cache = {}

    df = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Return_Platform": ["amazon"],
            "Return_Date": ["2026-03-10"],
            "Return_Units": [25],
        }
    )
    df.to_parquet(warm / "po_return_overlay_df.parquet", index=False)

    sess = AppSession()
    assert sess.po_return_overlay_df.empty
    _hydrate_session_for_intelligence(sess)
    assert not sess.po_return_overlay_df.empty
    assert int(sess.po_return_overlay_df["Return_Units"].sum()) == 25

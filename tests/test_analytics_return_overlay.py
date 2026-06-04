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

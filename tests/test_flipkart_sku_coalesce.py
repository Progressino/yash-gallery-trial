"""Flipkart listing SKU coalescing when primary SKU cell is a dash placeholder."""

import pandas as pd

from backend.services.flipkart import _fk_coalesced_listing_sku_series


def test_coalesce_finds_variant_sku_column_when_primary_is_dash():
    xl = pd.DataFrame(
        {
            "SKU": ["-", "-"],
            "Skuu": ["1316YKGREEN-3XL", "165YK251MUSTRAD-3XL"],
        }
    )
    out = _fk_coalesced_listing_sku_series(xl)
    assert list(out) == ["1316YKGREEN-3XL", "165YK251MUSTRAD-3XL"]


def test_coalesce_prefers_seller_sku_over_dash():
    xl = pd.DataFrame(
        {
            "SKU": ["-", "X"],
            "Seller SKU": ["165YK251MUSTRAD-M", "165YK251MUSTRAD-M"],
        }
    )
    out = _fk_coalesced_listing_sku_series(xl)
    assert out.iloc[0] == "165YK251MUSTRAD-M"
    assert out.iloc[1] == "X"

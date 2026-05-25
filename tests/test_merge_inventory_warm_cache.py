"""Inventory snapshot must enter warm cache so reload / new sessions restore it."""
import pandas as pd

from backend.main import merge_inventory_into_warm_cache
from backend.routers.data import _restore_inventory_from_warm
from backend.session import AppSession


def test_merge_inventory_into_warm_cache_and_restore():
    import backend.main as main

    main._warm_cache = {}
    sess = AppSession()
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU1"], "Flipkart_Inventory": [10], "Total_Inventory": [10]}
    )
    sess.inventory_df_parent = sess.inventory_df_variant.copy()

    merge_inventory_into_warm_cache(sess)
    assert not main._warm_cache["inventory_df_variant"].empty
    assert main._warm_cache["inventory_df_variant"].iloc[0]["OMS_SKU"] == "SKU1"

    empty = AppSession()
    assert empty.inventory_df_variant.empty
    _restore_inventory_from_warm(empty)
    assert not empty.inventory_df_variant.empty
    assert empty.inventory_df_variant.iloc[0]["Flipkart_Inventory"] == 10

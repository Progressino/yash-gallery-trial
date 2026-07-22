"""Warm-cache publish must merge platform history, never shrink bulk uploads."""
import pandas as pd

import backend.main as main_mod
from backend.session import AppSession
from tests.test_daily_auto_merge_preserves_bulk import _tier3_line_df


def _reset_warm() -> None:
    main_mod._warm_cache = {}


def test_publish_warm_cache_does_not_clobber_newer_inventory_by_row_count():
    """Regression: older session with more SKU rows must not overwrite a fresh upload.

    Prod symptom: UI showed the new snapshot date (meta) while OMS/Amazon totals
    stayed on the previous day's numbers (frame replaced by row-count publish).
    """
    _reset_warm()
    fresh = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B"],
            "OMS_Inventory": [10, 20],
            "Amazon_Inventory": [5, 5],
            "Total_Inventory": [15, 25],
        }
    )
    main_mod._warm_cache["inventory_df_variant"] = fresh.copy()
    main_mod._warm_cache["inventory_df_parent"] = fresh.copy()
    main_mod._warm_cache[main_mod._INVENTORY_META_WARM_KEY] = {
        "inventory_snapshot_uploaded_at": "2026-07-21T12:00:00.000000Z",
        "inventory_snapshot_date_label": "21 Jul 2026",
    }

    stale = AppSession()
    stale.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B", "C"],
            "OMS_Inventory": [100, 100, 100],
            "Amazon_Inventory": [40, 40, 40],
            "Total_Inventory": [140, 140, 140],
        }
    )
    stale.inventory_df_parent = stale.inventory_df_variant.copy()
    stale.inventory_snapshot_uploaded_at = "2026-07-20T12:00:00.000000Z"

    main_mod.publish_warm_cache_from_session(stale)

    warm = main_mod._warm_cache["inventory_df_variant"]
    assert float(warm["OMS_Inventory"].sum()) == 30.0
    assert float(warm["Amazon_Inventory"].sum()) == 10.0
    assert main_mod._warm_cache[main_mod._INVENTORY_META_WARM_KEY][
        "inventory_snapshot_uploaded_at"
    ].startswith("2026-07-21")


def test_publish_does_not_shrink_warm_cache_to_partial_session():
    """Regression: partial PG session must not wipe bulk warm-cache history."""
    _reset_warm()
    main_mod._warm_cache["myntra_df"] = _tier3_line_df("myntra", 190)

    sess = AppSession()
    sess.myntra_df = _tier3_line_df("myntra", 84, start=9000)

    main_mod.publish_warm_cache_from_session(sess)

    assert len(main_mod._warm_cache["myntra_df"]) >= 190


def test_publish_warm_cache_grows_when_session_has_more_rows():
    _reset_warm()
    main_mod._warm_cache["mtr_df"] = _tier3_line_df("amazon", 50)

    sess = AppSession()
    sess.mtr_df = _tier3_line_df("amazon", 120, start=200)

    main_mod.publish_warm_cache_from_session(sess)

    assert len(main_mod._warm_cache["mtr_df"]) == 170


def test_session_needs_warm_cache_topup_when_warm_has_more_rows():
    _reset_warm()
    main_mod._warm_cache["myntra_df"] = _tier3_line_df("myntra", 200)
    main_mod._warm_cache["sales_df"] = pd.DataFrame({"Sku": ["A"], "TxnDate": ["2024-01-01"]})

    sess = AppSession()
    sess.myntra_df = _tier3_line_df("myntra", 80)
    sess.sales_df = main_mod._warm_cache["sales_df"].copy()

    assert main_mod.session_needs_warm_cache_topup(sess) is True


def test_warm_cache_has_more_detects_row_gap():
    warm = _tier3_line_df("flipkart", 100)
    sess = _tier3_line_df("flipkart", 40)
    assert main_mod._warm_frame_has_more_rows(warm, sess) is True
    assert main_mod._warm_frame_has_more_rows(sess, warm) is False


def test_copy_warm_cache_does_not_fail_when_platform_merge_grows_cache():
    """Regression: merging session rows into warm cache must not mutate _warm_cache during iteration."""
    _reset_warm()
    main_mod._warm_cache["mtr_df"] = _tier3_line_df("amazon", 50)
    main_mod._warm_cache["sales_df"] = pd.DataFrame({"Sku": ["A"], "TxnDate": ["2024-01-01"]})

    sess = AppSession()
    sess.mtr_df = _tier3_line_df("amazon", 120, start=200)

    assert main_mod._copy_warm_cache_to_session(sess) is True
    assert len(sess.mtr_df) >= 120
    assert len(main_mod._warm_cache["mtr_df"]) >= 120

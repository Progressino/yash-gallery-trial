"""Cross-user hydration: newer server uploads must win over stale per-user sessions."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def test_daily_inventory_meta_is_newer_by_max_date():
    from backend.services.daily_inventory_history import daily_inventory_meta_is_newer

    sess = SimpleNamespace(
        daily_inventory_history_uploaded_at="2026-05-30 10:00:00",
        daily_inventory_history_filename="old.xlsx",
        daily_inventory_history_df=pd.DataFrame(
            {
                "OMS_SKU": ["A-1"],
                "Date": pd.to_datetime(["2026-05-30"]),
                "Qty": [1.0],
            }
        ),
    )
    meta = {
        "daily_inventory_history_uploaded_at": "2026-06-25 14:00:00",
        "daily_inventory_history_max_date": "2026-06-25",
    }
    assert daily_inventory_meta_is_newer(meta, sess) is True


def test_session_should_keep_existing_po_false_when_disk_generation_newer():
    from backend.services.existing_po import session_should_keep_existing_po

    sess = SimpleNamespace(
        existing_po_df=pd.DataFrame({"OMS_SKU": ["STYLE-1-XL"], "PO_Pipeline_Total": [10]}),
        existing_po_generation=2,
        existing_po_uploaded_at="2026-05-01T00:00:00Z",
        existing_po_filename="old-po.xlsx",
    )
    warm_df = pd.DataFrame({"OMS_SKU": ["STYLE-1-XL"], "PO_Pipeline_Total": [10]})
    import backend.main as _main

    _main._warm_cache[_main._EXISTING_PO_META_WARM_KEY] = {
        "existing_po_generation": 5,
        "existing_po_uploaded_at": "2026-06-25T10:00:00Z",
        "existing_po_filename": "new-po.xlsx",
    }
    assert session_should_keep_existing_po(sess, warm_df) is False

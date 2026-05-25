"""Deleting raise ledger must reduce confirmed pipeline used in PO math."""
import pandas as pd

from backend.services.po_raise_ledger import aggregate_raise_ledger_for_po
from backend.services.po_raise_remove import invalidate_po_calculate_result, remove_raise_ledger_day


def test_delete_day_removes_confirmed_pipeline_units():
    ledger = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "Raised_Qty": [100, 50],
            "Raised_Date": pd.to_datetime(["2026-05-22", "2026-05-21"]),
        }
    )
    with_22 = aggregate_raise_ledger_for_po(
        ledger,
        None,
        pd.Timestamp("2026-05-25"),
        lookback_days=14,
    )
    assert int(with_22.loc[with_22["OMS_SKU"] == "SKU-A", "PO_Confirmed_Raise_Pipeline"].iloc[0]) == 100

    from backend.session import AppSession

    sess = AppSession()
    sess.po_raise_ledger_df = ledger.copy()
    out = remove_raise_ledger_day(sess, "2026-05-22", session_id="test-session-delete-22")
    assert out["ok"] is True
    assert sess.po_calculate_status == "idle"
    assert sess.po_calculate_result_df.empty

    without_22 = aggregate_raise_ledger_for_po(
        sess.po_raise_ledger_df,
        None,
        pd.Timestamp("2026-05-25"),
        lookback_days=14,
    )
    sku_a = without_22[without_22["OMS_SKU"] == "SKU-A"]
    assert sku_a.empty or int(sku_a["PO_Confirmed_Raise_Pipeline"].fillna(0).iloc[0]) == 0
    assert int(
        without_22.loc[without_22["OMS_SKU"] == "SKU-B", "PO_Confirmed_Raise_Pipeline"].fillna(0).iloc[0]
    ) == 50

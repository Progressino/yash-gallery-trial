"""Return overlay refunds land on return_created_date, not upload/filename day."""
from __future__ import annotations

import pandas as pd

from backend.services.sales import return_sheet_refund_rows_from_overlay


def test_return_overlay_spreads_refunds_by_return_date():
    overlay = pd.DataFrame(
        [
            {
                "OMS_SKU": "SKU-A",
                "Return_Units": 10,
                "Return_Platform": "myntra",
                "Return_Date": "2026-05-30",
            },
            {
                "OMS_SKU": "SKU-B",
                "Return_Units": 5,
                "Return_Platform": "myntra",
                "Return_Date": "2026-05-31",
            },
        ]
    )
    synth = return_sheet_refund_rows_from_overlay(
        overlay, {}, ref_txn_date=pd.Timestamp("2026-06-03")
    )
    assert len(synth) == 2
    days = set(
        pd.to_datetime(synth["TxnDate"]).dt.strftime("%Y-%m-%d").tolist()
    )
    assert days == {"2026-05-30", "2026-05-31"}
    assert int(synth["Quantity"].sum()) == 15
    assert (synth["Transaction Type"] == "Refund").all()

"""Streaming quarterly aggregator."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import pandas as pd

from backend.services.po_quarterly_fast import _accumulate_shipment_frame


def test_accumulate_frame_counts_quarters():
    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025", (2025, 3): "Oct-Dec 2024"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-11-15"]),
            "SKU": ["1001YKBEIGE-M"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [40],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "amazon",
        None,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert quarter_sums[("1001YKBEIGE-M", "Oct-Dec 2024")] == 40

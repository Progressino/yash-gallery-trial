"""Amazon twin exports (root CSV + Sales-folder copy) must coalesce to one blob."""
import io

import pandas as pd

from backend.services.daily_store import _coalesce_tier3_upload_rows


def _mini_parquet(days_qty: list[tuple[str, int]], segment: str = "YG") -> bytes:
    rows = []
    for day, qty in days_qty:
        rows.append(
            {
                "Date": day,
                "Transaction_Type": "Shipment",
                "Quantity": qty,
                "DSR_Segment": segment,
            }
        )
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_coalesce_amazon_export_twins_drops_numeric_subset():
    blob_named = _mini_parquet(
        [("2026-06-18", 10), ("2026-06-19", 20), ("2026-06-20", 100), ("2026-06-21", 50)]
    )
    blob_numeric = _mini_parquet([("2026-06-20", 100), ("2026-06-21", 50)])
    rows = [
        ("984227020626.csv", blob_numeric),
        ("Sales 18-22-6-26/984527020627_YG Amazon 18-22-6-26.csv", blob_named),
    ]
    out = _coalesce_tier3_upload_rows(rows, platform="amazon")
    assert len(out) == 1
    assert "Sales" in out[0][0]


def test_coalesce_amazon_twin_with_one_growing_cutoff_day():
    """USA twin case: earlier export matches on all days except the cut-off day."""
    blob_numeric = _mini_parquet(
        [("2026-06-18", 33), ("2026-06-19", 30), ("2026-06-20", 26), ("2026-06-21", 28)]
    )
    blob_named = _mini_parquet(
        [
            ("2026-06-18", 33),
            ("2026-06-19", 30),
            ("2026-06-20", 26),
            ("2026-06-21", 32),
            ("2026-06-22", 36),
        ]
    )
    rows = [
        ("144406020626.csv", blob_numeric),
        ("Sales 18-22-6-26/144474020627_USA  Amazon18-22-6-26.csv", blob_named),
    ]
    out = _coalesce_tier3_upload_rows(rows, platform="amazon")
    assert len(out) == 1
    assert "Sales" in out[0][0]


def test_coalesce_amazon_keeps_distinct_accounts():
    yg = _mini_parquet([("2026-06-20", 100)], "YG")
    ak = _mini_parquet([("2026-06-20", 20)], "Akiko")
    out = _coalesce_tier3_upload_rows(
        [("a_YG Amazon.csv", yg), ("b_Akiko Amazon.csv", ak)],
        platform="amazon",
    )
    assert len(out) == 2

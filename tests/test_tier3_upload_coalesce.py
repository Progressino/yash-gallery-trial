"""Re-uploaded daily folders must not double-count the same inner CSV."""
import io

import pandas as pd

from backend.services.daily_store import _coalesce_tier3_upload_rows


def test_coalesce_tier3_upload_rows_keeps_latest_basename():
    blob_a = b"a"
    blob_b = b"b"
    rows = [
        ("Sales 10-jul-26/988826020645_YG Amazon 10-7-26.csv", blob_a),
        ("Sales 11-Jul-26/988826020645_YG Amazon 10-7-26.csv", blob_b),
    ]
    out = _coalesce_tier3_upload_rows(rows)
    assert len(out) == 1
    assert out[0][0].endswith("11-Jul-26/988826020645_YG Amazon 10-7-26.csv")
    assert out[0][1] == blob_b


def test_coalesce_distinct_basenames_kept():
    rows = [
        ("folder/a.csv", b"1"),
        ("folder/b.csv", b"2"),
    ]
    assert len(_coalesce_tier3_upload_rows(rows)) == 2

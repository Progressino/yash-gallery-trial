#!/usr/bin/env python3
"""Repair Myntra Tier-3: clamp multi-day seller blobs to filename window; drop single-day spill."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import (  # noqa: E402
    _df_to_parquet,
    _extract_date_range,
    _filter_myntra_tier3_upload_frame,
    _get_conn,
    _myntra_authoritative_single_days_from_filenames,
    clear_tier3_range_cache,
)
from backend.services.myntra import _myntra_seller_single_report_day  # noqa: E402


def main() -> int:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, data_parquet FROM daily_uploads WHERE platform='myntra'"
    ).fetchall()
    auth = _myntra_authoritative_single_days_from_filenames([r[1] for r in rows])
    updated = 0
    for row_id, filename, blob in rows:
        df = pd.read_parquet(io.BytesIO(blob))
        if df.empty:
            continue
        before = len(df)
        fixed = _filter_myntra_tier3_upload_frame(filename, df, auth_days=auth)
        if len(fixed) == before:
            continue
        date_from, date_to = _extract_date_range(fixed)
        conn.execute(
            "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=? WHERE id=?",
            (_df_to_parquet(fixed), len(fixed), date_from, date_to, row_id),
        )
        updated += 1
        tag = "single" if _myntra_seller_single_report_day(filename) else "multi"
        print(
            "updated",
            tag,
            filename.split("/")[-1][:72],
            before,
            "->",
            len(fixed),
        )
    conn.commit()
    conn.close()
    clear_tier3_range_cache()
    print("done", updated, "uploads", "auth_days", len(auth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

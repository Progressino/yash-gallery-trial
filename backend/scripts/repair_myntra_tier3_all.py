#!/usr/bin/env python3
"""
Repair all Myntra Tier-3 uploads in place (no per-file save_daily_file trim/side effects):
- Forward seller sales CSVs → every row Shipment (returns are separate RT files).
- Single-day Seller_Orders_Report → bucket entire file to the report day in the filename.
"""
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
    _get_conn,
    clear_tier3_range_cache,
)
from backend.services.myntra import _myntra_seller_single_report_day  # noqa: E402


def _is_return_upload(filename: str) -> bool:
    base = Path(str(filename).replace("\\", "/")).name
    return base.upper().startswith("RT ")


def main() -> int:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, data_parquet FROM daily_uploads WHERE platform='myntra'"
    ).fetchall()
    single = multi = other = updated = 0
    for row_id, filename, blob in rows:
        df = pd.read_parquet(io.BytesIO(blob))
        if df.empty or "Date" not in df.columns:
            continue
        changed = False
        df = df.copy()
        if not _is_return_upload(filename) and "TxnType" in df.columns:
            bad = df["TxnType"].astype(str).str.strip() != "Shipment"
            if bad.any():
                df.loc[bad.index, "TxnType"] = "Shipment"
                changed = True
        report_day = _myntra_seller_single_report_day(filename)
        if report_day is not None:
            single += 1
            before = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            target = pd.Timestamp(report_day).normalize()
            if not (before == target).all():
                df["Date"] = report_day
                changed = True
        elif "seller_orders_report" in filename.lower().replace("-", "_"):
            multi += 1
        else:
            other += 1
        if not changed:
            continue
        date_from, date_to = _extract_date_range(df)
        conn.execute(
            "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=? WHERE id=?",
            (_df_to_parquet(df), len(df), date_from, date_to, row_id),
        )
        updated += 1
        print("updated", filename.split("/")[-1][:72], "rows", len(df))
    conn.commit()
    conn.close()
    clear_tier3_range_cache()
    try:
        from backend.services.tier3_session_merge import invalidate_platform_build_cache

        invalidate_platform_build_cache()
    except Exception:
        pass
    print(
        "summary uploads",
        len(rows),
        "updated",
        updated,
        "single_day_seller",
        single,
        "multi_day_seller",
        multi,
        "other",
        other,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

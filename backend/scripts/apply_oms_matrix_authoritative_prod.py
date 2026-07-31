#!/usr/bin/env python3
"""Replace OMS inventory-history days from an authoritative wide matrix CSV.

Use when Inv History / Eff_Days diverge from the operator's source matrix.

  docker exec -e PYTHONPATH=/srv -w /srv progressino-backend-1 \\
    python backend/scripts/apply_oms_matrix_authoritative_prod.py \\
    /tmp/inventory-matrix-oms-2026-07-source-authoritative.csv

Only OMS channel rows for dates present in the file are replaced; Amazon and
non-overlapping days are preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: apply_oms_matrix_authoritative_prod.py <matrix.csv>", flush=True)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"Missing file: {path}", flush=True)
        return 2

    from io import BytesIO

    import pandas as pd

    from backend.services.daily_inventory_history import (
        merge_inventory_history_preserving_channels,
        parse_daily_inventory_history_upload,
        persist_inventory_history_authoritative,
    )
    from backend.session import AppSession

    incoming = parse_daily_inventory_history_upload(
        BytesIO(path.read_bytes()), path.name, sku_mapping=None
    )
    if incoming is None or incoming.empty:
        print("Parse produced no rows", flush=True)
        return 1
    incoming = incoming.copy()
    incoming["Source"] = "snapshot"
    incoming["Channel"] = "oms"
    print(
        f"Incoming: {len(incoming):,} rows, "
        f"dates {incoming['Date'].min().date()}..{incoming['Date'].max().date()}, "
        f"skus {incoming['OMS_SKU'].nunique():,}",
        flush=True,
    )

    hist_path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    if not hist_path.is_file():
        print(f"Missing {hist_path}", flush=True)
        return 1
    existing = pd.read_parquet(hist_path)
    print(f"Existing: {len(existing):,} rows", flush=True)

    merged = merge_inventory_history_preserving_channels(existing, incoming)
    print(f"Merged: {len(merged):,} rows", flush=True)

    sess = AppSession()
    sess.daily_inventory_history_df = merged
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        sess.daily_inventory_history_uploaded_at = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).isoformat(timespec="seconds")
    except Exception:
        from datetime import datetime, timezone

        sess.daily_inventory_history_uploaded_at = datetime.now(timezone.utc).isoformat()
    try:
        sess.daily_inventory_history_filename = path.name
    except Exception:
        pass
    # Force accept by clearing disk meta uploaded_at comparison path when needed
    ok = persist_inventory_history_authoritative(sess, merged)
    print(f"Persisted={ok}", flush=True)

    # Spot-check Jul 30 OMS total if present
    m = merged.copy()
    m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
    m["Channel"] = m.get("Channel", "oms").astype(str).str.lower()
    d30 = m[(m["Date"] == "2026-07-30") & (m["Channel"].isin(["oms", "", "nan"]))]
    if not d30.empty:
        qty = pd.to_numeric(d30["Qty"], errors="coerce").fillna(0)
        print(
            f"OMS 2026-07-30: skus={d30['OMS_SKU'].nunique()} qty_sum={float(qty.sum()):.0f} "
            f"instock={(qty > 0).sum()}",
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

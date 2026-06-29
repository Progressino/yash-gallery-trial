#!/usr/bin/env python3
"""Replace wrong Jun-27 raise ledger (full pipeline) with New Order qty only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import pandas as pd

    from backend.db import po_raised_db
    from backend.services.existing_po import (
        ensure_existing_po_hydrated,
        existing_po_pipeline_totals,
        persist_existing_po_to_disk,
    )
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_raise_import import (
        hydrate_session_ledger_from_db,
        seed_ledger_from_manual_existing_po_upload,
    )
    from backend.services.po_session_hydrate import hydrate_po_session_for_calculate
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.session import AppSession

    day = "2026-06-27"
    planning = "2026-06-29"

    po_raised_db.init_db()
    before = po_raised_db.ledger_rows_as_dataframe(start_date=day, end_date=day)
    before_units = (
        int(pd.to_numeric(before["Raised_Qty"], errors="coerce").fillna(0).sum())
        if before is not None and not before.empty
        else 0
    )
    print(f"BEFORE {day}: {before_units:,} units in DB", flush=True)

    po_raised_db.delete_raises_for_date(day)
    po_raised_db.clear_raise_date_suppression(day)

    sess = AppSession()
    hydrate_po_session_for_calculate(sess)
    ensure_existing_po_hydrated(sess)
    ep = getattr(sess, "existing_po_df", None)
    if ep is None or ep.empty:
        cache = Path("/data/warm_cache/existing_po_df.parquet")
        if cache.is_file():
            sess.existing_po_df = pd.read_parquet(cache)
            ep = sess.existing_po_df
    if ep is None or ep.empty:
        print("ERROR: no existing_po_df", flush=True)
        return 1

    ordered = int(pd.to_numeric(ep["PO_Qty_Ordered"], errors="coerce").fillna(0).sum())
    pipe_sum, pipe_skus = existing_po_pipeline_totals(ep)
    print(f"EXISTING PO: new_order={ordered:,} pipeline={pipe_sum:,} ({pipe_skus} skus)", flush=True)

    sess.existing_po_filename = str(getattr(sess, "existing_po_filename", "") or "Po 27-Jun-26.xlsx")
    sess.existing_po_manual_upload = True
    seed = seed_ledger_from_manual_existing_po_upload(
        sess, raised_date=pd.Timestamp(day), replace_day=True
    )
    print("SEED:", json.dumps(seed, indent=2, default=str), flush=True)

    hydrate_session_ledger_from_db(sess, planning, lookback_days=45, authoritative=True)
    after = po_raised_db.ledger_rows_as_dataframe(start_date=day, end_date=day)
    after_units = (
        int(pd.to_numeric(after["Raised_Qty"], errors="coerce").fillna(0).sum())
        if after is not None and not after.empty
        else 0
    )
    print(f"AFTER {day}: {after_units:,} units in DB ({len(after) if after is not None else 0} rows)", flush=True)

    try:
        persist_existing_po_to_disk(sess)
    except Exception:
        pass

    invalidate_all_shared_caches()

    body = {
        "period_days": 30,
        "lead_time": 60,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "use_ly_fallback": True,
        "group_by_parent": False,
        "min_denominator": 7,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 45,
        "use_shared_cache": False,
        "planning_date": planning,
    }
    res = execute_po_calculate(sess, body, session_id="reconcile-jun27")
    if not res.get("ok"):
        print("PO CALC FAILED:", res.get("message"), flush=True)
        return 1
    summary = res.get("summary") or {}
    print(
        "PO SUMMARY:",
        json.dumps(
            {
                "new_po_qty_sum": summary.get("new_po_qty_sum"),
                "new_po_sku_count": summary.get("new_po_sku_count"),
                "pipeline_qty_sum": summary.get("pipeline_qty_sum"),
                "sheet_po_ordered_sum": summary.get("sheet_po_ordered_sum"),
            },
            indent=2,
        ),
        flush=True,
    )

    out_dir = Path("/data/warm_cache/po_exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is not None and not po_df.empty:
        out_path = out_dir / f"po_recommendation_{planning}.csv"
        po_df.to_csv(out_path, index=False)
        pos = po_df[pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0) > 0]
        pos.to_csv(out_dir / f"po_recommendation_{planning}_positive_only.csv", index=False)
        print(f"WROTE {out_path} ({len(po_df)} rows, {int(pos['PO_Qty'].sum())} PO units)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Recover raise ledger from archived PO exports and re-run PO calculate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import json
    from pathlib import Path

    import pandas as pd

    from backend.services.existing_po import existing_po_pipeline_totals
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_raise_import import (
        bootstrap_all_archives_to_db,
        hydrate_session_ledger_from_db,
        seed_ledger_from_manual_existing_po_upload,
    )
    from backend.services.po_session_hydrate import hydrate_po_session_for_calculate
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.session import AppSession

    boot = bootstrap_all_archives_to_db()
    print("BOOTSTRAP:", json.dumps(boot, indent=2, default=str), flush=True)

    cache = Path("/data/warm_cache")
    sess = AppSession()
    hydrate_po_session_for_calculate(sess)
    for key in (
        "daily_inventory_history_df",
        "existing_po_df",
        "sales_df",
        "inventory_df_variant",
    ):
        path = cache / f"{key}.parquet"
        if path.is_file():
            setattr(sess, key, pd.read_parquet(path))
    inv_meta = cache / "inventory_session_meta.json"
    if inv_meta.is_file():
        meta = json.loads(inv_meta.read_text(encoding="utf-8"))
        sess.inventory_snapshot_date = str(meta.get("inventory_snapshot_date") or "")
    ep_meta = cache / "existing_po_meta.json"
    if ep_meta.is_file():
        em = json.loads(ep_meta.read_text(encoding="utf-8"))
        sess.existing_po_filename = str(em.get("existing_po_filename") or "")
        sess.existing_po_generation = int(em.get("existing_po_generation") or 0)
    hydrate_session_ledger_from_db(sess, "2026-06-26", lookback_days=45, authoritative=True)

    sess.existing_po_manual_upload = True
    seed = seed_ledger_from_manual_existing_po_upload(sess, replace_day=True)
    print("SEED FROM EXISTING PO:", json.dumps(seed, indent=2, default=str), flush=True)
    if seed.get("ok"):
        hydrate_session_ledger_from_db(sess, "2026-06-26", lookback_days=45, authoritative=True)
        from backend.services.existing_po import persist_manual_raise_skus

        persist_manual_raise_skus(sess)
        meta_path = cache / "existing_po_meta.json"
        if meta_path.is_file():
            em = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            em = {}
        em.update(
            {
                "existing_po_manual_raise_date": str(getattr(sess, "existing_po_manual_raise_date", "") or ""),
                "existing_po_manual_raise_skus_count": len(
                    getattr(sess, "existing_po_manual_raise_skus", []) or []
                ),
                "existing_po_manual_upload": True,
            }
        )
        meta_path.write_text(json.dumps(em, indent=2), encoding="utf-8")

    pipe_sum, pipe_skus = existing_po_pipeline_totals(getattr(sess, "existing_po_df", None))
    print("PIPELINE FROM SHEET:", pipe_sum, "skus", pipe_skus, flush=True)

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
        "planning_date": "2026-06-26",
    }
    res = execute_po_calculate(sess, body, session_id="bootstrap-ledger")
    summary = res.get("summary") or {}
    print(
        "PO:",
        json.dumps(
            {
                "new_po_qty_sum": summary.get("new_po_qty_sum"),
                "pipeline_qty_sum": summary.get("pipeline_qty_sum"),
                "raise_ledger_rows": res.get("raise_ledger_rows"),
            },
            indent=2,
        ),
        flush=True,
    )

    led = getattr(sess, "po_raise_ledger_df", None)
    if led is not None and not led.empty:
        led = led.copy()
        led["Raised_Date"] = pd.to_datetime(led["Raised_Date"]).dt.normalize()
        j24 = led[led["Raised_Date"] == pd.Timestamp("2026-06-24")]
        print(
            "LEDGER Jun24:",
            len(j24),
            "units",
            int(pd.to_numeric(j24["Raised_Qty"], errors="coerce").fillna(0).sum()),
            flush=True,
        )

    new_po = int(summary.get("new_po_qty_sum") or 0)
    if boot.get("ledger_rows", 0) == 0 and not seed.get("ok"):
        print("WARN: no ledger rows recovered", flush=True)
    print("RESULT: new_po", new_po, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

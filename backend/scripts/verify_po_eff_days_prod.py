#!/usr/bin/env python3
"""One-shot PO verify on prod: Eff_Days channel + save ui_default shared cache."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import pandas as pd

    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_shared_cache import invalidate_all_shared_caches, save_shared_cache
    from backend.services.daily_inventory_history import (
        apply_daily_inventory_history_meta,
        effective_days_from_history,
        read_daily_inventory_history_disk_meta,
    )

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1
    main_mod._warm_cache = data
    main_mod._warm_cache_generation = max(int(getattr(main_mod, "_warm_cache_generation", 0) or 0), 2)

    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "warm_cache_copy_failed"}))
        return 1
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    meta = read_daily_inventory_history_disk_meta() or {}
    apply_daily_inventory_history_meta(sess, meta)
    hist_path = Path(main_mod._DISK_CACHE_DIR) / "daily_inventory_history_df.parquet"
    if hist_path.is_file():
        sess.daily_inventory_history_df = pd.read_parquet(hist_path)

    if sess.sales_df is None or sess.sales_df.empty:
        print(json.dumps({"ok": False, "error": "missing_sales"}))
        return 1
    if sess.inventory_df_variant is None or sess.inventory_df_variant.empty:
        print(json.dumps({"ok": False, "error": "missing_inventory"}))
        return 1

    hist = sess.daily_inventory_history_df
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.normalize()
    print(
        "hist_max",
        str(hist["Date"].max().date()),
        "rows",
        len(hist),
        "sales",
        len(sess.sales_df),
        flush=True,
    )
    assert str(hist["Date"].max().date()) == "2026-07-23"

    sku = "1072YKBLACK-4XL"
    start, end = pd.Timestamp("2026-06-24"), pd.Timestamp("2026-07-23")
    pre = {}
    for ch in ("combined", "oms", "amazon"):
        r = effective_days_from_history(hist, start, end, channel=ch)
        row = r[r["OMS_SKU"] == sku]
        pre[ch] = None if row.empty else int(row.iloc[0]["Eff_Days_Inventory"])
    print("pre_eff", sku, pre, flush=True)

    invalidate_all_shared_caches()
    body = {
        "period_days": 30,
        "lead_time": 45,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": False,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": False,
        "raise_ledger_lookback_days": 14,
        "inventory_history_channel": "combined",
        "use_shared_cache": False,
        "planning_date": "2026-07-25",
    }
    sid = f"po-verify-{uuid.uuid4().hex[:8]}"
    print("calculate_start", sid, flush=True)
    result = execute_po_calculate(sess, body, session_id=sid)
    if not result.get("ok"):
        print(json.dumps({"ok": False, "error": result.get("message")}))
        return 1
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        print(json.dumps({"ok": False, "error": "empty_po"}))
        return 1

    key = save_shared_cache(sess, body, po_df, result)
    eff_col = "Eff_Days_Inventory" if "Eff_Days_Inventory" in po_df.columns else "Eff_Days"
    sample = po_df[po_df["OMS_SKU"].astype(str) == sku]
    po_eff = None if sample.empty else int(float(sample.iloc[0].get(eff_col) or 0))
    print(
        json.dumps(
            {
                "ok": True,
                "cache_key": key,
                "total_rows": int(len(po_df)),
                "sku": sku,
                "pre_eff": pre,
                "po_eff_combined": po_eff,
                "hist_max": "2026-07-23",
            },
            indent=2,
        ),
        flush=True,
    )
    if po_eff is not None and pre.get("combined") is not None:
        # Allow 1-day window anchor drift
        assert abs(po_eff - pre["combined"]) <= 2, (po_eff, pre["combined"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

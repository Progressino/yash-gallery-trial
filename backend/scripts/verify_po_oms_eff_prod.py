#!/usr/bin/env python3
"""PO calculate on prod with OMS Eff_Days; verify against 413 list + sample sales."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd


def main() -> int:
    import backend.main as m
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_shared_cache import invalidate_all_shared_caches, save_shared_cache
    from backend.services.daily_inventory_history import (
        apply_daily_inventory_history_meta,
        effective_days_from_history,
        read_daily_inventory_history_disk_meta,
    )

    ok, data = m._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1
    m._warm_cache = data
    m._warm_cache_generation = max(int(getattr(m, "_warm_cache_generation", 0) or 0), 2)

    sess = AppSession()
    if not m._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "warm_cache_copy_failed"}))
        return 1
    m.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    meta = read_daily_inventory_history_disk_meta() or {}
    apply_daily_inventory_history_meta(sess, meta)
    hist_path = Path(m._DISK_CACHE_DIR) / "daily_inventory_history_df.parquet"
    sess.daily_inventory_history_df = pd.read_parquet(hist_path)
    hist = sess.daily_inventory_history_df
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.normalize()
    end = hist["Date"].max()
    start = end - pd.Timedelta(days=29)
    print("hist_window", str(start.date()), str(end.date()), flush=True)

    sku = "1024YKMUSTARD-4XL"
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
        "demand_basis": "Net",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": False,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": False,
        "raise_ledger_lookback_days": 14,
        "inventory_history_channel": "oms",
        "use_shared_cache": False,
        "use_ly_fallback": True,
    }
    sid = f"po-verify-oms-{uuid.uuid4().hex[:8]}"
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
    eff413 = pd.read_excel("/tmp/413_eff.xlsx")
    cols = [c for c in ("OMS_SKU", "Eff_Days", "Eff_Days_Inventory", "Jan-Mar 2025") if c in po_df.columns]
    mrg = eff413.merge(po_df[cols], on="OMS_SKU", how="left", suffixes=("_old", "_new"))
    eff_new = mrg["Eff_Days_new"] if "Eff_Days_new" in mrg.columns else mrg["Eff_Days"]
    match = int((eff_new == mrg["FSAD"]).sum())
    print("413 Eff match FSAD", match, "/", len(mrg), flush=True)
    print(
        mrg[["OMS_SKU", "Eff_Days_old", "FSAD"]].assign(Eff_Days_new=eff_new).head(8).to_string(index=False),
        flush=True,
    )

    samples = {}
    for s in ("1024YKMUSTARD-4XL", "1037YKBLUE-M", "1592YKBLUE-XL", "1057YKBLUE-3XL", "1338YKBLACK-5XL"):
        hit = po_df[po_df["OMS_SKU"].astype(str) == s]
        if hit.empty:
            samples[s] = None
            continue
        r = hit.iloc[0]
        samples[s] = {
            "Eff_Days": int(float(r.get("Eff_Days") or 0)),
            "Eff_Days_Inventory": int(float(r.get("Eff_Days_Inventory") or 0)),
            "Jan-Mar 2025": r.get("Jan-Mar 2025"),
        }
    print(
        json.dumps(
            {
                "ok": True,
                "cache_key": key,
                "total_rows": int(len(po_df)),
                "pre_eff": pre,
                "match_413": match,
                "samples": samples,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    assert match == len(mrg), (match, len(mrg))
    assert samples["1024YKMUSTARD-4XL"]["Eff_Days"] == 11
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

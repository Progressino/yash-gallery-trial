#!/usr/bin/env python3
"""In-container smoke: full_restore_session → PO calculate → print checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import backend.main as main_mod
    from backend.routers.data import _build_coverage_response, full_restore_session
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.session import AppSession

    sess = AppSession()
    _missing, steps, msg = full_restore_session(sess)
    cov = _build_coverage_response(sess)
    print("RESTORE:", msg, flush=True)
    print("STEPS:", steps, flush=True)
    summary = {k: getattr(cov, k, None) for k in (
        "inventory",
        "sales",
        "existing_po",
        "daily_inventory_history",
        "daily_inventory_history_rows",
        "daily_inventory_history_max_date",
        "daily_inventory_history_min_date",
        "existing_po_rows",
        "inventory_snapshot_date",
    )}
    print("COVERAGE:", json.dumps(summary, indent=2, default=str), flush=True)

    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    body = {
        "period_days": 30,
        "lead_time": 45,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "use_ly_fallback": True,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 45,
        "use_shared_cache": False,
    }
    res = execute_po_calculate(sess, body, session_id="prod-smoke-test")
    po = sess.po_calculate_result_df
    print(
        "PO:",
        json.dumps(
            {k: res.get(k) for k in ("ok", "message", "total_rows", "planning_date")},
            default=str,
        ),
        flush=True,
    )

    ok = True
    for sku in ("1001YKBEIGE-3XL", "1001YKBEIGE-5XL", "1001YKBEIGE-8XL"):
        row = po[po["OMS_SKU"].astype(str) == sku]
        if row.empty:
            print(sku, "NOT FOUND", flush=True)
            ok = False
            continue
        r = row.iloc[0]
        sample = {
            "Total_Inventory": int(r.get("Total_Inventory") or 0),
            "Eff_Days": float(r.get("Eff_Days") or 0),
            "Eff_Days_Inventory": int(r.get("Eff_Days_Inventory") or 0),
            "ADS": float(r.get("ADS") or 0),
            "PO_Pipeline_Total": int(r.get("PO_Pipeline_Total") or 0),
            "PO_Qty": int(r.get("PO_Qty") or 0),
        }
        print(sku, json.dumps(sample), flush=True)
        if sku == "1001YKBEIGE-5XL":
            eff = sample["Eff_Days"]
            eff_i = sample["Eff_Days_Inventory"]
            if eff_i >= 25 and eff < eff_i - 5:
                print(f"FAIL: 5XL Eff_Days {eff} << Eff_Days_Inventory {eff_i}", flush=True)
                ok = False

    hist = getattr(sess, "daily_inventory_history_df", None)
    if hist is not None and not hist.empty:
        import pandas as pd

        h = hist.copy()
        h["Date"] = pd.to_datetime(h["Date"])
        print(
            "SESSION HISTORY:",
            len(h),
            "skus",
            h.OMS_SKU.nunique(),
            "max",
            h.Date.max(),
            flush=True,
        )
    else:
        print("SESSION HISTORY: empty", flush=True)
        ok = False

    if not summary.get("existing_po"):
        print("FAIL: existing_po not loaded", flush=True)
        ok = False
    max_d = str(summary.get("daily_inventory_history_max_date") or "")
    if max_d and max_d < "2026-06-20":
        print(f"FAIL: history max_date too old ({max_d})", flush=True)
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

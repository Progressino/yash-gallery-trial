#!/usr/bin/env python3
"""One-shot PO Fresh default profile while API is stopped — fills shared cache."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_shared_cache import save_shared_cache

    # Match PO Fresh UI defaults (see run_po_calculate_production._PROFILES).
    body = {
        "period_days": 30,
        "lead_time": 60,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": True,
        "use_ly_fallback": True,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 45,
        "use_shared_cache": False,
    }

    print("loading warm cache…", flush=True)
    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1

    sales = data.get("sales_df")
    if sales is not None and not getattr(sales, "empty", True):
        for heavy in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
            if heavy in data:
                data[heavy] = pd.DataFrame()
                print("cleared", heavy, flush=True)

    main_mod._warm_cache = data
    main_mod._warm_cache_generation = max(int(getattr(main_mod, "_warm_cache_generation", 0) or 0), 2)

    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "copy_failed"}))
        return 1
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)
    print("sales", len(sess.sales_df), "inv", len(sess.inventory_df_variant), flush=True)

    sid = f"po-fresh-oneshot-{uuid.uuid4().hex[:8]}"
    print("execute_po_calculate (PO Fresh defaults)…", flush=True)
    result = execute_po_calculate(sess, body, session_id=sid)
    print("ok", result.get("ok"), "msg", str(result.get("message") or "")[:200], flush=True)
    if not result.get("ok"):
        print(json.dumps(result, default=str)[:2000])
        return 2
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        print("empty po")
        return 3
    key = save_shared_cache(sess, body, po_df, result)
    out = Path("/data/warm_cache/po_sold_result.csv")
    try:
        po_df.to_csv(out, index=False)
    except Exception:
        po_df.to_csv("/tmp/po_sold_result.csv", index=False)
    print(json.dumps({"ok": True, "cache_key": key, "rows": int(len(po_df))}, indent=2))
    print("ONESHOT_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

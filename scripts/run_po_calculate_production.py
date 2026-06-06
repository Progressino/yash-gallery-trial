#!/usr/bin/env python3
"""Run a full PO calculate on production using server warm-cache data.

Usage (inside backend container):
  python scripts/run_po_calculate_production.py

Invalidates shared PO cache, rebuilds from warm cache + sidecars, saves a fresh
shared result for all sessions, and prints sanity checks (incl. 1361YKBLUE-XL).
"""
from __future__ import annotations

import json
import re
import sys
import uuid

_DUP_SUFFIX_RE = re.compile(
    r"-(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL|7XL|8XL)-\1$"
)


def _body() -> dict:
    return {
        "period_days": 30,
        "lead_time": 45,
        "target_days": 135,
        "demand_basis": "Net",
        "use_seasonality": False,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": False,
        "enforce_lead_time_release_gate": False,
        "urgent_all_sizes_days": 45,
        "use_shared_cache": False,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 14,
    }


def main() -> int:
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_shared_cache import invalidate_all_shared_caches, save_shared_cache

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1

    main_mod._warm_cache = data
    main_mod._warm_cache_generation = max(int(getattr(main_mod, "_warm_cache_generation", 0) or 0), 2)

    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "warm_cache_copy_to_session_failed"}))
        return 1

    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    removed = invalidate_all_shared_caches()
    session_id = f"po-recalc-{uuid.uuid4().hex[:12]}"

    if sess.sales_df.empty:
        print(json.dumps({"ok": False, "error": "sales_df_empty"}))
        return 1
    if sess.inventory_df_variant.empty:
        print(json.dumps({"ok": False, "error": "inventory_df_variant_empty"}))
        return 1

    body = _body()
    result = execute_po_calculate(sess, body, session_id=session_id)
    if not result.get("ok"):
        print(json.dumps({"ok": False, "error": result.get("message") or "calculate_failed"}))
        return 1

    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        print(json.dumps({"ok": False, "error": "po_result_df_empty"}))
        return 1

    save_shared_cache(sess, body, po_df, result)

    skus = po_df["OMS_SKU"].astype(str).tolist()
    dup_suffix = [s for s in skus if _DUP_SUFFIX_RE.search(s)]
    yk1361 = po_df[po_df["OMS_SKU"].astype(str).str.contains("1361YKBLUE", na=False)]
    xl_rows = yk1361[yk1361["OMS_SKU"].astype(str) == "1361YKBLUE-XL"]

    out = {
        "ok": True,
        "shared_cache_invalidated": removed,
        "total_rows": int(len(po_df)),
        "duplicate_suffix_skus": dup_suffix[:20],
        "duplicate_suffix_count": len(dup_suffix),
        "sales_rows": int(len(sess.sales_df)),
        "inventory_rows": int(len(sess.inventory_df_variant)),
        "inv_history_rows": int(len(getattr(sess, "daily_inventory_history_df", []))),
        "existing_po_rows": int(len(getattr(sess, "existing_po_df", []))),
    }
    if not xl_rows.empty:
        xl = xl_rows.iloc[0]
        out["1361YKBLUE-XL"] = {
            "Eff_Days": float(xl.get("Eff_Days") or 0),
            "Eff_Days_Inventory": float(xl.get("Eff_Days_Inventory") or 0),
            "ADS": float(xl.get("ADS") or 0),
            "PO_Qty": int(xl.get("PO_Qty") or 0),
            "Gross_PO_Qty": int(xl.get("Gross_PO_Qty") or 0),
            "Net_Units": int(xl.get("Net_Units") or 0),
        }

    print(json.dumps(out, indent=2))

    if dup_suffix:
        print("FAIL: duplicate-suffix OMS_SKU rows remain", file=sys.stderr)
        return 2
    if xl_rows.empty:
        print("WARN: 1361YKBLUE-XL row missing", file=sys.stderr)
    elif float(xl_rows.iloc[0].get("Eff_Days") or 0) < 1:
        print("FAIL: 1361YKBLUE-XL Eff_Days still near zero", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

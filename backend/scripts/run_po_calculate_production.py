#!/usr/bin/env python3
"""Run full PO calculate on production and save shared cache for UI param profiles."""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))

_DUP_SUFFIX_RE = re.compile(
    r"-(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL|7XL|8XL)-\1$"
)

# Profiles that match PO Engine UI defaults / common operator settings.
_PROFILES: tuple[dict, ...] = (
    {
        "label": "ui_default_30",
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
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 14,
    },
    {
        "label": "po_fresh_default",
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
    },
    {
        "label": "no_lead_gate",
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
        "enforce_lead_time_release_gate": False,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": True,
        "raise_ledger_lookback_days": 14,
    },
)


def _run_profile(sess, body: dict, session_id: str) -> tuple[dict, "pd.DataFrame"]:
    from backend.services.po_calculate_run import execute_po_calculate

    result = execute_po_calculate(sess, body, session_id=session_id)
    if not result.get("ok"):
        raise RuntimeError(result.get("message") or "calculate_failed")
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        raise RuntimeError("po_result_df_empty")
    return result, po_df


def main() -> int:
    import pandas as pd

    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
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

    if sess.sales_df.empty or sess.inventory_df_variant.empty:
        print(json.dumps({"ok": False, "error": "missing_sales_or_inventory"}))
        return 1

    removed = invalidate_all_shared_caches()
    saved: list[dict] = []
    checks: dict = {}

    for profile in _PROFILES:
        body = {k: v for k, v in profile.items() if k != "label"}
        body["use_shared_cache"] = False
        sid = f"po-recalc-{profile.get('label', 'profile')}-{uuid.uuid4().hex[:8]}"
        result, po_df = _run_profile(sess, body, sid)
        key = save_shared_cache(sess, body, po_df, result)
        saved.append(
            {
                "label": profile.get("label", "profile"),
                "cache_key": key,
                "total_rows": int(len(po_df)),
            }
        )
        if profile.get("label") == "ui_default_30":
            skus = po_df["OMS_SKU"].astype(str).tolist()
            dups = [s for s in skus if _DUP_SUFFIX_RE.search(s)]
            xl = po_df[po_df["OMS_SKU"].astype(str) == "1361YKBLUE-XL"]
            x8 = po_df[po_df["OMS_SKU"].astype(str) == "1001YKBEIGE-8XL"]
            checks = {
                "duplicate_suffix_count": len(dups),
                "1361YKBLUE-XL": (
                    {
                        "Eff_Days": float(xl.iloc[0].get("Eff_Days") or 0),
                        "Eff_Days_Inventory": int(xl.iloc[0].get("Eff_Days_Inventory") or 0),
                        "Net_Units": int(xl.iloc[0].get("Net_Units") or 0),
                        "ADS": float(xl.iloc[0].get("ADS") or 0),
                        "PO_Qty": int(xl.iloc[0].get("PO_Qty") or 0),
                    }
                    if not xl.empty
                    else None
                ),
                "1001YKBEIGE-8XL": (
                    {
                        "Net_Units": int(x8.iloc[0].get("Net_Units") or 0),
                        "Eff_Days": float(x8.iloc[0].get("Eff_Days") or 0),
                        "Eff_Days_Inventory": int(x8.iloc[0].get("Eff_Days_Inventory") or 0),
                    }
                    if not x8.empty
                    else None
                ),
            }

    out = {
        "ok": True,
        "shared_cache_invalidated": removed,
        "profiles_saved": saved,
        "sales_rows": int(len(sess.sales_df)),
        "inventory_rows": int(len(sess.inventory_df_variant)),
        "checks": checks,
    }
    print(json.dumps(out, indent=2))

    xl = (checks.get("1361YKBLUE-XL") or {})
    if checks.get("duplicate_suffix_count", 0) > 0:
        print("FAIL: duplicate-suffix SKUs remain", file=sys.stderr)
        return 2
    if not xl or float(xl.get("Eff_Days") or 0) < 1:
        print("FAIL: 1361YKBLUE-XL Eff_Days still near zero", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verify PO inputs on prod: inventory channel totals, platform sales, snapshot gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))

import backend.main as main_mod  # noqa: E402
from backend.session import AppSession  # noqa: E402
from backend.services.inventory import recompute_inventory_totals  # noqa: E402
from backend.services.manual_intransit_sheet import ensure_manual_intransit_overlay_applied  # noqa: E402
from backend.services.po_pipeline import check_calculate_gate, prepare_po_snapshot  # noqa: E402


def _sum_col(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return -1
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def main() -> int:
    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_load_failed"}))
        return 1
    main_mod._warm_cache = data
    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "warm_copy_failed"}))
        return 1
    main_mod.restore_po_sidecars_from_warm(sess)

    ensure_manual_intransit_overlay_applied(sess)
    inv = recompute_inventory_totals(sess.inventory_df_variant.copy())
    inv_totals = {
        "OMS_Inventory": _sum_col(inv, "OMS_Inventory"),
        "Amazon_Inventory": _sum_col(inv, "Amazon_Inventory"),
        "Flipkart_Inventory": _sum_col(inv, "Flipkart_Inventory"),
        "Myntra_Other_Inventory": _sum_col(inv, "Myntra_Other_Inventory"),
        "Myntra_YG_Inventory": _sum_col(inv, "Myntra_YG_Inventory"),
        "Total_Inventory": _sum_col(inv, "Total_Inventory"),
        "variant_rows": len(inv),
    }

    sales_info: dict = {"rows": len(sess.sales_df)}
    sales = sess.sales_df
    if not sales.empty and "TxnDate" in sales.columns:
        s = sales.copy()
        s["TxnDate"] = pd.to_datetime(s["TxnDate"], errors="coerce")
        end = s["TxnDate"].max()
        start = end - pd.Timedelta(days=30)
        sub = s[s["TxnDate"] >= start]
        sales_info["max_date"] = str(end.date())[:10]
        sales_info["rows_30d"] = len(sub)
        if "Source" in sub.columns and "Units_Effective" in sub.columns:
            sales_info["units_30d_by_source"] = {
                str(k): float(v)
                for k, v in sub.groupby(sub["Source"].astype(str))["Units_Effective"]
                .sum()
                .items()
            }

    body = {
        "period_days": 30,
        "lead_time": 60,
        "target_days": 180,
        "demand_basis": "Sold",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "use_ly_fallback": True,
        "min_denominator": 7,
    }
    gate = check_calculate_gate(sess)
    snap = prepare_po_snapshot(sess, body, progress_cb=None)
    po_inv_total = _sum_col(snap.inv_df, "Total_Inventory") if not snap.inv_df.empty else 0

    out = {
        "ok": True,
        "inventory": inv_totals,
        "sales": sales_info,
        "gate": {
            "calculate_allowed": gate.get("calculate_allowed"),
            "blockers": gate.get("blockers") or [],
            "warnings": (gate.get("warnings") or [])[:8],
        },
        "snapshot": {
            "ads_label": snap.ads_label,
            "ads_rows": len(snap.sales_df),
            "inv_rows": len(snap.inv_df),
            "po_inv_total": po_inv_total,
            "snapshot_blockers": snap.blockers,
        },
    }
    print(json.dumps(out, indent=2))

    # Expected ops-sheet inventory (21-Jul-26) — allow small drift from overlay/history.
    expected = {
        "OMS_Inventory": 172115,
        "Amazon_Inventory": 40744,
        "Flipkart_Inventory": 412,
    }
    myntra = inv_totals.get("Myntra_Other_Inventory", 0) + max(
        0, inv_totals.get("Myntra_YG_Inventory", 0)
    )
    if myntra >= 0:
        expected_myntra = 2198
        if abs(myntra - expected_myntra) > 50:
            print(
                f"WARN: Myntra inventory sum {myntra} vs expected ~{expected_myntra}",
                file=sys.stderr,
            )
    for k, exp in expected.items():
        got = inv_totals.get(k, -1)
        if got >= 0 and abs(got - exp) > 100:
            print(f"WARN: {k} {got} vs expected {exp}", file=sys.stderr)

    if not gate.get("calculate_allowed"):
        return 2
    if snap.blockers:
        return 3
    if sales_info.get("rows", 0) < 1000:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

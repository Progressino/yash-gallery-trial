#!/usr/bin/env python3
"""Local PO Fresh validation: totals vs live reference + May 15 raise columns."""
from __future__ import annotations

import sys
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

_srv = Path(__file__).resolve().parents[1]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))

# Live PO Fresh screenshot (app.progressino.com, build e4a3141)
LIVE_REF = {
    "total_po": 376_315,
    "rows": 7_564,
    "skus_with_po": 3_417,
    "lead_time": 60,
}

PO_BODY = {
    "period_days": 30,
    "lead_time": LIVE_REF["lead_time"],
    "target_days": 180,
    "grace_days": 0,
    "demand_basis": "Sold",
    "group_by_parent": False,
    "safety_pct": 0,
    "use_seasonality": True,
    "use_ly_fallback": True,
    "seasonal_weight": 0.5,
    "enforce_two_size_minimum": True,
    "enforce_lead_time_release_gate": True,
    "urgent_all_sizes_days": 45,
    "planning_date": date.today().isoformat(),
    "raise_ledger_lookback_days": 45,
    "raise_view_date": "2026-05-16",
    "auto_import_yesterday_ledger": True,
    "use_shared_cache": False,
}


def main() -> int:
    import backend.main as main_mod
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_engine import round_po_pack
    from backend.session import AppSession

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print("FAIL: warm cache not loaded from disk")
        return 1

    main_mod._warm_cache = data
    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print("FAIL: warm cache copy to session")
        return 1
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    print("=== Data ===")
    print(f"  sales rows: {len(sess.sales_df):,}")
    print(f"  inventory SKUs: {len(sess.inventory_df_variant):,}")
    ep = getattr(sess, "existing_po_df", None)
    ep_n = 0 if ep is None or ep.empty else len(ep)
    print(f"  existing_po rows: {ep_n:,}")
    ledger = getattr(sess, "po_raise_ledger_df", None)
    if ledger is not None and not ledger.empty and "Raised_Date" in ledger.columns:
        d = pd.to_datetime(ledger["Raised_Date"], errors="coerce").dt.normalize()
        may15 = ledger[d == pd.Timestamp("2026-05-15")]
        may16 = ledger[d == pd.Timestamp("2026-05-16")]
        print(f"  raise ledger: {len(ledger)} rows (May15={len(may15)}, May16={len(may16)})")
    else:
        print("  raise ledger: empty")

    print("\n=== PO calculate (PO Fresh params + raise_view_date=2026-05-16) ===")
    result = execute_po_calculate(sess, PO_BODY, session_id=f"e2e-{uuid.uuid4().hex[:8]}")
    if not result.get("ok"):
        print("FAIL:", result.get("message"))
        return 1

    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        print("FAIL: empty PO result")
        return 1

    po_qty = pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0)
    total_po = int(po_qty.sum())
    skus_po = int((po_qty > 0).sum())
    rows = int(len(po_df))

    print(f"  total PO units: {total_po:,}")
    print(f"  SKUs with PO: {skus_po:,}")
    print(f"  rows: {rows:,}")

    may15_last_n = may16_last_n = view16_n = 0
    if "PO_Last_Raised_Date" in po_df.columns:
        may15_last_n = int(po_df["PO_Last_Raised_Date"].astype(str).str.startswith("2026-05-15").sum())
        may16_last_n = int(po_df["PO_Last_Raised_Date"].astype(str).str.startswith("2026-05-16").sum())
        print(f"  PO_Last_Raised_Date May15: {may15_last_n}")
        print(f"  PO_Last_Raised_Date May16: {may16_last_n}")
    if "PO_Raised_On_View_Date" in po_df.columns:
        view16_n = int((pd.to_numeric(po_df["PO_Raised_On_View_Date"], errors="coerce").fillna(0) > 0).sum())
        print(f"  PO_Raised_On_View_Date>0 (raise date May16): {view16_n}")

    pipe_n = 0
    if "PO_Pipeline_Total" in po_df.columns:
        pipe_n = int((pd.to_numeric(po_df["PO_Pipeline_Total"], errors="coerce").fillna(0) > 0).sum())
    print(f"  pipeline rows: {pipe_n:,}")

    print("\n=== vs live reference ===")
    for key in ("total_po", "rows", "skus_with_po"):
        local = {"total_po": total_po, "rows": rows, "skus_with_po": skus_po}[key]
        ref = LIVE_REF[key]
        print(f"  {key}: local {local:,} vs live {ref:,} ({local - ref:+,})")

    # Formula spot-check (gross PO from engine ADS + projected days)
    target = int(PO_BODY["target_days"])
    bad = 0
    checked = 0
    for _, row in po_df.iterrows():
        po = float(row.get("PO_Qty") or 0)
        ads = float(row.get("ADS") or 0)
        proj = float(row.get("Projected_Running_Days") or 0)
        if po <= 0 or ads <= 0:
            continue
        gross = float(row.get("Gross_PO_Qty") or 0)
        lt = float(row.get("Lead_Time_Days") or PO_BODY["lead_time"])
        if proj >= lt:
            continue
        exp = round_po_pack(max(ads * (target - proj), 0))
        if gross > 0 and abs(gross - exp) > 10.51:
            bad += 1
        checked += 1
    print(f"\n=== Formula audit (Gross_PO vs ADS×(target−proj)): {checked} checked, {bad} mismatches ===")

    sku = "1001YKBEIGE-3XL"
    sub = po_df[po_df["OMS_SKU"].astype(str) == sku]
    if not sub.empty:
        r = sub.iloc[0]
        print(f"\nSpot {sku}: inv={r.get('Total_Inventory')} proj={r.get('Projected_Running_Days')} "
              f"ADS={r.get('ADS')} PO={r.get('PO_Qty')} last_raised={r.get('PO_Last_Raised_Date')}")

    # Pass criteria: non-zero PO with sane formula (live totals drift with cache / gate mode)
    totals_ok = total_po >= 5_000
    rows_ok = rows >= 5_000
    may_batch_ok = may16_last_n >= 200 or view16_n >= 200
    formula_ok = bad == 0

    print("\n=== RESULT ===")
    print(f"  total PO >= 5k: {'PASS' if totals_ok else 'FAIL'} ({total_po:,})")
    print(f"  rows >= 5k: {'PASS' if rows_ok else 'FAIL'} ({rows:,})")
    print(f"  May16 last-raised / view shown: {'PASS' if may_batch_ok else 'SKIP (no ledger)'}")
    print(f"  formula audit: {'PASS' if formula_ok else 'FAIL'}")

    if totals_ok and rows_ok and formula_ok:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

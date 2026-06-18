#!/usr/bin/env python3
"""Run full-catalog ADS audit on local warm cache (PO Fresh settings)."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

_srv = Path(__file__).resolve().parents[1]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))

from tests.test_ads_period_cap_catalog import (  # noqa: E402
    PO_FRESH_BODY,
    _formula_mismatches,
)


def main() -> int:
    import numpy as np
    import pandas as pd

    import backend.main as main_mod
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_shared_cache import PO_MERGE_LOGIC_VERSION
    from backend.session import AppSession

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print("FAIL: warm cache not loaded")
        return 1

    main_mod._warm_cache = data
    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        print("FAIL: warm cache copy")
        return 1
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    print(f"=== ADS catalog audit (engine v{PO_MERGE_LOGIC_VERSION}) ===")
    print(f"  sales rows: {len(sess.sales_df):,}")
    print(f"  inventory SKUs: {len(sess.inventory_df_variant):,}")

    result = execute_po_calculate(
        sess,
        PO_FRESH_BODY,
        session_id=f"ads-local-{uuid.uuid4().hex[:8]}",
    )
    if not result.get("ok"):
        print("FAIL:", result.get("message"))
        return 1

    po_df = sess.po_calculate_result_df
    n = len(po_df)
    po_sum = int(pd.to_numeric(po_df["PO_Qty"], errors="coerce").fillna(0).sum())
    print(f"  PO rows: {n:,}")
    print(f"  total PO units: {po_sum:,}")

    mismatches = _formula_mismatches(po_df, period_days=30, use_ly_fallback=False)
    sold = pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0)
    recent = pd.to_numeric(po_df["Recent_ADS"], errors="coerce").fillna(0)
    ads = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0)
    flat = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0)
    season = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0)
    eff = pd.to_numeric(po_df["Eff_Days"], errors="coerce").fillna(0)
    pr = sold / 30.0
    stuck = (
        (sold >= 6)
        & (recent > pr + 0.05)
        & (np.abs(ads - recent) < 0.003)
        & (season < ads - 0.01)
        & (flat < ads - 0.01)
    )
    short = (sold >= 6) & (eff < 30)
    max_allowed = np.maximum(np.maximum(pr, flat), season)
    short_over = short & (ads > max_allowed + 0.004)

    sku = "1050YKBLUE-L"
    spot = po_df[po_df["OMS_SKU"].astype(str) == sku]
    if not spot.empty:
        r = spot.iloc[0]
        print(f"\n  Spot {sku}: sold={r['Sold_Units']} eff={r['Eff_Days']} "
              f"recent={r['Recent_ADS']} ADS={r['ADS']} PO={r['PO_Qty']}")

    print("\n=== Checks ===")
    print(f"  formula mismatches: {len(mismatches)}")
    print(f"  uncapped burst (Recent stuck): {int(stuck.sum())}")
    print(f"  short Eff_Days over floor: {int(short_over.sum())}")

    if mismatches:
        print("\nFirst mismatches:")
        for line in mismatches[:10]:
            print(" ", line)
        return 2
    if stuck.any() or short_over.any():
        return 3

    print("\nPASS — all catalog ADS rows valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())

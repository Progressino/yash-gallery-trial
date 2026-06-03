#!/usr/bin/env python3
"""Run inside production backend container to verify June Tier-3 → Intelligence path."""
from __future__ import annotations

import json
import sys

from backend.services.daily_store import (
    get_summary,
    load_platform_data_for_report_range,
    platforms_with_uploads_in_range,
)
from backend.services.sales import _compute_platform_metrics


def main() -> int:
    start, end = "2026-06-01", "2026-06-04"
    uploaded = platforms_with_uploads_in_range(start, end)
    summary = get_summary()
    out = {
        "window": [start, end],
        "platforms_with_uploads": uploaded,
        "tier3_summary": summary,
        "metrics": {},
    }
    specs = {
        "amazon": ("Amazon", "Date", "SKU", "Transaction_Type"),
        "meesho": ("Meesho", "Date", "OMS_SKU", "TxnType"),
    }
    for plat in uploaded:
        df = load_platform_data_for_report_range(plat, start, end, dedup=False)
        if plat in specs:
            name, dc, sku, txn = specs[plat]
            m = _compute_platform_metrics(df, name, sku, txn, start_date=start, end_date=end)
            out["metrics"][plat] = {
                "rows": len(df),
                "total_units": m.get("total_units"),
                "loaded": m.get("loaded"),
            }
        else:
            out["metrics"][plat] = {"rows": len(df)}
    print(json.dumps(out, indent=2))
    return 0 if uploaded and any(
        (out["metrics"].get(p) or {}).get("total_units", 0) > 0 for p in uploaded
    ) else 1


if __name__ == "__main__":
    sys.exit(main())

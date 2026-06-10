#!/usr/bin/env python3
"""Deep audit of a PO Engine CSV export for client-ready justification."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_DAYS = 135
PACK = 5


def _expected_po(row: pd.Series) -> int:
    inv = float(row.get("Total_Inventory", 0) or 0)
    pipe = float(row.get("PO_Pipeline_Effective", row.get("PO_Pipeline_Total", 0)) or 0)
    ads = float(row.get("ADS", 0) or 0)
    overlay = int(row.get("Return_Overlay_Units", 0) or 0)
    if ads <= 0:
        return 0
    proj = (inv + pipe) / ads
    raw = ads * max(0.0, TARGET_DAYS - proj)
    gross = int(np.floor(np.ceil(max(raw, 0.0) / PACK) * PACK))
    return max(0, gross - overlay)


def audit(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.columns[0].startswith("\ufeff"):
        df.columns = [c.lstrip("\ufeff") for c in df.columns]

    hot = df[df["PO_Qty"] > 0].copy()
    hot["exp_po"] = hot.apply(_expected_po, axis=1)
    formula_bad = hot[hot["exp_po"] != hot["PO_Qty"]]

    closed_po = hot[
        hot["SKU_Sheet_Status"].astype(str).str.contains("Closed|closed", na=False, regex=True)
    ]

    zero_sold_high = hot[(hot["Sold_Units"] == 0) & (hot["PO_Qty"] >= 300)]

    post = np.where(
        hot["ADS"] > 0,
        ((hot["Total_Inventory"] + hot["PO_Pipeline_Effective"] + hot["PO_Qty"]) / hot["ADS"]).round(1),
        999,
    )

    return {
        "file": str(path),
        "rows": int(len(df)),
        "po_skus": int(len(hot)),
        "total_po_qty": int(hot["PO_Qty"].sum()),
        "priority": df["Priority"].value_counts().to_dict(),
        "formula_mismatches": int(len(formula_bad)),
        "closed_status_with_po": {
            "skus": int(len(closed_po)),
            "units": int(closed_po["PO_Qty"].sum()) if len(closed_po) else 0,
            "note": "Sales After Closed is informational — not auto-blocked unless SKU_Sheet_Closed",
        },
        "zero_period_sales_po_300plus": zero_sold_high[
            ["OMS_SKU", "PO_Qty", "ADS", "Recent_ADS", "Ship_Units_150d", "PO_Pipeline_Total"]
        ].head(10).to_dict("records"),
        "post_cover_days": {
            "median": float(np.median(post)),
            "pct_at_least_135": float((post >= 134.5).mean() * 100),
            "pct_over_150": float((post > 150).mean() * 100),
        },
        "pipeline_skus": int((df["PO_Pipeline_Total"] > 0).sum()),
        "pipeline_units": int(pd.to_numeric(df["PO_Pipeline_Total"], errors="coerce").fillna(0).sum()),
        "top_po": hot.nlargest(10, "PO_Qty")[
            ["OMS_SKU", "PO_Qty", "Total_Inventory", "ADS", "PO_Pipeline_Total", "SKU_Sheet_Status"]
        ].to_dict("records"),
        "formula_ok": len(formula_bad) == 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    args = p.parse_args()
    out = audit(args.csv)
    print(json.dumps(out, indent=2))
    return 0 if out["formula_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

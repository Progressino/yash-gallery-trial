#!/usr/bin/env python3
"""Verify OMS Inv History July matrix + Eff_Days against source fixtures on prod."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _oms(h: pd.DataFrame) -> pd.DataFrame:
    out = h.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    ch = (
        out["Channel"].astype(str).str.lower()
        if "Channel" in out.columns
        else pd.Series(["oms"] * len(out), index=out.index)
    )
    return out[ch.isin(["oms", "", "nan", "none"])].copy()


def main() -> int:
    matrix = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/tmp/inventory-matrix-oms-2026-07-source-authoritative.csv"
    )
    mismatch = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/inv-mismatch-jul31.csv")
    hist_path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    status_path = Path("/data/warm_cache/sku_status_lead_df.parquet")

    if not hist_path.is_file():
        print(f"MISSING {hist_path}", flush=True)
        return 1
    if not matrix.is_file():
        print(f"MISSING {matrix}", flush=True)
        return 1

    h = pd.read_parquet(hist_path)
    oms = _oms(h)
    print(
        "range",
        oms["Date"].min().date(),
        oms["Date"].max().date(),
        "rows",
        len(oms),
        flush=True,
    )

    for d in ["2026-07-16", "2026-07-30", "2026-07-31"]:
        sub = oms[oms["Date"] == d]
        qty = pd.to_numeric(sub["Qty"], errors="coerce").fillna(0)
        print(
            d,
            "skus",
            int(sub["OMS_SKU"].nunique()),
            "qty",
            float(qty.sum()),
            "instock",
            int((qty > 0).sum()),
            "max",
            float(qty.max()) if len(qty) else 0.0,
            flush=True,
        )

    yr = oms[
        (oms["OMS_SKU"].astype(str).str.upper().str.startswith("YR02"))
        & (oms["Date"] == "2026-07-16")
    ]
    if not yr.empty:
        bad = yr[pd.to_numeric(yr["Qty"], errors="coerce").fillna(0) > 10000]
        print(
            "YR Jul16 rows",
            len(yr),
            "corrupt>10k",
            len(bad),
            flush=True,
        )
        if len(bad):
            print(bad[["OMS_SKU", "Qty"]].head(10).to_string(index=False), flush=True)
            print("FAIL: Jul16 YR corruption still present", flush=True)
            return 1
    else:
        print("YR021-027 Jul16 gone or not in OMS channel", flush=True)

    src = pd.read_csv(matrix)
    src["OMS_SKU"] = src["OMS_SKU"].astype(str).str.strip().str.upper()
    src30 = pd.to_numeric(src["2026-07-30"], errors="coerce").fillna(0)
    print("source Jul30 total", float(src30.sum()), flush=True)

    d30 = oms[oms["Date"] == "2026-07-30"]
    got = d30.groupby(d30["OMS_SKU"].astype(str).str.upper())["Qty"].sum()
    exp = pd.to_numeric(src.set_index("OMS_SKU")["2026-07-30"], errors="coerce").fillna(0)
    mism = phant = 0
    for sku in set(got.index) | set(exp.index):
        a = float(exp.get(sku, 0) or 0)
        b = float(got.get(sku, 0) or 0)
        if abs(a - b) > 0.5:
            mism += 1
            if b > 0 and a == 0:
                phant += 1
    print("Jul30 SKU mismatches vs source", mism, "phantoms", phant, flush=True)
    if mism > 50:
        print(f"FAIL: too many Jul30 mismatches: {mism}", flush=True)
        return 1

    # Eff_Days vs operator mismatch sheet (Unnamed:32 = expected from source)
    if mismatch.is_file():
        from backend.services.daily_inventory_history import effective_days_from_history

        m = pd.read_csv(mismatch)
        m = m[m["SKU"].astype(str).str.upper() != "TOTAL INV."].copy()
        m["SKU"] = m["SKU"].astype(str).str.strip().str.upper()
        exp_eff = pd.to_numeric(m.get("Unnamed: 32"), errors="coerce")
        flagged = m[exp_eff.notna()].copy()
        flagged["exp_eff"] = exp_eff[exp_eff.notna()].astype(int)
        start = pd.Timestamp("2026-07-02")
        end = pd.Timestamp("2026-07-31")
        eff = effective_days_from_history(h, start, end, channel="oms")
        eff["OMS_SKU"] = eff["OMS_SKU"].astype(str).str.strip().str.upper()
        joined = flagged.merge(
            eff.rename(columns={"OMS_SKU": "SKU", "Eff_Days_Inventory": "got_eff"}),
            on="SKU",
            how="left",
        )
        joined["got_eff"] = pd.to_numeric(joined["got_eff"], errors="coerce").fillna(0).astype(int)
        bad_eff = joined[joined["exp_eff"] != joined["got_eff"]]
        print(
            "Eff_Days compare rows",
            len(joined),
            "mismatches",
            len(bad_eff),
            "avg_abs_diff",
            float((joined["exp_eff"] - joined["got_eff"]).abs().mean()) if len(joined) else 0,
            flush=True,
        )
        if len(bad_eff):
            print(bad_eff[["SKU", "exp_eff", "got_eff"]].head(15).to_string(index=False), flush=True)
        # After source matrix apply, expect near-zero mismatches on flagged set
        if len(bad_eff) > 50:
            print(f"FAIL: Eff_Days mismatches still high: {len(bad_eff)}", flush=True)
            return 1
    else:
        print(f"SKIP Eff_Days sheet (missing {mismatch})", flush=True)

    if status_path.is_file():
        st = pd.read_parquet(status_path)
        print("status_rows", len(st), "cols", list(st.columns), flush=True)
        col = "SKU_Sheet_Status" if "SKU_Sheet_Status" in st.columns else None
        if col:
            print(st[col].astype(str).value_counts().head(10).to_string(), flush=True)
        if len(st) < 100:
            print("FAIL: status sheet too small", flush=True)
            return 1
    else:
        print(f"WARN missing {status_path}", flush=True)

    print("VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

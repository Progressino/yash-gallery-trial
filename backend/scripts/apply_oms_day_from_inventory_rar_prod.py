#!/usr/bin/env python3
"""Replace one OMS inventory-history day from a daily Inventory RAR / OMS CSV.

Example (on VPS backend container):

  python backend/scripts/apply_oms_day_from_inventory_rar_prod.py \\
    /tmp/Inventory_27-Aug-26.rar --date 2026-08-27

Parses the RAR with the same loader as daily inventory upload, writes OMS
channel rows for that date (Amazon preserved), refreshes the day snapshot,
and invalidates PO caches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Inventory RAR or OMS CSV path")
    p.add_argument("--date", required=True, help="Snapshot date YYYY-MM-DD (e.g. 2026-08-27)")
    args = p.parse_args(argv)
    path = Path(args.path)
    day = str(args.date).strip()[:10]
    if not path.is_file():
        print(f"Missing file: {path}", flush=True)
        return 2

    import pandas as pd

    from backend.services.daily_inventory_history import (
        archive_inventory_day_snapshot,
        coalesce_inventory_history_sku_aliases,
        merge_inventory_history_preserving_channels,
        persist_inventory_history_authoritative,
        repair_inventory_history_integrity,
    )
    from backend.services.inventory import load_inventory_consolidated
    from backend.services.sku_mapping import load_sku_mapping_from_disk
    from backend.session import AppSession

    raw = path.read_bytes()
    name = path.name.lower()
    mapping = load_sku_mapping_from_disk() or {}
    if name.endswith((".rar", ".zip")):
        # Daily Inventory RAR is passed as the archive argument (same as UI upload).
        result = load_inventory_consolidated(
            None, None, None, raw, mapping, group_by_parent=False, return_debug=True
        )
    else:
        result = load_inventory_consolidated(
            raw, None, None, None, mapping, group_by_parent=False, return_debug=True
        )
    if isinstance(result, tuple):
        variant, debug = result
    else:
        variant, debug = result, {}
    if variant is None or getattr(variant, "empty", True):
        print(f"Parse produced no inventory rows: {debug}", flush=True)
        return 1
    print(f"Parsed variant rows={len(variant)} debug={debug}", flush=True)

    qty_col = "OMS_Inventory" if "OMS_Inventory" in variant.columns else (
        "Total_Inventory" if "Total_Inventory" in variant.columns else None
    )
    if not qty_col or "OMS_SKU" not in variant.columns:
        print("Missing OMS_SKU / OMS_Inventory columns", flush=True)
        return 1

    work = variant[["OMS_SKU", qty_col]].copy()
    work["OMS_SKU"] = work["OMS_SKU"].astype(str).str.strip().str.upper()
    work["Qty"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work[work["OMS_SKU"].astype(str).str.len() > 0]
    work = work.groupby("OMS_SKU", as_index=False)["Qty"].sum()
    incoming = pd.DataFrame(
        {
            "OMS_SKU": work["OMS_SKU"],
            "Date": pd.to_datetime(day),
            "Channel": "oms",
            "Qty": work["Qty"],
            "Source": "snapshot",
        }
    )
    print(
        f"Incoming OMS {day}: skus={incoming['OMS_SKU'].nunique()} "
        f"qty_sum={float(incoming['Qty'].sum()):.0f}",
        flush=True,
    )

    hist_path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    if not hist_path.is_file():
        print(f"Missing {hist_path}", flush=True)
        return 1
    existing = pd.read_parquet(hist_path)
    merged = merge_inventory_history_preserving_channels(existing, incoming)
    merged = coalesce_inventory_history_sku_aliases(
        merged, mapping
    )
    merged, report = repair_inventory_history_integrity(merged, persist_report=False)
    print(f"Merged rows={len(merged)} repair={report.get('actions')}", flush=True)

    sess = AppSession()
    sess.daily_inventory_history_df = merged
    sess.inventory_df_variant = variant
    try:
        sess.inventory_snapshot_date = day
        sess.inventory_snapshot_date_label = day
    except Exception:
        pass
    ok = persist_inventory_history_authoritative(sess, merged, force=True)
    print(f"Persisted history={ok}", flush=True)
    try:
        snap = archive_inventory_day_snapshot(variant, day)
        print(f"Day snapshot={snap}", flush=True)
    except Exception as exc:
        print(f"Day snapshot skipped: {exc}", flush=True)
    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        invalidate_all_shared_caches()
        print("PO shared caches invalidated", flush=True)
    except Exception as exc:
        print(f"PO cache invalidate skipped: {exc}", flush=True)

    # Spot-check
    m = merged.copy()
    m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
    m["Channel"] = m.get("Channel", "oms").astype(str).str.lower()
    d = m[(m["Date"] == day) & (m["Channel"].isin(["oms", "", "nan"]))]
    print(
        f"Verify OMS {day}: skus={d['OMS_SKU'].nunique()} "
        f"qty_sum={float(pd.to_numeric(d['Qty'], errors='coerce').fillna(0).sum()):.0f}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

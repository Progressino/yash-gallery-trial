#!/usr/bin/env python3
"""Full prod check: merge SKU map, Inv History aliases, run PO calculate, assert core SKUs."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd


def main() -> int:
    import backend.main as m
    from backend.session import AppSession
    from backend.services.daily_inventory_history import (
        coalesce_inventory_history_sku_aliases,
        combine_inventory_channels,
        filter_inventory_history_channel,
        persist_inventory_history_authoritative,
        repair_inventory_history_integrity,
    )
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate
    from backend.services.po_engine import canonical_oms_key
    from backend.services.po_shared_cache import invalidate_all_shared_caches, save_shared_cache
    from backend.services.sku_mapping import (
        ensure_sku_mapping_merged_globally,
        load_bundled_sku_mapping,
        load_sku_mapping_from_disk,
    )

    ok, data = m._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1
    m._warm_cache = data
    m._warm_cache_generation = max(int(getattr(m, "_warm_cache_generation", 0) or 0), 2)

    sess = AppSession()
    if not m._copy_warm_cache_to_session(sess):
        print(json.dumps({"ok": False, "error": "warm_cache_copy_failed"}))
        return 1
    m.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)

    bundled_n = len(load_bundled_sku_mapping())
    before_disk = len(load_sku_mapping_from_disk() or {})
    merged_map = ensure_sku_mapping_merged_globally(sess)
    after_disk = len(load_sku_mapping_from_disk() or {})
    print(
        f"sku_map bundled={bundled_n} disk_before={before_disk} "
        f"merged={len(merged_map)} disk_after={after_disk}",
        flush=True,
    )
    assert "1415YKBALCK-6XL" in merged_map or canonical_oms_key("1415YKBALCK-6XL", merged_map) == "1415YKBLACK-6XL"
    assert canonical_oms_key("1415YKBALCK-6XL", merged_map) == "1415YKBLACK-6XL"

    hist_path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    hist = pd.read_parquet(hist_path)
    hist = coalesce_inventory_history_sku_aliases(hist, merged_map)
    hist, repair = repair_inventory_history_integrity(hist, persist_report=False)
    sess.daily_inventory_history_df = hist
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        sess.daily_inventory_history_uploaded_at = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).isoformat(timespec="seconds")
    except Exception:
        pass
    persist_inventory_history_authoritative(sess, hist)
    print(f"hist_rows={len(hist)} repair_actions={repair.get('actions')}", flush=True)

    balck_left = hist["OMS_SKU"].astype(str).str.upper().str.contains("BALCK", na=False).sum()
    print(f"BALCK_rows_remaining={int(balck_left)}", flush=True)
    assert int(balck_left) == 0, "BALCK typo SKUs must be remapped in history"

    def _day_total(df, day, channel):
        view = (
            combine_inventory_channels(df)
            if channel == "combined"
            else filter_inventory_history_channel(df, channel)
        )
        view = view.copy()
        view["Date"] = pd.to_datetime(view["Date"]).dt.normalize()
        sub = view[view["Date"] == day]
        return float(pd.to_numeric(sub["Qty"], errors="coerce").fillna(0).sum())

    totals = {
        "oms_0730": _day_total(hist, "2026-07-30", "oms"),
        "oms_0731": _day_total(hist, "2026-07-31", "oms"),
        "amz_0731": _day_total(hist, "2026-07-31", "amazon"),
        "comb_0730": _day_total(hist, "2026-07-30", "combined"),
        "comb_0731": _day_total(hist, "2026-07-31", "combined"),
    }
    print("inv_totals", json.dumps(totals), flush=True)
    assert 180_000 <= totals["oms_0730"] <= 200_000, totals["oms_0730"]
    assert totals["comb_0730"] < 210_000, f"Jul30 combined spike still present: {totals['comb_0730']}"
    assert totals["comb_0731"] >= totals["oms_0731"], "combined must be >= OMS (max semantics)"
    assert totals["comb_0731"] < totals["oms_0731"] + totals["amz_0731"], "combined must be < OMS+Amazon sum"

    # Coverage: history OMS SKUs should canonicalize; track how many differ from raw
    oms = filter_inventory_history_channel(hist, "oms")
    skus = sorted(set(oms["OMS_SKU"].astype(str).str.upper()))
    remapped = 0
    for s in skus:
        c = canonical_oms_key(s, merged_map)
        if c and c != s:
            remapped += 1
    print(f"history_oms_skus={len(skus)} canonical_remapped={remapped}", flush=True)

    invalidate_all_shared_caches()
    body = {
        "period_days": 30,
        "lead_time": 45,
        "target_days": 180,
        "demand_basis": "Net",
        "use_seasonality": True,
        "seasonal_weight": 0.5,
        "group_by_parent": False,
        "min_denominator": 7,
        "grace_days": 0,
        "safety_pct": 0.0,
        "enforce_two_size_minimum": True,
        "enforce_lead_time_release_gate": False,
        "urgent_all_sizes_days": 45,
        "auto_import_yesterday_ledger": False,
        "raise_ledger_lookback_days": 14,
        "inventory_history_channel": "combined",
        "use_shared_cache": False,
        "use_ly_fallback": True,
    }
    sid = f"po-full-verify-{uuid.uuid4().hex[:8]}"
    print("po_calculate_start", sid, flush=True)
    result = execute_po_calculate(sess, body, session_id=sid)
    if not result.get("ok"):
        print(json.dumps({"ok": False, "error": result.get("message")}))
        return 1
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        print(json.dumps({"ok": False, "error": "empty_po"}))
        return 1

    key = save_shared_cache(sess, body, po_df, result)
    po_skus = po_df["OMS_SKU"].astype(str).str.upper()
    balck_po = int(po_skus.str.contains("BALCK", na=False).sum())
    black_rows = po_df[po_skus.str.startswith("1415YKBLACK")]
    bottel_po = int(po_skus.str.contains("BOTTELGREEN", na=False).sum())

    inv_col = "Total_Inventory" if "Total_Inventory" in po_df.columns else None
    tot_inv = float(pd.to_numeric(po_df[inv_col], errors="coerce").fillna(0).sum()) if inv_col else None

    samples = {}
    for s in (
        "1415YKBLACK-6XL",
        "1415YKBLACK-7XL",
        "1415YKBLACK-8XL",
        "1024YKMUSTARD-4XL",
        "DPT21MULTI",
        "5041YKBOTTLEGREEN-6XL",
    ):
        hit = po_df[po_skus == s]
        if hit.empty:
            samples[s] = None
            continue
        r = hit.iloc[0]
        samples[s] = {
            "Eff_Days": int(float(r.get("Eff_Days") or 0)),
            "Total_Inventory": float(r.get("Total_Inventory") or 0) if "Total_Inventory" in hit.columns else None,
            "Priority": str(r.get("Priority") or r.get("priority") or ""),
        }

    out = {
        "ok": True,
        "cache_key": key,
        "po_rows": int(len(po_df)),
        "po_total_inventory": tot_inv,
        "balck_in_po": balck_po,
        "bottel_in_po": bottel_po,
        "1415ykblack_sizes": int(len(black_rows)),
        "sku_map_size": len(merged_map),
        "inv_totals": totals,
        "samples": samples,
    }
    print(json.dumps(out, indent=2, default=str), flush=True)

    assert balck_po == 0, "PO must not contain BALCK typo SKUs"
    assert len(black_rows) >= 3, "1415YKBLACK sizes must appear in PO"
    assert samples.get("1415YKBLACK-6XL") is not None, "1415YKBLACK-6XL missing from PO"
    assert int(len(po_df)) > 1000, "PO too small"
    print("PO_FULL_VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

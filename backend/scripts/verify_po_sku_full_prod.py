#!/usr/bin/env python3
"""Light prod check (no second warm-cache load): merge SKU map + Inv History aliases."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def main() -> int:
    from backend.services.daily_inventory_history import (
        coalesce_inventory_history_sku_aliases,
        combine_inventory_channels,
        filter_inventory_history_channel,
        repair_inventory_history_integrity,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.services.po_engine import canonical_oms_key
    from backend.services.sku_mapping import (
        ensure_sku_mapping_merged_globally,
        load_bundled_sku_mapping,
        load_sku_mapping_from_disk,
    )

    bundled_n = len(load_bundled_sku_mapping())
    before_disk = len(load_sku_mapping_from_disk() or {})
    merged_map = ensure_sku_mapping_merged_globally(None)
    after_disk = len(load_sku_mapping_from_disk() or {})
    print(
        f"sku_map bundled={bundled_n} disk_before={before_disk} "
        f"merged={len(merged_map)} disk_after={after_disk}",
        flush=True,
    )
    assert canonical_oms_key("1415YKBALCK-6XL", merged_map) == "1415YKBLACK-6XL"
    assert canonical_oms_key("1415YKBALCK-7XL", merged_map) == "1415YKBLACK-7XL"
    assert canonical_oms_key("1415YKBALCK-8XL", merged_map) == "1415YKBLACK-8XL"
    from backend.services.inventory import _inventory_alias_oms_key

    assert _inventory_alias_oms_key("5041YKBOTTELGREEN-6XL", merged_map) == "5041YKBOTTLEGREEN-6XL"
    assert canonical_oms_key("1415YKCBLACK-XXL", merged_map).startswith("1415YKBLACK")

    hist_path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    meta_path = Path("/data/warm_cache/daily_inventory_history_meta.json")
    hist = pd.read_parquet(hist_path)
    hist = coalesce_inventory_history_sku_aliases(hist, merged_map)
    hist, repair = repair_inventory_history_integrity(hist, persist_report=False)
    _coerce_df_for_parquet(hist).to_parquet(hist_path, index=False)
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["daily_inventory_history_uploaded_at"] = datetime.now(
                ZoneInfo("Asia/Kolkata")
            ).isoformat(timespec="seconds")
            meta_path.write_text(json.dumps(meta, default=str, indent=2), encoding="utf-8")
        except Exception as exc:
            print("meta_update_skip", exc, flush=True)
    print(f"hist_rows={len(hist)} repair_actions={repair.get('actions')}", flush=True)

    balck_left = int(hist["OMS_SKU"].astype(str).str.upper().str.contains("BALCK", na=False).sum())
    print(f"BALCK_rows_remaining={balck_left}", flush=True)
    assert balck_left == 0

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
    assert totals["comb_0730"] < 210_000, totals["comb_0730"]
    assert totals["comb_0731"] >= totals["oms_0731"]
    assert totals["comb_0731"] < totals["oms_0731"] + totals["amz_0731"]

    oms = filter_inventory_history_channel(hist, "oms")
    skus = set(oms["OMS_SKU"].astype(str).str.upper())
    remapped = sum(1 for s in skus if (c := canonical_oms_key(s, merged_map)) and c != s)
    print(
        json.dumps(
            {
                "ok": True,
                "sku_map_size": len(merged_map),
                "history_oms_skus": len(skus),
                "canonical_remapped": remapped,
                "inv_totals": totals,
            },
            indent=2,
        ),
        flush=True,
    )
    print("SKU_HIST_VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

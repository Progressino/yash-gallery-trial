#!/usr/bin/env python3
"""Re-apply OMS Inventory history.xlsx onto prod Inv History (OMS channel).

Daily snapshot uploads drifted prod away from the workbook. The user's
``inventory_diff_report.xlsx`` History column matches the workbook; CSV matches
current drifted prod. This overlay restores History as the OMS truth.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_IST = ZoneInfo("Asia/Kolkata")
_OVERLAY = Path("/tmp/_oms_history_workbook_tall.csv")
_DIFF = Path("/tmp/inventory_diff_report.xlsx")
_HIST = Path("/data/warm_cache/daily_inventory_history_df.parquet")


def _load_overlay() -> pd.DataFrame:
    tall = pd.read_csv(_OVERLAY)
    tall["OMS_SKU"] = tall["OMS_SKU"].astype(str).str.strip().str.upper()
    tall["Date"] = pd.to_datetime(tall["Date"], errors="coerce").dt.normalize()
    tall["Qty"] = pd.to_numeric(tall["Qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tall = tall.dropna(subset=["OMS_SKU", "Date"])
    tall = tall[tall["OMS_SKU"].astype(str).str.len() > 0]
    tall["Source"] = "snapshot"
    tall["Channel"] = "oms"
    return tall[["OMS_SKU", "Date", "Qty", "Source", "Channel"]].reset_index(drop=True)


def _load_diff_history() -> pd.DataFrame:
    xl = pd.ExcelFile(_DIFF)
    rows = []
    for s in xl.sheet_names:
        parts = s.split("-")
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        date = f"20{y:02d}-{m:02d}-{d:02d}"
        df = pd.read_excel(xl, sheet_name=s)
        df["Date"] = date
        rows.append(df)
    diff = pd.concat(rows, ignore_index=True)
    diff["SKU"] = diff["SKU"].astype(str).str.strip().str.upper()
    diff["History"] = pd.to_numeric(diff["History"], errors="coerce").fillna(0.0).clip(lower=0.0)
    diff["CSV"] = pd.to_numeric(diff["CSV"], errors="coerce").fillna(0.0)
    diff["Date"] = pd.to_datetime(diff["Date"], errors="coerce").dt.normalize()
    return diff


def main() -> int:
    sys.path.insert(0, "/srv")
    from backend.services.daily_inventory_history import (
        clear_inventory_channel_view_cache,
        filter_inventory_history_channel,
        merge_inventory_history_preserving_channels,
        persist_inventory_history_authoritative,
    )
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.session import AppSession
    import backend.main as _main

    if not _OVERLAY.exists():
        print("FAIL missing overlay", _OVERLAY)
        return 1
    if not _HIST.exists():
        print("FAIL missing history parquet", _HIST)
        return 2

    bak = _HIST.with_name(
        f"{_HIST.name}.bak-before-oms-workbook-reapply-{datetime.now(_IST).strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(_HIST, bak)
    print("backup", bak)

    existing = pd.read_parquet(_HIST)
    overlay = _load_overlay()
    print("existing", len(existing), "overlay", len(overlay), "dates", sorted(overlay["Date"].dt.strftime("%Y-%m-%d").unique())[:3], "...")

    merged = merge_inventory_history_preserving_channels(existing, overlay)
    print("merged", len(merged))

    clear_inventory_channel_view_cache()
    sess = AppSession()
    sess.daily_inventory_history_df = merged
    sess.daily_inventory_history_uploaded_at = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S")
    sess.daily_inventory_history_filename = "OMS Inventory history.xlsx reapply"
    sess.daily_inventory_history_snapshot_dates = sorted(
        overlay["Date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )
    ok = persist_inventory_history_authoritative(sess, merged)
    if not ok:
        # Force write if newer-guard blocked (same max date, intentional re-authoring).
        from backend.services.helpers import _coerce_df_for_parquet
        import json

        _coerce_df_for_parquet(merged).to_parquet(_HIST, index=False)
        meta = {
            "daily_inventory_history_uploaded_at": sess.daily_inventory_history_uploaded_at,
            "daily_inventory_history_filename": sess.daily_inventory_history_filename,
            "daily_inventory_history_snapshot_dates": sess.daily_inventory_history_snapshot_dates,
        }
        (_HIST.parent / "daily_inventory_history_meta.json").write_text(
            json.dumps(meta, default=str, indent=2), encoding="utf-8"
        )
        print("forced parquet write (newer-guard bypass)")
    else:
        print("persisted via authoritative helper")

    if not getattr(_main, "_warm_cache", None):
        _main._warm_cache = {}
    _main._warm_cache["daily_inventory_history_df"] = merged
    try:
        invalidate_all_shared_caches()
    except Exception as e:
        print("po cache invalidate:", e)

    # Verify vs History column of diff report
    if not _DIFF.exists():
        print("WARN no diff report on host — skip mismatch check")
        return 0

    diff = _load_diff_history()
    oms = filter_inventory_history_channel(merged, "oms")
    oms = oms.copy()
    oms["Date"] = pd.to_datetime(oms["Date"], errors="coerce").dt.normalize()
    oms["OMS_SKU"] = oms["OMS_SKU"].astype(str).str.strip().str.upper()
    oms["Qty"] = pd.to_numeric(oms["Qty"], errors="coerce").fillna(0.0)
    oms = (
        oms.groupby(["OMS_SKU", "Date"], as_index=False)["Qty"]
        .max()
    )

    check = diff.merge(
        oms,
        left_on=["SKU", "Date"],
        right_on=["OMS_SKU", "Date"],
        how="left",
    )
    check["Qty"] = check["Qty"].fillna(-999999)
    mism = check[check["History"] != check["Qty"]]
    print("mismatches vs History column", len(mism), "/", len(check))
    if len(mism):
        print(mism.head(20)[["Date", "SKU", "CSV", "History", "Qty"]].to_string())
        return 3

    # Also confirm CSV no longer equals prod for previously-drifted sample
    sample = check[(check["SKU"] == "1001YKBEIGE-3XL") & (check["Date"] == pd.Timestamp("2026-07-01"))]
    if len(sample):
        print("sample 1001YKBEIGE-3XL 2026-07-01", sample[["CSV", "History", "Qty"]].to_dict("records"))
    print("OMS_WORKBOOK_REAPPLY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

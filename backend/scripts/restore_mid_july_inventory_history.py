#!/usr/bin/env python3
"""Restore mid-July continuum from bak-preflatfix, then roll forward to snapshot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_CACHE = Path("/data/warm_cache")
_BAK = _CACHE / "daily_inventory_history_df.parquet.bak-preflatfix-202607240520"


def main() -> int:
    import pandas as pd

    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        filter_inventory_history_channel,
        inventory_history_wide_matrix,
        persist_daily_inventory_history_meta,
        refresh_inventory_history_rollforward,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.session import AppSession

    if not _BAK.is_file():
        print("FAIL: missing", _BAK, flush=True)
        return 1

    bak = pd.read_parquet(_BAK)
    bak["Date"] = pd.to_datetime(bak["Date"], errors="coerce").dt.normalize()
    bak = bak.dropna(subset=["Date"]).copy()
    src = bak["Source"].astype(str).str.strip().str.lower()
    # Keep matrix through Jul 16 + Jul 17 snapshot; drop flat Jul 18–24 (re-append snap).
    keep = (bak["Date"] <= pd.Timestamp("2026-07-16")) | (
        (bak["Date"] == pd.Timestamp("2026-07-17")) & (src == "snapshot")
    )
    work = bak.loc[keep].copy()
    mask_mid = (work["Date"] >= pd.Timestamp("2026-07-08")) & (
        work["Date"] <= pd.Timestamp("2026-07-16")
    )
    work.loc[mask_mid, "Source"] = "uploaded"

    from datetime import datetime, timezone

    sess = AppSession()
    sess.daily_inventory_history_df = work
    sess.daily_inventory_history_filename = "Daily Inventory History restored mid-Jul.xlsx"
    sess.daily_inventory_history_wide_end_date = "2026-07-16"
    sess.daily_inventory_history_snapshot_dates = ["2026-07-17"]
    sess.daily_inventory_history_uploaded_at = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    inv_meta_path = _CACHE / "inventory_session_meta.json"
    if inv_meta_path.is_file():
        inv_meta = json.loads(inv_meta_path.read_text(encoding="utf-8"))
        sess.inventory_snapshot_date = str(
            inv_meta.get("inventory_snapshot_date") or "2026-07-24"
        )[:10]
    else:
        sess.inventory_snapshot_date = "2026-07-24"

    if (_CACHE / "inventory_df_variant.parquet").is_file():
        sess.inventory_df_variant = pd.read_parquet(_CACHE / "inventory_df_variant.parquet")
    sales = None
    if (_CACHE / "sales_df.parquet").is_file():
        sales = pd.read_parquet(_CACHE / "sales_df.parquet")

    result = refresh_inventory_history_rollforward(
        sess, include_snapshot=True, sales_df=sales
    )
    print("REFRESH", json.dumps(result, default=str), flush=True)
    if not result.get("ok"):
        return 1

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    _coerce_df_for_parquet(sess.daily_inventory_history_df).to_parquet(
        hist_path, index=False
    )
    persist_daily_inventory_history_meta(sess)
    meta = daily_inventory_history_meta_bundle(sess)
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, default=str, indent=2), encoding="utf-8"
    )

    df = sess.daily_inventory_history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    g = df.groupby("Date")["Qty"].sum().sort_index()
    for d, q in g.items():
        if d >= pd.Timestamp("2026-07-08"):
            print(str(d.date()), int(q), flush=True)
    print(
        "rows",
        len(df),
        "max",
        int(g.max()),
        "any>500k",
        bool((g > 500_000).any()),
        flush=True,
    )

    oms = filter_inventory_history_channel(df, "oms")
    wide = inventory_history_wide_matrix(df, days=20, end_date="2026-07-24", channel="oms")
    print(
        "OMS_rows",
        len(oms),
        "OMS_last_totals",
        list(zip(wide.get("dates", [])[-5:], wide.get("date_totals", [])[-5:])),
        flush=True,
    )
    if not oms.empty and sum(wide.get("date_totals") or [0]) <= 0:
        print("FAIL: OMS matrix still empty", flush=True)
        return 1
    if any(t > 500_000 for t in (wide.get("date_totals") or [])):
        print("FAIL: OMS spike >500k", flush=True)
        return 1
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

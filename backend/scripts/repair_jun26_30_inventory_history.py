#!/usr/bin/env python3
"""Repair Jun 26–30 Inv History on prod so uploaded OMS values reflect + aren't CARRIED.

- Promote legacy ``uploaded`` / blank-channel census rows to ``snapshot`` + ``oms``
- Upsert spreadsheet OMS qtys for 165YK251MUSTRAD-XXL on Jun 29–30 (3, 10)
- Persist warm cache, clear channel/PO caches
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_IST = ZoneInfo("Asia/Kolkata")
_SKU = "165YK251MUSTRAD-XXL"
# From user spreadsheet (OMS warehouse wide matrix)
_SHEET_OMS = {
    "2026-06-26": 26.0,
    "2026-06-27": 22.0,
    "2026-06-28": 12.0,
    "2026-06-29": 3.0,
    "2026-06-30": 10.0,
}


def main() -> None:
    from backend.services.daily_inventory_history import (
        clear_inventory_channel_view_cache,
        inventory_history_wide_matrix,
        persist_inventory_history_authoritative,
    )
    from backend.services.po_shared_cache import invalidate_all_shared_caches
    from backend.session import AppSession
    import backend.main as _main

    path = Path("/data/warm_cache/daily_inventory_history_df.parquet")
    bak = path.with_name(
        f"{path.name}.bak-before-jun26-30-repair-{datetime.now(_IST).strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(path, bak)
    print("backup", bak)

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["OMS_SKU"] = df["OMS_SKU"].astype(str).str.strip().str.upper()
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "Source" not in df.columns:
        df["Source"] = "uploaded"
    if "Channel" not in df.columns:
        df["Channel"] = ""
    df["Source"] = df["Source"].astype(str).str.strip().str.lower().replace({"nan": "uploaded", "none": "uploaded"})
    df["Channel"] = df["Channel"].astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})

    # 1) Promote file rows: uploaded → snapshot
    up_mask = df["Source"].eq("uploaded")
    print("promote uploaded→snapshot", int(up_mask.sum()))
    df.loc[up_mask, "Source"] = "snapshot"

    # 2) Blank channel on census days with no OMS row → oms
    #    (do not convert blanks on days that already have explicit OMS — those blanks
    #    are legacy combined totals and would inflate the OMS tab).
    oms_dates = set(df.loc[df["Channel"].eq("oms"), "Date"].dropna().unique())
    blank = df["Channel"].eq("") & ~df["Date"].isin(oms_dates)
    print("blank→oms", int(blank.sum()), "oms_dates", len(oms_dates))
    df.loc[blank, "Channel"] = "oms"

    # 3) Upsert spreadsheet OMS for 165YK Jun 26–30
    for day_s, qty in _SHEET_OMS.items():
        day = pd.Timestamp(day_s)
        mask = (
            df["OMS_SKU"].eq(_SKU)
            & df["Date"].eq(day)
            & df["Channel"].isin(["oms", ""])
        )
        if mask.any():
            df = df.loc[~mask].copy()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "OMS_SKU": _SKU,
                            "Date": day,
                            "Qty": float(qty),
                            "Source": "snapshot",
                            "Channel": "oms",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        print("upsert", _SKU, day_s, qty)

    # Drop pure-derived duplicates for this SKU on those days (snapshot wins via coalesce later)
    days = {pd.Timestamp(d) for d in _SHEET_OMS}
    drop_der = (
        df["OMS_SKU"].eq(_SKU)
        & df["Date"].isin(days)
        & df["Source"].eq("derived")
        & df["Channel"].isin(["oms", ""])
    )
    if drop_der.any():
        print("drop derived oms rows", int(drop_der.sum()))
        df = df.loc[~drop_der].reset_index(drop=True)

    clear_inventory_channel_view_cache()
    sess = AppSession()
    sess.daily_inventory_history_df = df
    sess.daily_inventory_history_uploaded_at = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S")
    sess.daily_inventory_history_filename = "jun26-30-oms-repair"
    assert persist_inventory_history_authoritative(sess, df)
    if not getattr(_main, "_warm_cache", None):
        _main._warm_cache = {}
    _main._warm_cache["daily_inventory_history_df"] = df
    try:
        invalidate_all_shared_caches()
    except Exception as e:
        print("po cache invalidate:", e)

    # Verify
    wide_c = inventory_history_wide_matrix(df, q=_SKU, days=30, end_date="2026-07-25", channel="combined")
    wide_o = inventory_history_wide_matrix(df, q=_SKU, days=30, end_date="2026-07-25", channel="oms")
    dates = wide_c.get("dates") or []
    gaps = set(wide_c.get("gap_dates") or [])
    row_c = (wide_c.get("rows") or [{}])[0]
    row_o = (wide_o.get("rows") or [{}])[0]
    qtys_c = dict(zip(dates, row_c.get("qtys") or []))
    qtys_o = dict(zip(dates, row_o.get("qtys") or []))
    print("gap Jun26-30", sorted(d for d in _SHEET_OMS if d in gaps))
    for d, expect in _SHEET_OMS.items():
        print(
            d,
            "combined",
            qtys_c.get(d),
            "oms",
            qtys_o.get(d),
            "expect",
            expect,
            "carried" if d in gaps else "uploaded",
        )
        assert d not in gaps, f"{d} still marked carried"
        assert abs(float(qtys_c.get(d, -1)) - expect) < 0.01
        assert abs(float(qtys_o.get(d, -1)) - expect) < 0.01
    print("PASS")


if __name__ == "__main__":
    main()

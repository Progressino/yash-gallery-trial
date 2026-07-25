#!/usr/bin/env python3
"""Trim non-uploaded tail days and blank-channel duplicate rows from Inv History.

- Deletes every row after ``--cutoff`` (default 2026-07-23) so days that were
  never actually uploaded (e.g. a flat-copied phantom snapshot, or read-path
  forward-fill to today) stop showing as real census columns.
- Drops legacy blank-``Channel`` rows on any date that already carries explicit
  ``oms``/``amazon`` rows — those stale duplicates inflate the Combined total.
- Rewrites warm-cache history + meta (max/matrix/wide end = cutoff), invalidates
  shared PO caches, and hot-syncs the running backend sidecar.

Usage (inside progressino-backend container):
  python -m backend.scripts.cleanup_inventory_history_tail --cutoff 2026-07-23
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_CACHE = Path("/data/warm_cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-07-23", help="Keep rows with Date <= cutoff")
    ap.add_argument("--apply", action="store_true", help="Write changes (otherwise dry-run)")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff).normalize()

    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        inventory_history_wide_matrix,
    )
    from backend.services.helpers import _coerce_df_for_parquet
    from backend.session import AppSession

    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        print("FAIL: missing history", hist_path, flush=True)
        return 1

    df = pd.read_parquet(hist_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).copy()
    if "Channel" not in df.columns:
        df["Channel"] = ""
    if "Source" not in df.columns:
        df["Source"] = "uploaded"

    ch = df["Channel"].astype(str).str.strip().str.lower()
    total_before = len(df)

    # 1) drop everything after the cutoff
    after_mask = df["Date"] > cutoff
    print("rows_after_cutoff", int(after_mask.sum()), flush=True)
    for d in sorted(df.loc[after_mask, "Date"].dt.date.unique()):
        print("  drop_day", d, flush=True)

    # 2) drop blank-channel rows on dates that also have explicit oms/amazon
    explicit_dates = set(df.loc[ch.isin(["oms", "amazon"]), "Date"].unique())
    blank_dup_mask = (ch == "") & df["Date"].isin(explicit_dates)
    print("blank_dup_rows", int(blank_dup_mask.sum()), flush=True)

    keep = ~after_mask & ~blank_dup_mask
    out = df.loc[keep].reset_index(drop=True)
    print("rows", total_before, "->", len(out), "new_max", str(out["Date"].max().date()), flush=True)

    if not args.apply:
        print("DRY-RUN (pass --apply to write)", flush=True)
        return 0

    backup = hist_path.with_suffix(
        hist_path.suffix + f".bak-before-tail-cleanup-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    backup.write_bytes(hist_path.read_bytes())
    print("backup", backup.name, flush=True)

    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    live_max = pd.Timestamp(out["Date"].max())
    snap_dates = sorted(
        {
            str(pd.Timestamp(d).date())
            for d in out.loc[
                out["Source"].astype(str).str.lower().eq("snapshot"), "Date"
            ].unique()
        }
    )
    sess = AppSession()
    sess.daily_inventory_history_df = out
    sess.daily_inventory_history_filename = "Daily Inventory History tail cleanup"
    sess.daily_inventory_history_wide_end_date = str(live_max.date())
    sess.daily_inventory_history_snapshot_dates = snap_dates
    sess.daily_inventory_history_matrix_max_date = str(live_max.date())
    sess.daily_inventory_history_uploaded_at = uploaded_at

    _coerce_df_for_parquet(out).to_parquet(hist_path, index=False)
    meta = daily_inventory_history_meta_bundle(sess)
    meta["daily_inventory_history_uploaded_at"] = uploaded_at
    (_CACHE / "daily_inventory_history_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    good = hist_path.with_suffix(
        hist_path.suffix + f".bak-good-tail-cleanup-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    )
    good.write_bytes(hist_path.read_bytes())
    print("good_backup", good.name, flush=True)

    for channel in ("combined", "oms", "amazon"):
        w = inventory_history_wide_matrix(out, days=8, end_date=str(live_max.date()), channel=channel)
        print(
            "tail",
            channel,
            list(zip(w.get("dates", [])[-4:], [int(x) for x in w.get("date_totals", [])[-4:]])),
            "gaps",
            w.get("gap_dates"),
            flush=True,
        )

    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        invalidate_all_shared_caches()
        print("shared_po_cache_invalidated", flush=True)
    except Exception as exc:
        print("shared_cache_invalidate_warn", exc, flush=True)

    print("DONE new_max", str(live_max.date()), "rows", len(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

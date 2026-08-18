#!/usr/bin/env python3
"""Restore Inv History census days from day-archive files and bak-* parquets.

Daily snapshot append can leave mid-window days as Source=derived (CARRIED)
even when the original upload still exists as inventory_day_snapshots/*.parquet
or in a bak-* history file. Overlay those days, persist with force, and
invalidate PO caches so Eff_Days uses the restored census.

Run on prod (after the backend image includes this script):

  docker exec -e PYTHONPATH=/srv -e WARM_CACHE_DIR=/data/warm_cache -w /srv \\
    progressino-backend-1 python backend/scripts/repair_inv_history_from_disk_archives.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
_CACHE = Path("/data/warm_cache")
_HIST = _CACHE / "daily_inventory_history_df.parquet"


def _auth_dates(df) -> list[str]:
    import pandas as pd

    if df is None or getattr(df, "empty", True) or "Date" not in df.columns:
        return []
    if "Source" not in df.columns:
        return []
    src = df["Source"].astype(str).str.strip().str.lower()
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return sorted({str(pd.Timestamp(d).date()) for d in dates[src.isin(["snapshot", "uploaded"])].dropna().unique()})


def _print_jul_aug(df, snap_dir: Path) -> None:
    import pandas as pd

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    print("--- JUL/AUG DAYS ---", flush=True)
    for d in pd.date_range("2026-07-01", "2026-08-17"):
        day = pd.Timestamp(d).normalize()
        sub = work[work["Date"] == day]
        counts = (
            sub["Source"].astype(str).str.lower().value_counts().to_dict()
            if not sub.empty and "Source" in sub.columns
            else {}
        )
        qty = float(pd.to_numeric(sub["Qty"], errors="coerce").fillna(0).sum()) if not sub.empty else 0.0
        has_file = (snap_dir / f"{day.date()}.parquet").is_file()
        print(
            f"DAY {day.date()} file={int(has_file)} rows={len(sub)} qty={qty:.0f} src={counts}",
            flush=True,
        )


def main() -> int:
    import pandas as pd

    from backend.services.daily_inventory_history import (
        _inventory_day_snapshot_dir,
        overlay_persisted_inventory_snapshots,
        persist_inventory_history_authoritative,
        restore_inventory_history_from_best_disk_backups,
    )
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    if not _HIST.is_file():
        print("FAIL missing history parquet", _HIST, flush=True)
        return 1

    stamp = datetime.now(_IST).strftime("%Y%m%d%H%M%S")
    bak_out = _HIST.with_name(f"{_HIST.name}.bak-before-archive-repair-{stamp}")
    shutil.copy2(_HIST, bak_out)
    print("backup", bak_out, flush=True)

    current = pd.read_parquet(_HIST)
    before = _auth_dates(current)
    snap_dir = _inventory_day_snapshot_dir()
    print("BEFORE auth_days", len(before), before[:12], "...", flush=True)
    _print_jul_aug(current, snap_dir)

    sess = AppSession()
    sess.daily_inventory_history_df = current
    meta_path = _CACHE / "daily_inventory_history_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in (
                "daily_inventory_history_uploaded_at",
                "daily_inventory_history_filename",
                "daily_inventory_history_matrix_max_date",
                "daily_inventory_history_wide_end_date",
            ):
                if meta.get(key):
                    setattr(sess, key, meta[key])
        except Exception as exc:
            print("meta_skip", exc, flush=True)

    dih._MATRIX_DAY_OVERLAY_MTIME = None
    n_overlay = overlay_persisted_inventory_snapshots(sess)
    print("overlay_days", n_overlay, flush=True)

    restored = restore_inventory_history_from_best_disk_backups(sess.daily_inventory_history_df)
    if restored is not None:
        after_overlay = _auth_dates(sess.daily_inventory_history_df)
        after_bak = _auth_dates(restored)
        if len(after_bak) > len(after_overlay):
            print(
                "bak_restore auth_days",
                len(after_overlay),
                "->",
                len(after_bak),
                flush=True,
            )
            sess.daily_inventory_history_df = restored
        else:
            print("bak_restore skipped (no extra census days)", flush=True)
    else:
        print("bak_restore none", flush=True)

    wrote = persist_inventory_history_authoritative(sess, force=True)
    print("persist", wrote, flush=True)
    if not wrote:
        return 2

    dih._MATRIX_DAY_OVERLAY_MTIME = None

    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        invalidate_all_shared_caches()
        print("po_cache_invalidated", flush=True)
    except Exception as exc:
        print("po_cache_skip", exc, flush=True)

    try:
        import backend.main as _main

        if not _main._warm_cache:
            _main._warm_cache = {}
        _main._warm_cache["daily_inventory_history_df"] = sess.daily_inventory_history_df
    except Exception as exc:
        print("warm_seed_skip", exc, flush=True)

    after = _auth_dates(sess.daily_inventory_history_df)
    print("AFTER auth_days", len(after), after, flush=True)
    _print_jul_aug(sess.daily_inventory_history_df, snap_dir)
    added = [d for d in after if d not in before]
    print("RESTORED_DATES", added, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

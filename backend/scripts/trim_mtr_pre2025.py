#!/usr/bin/env python3
"""Drop Amazon MTR rows before 2025-01-01 to keep warm cache lean (2025+ only)."""
from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.services.helpers import _coerce_df_for_parquet


def main() -> int:
    cache = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    pq = cache / "mtr_df.parquet"
    if not pq.is_file():
        print(f"Missing {pq}")
        return 1

    cutoff = pd.Timestamp("2025-01-01")
    print(f"Loading {pq}…", flush=True)
    mtr = pd.read_parquet(pq)
    before = len(mtr)
    d = pd.to_datetime(mtr["Date"], errors="coerce")
    if "Reporting_Date" not in mtr.columns:
        inv = (
            pd.to_datetime(mtr["Invoice_Date_Text"], errors="coerce")
            if "Invoice_Date_Text" in mtr.columns
            else pd.Series(pd.NaT, index=mtr.index)
        )
        mtr["Reporting_Date"] = inv.where(inv.notna(), d)
    rd = pd.to_datetime(mtr["Reporting_Date"], errors="coerce")
    keep = (d >= cutoff) | (rd >= cutoff)
    mtr = mtr.loc[keep].copy()
    after = len(mtr)
    print(f"Trimmed {before:,} → {after:,} (dropped {before - after:,} pre-2025 rows)", flush=True)

    backup = cache / "mtr_df.parquet.bak-pre2025-trim"
    if not backup.exists() and (cache / "mtr_df.parquet").exists():
        (cache / "mtr_df.parquet").replace(backup)
        print(f"Backup at {backup}", flush=True)

    _coerce_df_for_parquet(mtr).to_parquet(pq, index=False)
    mf = cache / "_manifest.json"
    keys: set[str] = set()
    if mf.is_file():
        try:
            keys = set(json.loads(mf.read_text()).get("keys") or [])
        except Exception:
            pass
    keys.add("mtr_df")
    mf.write_text(
        json.dumps(
            {"saved_at": datetime.now(timezone.utc).isoformat(), "keys": sorted(keys)}
        )
    )
    del mtr
    gc.collect()
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

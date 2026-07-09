#!/usr/bin/env python3
"""OOM-safe Tier-1 Amazon ingest: zip archive + daily tail after zip max date."""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.services.daily_store import merge_platform_data
from backend.services.helpers import _coerce_df_for_parquet
from backend.services.mtr import dedup_amazon_mtr_dataframe, load_mtr_from_zip


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: ingest_amazon_2yr_oomsafe.py /path/to/amazon.zip")
        return 1

    zip_path = Path(sys.argv[1])
    if not zip_path.is_file():
        print(f"Missing {zip_path}")
        return 1

    cache = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    existing_pq = cache / "mtr_df.parquet"
    tmp_zip_pq = cache / "_mtr_zip_parsed.parquet"

    if tmp_zip_pq.is_file():
        print(f"Reusing parsed zip parquet {tmp_zip_pq}", flush=True)
        zip_max = pd.to_datetime(pd.read_parquet(tmp_zip_pq, columns=["Date"])["Date"]).max()
        print(f"zip max date {zip_max}", flush=True)
    else:
        print("Parsing zip...", flush=True)
        raw, n_csv, skipped = load_mtr_from_zip(zip_path.read_bytes())
        print(f"parsed {len(raw):,} rows from {n_csv} csvs, skipped={len(skipped)}", flush=True)
        raw = dedup_amazon_mtr_dataframe(raw)
        print(f"after dedup {len(raw):,}", flush=True)
        _coerce_df_for_parquet(raw).to_parquet(tmp_zip_pq, index=False)
        zip_max = pd.to_datetime(raw["Date"]).max()
        print(f"zip max date {zip_max}", flush=True)
        del raw
        gc.collect()

    print("Loading tail from existing parquet (Date > zip max)...", flush=True)
    if existing_pq.is_file():
        tail = pd.read_parquet(existing_pq, filters=[("Date", ">", zip_max)])
    else:
        tail = pd.DataFrame()
    print(f"tail rows {len(tail):,}", flush=True)

    print("Loading zip base from temp parquet...", flush=True)
    base = pd.read_parquet(tmp_zip_pq)
    print(f"base rows {len(base):,}", flush=True)

    if tail.empty:
        merged = base
    else:
        # Tail is strictly after zip max date — no overlapping Tier-1 history to dedup.
        tail_dates = pd.to_datetime(tail["Date"], errors="coerce")
        if bool((tail_dates <= zip_max).any()):
            print("Tail overlaps zip dates — running full merge dedup", flush=True)
            merged = merge_platform_data(
                base, tail, "amazon", source_filename=f"{zip_path.name} + daily tail"
            )
        else:
            print("Tail is post-zip only — concat without merge (OOM-safe)", flush=True)
            merged = pd.concat([base, tail], ignore_index=True)
    print(
        f"merged {len(merged):,} (base {len(base):,} + tail {len(tail):,})",
        flush=True,
    )
    del base, tail
    gc.collect()

    if len(merged) < 900_000:
        print(f"ABORT: merged frame too small ({len(merged):,})")
        return 2

    backup = cache / "mtr_df.parquet.bak-pre-2yr-import"
    if existing_pq.is_file() and not backup.exists():
        existing_pq.replace(backup)
        print(f"backed up existing to {backup}", flush=True)

    _coerce_df_for_parquet(merged).to_parquet(existing_pq, index=False)
    d = pd.to_datetime(merged["Date"], errors="coerce")
    print(
        f"written {existing_pq} rows {len(merged):,} range {d.min()} → {d.max()}",
        flush=True,
    )
    print(f"2024 rows {int((d.dt.year == 2024).sum()):,}", flush=True)

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
    tmp_zip_pq.unlink(missing_ok=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

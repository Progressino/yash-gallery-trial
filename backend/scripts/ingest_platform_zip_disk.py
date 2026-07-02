#!/usr/bin/env python3
"""Ingest historical platform ZIP(s) on the server.

Default mode = **Tier-1** (bulk historical): parse → optional date filter →
merge into the warm-cache platform frame (``{platform}_df``) → persist durable
parquet in ``WARM_CACHE_DIR``. Tier-1 bulk frames are the primary quarterly PO
source and survive restarts/redeploys (Docker volume), which is where multi-year
historical archives belong.

Use ``--tier3`` only for incremental daily uploads that should live in the
capped SQLite ``daily_uploads`` store.

PO uses the last 8 quarters, so historical archives are typically filtered with
``--min-date 2025-01-01`` to drop stale 2024 rows.
"""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("ingest_platform_zip_disk")

PLATFORMS = ("myntra", "flipkart", "meesho", "snapdeal", "amazon")
PLATFORM_ATTR = {
    "myntra": "myntra_df",
    "flipkart": "flipkart_df",
    "meesho": "meesho_df",
    "snapdeal": "snapdeal_df",
    "amazon": "mtr_df",
}


def _mapping() -> dict:
    from backend.services.sku_mapping import (
        load_bundled_sku_mapping,
        load_sku_mapping_from_disk,
        merge_sku_mapping_upload,
    )

    return merge_sku_mapping_upload(load_bundled_sku_mapping(), load_sku_mapping_from_disk())


def _parse_zip(platform: str, path: Path, mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    if platform == "myntra":
        from backend.services.myntra import load_myntra_from_zip

        df, _n, skipped = load_myntra_from_zip(path.read_bytes(), mapping, path.name)
        return df, skipped
    if platform == "flipkart":
        from backend.services.flipkart import load_flipkart_from_zip

        df, _n, skipped = load_flipkart_from_zip(path, mapping, source_filename=path.name)
        return df, skipped
    if platform == "meesho":
        from backend.services.meesho import load_meesho_from_zip

        df, _n, skipped = load_meesho_from_zip(path.read_bytes(), source_filename=path.name)
        return df, skipped
    if platform == "snapdeal":
        from backend.services.snapdeal import load_snapdeal_from_zip

        df, _n, skipped, _info = load_snapdeal_from_zip(path.read_bytes(), mapping, filename=path.name)
        return df, skipped
    if platform == "amazon":
        # India Amazon MTR (GST B2C/B2B). Accepts a master ZIP (nested monthly
        # ZIPs / CSVs) or a directory of extracted month ZIPs / CSVs.
        if path.is_dir():
            from backend.services.mtr import load_mtr_from_extracted_files

            files = [(f.name, f.read_bytes()) for f in sorted(path.rglob("*")) if f.is_file()]
            df, _n, skipped = load_mtr_from_extracted_files(files)
        else:
            from backend.services.mtr import load_mtr_from_zip

            df, _n, skipped = load_mtr_from_zip(path.read_bytes())
        return df, skipped
    raise ValueError(f"Unsupported platform: {platform}")


def _has_usable_sku(df: pd.DataFrame) -> bool:
    # Marketplace frames carry OMS_SKU; Amazon MTR carries the raw seller SKU.
    if df.empty:
        return False
    col = "OMS_SKU" if "OMS_SKU" in df.columns else ("SKU" if "SKU" in df.columns else None)
    if col is None:
        return False
    s = df[col].astype(str).str.strip().str.upper()
    good = ~(s.eq("") | s.isin(["NAN", "NONE", "UNKNOWN"]))
    return bool(good.any())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("platform", choices=PLATFORMS)
    p.add_argument("zip_path", type=Path)
    p.add_argument("--filename", default="", help="Label for the archive (default: zip_path.name).")
    p.add_argument(
        "--min-date",
        default="",
        help="Drop rows before this ISO date (e.g. 2025-01-01 to exclude 2024).",
    )
    p.add_argument(
        "--tier3",
        action="store_true",
        help="Persist to Tier-3 SQLite daily_uploads instead of Tier-1 bulk frames.",
    )
    p.add_argument(
        "--allow-empty-sku",
        action="store_true",
        help="Proceed even when parsed rows have no usable OMS_SKU (not recommended).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + merge in memory and report before/after counts, but do NOT persist.",
    )
    args = p.parse_args(argv)

    path = args.zip_path
    label = (args.filename or path.name).strip()
    if not path.exists():
        _log.error("Missing %s", path)
        return 1
    if path.is_dir() and args.platform != "amazon":
        _log.error("Directory input is only supported for amazon MTR (%s given)", args.platform)
        return 1

    mapping = _mapping()
    _log.info("SKU mapping keys: %s", f"{len(mapping):,}")

    df, skipped = _parse_zip(args.platform, path, mapping)
    if df is None or df.empty:
        _log.error("No rows parsed from %s — %s", label, skipped[:5])
        return 1

    d = pd.to_datetime(df["Date"], errors="coerce")
    _log.info("Parsed %s: %s rows, %s → %s", label, f"{len(df):,}", str(d.min())[:10], str(d.max())[:10])
    if skipped:
        _log.info("Parse notes (%s):", len(skipped))
        for s in skipped[:8]:
            _log.info("  %s", s)

    if not _has_usable_sku(df) and not args.allow_empty_sku:
        _log.error(
            "ABORT: parsed rows have no usable OMS_SKU — this looks like a tax/settlement "
            "export, not a per-SKU sales/order report. Use --allow-empty-sku to override."
        )
        return 2

    if args.min_date:
        cutoff = pd.Timestamp(args.min_date)
        before = len(df)
        df = df.loc[d >= cutoff].copy()
        _log.info("Filtered < %s: dropped %s rows, kept %s", args.min_date, f"{before - len(df):,}", f"{len(df):,}")
        if df.empty:
            _log.error("Nothing left after min-date filter.")
            return 1

    from backend.session import AppSession
    import backend.main as main_mod

    attr = PLATFORM_ATTR[args.platform]

    if args.dry_run:
        from backend.services.daily_store import merge_platform_data

        existing = (main_mod._warm_cache or {}).get(attr)
        if not isinstance(existing, pd.DataFrame):
            from pathlib import Path as _P
            import os as _os

            pq = _P(_os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")) / f"{attr}.parquet"
            existing = pd.read_parquet(pq) if pq.is_file() else pd.DataFrame()
        merged = merge_platform_data(existing, df, args.platform, source_filename=label)
        ed = pd.to_datetime(existing["Date"], errors="coerce") if "Date" in existing.columns and len(existing) else pd.Series([], dtype="datetime64[ns]")
        md = pd.to_datetime(merged["Date"], errors="coerce")
        _log.info(
            "DRY-RUN %s: BEFORE %s rows (%s→%s) + archive %s rows → AFTER %s rows (%s→%s); net new = %s",
            attr,
            f"{len(existing):,}",
            str(ed.min())[:10] if len(ed) and ed.notna().any() else "NA",
            str(ed.max())[:10] if len(ed) and ed.notna().any() else "NA",
            f"{len(df):,}",
            f"{len(merged):,}",
            str(md.min())[:10] if md.notna().any() else "NA",
            str(md.max())[:10] if md.notna().any() else "NA",
            f"{len(merged) - len(existing):,}",
        )
        return 0

    if args.tier3:
        from backend.services.daily_store import load_platform_data, save_daily_file

        file_date, saved_rows, block = save_daily_file(args.platform, label, df)
        if block:
            _log.error("save_daily_file blocked: %s", block)
            return 1
        _log.info("Tier-3 saved: file_date=%s rows=%s", file_date, saved_rows)
        full = load_platform_data(args.platform, dedup=True)
    else:
        # Tier-1: merge archive into the durable warm-cache bulk frame.
        from backend.services.daily_store import merge_platform_data

        existing = (main_mod._warm_cache or {}).get(attr)
        if not isinstance(existing, pd.DataFrame):
            existing = pd.DataFrame()
        full = merge_platform_data(existing, df, args.platform, source_filename=label)
        _log.info("Tier-1 merged %s: %s existing + archive → %s rows", attr, f"{len(existing):,}", f"{len(full):,}")

    sess = AppSession()
    sess.sku_mapping = mapping
    setattr(sess, attr, full)
    main_mod.publish_warm_cache_from_session(sess)
    try:
        main_mod._save_warm_cache_to_disk(main_mod._warm_cache)
    except Exception:
        _log.exception("_save_warm_cache_to_disk failed")

    warm_rows = len((main_mod._warm_cache or {}).get(attr, pd.DataFrame()))
    _log.info("Tier-1 warm cache %s rows: %s (persisted to WARM_CACHE_DIR)", attr, f"{warm_rows:,}")
    del df, full
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

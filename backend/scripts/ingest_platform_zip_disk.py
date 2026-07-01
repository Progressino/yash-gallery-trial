#!/usr/bin/env python3
"""Ingest platform ZIP(s) on the server: Tier-3 SQLite + warm-cache publish."""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("ingest_platform_zip_disk")


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
    raise ValueError(platform)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("platform", choices=["myntra", "flipkart"])
    p.add_argument("zip_path", type=Path)
    p.add_argument(
        "--filename",
        default="",
        help="Tier-3 filename label (default: zip_path.name).",
    )
    p.add_argument(
        "--save-only",
        action="store_true",
        help="Only persist to Tier-3; skip warm-cache reload (use for batch ingests).",
    )
    args = p.parse_args(argv)

    path = args.zip_path
    save_name = (args.filename or path.name).strip()
    if not path.is_file():
        _log.error("Missing %s", path)
        return 1

    mapping = _mapping()
    _log.info("SKU mapping keys: %s", f"{len(mapping):,}")

    df, skipped = _parse_zip(args.platform, path, mapping)
    if df.empty:
        _log.error("No rows parsed from %s — %s", save_name, skipped[:5])
        return 1

    d = pd.to_datetime(df["Date"], errors="coerce")
    _log.info(
        "Parsed %s: %s rows, %s → %s",
        save_name,
        f"{len(df):,}",
        str(d.min())[:10],
        str(d.max())[:10],
    )
    if skipped:
        _log.info("Parse notes (%s):", len(skipped))
        for s in skipped[:8]:
            _log.info("  %s", s)

    from backend.services.daily_store import load_platform_data, save_daily_file
    from backend.session import AppSession
    import backend.main as main_mod

    file_date, saved_rows, block = save_daily_file(args.platform, save_name, df)
    if block:
        _log.error("save_daily_file blocked: %s", block)
        return 1
    _log.info("Tier-3 saved: file_date=%s rows=%s", file_date, saved_rows)

    if args.save_only:
        return 0

    attr = f"{args.platform}_df"
    sess = AppSession()
    sess.sku_mapping = mapping
    full = load_platform_data(args.platform, dedup=True)
    setattr(sess, attr, full)
    _log.info("Tier-3 reload %s: %s rows", args.platform, f"{len(full):,}")

    main_mod.publish_warm_cache_from_session(sess)
    try:
        main_mod._save_warm_cache_to_disk(main_mod._warm_cache)
    except Exception:
        _log.exception("_save_warm_cache_to_disk failed")

    warm_rows = len((main_mod._warm_cache or {}).get(attr, pd.DataFrame()))
    _log.info("Warm cache %s rows: %s", attr, f"{warm_rows:,}")
    del df, full
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

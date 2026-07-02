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
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("ingest_platform_zip_disk")


def _warm_cache_dir() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def _existing_frame(attr: str, main_mod) -> pd.DataFrame:
    """Current Tier-1 frame for *attr*: prefer live warm cache, else disk parquet.

    Standalone ingest runs in a separate process where ``_warm_cache`` is empty,
    so we must fall back to the durable parquet to avoid overwriting bulk history.
    """
    existing = (getattr(main_mod, "_warm_cache", None) or {}).get(attr)
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        return existing
    pq = _warm_cache_dir() / f"{attr}.parquet"
    if pq.is_file():
        try:
            return pd.read_parquet(pq)
        except Exception:
            _log.exception("Could not read existing parquet %s", pq)
    return pd.DataFrame()


def _refresh_manifest(attr: str) -> None:
    """Add *attr* to the disk manifest keys and bump saved_at (fresh fast-path)."""
    mf = _warm_cache_dir() / "_manifest.json"
    keys: set[str] = set()
    if mf.is_file():
        try:
            keys = set(json.loads(mf.read_text()).get("keys") or [])
        except Exception:
            pass
    keys.add(attr)
    mf.write_text(json.dumps({"saved_at": datetime.now(timezone.utc).isoformat(), "keys": sorted(keys)}))

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

        existing = _existing_frame(attr, main_mod)
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

        sess = AppSession()
        sess.sku_mapping = mapping
        setattr(sess, attr, full)
        main_mod.publish_warm_cache_from_session(sess)
        try:
            main_mod._save_warm_cache_to_disk(main_mod._warm_cache)
        except Exception:
            _log.exception("_save_warm_cache_to_disk failed")
        _log.info("Tier-3 warm cache %s rows: %s", attr, f"{len(full):,}")
        del df, full
        gc.collect()
        return 0

    # ── Tier-1 (default): process-safe single-frame durable write ──────────────
    # Load existing from live warm cache OR disk parquet, merge the archive, and
    # write ONLY this platform's parquet. We deliberately do NOT publish the full
    # warm cache or rewrite sku_mapping/sales_df here: this runs as a separate
    # process, so touching those would risk shrinking maps or corrupting derived
    # frames. The running backend picks up the new parquet on its next restart.
    from backend.services.daily_store import merge_platform_data
    from backend.services.helpers import _coerce_df_for_parquet

    existing = _existing_frame(attr, main_mod)
    full = merge_platform_data(existing, df, args.platform, source_filename=label)
    _log.info(
        "Tier-1 merge %s: existing %s + archive %s → %s rows (net new %s)",
        attr,
        f"{len(existing):,}",
        f"{len(df):,}",
        f"{len(full):,}",
        f"{len(full) - len(existing):,}",
    )
    if len(full) < len(existing):
        _log.error(
            "ABORT: merged frame (%s) is smaller than existing (%s) — refusing to shrink Tier-1.",
            f"{len(full):,}",
            f"{len(existing):,}",
        )
        return 2

    parquet_path = _warm_cache_dir() / f"{attr}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_df_for_parquet(full).to_parquet(parquet_path, index=False)
    _refresh_manifest(attr)
    fd = pd.to_datetime(full["Date"], errors="coerce") if "Date" in full.columns else pd.Series([], dtype="datetime64[ns]")
    _log.info(
        "Tier-1 wrote %s: %s rows (%s→%s) → %s",
        attr,
        f"{len(full):,}",
        str(fd.min())[:10] if len(fd) and fd.notna().any() else "NA",
        str(fd.max())[:10] if len(fd) and fd.notna().any() else "NA",
        parquet_path,
    )
    del df, full
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

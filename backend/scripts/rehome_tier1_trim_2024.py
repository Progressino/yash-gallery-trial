"""Re-home platform history into Tier-1 and trim pre-cutoff (2024) rows.

For each selected platform this script makes all three storage layers
consistent at ``Date >= cutoff`` (default 2025-01-01), which is required so a
backend restart does NOT resurrect the trimmed rows via the
stale-vs-Tier-3 / stale-vs-GitHub full-rebuild guards:

  1. Tier-1 (durable warm-cache parquet in WARM_CACHE_DIR): rebuilt from the
     UNION of the existing disk frame + full Tier-3 history, then filtered to
     >= cutoff and written directly (bypassing the app's downgrade guard).
  2. Tier-3 (SQLite daily_uploads): every blob is row-filtered to >= cutoff;
     emptied blobs are deleted.
  3. GitHub cache (--push-github): the trimmed disk snapshot + merged SKU map
     is re-uploaded so the manifest row_counts match the trimmed frames.

Run with --dry-run first to preview per-platform before/after counts.

Usage:
    python -m backend.scripts.rehome_tier1_trim_2024 \
        --platforms myntra flipkart meesho snapdeal \
        --cutoff 2025-01-01 --dry-run
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("rehome_trim")

PLATFORM_ATTR = {
    "amazon": "mtr_df",
    "myntra": "myntra_df",
    "meesho": "meesho_df",
    "flipkart": "flipkart_df",
    "snapdeal": "snapdeal_df",
}


def _date_series(df: pd.DataFrame) -> pd.Series:
    for col in ("Date", "TxnDate", "_Date"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series([pd.NaT] * len(df))


def _fmt(df: pd.DataFrame) -> str:
    if df is None or getattr(df, "empty", True):
        return "0 rows"
    d = _date_series(df)
    lo = str(d.min())[:10] if d.notna().any() else "NA"
    hi = str(d.max())[:10] if d.notna().any() else "NA"
    return f"{len(df):,} rows {lo}->{hi}"


def rehome_tier1(platform: str, cutoff: pd.Timestamp, cache_dir: Path, dry: bool) -> pd.DataFrame:
    from backend.services.daily_store import load_platform_data, merge_platform_data
    from backend.services.helpers import _coerce_df_for_parquet

    attr = PLATFORM_ATTR[platform]
    parquet_path = cache_dir / f"{attr}.parquet"

    disk = pd.read_parquet(parquet_path) if parquet_path.is_file() else pd.DataFrame()
    tier3 = load_platform_data(platform, months=None, dedup=True)
    if tier3 is None:
        tier3 = pd.DataFrame()

    _log.info("[%s] Tier-1 disk : %s", platform, _fmt(disk))
    _log.info("[%s] Tier-3 full : %s", platform, _fmt(tier3))

    if disk.empty and tier3.empty:
        _log.warning("[%s] no data in either layer — skipping", platform)
        return pd.DataFrame()

    merged = merge_platform_data(disk, tier3, platform)
    d = _date_series(merged)
    trimmed = merged.loc[d >= cutoff].copy()
    _log.info("[%s] union       : %s", platform, _fmt(merged))
    _log.info("[%s] trimmed(>=%s): %s", platform, cutoff.date(), _fmt(trimmed))

    if not dry:
        if trimmed.empty:
            _log.warning("[%s] trimmed frame empty — NOT overwriting disk parquet", platform)
        else:
            _coerce_df_for_parquet(trimmed).to_parquet(parquet_path, index=False)
            _log.info("[%s] wrote Tier-1 parquet → %s", platform, parquet_path)
    return trimmed


def trim_tier3(platform: str, cutoff: pd.Timestamp, dry: bool) -> dict:
    from backend.services.daily_store import _df_to_parquet, _extract_date_range, _get_conn

    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, data_parquet FROM daily_uploads WHERE platform=?",
        (platform,),
    ).fetchall()
    deleted = rewritten = kept = dropped = 0
    for _id, _fname, blob in rows:
        try:
            df = pd.read_parquet(io.BytesIO(blob))
        except Exception:
            kept += 1
            continue
        d = _date_series(df)
        keep = df.loc[d >= cutoff]
        if len(keep) == len(df):
            kept += 1
            continue
        dropped += len(df) - len(keep)
        if keep.empty:
            deleted += 1
            if not dry:
                conn.execute("DELETE FROM daily_uploads WHERE id=?", (_id,))
        else:
            rewritten += 1
            if not dry:
                df_from, df_to = _extract_date_range(keep)
                conn.execute(
                    "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=? WHERE id=?",
                    (_df_to_parquet(keep), len(keep), df_from, df_to, _id),
                )
    if not dry:
        conn.commit()
    conn.close()
    _log.info(
        "[%s] Tier-3 trim: %s blobs untouched, %s rewritten, %s deleted, %s rows dropped",
        platform,
        kept,
        rewritten,
        deleted,
        f"{dropped:,}",
    )
    return {"kept": kept, "rewritten": rewritten, "deleted": deleted, "dropped": dropped}


def push_github(cache_dir: Path, dry: bool) -> None:
    if dry:
        _log.info("DRY-RUN: skipping GitHub push")
        return
    import backend.main as m
    from backend.services.github_cache import save_cache_to_drive
    from backend.services.sku_mapping import load_sku_mapping_from_disk

    ok, disk_data = m._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not disk_data:
        _log.error("could not load disk warm cache for GitHub snapshot — SKIPPING push")
        return
    sku = load_sku_mapping_from_disk()
    if sku:
        disk_data["sku_mapping"] = sku
    _log.info("Pushing snapshot to GitHub cache (%d keys)…", len(disk_data))
    ok2, msg = save_cache_to_drive(disk_data)
    _log.info("GitHub push: ok=%s %s", ok2, msg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--platforms", nargs="+", default=["myntra", "flipkart", "meesho", "snapdeal"], choices=list(PLATFORM_ATTR))
    p.add_argument("--cutoff", default="2025-01-01")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--push-github", action="store_true")
    p.add_argument("--skip-tier3", action="store_true")
    args = p.parse_args(argv)

    cutoff = pd.Timestamp(args.cutoff)
    cache_dir = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    _log.info("=== rehome+trim  cutoff=%s  platforms=%s  dry_run=%s ===", cutoff.date(), args.platforms, args.dry_run)

    for plat in args.platforms:
        _log.info("----- %s -----", plat)
        rehome_tier1(plat, cutoff, cache_dir, args.dry_run)
        if not args.skip_tier3:
            trim_tier3(plat, cutoff, args.dry_run)

    if not args.dry_run:
        manifest_path = cache_dir / "_manifest.json"
        try:
            import json
            from datetime import datetime

            from backend.main import IST, _WARM_PLATFORM_KEYS

            existing = set()
            if manifest_path.is_file():
                existing = set(json.loads(manifest_path.read_text()).get("keys") or [])
            keys = sorted(existing | {PLATFORM_ATTR[p] for p in args.platforms})
            manifest_path.write_text(json.dumps({"saved_at": datetime.now(IST).isoformat(), "keys": keys}))
            _log.info("Refreshed manifest saved_at (keys=%d)", len(keys))
        except Exception:
            _log.exception("manifest refresh failed")

    if args.push_github:
        push_github(cache_dir, args.dry_run)

    _log.info("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Ingest Meesho monthly TCS-sales archives into Tier-1 warm cache.

Input: master ZIP files (ZIP-of-ZIPs where each inner ZIP is a month folder).
These are pre-built locally from the original RAR archives since unrar is not
available inside the container.

Run inside the backend container (PYTHONPATH=/srv, workdir /srv):
  python3 backend/scripts/_ingest_meesho_rar_tier1.py \
      --zip /tmp/meesho_pe.zip --zip /tmp/meesho_ag.zip [--dry-run]
"""

import argparse, io, json, sys, zipfile, logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/srv")
from backend.services.meesho import load_meesho_from_zip
from backend.services.daily_store import merge_platform_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

WARM_CACHE_DIR = Path("/data/warm_cache")
PARQUET_PATH   = WARM_CACHE_DIR / "meesho_df.parquet"
MANIFEST_PATH  = WARM_CACHE_DIR / "manifest.json"
MIN_DATE       = pd.Timestamp("2025-01-01")


def parse_zip_meesho(zip_path: str, label: str) -> pd.DataFrame:
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    df, inner_count, skipped = load_meesho_from_zip(zip_bytes, source_filename=label)
    _log.info("  Parsed %d inner ZIPs, %d rows; skipped: %s", inner_count, len(df), skipped)
    return df


def qsummary(df: pd.DataFrame, label: str = "") -> None:
    if df is None or df.empty:
        _log.info("%s: EMPTY", label)
        return
    ship = df[df.get("TxnType", df.get("txntype", pd.Series(dtype=str))).astype(str).str.lower().eq("shipment")]
    dt = pd.to_datetime(ship["Date"], errors="coerce")
    ship = ship[dt.notna()]
    dt   = dt[dt.notna()]
    if ship.empty:
        _log.info("%s: no shipments", label)
        return
    q = dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)
    n = pd.to_numeric(ship["Quantity"], errors="coerce").fillna(0).groupby(q).sum().astype(int)
    rows = " | ".join(f"{k}={v}" for k, v in sorted(n.items()) if v > 0)
    _log.info("%s: %s  [total ships=%d]", label, rows, int(n.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="append", required=True, help="Master ZIP file path (repeat for multiple)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Parse all ZIPs and merge
    new_df = pd.DataFrame()
    for zip_path in args.zip:
        label = Path(zip_path).name
        _log.info("Parsing %s", label)
        df = parse_zip_meesho(zip_path, label)
        if df.empty:
            _log.warning("  -> EMPTY after parse, skipping")
            continue
        qsummary(df, f"  {label}")
        new_df = merge_platform_data(new_df, df, "meesho", source_filename=label) if not new_df.empty else df

    if new_df.empty:
        _log.error("No data parsed from any RAR — aborting")
        sys.exit(1)

    qsummary(new_df, "NEW total (pre-merge with existing)")

    # Load existing Tier-1 meesho_df
    existing = pd.DataFrame()
    if PARQUET_PATH.exists():
        _log.info("Loading existing %s", PARQUET_PATH)
        existing = pd.read_parquet(PARQUET_PATH)
        qsummary(existing, "EXISTING meesho_df")
    else:
        _log.warning("No existing Tier-1 meesho_df found — will create fresh")

    # Merge
    merged = merge_platform_data(existing, new_df, "meesho", source_filename="meesho_rar_ingest")
    qsummary(merged, "MERGED")

    # Apply min-date filter
    dt_col = pd.to_datetime(merged["Date"], errors="coerce")
    before = len(merged)
    merged = merged[dt_col.ge(MIN_DATE) | dt_col.isna()].reset_index(drop=True)
    _log.info("Min-date filter (%s): %d -> %d rows", MIN_DATE.date(), before, len(merged))

    # Refuse to shrink (safety)
    if not existing.empty and len(merged) < len(existing):
        _log.error(
            "Merge would shrink from %d to %d rows — safety abort. "
            "Check for OMS_SKU mismatch or date filter issue.",
            len(existing), len(merged),
        )
        sys.exit(1)

    qsummary(merged, "FINAL (after filter)")
    _log.info("OMS_SKU fill: %d/%d", int(merged["OMS_SKU"].astype(str).str.strip().ne("").sum()), len(merged))

    if args.dry_run:
        _log.info("DRY RUN — no files written")
        return

    WARM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _log.info("Writing %s (%d rows)", PARQUET_PATH, len(merged))
    merged.to_parquet(PARQUET_PATH, index=False)

    # Update manifest
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    manifest["meesho_df"] = {
        "rows": len(merged),
        "updated": pd.Timestamp.utcnow().isoformat(),
        "source": "rar_ingest",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    _log.info("Done — meesho Tier-1 updated successfully")


if __name__ == "__main__":
    main()

"""Server-side SKU-mapping ingest.

Parses a multi-sheet SKU-mapping Excel file, overlays it onto the current
master map (uploaded keys win), and persists the merged map globally
(disk warm-cache JSON + in-memory warm cache). This is additive: existing
keys are preserved unless the upload provides a new value for the same key.

Usage:
    python -m backend.scripts.ingest_sku_mapping_disk /tmp/mapping.xlsx [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("ingest_sku_mapping")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("xlsx_path", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Parse + report, do not persist.")
    args = p.parse_args(argv)

    if not args.xlsx_path.is_file():
        _log.error("Missing file: %s", args.xlsx_path)
        return 2

    from backend.services.sku_mapping import (
        load_sku_mapping_from_disk,
        merge_sku_mapping_upload,
        parse_sku_mapping,
        persist_sku_mapping_globally,
    )

    raw = args.xlsx_path.read_bytes()
    parsed = parse_sku_mapping(raw)
    _log.info("Parsed %s mapping keys from %s", f"{len(parsed):,}", args.xlsx_path.name)
    if not parsed:
        _log.error("ABORT: no mapping keys parsed.")
        return 2

    existing = load_sku_mapping_from_disk()
    _log.info("Existing disk master map: %s keys", f"{len(existing):,}")

    new_keys = [k for k in parsed if k not in existing]
    changed = [k for k in parsed if k in existing and existing[k] != parsed[k]]
    _log.info(
        "Upload effect: +%s new keys, %s changed values, %s unchanged",
        f"{len(new_keys):,}",
        f"{len(changed):,}",
        f"{len(parsed) - len(new_keys) - len(changed):,}",
    )

    merged = merge_sku_mapping_upload(existing, parsed)
    _log.info("Merged master map: %s → %s keys", f"{len(existing):,}", f"{len(merged):,}")

    if args.dry_run:
        _log.info("DRY-RUN: not persisting.")
        return 0

    persist_sku_mapping_globally(merged)
    _log.info("Persisted merged SKU map to disk warm-cache (%s keys).", f"{len(merged):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

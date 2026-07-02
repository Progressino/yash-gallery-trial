"""Push the durable on-disk warm cache to the GitHub Release cache.

Startup Phase-2 MERGES the disk frames with the GitHub cache (union per
platform), so a trimmed/updated disk frame will be *re-polluted* on restart
unless the GitHub cache is updated to match. This script uploads the current
``WARM_CACHE_DIR`` frames (+ the disk SKU map) so the GitHub manifest row_counts
match disk.

Designed to run **detached** on the server with a progress log:

    docker exec -d <cid> sh -c 'python -m backend.scripts.push_warm_cache_github \
        > /tmp/gh_push.log 2>&1'
    # then poll /tmp/gh_push.log

Use --only to push a subset of platform keys (e.g. --only snapdeal_df mtr_df).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("push_warm_cache_github")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None, help="Subset of cache keys to push (default: all present).")
    args = p.parse_args(argv)

    import backend.main as m
    from backend.services.github_cache import save_cache_to_drive
    from backend.services.sku_mapping import load_sku_mapping_from_disk

    ok, disk_data = m._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not disk_data:
        _log.error("could not load disk warm cache — nothing to push")
        return 1
    sku = load_sku_mapping_from_disk()
    if sku:
        disk_data["sku_mapping"] = sku

    if args.only:
        keep = set(args.only) | {"sku_mapping"}
        disk_data = {k: v for k, v in disk_data.items() if k in keep}

    summary = []
    for k, v in disk_data.items():
        n = len(v) if hasattr(v, "__len__") else 0
        summary.append(f"{k}={n:,}")
    _log.info("Pushing %d keys: %s", len(disk_data), ", ".join(sorted(summary)))

    def cb(step: int, total: int, msg: str) -> None:
        _log.info("[%d/%d] %s", step + 1, total, msg)

    t0 = time.time()
    ok2, msg = save_cache_to_drive(disk_data, progress_callback=cb)
    _log.info("RESULT ok=%s (%.0fs): %s", ok2, time.time() - t0, msg)
    sys.stdout.flush()
    return 0 if ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())

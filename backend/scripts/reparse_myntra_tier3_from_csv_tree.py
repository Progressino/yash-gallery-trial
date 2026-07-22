#!/usr/bin/env python3
"""Walk a directory tree and re-save matching Myntra Tier-3 uploads from Seller CSV files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.services.daily_store import (  # noqa: E402
    _df_to_parquet,
    _extract_date_range,
    _get_conn,
    clear_tier3_range_cache,
)
from backend.services.myntra import _parse_myntra_csv  # noqa: E402
from backend.services.sku_mapping import (  # noqa: E402
    load_bundled_sku_mapping,
    load_sku_mapping_from_disk,
    merge_sku_mapping_upload,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Directories to search for Seller_Orders_Report CSVs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )
    conn = _get_conn()
    known: dict[str, tuple[int, str]] = {}
    for row_id, fn in conn.execute(
        "SELECT id, filename FROM daily_uploads WHERE platform='myntra'"
    ).fetchall():
        base = Path(str(fn).replace("\\", "/").split("/")[-1]).name
        known[base] = (row_id, fn)
    updated = 0
    seen: set[str] = set()
    for root in args.roots:
        for p in Path(root).rglob("*.csv"):
            if "seller_orders_report" not in p.name.lower().replace("-", "_"):
                continue
            tier = known.get(p.name)
            if not tier or p.name in seen:
                continue
            seen.add(p.name)
            row_id, tier_name = tier
            raw = p.read_bytes()
            df, msg = _parse_myntra_csv(raw, p.name, mapping)
            if df.empty:
                print("FAIL", p.name, msg)
                continue
            print("OK", p.name, "rows", len(df))
            if not args.dry_run:
                date_from, date_to = _extract_date_range(df)
                conn.execute(
                    "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=?, data_raw=? WHERE id=?",
                    (_df_to_parquet(df), len(df), date_from, date_to, raw, row_id),
                )
                updated += 1
    if not args.dry_run and updated:
        conn.commit()
    conn.close()
    if not args.dry_run and updated:
        clear_tier3_range_cache()
    print("updated", updated, "csv_matched", len(seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

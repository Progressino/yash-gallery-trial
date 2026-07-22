#!/usr/bin/env python3
"""Re-parse every Myntra Tier-3 upload that has stored raw CSV bytes (data_raw)."""
from __future__ import annotations

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
    mapping = merge_sku_mapping_upload(
        load_bundled_sku_mapping(), load_sku_mapping_from_disk()
    )
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, data_raw FROM daily_uploads "
        "WHERE platform='myntra' AND data_raw IS NOT NULL AND length(data_raw) > 0"
    ).fetchall()
    updated = skipped = 0
    for row_id, filename, raw in rows:
        df, msg = _parse_myntra_csv(raw, filename, mapping)
        if df.empty:
            print("SKIP", filename.split("/")[-1][:70], msg)
            skipped += 1
            continue
        date_from, date_to = _extract_date_range(df)
        conn.execute(
            "UPDATE daily_uploads SET data_parquet=?, rows=?, date_from=?, date_to=? WHERE id=?",
            (_df_to_parquet(df), len(df), date_from, date_to, row_id),
        )
        updated += 1
        print("OK", filename.split("/")[-1][:70], "rows", len(df))
    conn.commit()
    conn.close()
    clear_tier3_range_cache()
    print("with_raw", len(rows), "updated", updated, "skipped", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

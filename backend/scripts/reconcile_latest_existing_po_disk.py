"""
Re-finalize on-disk Existing PO parquet + meta after deploy.

Run on production:
  docker compose -p progressino -f docker-compose.prod.yml exec -T backend \
    python -m backend.scripts.reconcile_latest_existing_po_disk
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> int:
    from backend.services.existing_po import (
        _existing_po_disk_dir,
        existing_po_meta_bundle,
        existing_po_pipeline_totals,
        finalize_existing_po_dataframe,
        persist_existing_po_to_disk,
        read_existing_po_disk_meta,
        reconcile_existing_po_manual_raise_sidecar,
        seed_existing_po_warm_cache_from_disk,
    )
    from backend.session import AppSession

    path = _existing_po_disk_dir() / "existing_po_df.parquet"
    if not path.is_file():
        _log.warning("No existing_po_df.parquet at %s — nothing to reconcile", path)
        return 0

    import pandas as pd

    raw = pd.read_parquet(path)
    before_rows = len(raw)
    before_dupes = int(raw["OMS_SKU"].astype(str).duplicated().sum()) if "OMS_SKU" in raw.columns else 0
    fin = finalize_existing_po_dataframe(raw)
    after_rows = len(fin)
    units_before, _ = existing_po_pipeline_totals(raw)
    units_after, sku_n = existing_po_pipeline_totals(fin)

    meta_before = read_existing_po_disk_meta() or {}
    gen = int(meta_before.get("existing_po_generation") or 0)

    sess = AppSession()
    sess.existing_po_df = fin
    sess.existing_po_generation = gen
    for key in (
        "existing_po_uploaded_at",
        "existing_po_filename",
        "existing_po_manual_raise_date",
        "existing_po_manual_upload",
        "existing_po_sheet_pipeline_units",
        "existing_po_totals_match",
    ):
        if key in meta_before:
            setattr(sess, key, meta_before[key])

    reconcile_existing_po_manual_raise_sidecar(sess)
    if not persist_existing_po_to_disk(sess):
        _log.error("persist_existing_po_to_disk failed")
        return 1

    meta = existing_po_meta_bundle(sess)
    meta_path = _existing_po_disk_dir() / "existing_po_meta.json"
    meta_path.write_text(json.dumps(meta, default=str), encoding="utf-8")

    try:
        seed_existing_po_warm_cache_from_disk()
    except Exception:
        _log.exception("seed_existing_po_warm_cache_from_disk failed")

    _log.info(
        "Reconciled Existing PO gen=%s: rows %s→%s (dupes removed=%s), pipeline %s→%s (%s SKUs)",
        gen,
        before_rows,
        after_rows,
        before_dupes,
        units_before,
        units_after,
        sku_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

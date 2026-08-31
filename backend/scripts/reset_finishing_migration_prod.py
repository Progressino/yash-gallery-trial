#!/usr/bin/env python3
"""Prod: reset open Finishing migration JOs (preserve Ready-to-Finishing).

Usage:
  python3 backend/scripts/reset_finishing_migration_prod.py --dry-run
  python3 backend/scripts/reset_finishing_migration_prod.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _clear_existing_po_finishing_overlay() -> dict:
    """Clear Finishing_* columns from disk Existing PO (PO pipeline overlay)."""
    try:
        import pandas as pd
        from backend.services.finishing_receipt import (
            _FINISHING_META_COLS,
            _ensure_pipeline_columns,
        )
        from backend.services.existing_po import existing_po_merge_key, dedupe_po_rows_by_sku
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    root = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    path = root / "existing_po_df.parquet"
    if not path.is_file():
        return {"skipped": True, "reason": "no existing_po_df.parquet"}

    df = pd.read_parquet(path)
    if df.empty:
        return {"skipped": True, "reason": "empty parquet"}

    out = _ensure_pipeline_columns(df.copy())
    out["OMS_SKU"] = out["OMS_SKU"].map(existing_po_merge_key)
    mask = pd.to_numeric(out.get("Finishing_Received", 0), errors="coerce").fillna(0) > 0
    mask = mask | (pd.to_numeric(out.get("Finishing_Balance", 0), errors="coerce").fillna(0) > 0)
    mask = mask | (pd.to_numeric(out.get("Finishing_Issued", 0), errors="coerce").fillna(0) > 0)
    n = int(mask.sum())
    if n <= 0:
        return {"cleared_skus": 0, "path": str(path)}

    for col in _FINISHING_META_COLS:
        if col not in out.columns:
            continue
        if col.endswith(("No", "Date", "Status")):
            out.loc[mask, col] = ""
        else:
            out.loc[mask, col] = 0

    pending = pd.to_numeric(out.loc[mask, "Pending_Cutting"], errors="coerce").fillna(0).astype(int)
    out.loc[mask, "Balance_to_Dispatch"] = 0
    out.loc[mask, "PO_Pipeline_Total"] = pending
    out = dedupe_po_rows_by_sku(out.reset_index(drop=True))
    out.to_parquet(path, index=False)
    return {"cleared_skus": n, "path": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Execute reset (default is dry-run)")
    ap.add_argument("--skip-po-overlay", action="store_true")
    args = ap.parse_args()
    dry_run = not args.apply

    from backend.db import production_db

    stats = production_db.reset_finishing_migration(
        dry_run=dry_run,
        actor="prod-script",
    )
    print("FINISHING_RESET", json.dumps(stats, indent=2))

    if dry_run:
        print("DRY_RUN — pass --apply to execute")
        return 0

    overlay = {"skipped": True}
    if not args.skip_po_overlay:
        overlay = _clear_existing_po_finishing_overlay()
    print("PO_OVERLAY", json.dumps(overlay, indent=2))

    # Post-check
    import sqlite3

    db = os.environ.get("PRODUCTION_DB_PATH", "/data/production.db")
    conn = sqlite3.connect(db)
    open_n = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(planned_qty),0)
           FROM job_orders WHERE process='Finishing'
             AND IFNULL(status,'') NOT IN ('Cancelled','Closed')"""
    ).fetchone()
    dup = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT 1 FROM job_orders WHERE process='Finishing'
               AND IFNULL(status,'') NOT IN ('Cancelled','Closed')
             GROUP BY so_number, UPPER(TRIM(sku)) HAVING COUNT(*)>1
           )"""
    ).fetchone()[0]
    ready = conn.execute("SELECT COUNT(*) FROM ready_to_wip_imports").fetchone()[0]
    conn.close()
    print(
        "POST_CHECK",
        json.dumps(
            {
                "open_finishing_jos": open_n[0],
                "open_planned": open_n[1],
                "duplicate_keys": dup,
                "ready_to_wip_rows": ready,
            }
        ),
    )
    if int(open_n[0] or 0) > 0:
        print("WARN: open Finishing JOs remain — review before re-upload")
        return 1
    print("OK — Ready for clean Finishing JO re-upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

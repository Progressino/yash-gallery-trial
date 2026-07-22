#!/usr/bin/env python3
"""Re-parse inventory RAR on the server and write warm-cache snapshot (prod repair)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

import pandas as pd

from backend.services.inventory import (
    apply_inventory_snapshot_metadata,
    load_inventory_consolidated,
    refresh_inventory_api_cache,
)
from backend.services.sku_mapping import load_sku_mapping_from_disk
from backend.session import AppSession


def main() -> None:
    rar_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inv_21jul26.rar")
    if not rar_path.is_file():
        raise SystemExit(f"RAR not found: {rar_path}")
    raw = rar_path.read_bytes()
    mapping = load_sku_mapping_from_disk() or {}
    df, dbg = load_inventory_consolidated(None, None, None, raw, mapping, return_debug=True)
    if df.empty:
        raise SystemExit(f"Parse produced empty frame: {dbg}")
    sess = AppSession()
    sess.inventory_df_variant = df
    sess.inventory_df_parent = df.copy()
    sess.inventory_debug = dbg
    # Keep the user's current manual in-transit overlay when repairing.
    ov_path = Path("/data/warm_cache/manual_intransit_overlay_df.parquet")
    if ov_path.is_file():
        try:
            from backend.services.manual_intransit_sheet import (
                apply_manual_intransit_overlay_to_inventory,
            )

            ov = pd.read_parquet(ov_path)
            sess.manual_intransit_overlay_df = ov
            apply_manual_intransit_overlay_to_inventory(sess)
        except Exception as exc:
            print("overlay apply skipped:", exc)
    apply_inventory_snapshot_metadata(
        sess,
        file_parts=[(rar_path.name, raw)],
        debug=dbg,
    )
    # Force a fresh uploaded_at so disk/warm downgrade guards accept the repair.
    sess.inventory_snapshot_uploaded_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    refresh_inventory_api_cache(sess)
    totals = sess.inventory_api_totals or {}
    print("Total_Inventory", totals.get("Total_Inventory"))
    print("Amazon", totals.get("Amazon_Inventory"))
    print("OMS", totals.get("OMS_Inventory"))
    print("Manual", totals.get("Manual_InTransit"))
    print("NotIn", totals.get("Not_In_Inventory_Qty"))
    print("rows", len(sess.inventory_df_variant))
    checks = getattr(sess, "inventory_auto_checks", [])
    print("auto_checks", json.dumps(checks, indent=2))
    import backend.main as main_mod

    main_mod.merge_inventory_into_warm_cache(sess)
    warm = main_mod._warm_cache.get("inventory_df_variant")
    if warm is not None and not warm.empty:
        print(
            "warm after merge",
            {
                c: int(pd.to_numeric(warm[c], errors="coerce").fillna(0).sum())
                for c in warm.columns
                if any(k in c for k in ("Inventory", "InTransit", "Not_In", "Total"))
            },
        )
    print("meta", sess.inventory_snapshot_date, sess.inventory_snapshot_uploaded_at)


if __name__ == "__main__":
    main()

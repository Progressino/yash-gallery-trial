#!/usr/bin/env python3
"""Repair Amazon_Inventory on warm cache using pipeline snapshot (prod one-off)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from backend.services.inventory import (
    inventory_session_meta_bundle,
    recompute_inventory_totals,
    refresh_inventory_api_cache,
)
from backend.session import AppSession


def main() -> None:
    warm = Path("/data/warm_cache")
    cur_path = warm / "inventory_df_variant.parquet"
    pipe_path = warm / "pipeline" / "inventory_snapshot.parquet"
    cur = pd.read_parquet(cur_path).set_index("OMS_SKU")
    pipe = pd.read_parquet(pipe_path).set_index("OMS_SKU")
    shared = cur.index.intersection(pipe.index)
    cur.loc[shared, "Amazon_Inventory"] = pipe.loc[shared, "Amazon_Inventory"]
    missing = pipe.index.difference(cur.index)
    add = pipe.loc[missing]
    add = add[add["Amazon_Inventory"] > 0]
    out = pd.concat([cur, add])
    out = recompute_inventory_totals(out.reset_index())
    out = out[out["Total_Inventory"] > 0].reset_index(drop=True)
    out.to_parquet(cur_path, index=False)
    parent = out.copy()
    parent.to_parquet(warm / "inventory_df_parent.parquet", index=False)
    meta_path = warm / "inventory_session_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    sess = AppSession()
    sess.inventory_df_variant = out
    sess.inventory_df_parent = parent
    sess.inventory_debug = meta.get("inventory_debug") or {}
    from datetime import datetime, timezone

    sess.inventory_snapshot_uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    refresh_inventory_api_cache(sess)
    meta.update(inventory_session_meta_bundle(sess))
    meta_path.write_text(json.dumps(meta, indent=2))
    totals = sess.inventory_api_totals or {}
    print("Repaired Total_Inventory", totals.get("Total_Inventory"))
    print("Amazon", totals.get("Amazon_Inventory"))
    print("auto_checks", getattr(sess, "inventory_auto_checks", []))


if __name__ == "__main__":
    main()

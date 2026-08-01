#!/usr/bin/env python3
"""Coalesce YEAL/BALCK/BOTTEL inventory-history aliases on disk parquet (safe)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    try:
        import pandas as pd
        from backend.services.daily_inventory_history import (
            coalesce_inventory_history_sku_aliases,
            clear_inventory_channel_view_cache,
        )
        from backend.services.sku_mapping import load_bundled_sku_mapping, load_sku_mapping_from_disk

        candidates = [
            Path("/data/warm_cache/daily_inventory_history_df.parquet"),
            Path("/srv/data/warm_cache/daily_inventory_history_df.parquet"),
            Path("/app/data/warm_cache/daily_inventory_history_df.parquet"),
        ]
        # Discover via helper if present
        try:
            from backend.services.daily_inventory_history import _warm_cache_dir

            candidates.insert(0, _warm_cache_dir() / "daily_inventory_history_df.parquet")
        except Exception:
            pass

        hist_path = next((p for p in candidates if p.is_file()), None)
        if hist_path is None:
            print(json.dumps({"ok": False, "error": "NO_HISTORY_PARQUET", "tried": [str(p) for p in candidates]}))
            return 2

        hist = pd.read_parquet(hist_path)
        before = int(hist["OMS_SKU"].nunique()) if "OMS_SKU" in hist.columns else 0
        yeal_before = int(hist["OMS_SKU"].astype(str).str.contains("YEAL", case=False, na=False).sum())
        mapping = {}
        try:
            mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
        except Exception:
            pass
        out = coalesce_inventory_history_sku_aliases(hist, mapping)
        yeal_after = int(out["OMS_SKU"].astype(str).str.contains("YEAL", case=False, na=False).sum())
        after = int(out["OMS_SKU"].nunique())
        out.to_parquet(hist_path, index=False)
        try:
            clear_inventory_channel_view_cache()
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(hist_path),
                    "sku_unique_before": before,
                    "sku_unique_after": after,
                    "yeal_rows_before": yeal_before,
                    "yeal_rows_after": yeal_after,
                    "rows": int(len(out)),
                }
            )
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()[-2000:]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

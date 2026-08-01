#!/usr/bin/env python3
"""One-shot: coalesce YEAL/BALCK/BOTTEL history aliases and persist warm parquet."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace


def main() -> int:
    from backend.services.daily_inventory_history import (
        coalesce_inventory_history_sku_aliases,
        persist_inventory_history_authoritative,
    )
    from backend.services.sku_mapping import load_bundled_sku_mapping, load_sku_mapping_from_disk

    import backend.main as main

    main.bootstrap_warm_cache_if_empty()
    hist = (main._warm_cache or {}).get("daily_inventory_history_df")
    if hist is None or getattr(hist, "empty", True):
        print("NO_HISTORY")
        return 2
    before = int(hist["OMS_SKU"].nunique()) if "OMS_SKU" in hist.columns else 0
    yeal_before = int(hist["OMS_SKU"].astype(str).str.contains("YEAL", case=False, na=False).sum())
    mapping = {}
    try:
        mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
    except Exception:
        pass
    try:
        for sess in (getattr(main, "_sessions", None) or {}).values():
            mp = getattr(sess, "sku_mapping", None) or {}
            if mp:
                mapping = {**mapping, **mp}
                break
    except Exception:
        pass
    out = coalesce_inventory_history_sku_aliases(hist, mapping)
    yeal_after = int(out["OMS_SKU"].astype(str).str.contains("YEAL", case=False, na=False).sum())
    after = int(out["OMS_SKU"].nunique())

    sess = None
    try:
        for s in (getattr(main, "_sessions", None) or {}).values():
            sess = s
            break
    except Exception:
        pass
    if sess is None:
        sess = SimpleNamespace(
            daily_inventory_history_df=out,
            daily_inventory_history_meta={},
            sku_mapping=mapping,
        )
    else:
        sess.daily_inventory_history_df = out

    # Force overwrite even if dates look identical: bump uploaded_at via meta if present.
    try:
        meta = getattr(sess, "daily_inventory_history_meta", None) or {}
        from datetime import datetime, timezone

        meta = dict(meta)
        meta["daily_inventory_history_uploaded_at"] = datetime.now(timezone.utc).isoformat()
        sess.daily_inventory_history_meta = meta
    except Exception:
        pass

    ok = persist_inventory_history_authoritative(sess, out)
    try:
        if not getattr(main, "_warm_cache", None):
            main._warm_cache = {}
        main._warm_cache["daily_inventory_history_df"] = out
        main._warm_cache_generation = int(getattr(main, "_warm_cache_generation", 0) or 0) + 1
        for s in (getattr(main, "_sessions", None) or {}).values():
            s.daily_inventory_history_df = out
    except Exception as e:
        print("warm_refresh_warn", e)
    print(
        json.dumps(
            {
                "ok": True,
                "persisted": bool(ok),
                "sku_unique_before": before,
                "sku_unique_after": after,
                "yeal_rows_before": yeal_before,
                "yeal_rows_after": yeal_after,
                "rows": int(len(out)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

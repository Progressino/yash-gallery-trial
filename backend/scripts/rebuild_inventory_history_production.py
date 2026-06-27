#!/usr/bin/env python3
"""Roll daily inventory history forward on production and persist to warm cache.

Loads only history + sales + inventory parquets (no full_restore_session) so this
stays within container memory limits.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))

_IST = ZoneInfo("Asia/Kolkata")
_CACHE = Path("/data/warm_cache")


def _load_session_from_disk():
    import pandas as pd

    from backend.session import AppSession

    sess = AppSession()
    hist_path = _CACHE / "daily_inventory_history_df.parquet"
    if not hist_path.is_file():
        raise FileNotFoundError(f"Missing {hist_path}")
    sess.daily_inventory_history_df = pd.read_parquet(hist_path)

    sales_path = _CACHE / "sales_df.parquet"
    if sales_path.is_file():
        sess.sales_df = pd.read_parquet(sales_path)

    inv_path = _CACHE / "inventory_df_variant.parquet"
    if inv_path.is_file():
        sess.inventory_df_variant = pd.read_parquet(inv_path)

    inv_meta_path = _CACHE / "inventory_session_meta.json"
    if inv_meta_path.is_file():
        inv_meta = json.loads(inv_meta_path.read_text(encoding="utf-8"))
        sess.inventory_snapshot_date = str(inv_meta.get("inventory_snapshot_date") or "")
        sess.inventory_snapshot_date_label = str(inv_meta.get("inventory_snapshot_date_label") or "")

    sess.daily_inventory_history_uploaded_at = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S")
    sess.daily_inventory_history_filename = "rollforward-rebuild"
    return sess


def _persist_history(sess) -> None:
    from backend.services.daily_inventory_history import (
        daily_inventory_history_meta_bundle,
        persist_daily_inventory_history_meta,
    )

    df = getattr(sess, "daily_inventory_history_df", None)
    if df is None or df.empty:
        raise RuntimeError("empty history after refresh")

    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / "daily_inventory_history_df.parquet"
    df.to_parquet(path, index=False)

    meta = daily_inventory_history_meta_bundle(sess)
    meta_path = _CACHE / "daily_inventory_history_meta.json"
    meta_path.write_text(json.dumps(meta, default=str, indent=2), encoding="utf-8")
    persist_daily_inventory_history_meta(sess)

    # Touch manifest so startup reload picks up the file.
    manifest_path = _CACHE / "_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = set(manifest.get("keys") or [])
    keys.add("daily_inventory_history_df")
    manifest["keys"] = sorted(keys)
    manifest["saved_at"] = datetime.now(_IST).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        import backend.main as main_mod

        if not main_mod._warm_cache:
            main_mod._warm_cache = {}
        main_mod._warm_cache["daily_inventory_history_df"] = df.copy()
        main_mod._warm_cache[main_mod._DAILY_INV_META_WARM_KEY] = meta
    except Exception:
        pass


def main() -> int:
    from backend.services.daily_inventory_history import (
        read_daily_inventory_history_disk_meta,
        refresh_inventory_history_rollforward,
    )

    before_meta = read_daily_inventory_history_disk_meta() or {}
    print("BEFORE:", json.dumps(before_meta, indent=2, default=str), flush=True)

    sess = _load_session_from_disk()
    sales_df = getattr(sess, "sales_df", None)
    inv_df = getattr(sess, "inventory_df_variant", None)
    print(
        "LOADED:",
        f"history={len(sess.daily_inventory_history_df):,}",
        f"sales={len(sales_df) if sales_df is not None else 0:,}",
        f"inventory={len(inv_df) if inv_df is not None else 0:,}",
        f"snapshot={getattr(sess, 'inventory_snapshot_date', '')}",
        flush=True,
    )

    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    print("REFRESH:", json.dumps(result, indent=2, default=str), flush=True)
    if not result.get("ok"):
        print("FAIL: refresh returned not ok", flush=True)
        return 1

    _persist_history(sess)
    after_meta = read_daily_inventory_history_disk_meta() or {}
    print("AFTER:", json.dumps(after_meta, indent=2, default=str), flush=True)

    max_d = str(after_meta.get("daily_inventory_history_max_date") or result.get("max_date") or "")
    if max_d < "2026-06-20":
        print(f"FAIL: history max_date still too old ({max_d})", flush=True)
        return 1

    try:
        from backend.services.po_shared_cache import invalidate_all_shared_caches

        n = invalidate_all_shared_caches()
        print(f"PO shared cache invalidated ({n} entries)", flush=True)
    except Exception as exc:
        print("WARN: shared cache invalidate failed:", exc, flush=True)

    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Local development helpers (sync hydrate, env detection)."""
from __future__ import annotations

import os
from pathlib import Path


def local_dev_mode() -> bool:
    raw = (os.environ.get("LOCAL_DEV") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    wc = (os.environ.get("WARM_CACHE_DIR") or "").strip()
    if ".local-data" in wc or "/local-data/" in wc.replace("\\", "/"):
        return True
    try:
        return Path(wc).resolve().parts[-2:] == (".local-data", "warm_cache")
    except Exception:
        return False


def apply_local_dev_defaults() -> None:
    """Lightweight local defaults — must run before warm-cache disk load."""
    if not local_dev_mode():
        return
    os.environ.setdefault("WARM_CACHE_PO_SESSION_ONLY", "1")
    os.environ.setdefault("WARM_CACHE_SKIP_PHASE2_WHEN_DISK_FRESH", "1")
    os.environ.setdefault("SESSION_SHARED_FRAMES", "1")


def hydrate_session_from_warm_local(sess, warm_cache: dict, warm_cache_generation: int) -> int:
    """Fast local login hydrate — unified sales + inventory only (no 1M-row platform copy)."""
    from .services.shared_frames import attach_shared_frames, shared_frames_enabled
    from .session import resume_auto_data_restore

    if not warm_cache:
        return 0

    if shared_frames_enabled():
        attach_shared_frames(sess, warm_cache_generation=warm_cache_generation)
    else:
        for key in (
            "sales_df",
            "sku_mapping",
            "inventory_df_variant",
            "inventory_df_parent",
            "existing_po_df",
        ):
            val = warm_cache.get(key)
            if val is None:
                continue
            setattr(sess, key, val)

    meta = warm_cache.get("inventory_session_meta")
    if isinstance(meta, dict) and meta:
        try:
            from .services.inventory import apply_inventory_session_meta, ensure_inventory_snapshot_metadata

            apply_inventory_session_meta(sess, meta)
            ensure_inventory_snapshot_metadata(sess)
        except Exception:
            pass

    if not getattr(sess, "sku_mapping", None):
        try:
            from .services.sku_mapping import restore_sku_mapping_to_session

            restore_sku_mapping_to_session(sess)
        except Exception:
            pass

    sess._warm_cache_gen = int(warm_cache_generation or 0)
    sess._warm_cache_only = True
    sess._quarterly_cache.clear()
    sess.daily_restored = True
    resume_auto_data_restore(sess)

    sales = getattr(sess, "sales_df", None)
    if sales is not None and hasattr(sales, "__len__"):
        return int(len(sales))
    return 0

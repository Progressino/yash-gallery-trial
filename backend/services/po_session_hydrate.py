"""Hydrate PO-calculate essentials from server warm cache / disk.

POST /api/po/calculate uses a lightweight session path that skips PostgreSQL
restore and full warm-cache copy. Uploaded inventory and Existing PO still live
on disk under WARM_CACHE_DIR — this module loads them before calculate runs.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

_PO_CALC_PARQUET_KEYS = (
    "sales_df",
    "inventory_df_variant",
    "inventory_df_parent",
    "existing_po_df",
    "daily_inventory_history_df",
    "sku_status_lead_df",
    "po_raise_ledger_df",
    "po_return_overlay_df",
)

_PO_SIDECAR_KEYS = ("daily_inventory_history_df", "sku_status_lead_df")

# Placeholder uploads (e.g. a single ``Z-1`` test row) must not override real sheets.
_PLACEHOLDER_SKUS = frozenset({"Z-1", "TEST", "TEST-SKU", "DEMO-SKU"})


def _warm_cache_dir() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def _df_row_count(val: Any) -> int:
    if val is None or not hasattr(val, "empty"):
        return 0
    if getattr(val, "empty", True):
        return 0
    return int(len(val))


def _sidecar_sku_count(df: pd.DataFrame) -> int:
    if df is None or df.empty or "OMS_SKU" not in df.columns:
        return 0
    return int(df["OMS_SKU"].astype(str).nunique())


def _sidecar_looks_placeholder(key: str, df: pd.DataFrame | None) -> bool:
    if df is None or getattr(df, "empty", True):
        return False
    n = len(df)
    if key == "sku_status_lead_df":
        if n >= 100:
            return False
        if n <= 5:
            if "OMS_SKU" in df.columns:
                skus = set(df["OMS_SKU"].astype(str).str.strip().str.upper())
                if skus.issubset(_PLACEHOLDER_SKUS) or n == 1:
                    return True
            return False
        return n < 25
    if key == "daily_inventory_history_df":
        if n >= 5000:
            return False
        skus = _sidecar_sku_count(df)
        if n <= 10 or skus <= 5:
            if "OMS_SKU" in df.columns:
                skus_set = set(df["OMS_SKU"].astype(str).str.strip().str.upper())
                if skus_set.issubset(_PLACEHOLDER_SKUS):
                    return True
            return False
        return n < 500 and skus < 50
    return False


def _should_prefer_sidecar_backup(
    key: str,
    cur: pd.DataFrame | None,
    backup: pd.DataFrame | None,
) -> bool:
    cur_n = _df_row_count(cur)
    bak_n = _df_row_count(backup)
    if bak_n <= 0:
        return False
    if cur_n <= 0:
        return True
    if _sidecar_looks_placeholder(key, cur):
        return bak_n > cur_n
    return bak_n >= max(100, cur_n * 2) and cur_n < 500


def load_po_sidecar_backups_from_disk() -> dict[str, pd.DataFrame]:
    """Best sku_status / daily_inventory snapshots from warm_cache + github_cache dirs."""
    out: dict[str, tuple[int, pd.DataFrame]] = {}
    dirs: list[Path] = [_warm_cache_dir()]
    blob_root = Path(os.environ.get("GITHUB_BLOB_CACHE_DIR", "/data/github_cache"))
    if blob_root.is_dir():
        dirs.append(blob_root)
        dirs.extend(sorted(blob_root.glob("*/"), reverse=True))
    recovery = (os.environ.get("WARM_CACHE_RECOVERY_DIR") or "").strip()
    if recovery:
        dirs.append(Path(recovery))

    for base in dirs:
        if not base.is_dir():
            continue
        for key in _PO_SIDECAR_KEYS:
            path = base / f"{key}.parquet"
            if not path.is_file():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                _log.exception("po sidecar backup read failed: %s", path)
                continue
            n = _df_row_count(df)
            if n > out.get(key, (0, pd.DataFrame()))[0]:
                out[key] = (n, df)

    return {k: v for k, (_, v) in out.items() if _df_row_count(v) > 0}


def _persist_sidecars_to_warm_disk(frames: dict[str, pd.DataFrame]) -> None:
    if not frames:
        return
    try:
        import backend.main as _main

        _main._save_warm_cache_to_disk(frames)
    except Exception:
        _log.exception("persist restored PO sidecars to warm disk failed")


def ensure_po_sidecars_hydrated(sess) -> dict[str, int]:
    """Restore real SKU status + daily inventory matrix when placeholders overwrote them."""
    import backend.main as _main

    backups = load_po_sidecar_backups_from_disk()
    if not _main._warm_cache:
        _main._warm_cache = {}

    changed = False
    stats: dict[str, int] = {}

    for key in _PO_SIDECAR_KEYS:
        warm_df = _main._warm_cache.get(key)
        sess_df = getattr(sess, key, None)
        backup_df = backups.get(key)
        candidates = [d for d in (backup_df, warm_df, sess_df) if _df_row_count(d) > 0]
        if not candidates:
            stats[key] = 0
            continue
        non_placeholder = [d for d in candidates if not _sidecar_looks_placeholder(key, d)]
        best = max(non_placeholder or candidates, key=_df_row_count)

        if (
            _df_row_count(best) != _df_row_count(warm_df)
            or _sidecar_looks_placeholder(key, warm_df)
        ):
            _main._warm_cache[key] = best.copy()
            changed = True
        if _df_row_count(best) != _df_row_count(sess_df) or _sidecar_looks_placeholder(key, sess_df):
            setattr(sess, key, best.copy())
            changed = True
        stats[key] = _df_row_count(best)

    if changed:
        _persist_sidecars_to_warm_disk(
            {k: _main._warm_cache[k] for k in _PO_SIDECAR_KEYS if k in _main._warm_cache}
        )
        try:
            from .po_shared_cache import invalidate_all_shared_caches

            invalidate_all_shared_caches()
        except Exception:
            _log.exception("invalidate shared cache after PO sidecar restore failed")
        _log.info("PO sidecars restored (sku_status=%s daily_inv=%s)", stats.get("sku_status_lead_df"), stats.get("daily_inventory_history_df"))
        sess._quarterly_cache.clear()

    return stats


def effective_sku_status_df_for_engine(sess) -> pd.DataFrame | None:
    """Return status sheet for PO math, or None when only a placeholder row is loaded."""
    df = getattr(sess, "sku_status_lead_df", None)
    if df is None or getattr(df, "empty", True):
        return None
    if _sidecar_looks_placeholder("sku_status_lead_df", df):
        _log.warning(
            "Ignoring placeholder sku_status_lead_df (%s rows) — using global lead_time",
            len(df),
        )
        return None
    return df


def _should_replace_warm_frame(cur: Any, incoming: Any) -> bool:
    if incoming is None or not hasattr(incoming, "empty") or incoming.empty:
        return False
    if cur is None or not hasattr(cur, "empty") or cur.empty:
        return True
    return _df_row_count(incoming) > _df_row_count(cur)


def load_po_calc_essentials_from_disk() -> dict[str, Any]:
    """Read PO-calculate frames + metadata from WARM_CACHE_DIR (ignores manifest age)."""
    import backend.main as _main

    disk_dir = _warm_cache_dir()
    loaded: dict[str, Any] = dict(_main._warm_cache_loose_parquets_from_dir(disk_dir))

    for key in _PO_CALC_PARQUET_KEYS:
        if key in loaded:
            continue
        path = disk_dir / f"{key}.parquet"
        if path.is_file():
            try:
                loaded[key] = pd.read_parquet(path)
            except Exception:
                _log.exception("po hydrate: failed reading %s", path)

    inv_meta = disk_dir / "inventory_session_meta.json"
    if inv_meta.is_file():
        try:
            loaded[_main._INVENTORY_META_WARM_KEY] = json.loads(
                inv_meta.read_text(encoding="utf-8")
            )
        except Exception:
            _log.exception("po hydrate: inventory_session_meta read failed")

    ep_meta = disk_dir / "existing_po_meta.json"
    if ep_meta.is_file():
        try:
            loaded[_main._EXISTING_PO_META_WARM_KEY] = json.loads(
                ep_meta.read_text(encoding="utf-8")
            )
        except Exception:
            _log.exception("po hydrate: existing_po_meta read failed")

    return loaded


def ensure_po_calc_server_data_in_warm_cache() -> bool:
    """Top up in-memory warm cache with uploaded inventory / PO / sales from disk."""
    import backend.main as _main

    disk = load_po_calc_essentials_from_disk()
    if not disk:
        try:
            from .existing_po import seed_existing_po_warm_cache_from_disk

            return seed_existing_po_warm_cache_from_disk()
        except Exception:
            return False

    if not _main._warm_cache:
        _main._warm_cache = {}

    changed = False
    for key, val in disk.items():
        if key in (
            _main._INVENTORY_META_WARM_KEY,
            _main._EXISTING_PO_META_WARM_KEY,
        ):
            if isinstance(val, dict) and val:
                cur = _main._warm_cache.get(key)
                if not isinstance(cur, dict) or not cur:
                    _main._warm_cache[key] = dict(val)
                    changed = True
            continue
        cur = _main._warm_cache.get(key)
        if _should_replace_warm_frame(cur, val):
            _main._warm_cache[key] = val.copy() if hasattr(val, "copy") else val
            changed = True

    try:
        from .existing_po import seed_existing_po_warm_cache_from_disk

        changed = seed_existing_po_warm_cache_from_disk() or changed
    except Exception:
        _log.exception("seed_existing_po_warm_cache_from_disk during PO hydrate failed")

    if changed:
        _log.info(
            "PO hydrate: warm cache topped up from disk (inv=%s ep=%s sales=%s)",
            _df_row_count(_main._warm_cache.get("inventory_df_variant")),
            _df_row_count(_main._warm_cache.get("existing_po_df")),
            _df_row_count(_main._warm_cache.get("sales_df")),
        )
    return changed


def ensure_po_return_overlay_from_server(sess) -> bool:
    """Load saved return overlay from warm cache or disk when the session is empty."""
    cur = getattr(sess, "po_return_overlay_df", None)
    if cur is not None and not getattr(cur, "empty", True):
        return False
    import backend.main as _main

    try:
        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        _log.exception("restore_po_sidecars_from_warm for return overlay failed")
    cur = getattr(sess, "po_return_overlay_df", None)
    if cur is not None and not getattr(cur, "empty", True):
        return True

    path = _warm_cache_dir() / "po_return_overlay_df.parquet"
    if not path.is_file():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        _log.exception("read po_return_overlay_df.parquet failed")
        return False
    if df is None or getattr(df, "empty", True):
        return False
    sess.po_return_overlay_df = df.copy()
    if not isinstance(_main._warm_cache, dict):
        _main._warm_cache = {}
    _main._warm_cache["po_return_overlay_df"] = df.copy()
    _log.info("PO return overlay hydrated from disk (%s rows)", len(df))
    return True


def hydrate_po_session_for_calculate(sess) -> dict[str, int]:
    """Ensure session has sales, inventory, existing PO, and PO sidecars before calculate."""
    import backend.main as _main

    ensure_po_calc_server_data_in_warm_cache()
    ensure_po_return_overlay_from_server(sess)

    inv_before = _df_row_count(getattr(sess, "inventory_df_variant", None))
    sales_before = _df_row_count(getattr(sess, "sales_df", None))
    ep_before = _df_row_count(getattr(sess, "existing_po_df", None))

    if sales_before == 0:
        wc_sales = _main._warm_cache.get("sales_df")
        if wc_sales is not None and not wc_sales.empty:
            sess.sales_df = wc_sales.copy()

    try:
        from .inventory import sync_inventory_snapshot_from_warm

        sync_inventory_snapshot_from_warm(sess)
    except Exception:
        _log.exception("sync_inventory_snapshot_from_warm during PO hydrate failed")

    if _df_row_count(getattr(sess, "inventory_df_variant", None)) == 0:
        for key in _main._INVENTORY_WARM_KEYS:
            wc = _main._warm_cache.get(key)
            if wc is not None and not wc.empty:
                setattr(sess, key, wc.copy())
        meta = _main._warm_cache.get(_main._INVENTORY_META_WARM_KEY)
        if isinstance(meta, dict) and meta:
            try:
                from .inventory import apply_inventory_session_meta, refresh_inventory_api_cache

                apply_inventory_session_meta(sess, meta)
                refresh_inventory_api_cache(sess)
            except Exception:
                _log.exception("apply inventory meta during PO hydrate failed")

    sidecar_stats = ensure_po_sidecars_hydrated(sess)

    try:
        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        _log.exception("restore_po_sidecars_from_warm during PO hydrate failed")

    try:
        from .existing_po import ensure_existing_po_hydrated

        ensure_existing_po_hydrated(sess)
    except Exception:
        _log.exception("ensure_existing_po_hydrated during PO hydrate failed")

    stats = {
        "sales_rows": _df_row_count(getattr(sess, "sales_df", None)),
        "inventory_rows": _df_row_count(getattr(sess, "inventory_df_variant", None)),
        "existing_po_rows": _df_row_count(getattr(sess, "existing_po_df", None)),
        "return_overlay_rows": _df_row_count(getattr(sess, "po_return_overlay_df", None)),
        "sku_status_rows": sidecar_stats.get("sku_status_lead_df", _df_row_count(getattr(sess, "sku_status_lead_df", None))),
        "daily_inventory_rows": sidecar_stats.get(
            "daily_inventory_history_df",
            _df_row_count(getattr(sess, "daily_inventory_history_df", None)),
        ),
    }
    if (
        stats["inventory_rows"] > inv_before
        or stats["sales_rows"] > sales_before
        or stats["existing_po_rows"] > ep_before
    ):
        _log.info("PO session hydrated: %s", stats)
    return stats

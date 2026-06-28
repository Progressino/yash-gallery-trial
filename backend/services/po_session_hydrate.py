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

_PO_SIDECAR_KEYS = ("daily_inventory_history_df", "sku_status_lead_df", "manual_intransit_overlay_df")

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
    if key == "daily_inventory_history_df":
        try:
            from .daily_inventory_history import inventory_history_max_date

            cur_max = inventory_history_max_date(cur)
            bak_max = inventory_history_max_date(backup)
            if bak_max is not None and (cur_max is None or bak_max > cur_max):
                return True
            if (
                cur_max is not None
                and bak_max is not None
                and bak_max == cur_max
                and bak_n > cur_n
            ):
                return True
        except Exception:
            pass
    return bak_n >= max(100, cur_n * 2) and cur_n < 500


def _sidecar_candidate_score(key: str, df: pd.DataFrame | None, meta: dict | None = None) -> tuple:
    n = _df_row_count(df)
    meta_at = 0
    if isinstance(meta, dict):
        try:
            from .daily_inventory_history import upload_timestamp_epoch

            meta_at = int(upload_timestamp_epoch(str(meta.get("daily_inventory_history_uploaded_at") or "")))
        except Exception:
            meta_at = 0
    if key == "daily_inventory_history_df":
        try:
            from .daily_inventory_history import inventory_history_max_date

            mx = inventory_history_max_date(df)
            mx_ord = int(mx.toordinal()) if mx is not None else 0
            return (0 if _sidecar_looks_placeholder(key, df) else 1, meta_at, mx_ord, n)
        except Exception:
            pass
    return (0 if _sidecar_looks_placeholder(key, df) else 1, meta_at, n)


def load_po_sidecar_backups_from_disk() -> dict[str, pd.DataFrame]:
    """Best sku_status / daily_inventory snapshots from warm_cache + github_cache dirs.

    Only checks the primary warm-cache dir and the single most-recent GitHub blob-cache
    subdirectory — scanning ALL blob-cache subdirs was O(N × files) and slow when many
    manifests had accumulated (e.g. 50+ subdirs × 3 parquet reads each).
    """
    out: dict[str, tuple[int, pd.DataFrame]] = {}
    dirs: list[Path] = [_warm_cache_dir()]
    blob_root = Path(os.environ.get("GITHUB_BLOB_CACHE_DIR", "/data/github_cache"))
    if blob_root.is_dir():
        # Only add the most recent blob cache subdirectory, not all of them.
        try:
            latest_blob = max(blob_root.glob("*/"), key=lambda p: p.stat().st_mtime, default=None)
            if latest_blob:
                dirs.append(latest_blob)
        except Exception:
            pass
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
            prev = out.get(key)
            if prev is None or _sidecar_candidate_score(key, df) > _sidecar_candidate_score(key, prev[1]):
                out[key] = (_df_row_count(df), df)

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

    wc = _main._warm_cache or {}
    # Skip the expensive disk scan if the warm cache already has non-placeholder sidecar
    # data for all keys — the common fast path after a normal startup.
    def _inventory_sidecar_stale(df: pd.DataFrame | None) -> bool:
        try:
            from .daily_inventory_history import inventory_history_max_date, today_ist_timestamp

            mx = inventory_history_max_date(df)
            if mx is None:
                return True
            return (today_ist_timestamp() - mx).days > 2
        except Exception:
            return False

    try:
        from .daily_inventory_history import (
            apply_daily_inventory_history_meta,
            daily_inventory_meta_is_newer,
            read_daily_inventory_history_disk_meta,
        )

        disk_meta = read_daily_inventory_history_disk_meta()
        if daily_inventory_meta_is_newer(disk_meta, sess):
            need_backups = True
        else:
            need_backups = False
    except Exception:
        need_backups = False

    if not need_backups:
        need_backups = any(
            _sidecar_looks_placeholder(key, wc.get(key)) or _df_row_count(wc.get(key)) == 0
            for key in _PO_SIDECAR_KEYS
        )
    if not need_backups and _inventory_sidecar_stale(wc.get("daily_inventory_history_df")):
        need_backups = True
    backups = load_po_sidecar_backups_from_disk() if need_backups else {}
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
        meta_candidates = []
        if key == "daily_inventory_history_df":
            warm_meta = _main._warm_cache.get(_main._DAILY_INV_META_WARM_KEY)
            disk_meta = None
            try:
                from .daily_inventory_history import read_daily_inventory_history_disk_meta

                disk_meta = read_daily_inventory_history_disk_meta()
            except Exception:
                disk_meta = None
            meta_candidates = [m for m in (disk_meta, warm_meta) if isinstance(m, dict)]
        best_meta = meta_candidates[0] if meta_candidates else None
        best = max(
            non_placeholder or candidates,
            key=lambda d: _sidecar_candidate_score(
                key,
                d,
                meta=best_meta if key == "daily_inventory_history_df" else None,
            ),
        )

        if (
            _df_row_count(best) != _df_row_count(warm_df)
            or _sidecar_looks_placeholder(key, warm_df)
        ):
            _main._warm_cache[key] = best.copy()
            changed = True
        if _df_row_count(best) != _df_row_count(sess_df) or _sidecar_looks_placeholder(key, sess_df):
            setattr(sess, key, best.copy())
            changed = True
        if key == "daily_inventory_history_df":
            try:
                from .daily_inventory_history import (
                    apply_daily_inventory_history_meta,
                    read_daily_inventory_history_disk_meta,
                    recanonicalize_inventory_history_skus,
                )

                warm_meta = _main._warm_cache.get(_main._DAILY_INV_META_WARM_KEY)
                disk_meta = read_daily_inventory_history_disk_meta()
                for meta in (warm_meta, disk_meta):
                    if isinstance(meta, dict) and meta.get("daily_inventory_history_uploaded_at"):
                        apply_daily_inventory_history_meta(sess, meta)
                        if isinstance(warm_meta, dict):
                            _main._warm_cache[_main._DAILY_INV_META_WARM_KEY] = dict(meta)
                        break
                mapping = getattr(sess, "sku_mapping", None) or wc.get("sku_mapping") or {}
                if mapping and _df_row_count(best) > 0:
                    recan = recanonicalize_inventory_history_skus(best, mapping)
                    if _df_row_count(recan) > 0:
                        sku_changed = set(recan["OMS_SKU"].astype(str)) != set(
                            best["OMS_SKU"].astype(str)
                        )
                        if sku_changed or _df_row_count(recan) != _df_row_count(best):
                            best = recan
                            _main._warm_cache[key] = best.copy()
                            setattr(sess, key, best.copy())
                            changed = True
            except Exception:
                _log.exception("apply daily inventory history meta failed")
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


def ensure_inventory_history_authoritative_for_read(sess) -> pd.DataFrame:
    """Load newest on-disk matrix, re-key SKUs, and roll forward stale history."""
    import backend.main as _main

    from .daily_inventory_history import ensure_latest_daily_inventory_authoritative

    ensure_latest_daily_inventory_authoritative(sess)
    ensure_po_sidecars_hydrated(sess)
    try:
        from .sku_mapping import restore_sku_mapping_to_session

        restore_sku_mapping_to_session(sess)
    except Exception:
        pass

    df = getattr(sess, "daily_inventory_history_df", None)
    if df is None or getattr(df, "empty", True):
        wc = (_main._warm_cache or {}).get("daily_inventory_history_df")
        if wc is not None and not getattr(wc, "empty", True):
            df = wc.copy()
            sess.daily_inventory_history_df = df

    from .daily_inventory_history import (
        apply_daily_inventory_history_meta,
        inventory_history_authoritative_cap_date,
        inventory_history_max_date,
        read_daily_inventory_history_disk_meta,
        recanonicalize_inventory_history_skus,
        refresh_inventory_history_rollforward,
    )

    mapping = getattr(sess, "sku_mapping", None) or (_main._warm_cache or {}).get("sku_mapping") or {}
    if df is not None and not getattr(df, "empty", True) and mapping:
        recan = recanonicalize_inventory_history_skus(df, mapping)
        if _df_row_count(recan) > 0:
            sku_changed = set(recan["OMS_SKU"].astype(str)) != set(df["OMS_SKU"].astype(str))
            if sku_changed or _df_row_count(recan) != _df_row_count(df):
                df = recan
                sess.daily_inventory_history_df = df
                if not _main._warm_cache:
                    _main._warm_cache = {}
                _main._warm_cache["daily_inventory_history_df"] = df.copy()

    mx = inventory_history_max_date(df)
    cap = inventory_history_authoritative_cap_date(sess)
    snap = str(getattr(sess, "inventory_snapshot_date", "") or "").strip()[:10]
    snap_ts = pd.Timestamp(snap).normalize() if len(snap) == 10 else None
    # Only fill a gap up to the matrix upload end — never add a column after the wide sheet.
    if mx is not None and snap_ts is not None and snap_ts > mx and snap_ts <= cap:
        try:
            result = refresh_inventory_history_rollforward(
                sess, cap_date=snap_ts, include_snapshot=True
            )
            if result.get("ok"):
                df = getattr(sess, "daily_inventory_history_df", None)
                try:
                    _main.sync_daily_inventory_history_sidecar(sess)
                except Exception:
                    pass
        except Exception:
            _log.exception("inventory history roll-forward on read failed")
    elif mx is not None and mx.normalize() > cap.normalize():
        # Trim any synthetic rows beyond authoritative upload end (e.g. prior today roll-forward).
        from .daily_inventory_history import filter_inventory_history_window

        trimmed = filter_inventory_history_window(
            df,
            days=3650,
            end_date=str(cap.date()),
        )
        if _df_row_count(trimmed) > 0 and inventory_history_max_date(trimmed) != mx:
            sess.daily_inventory_history_df = trimmed
            if not _main._warm_cache:
                _main._warm_cache = {}
            _main._warm_cache["daily_inventory_history_df"] = trimmed.copy()

    disk_meta = read_daily_inventory_history_disk_meta()
    if isinstance(disk_meta, dict) and disk_meta.get("daily_inventory_history_uploaded_at"):
        apply_daily_inventory_history_meta(sess, disk_meta)

    df = getattr(sess, "daily_inventory_history_df", None)
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    from .daily_inventory_history import drop_zero_derived_rows, reconcile_daily_inventory_meta_if_session_newer

    try:
        reconcile_daily_inventory_meta_if_session_newer(sess)
    except Exception:
        _log.exception("reconcile_daily_inventory_meta_if_session_newer failed")

    return drop_zero_derived_rows(df)


_MIN_STATUS_ROWS_FOR_LARGE_CATALOG = 100


def _status_too_sparse_for_catalog(df: pd.DataFrame, sess) -> bool:
    """A tiny status fragment must not gate PO for the whole inventory catalog."""
    if len(df) >= _MIN_STATUS_ROWS_FOR_LARGE_CATALOG:
        return False
    inv_n = _df_row_count(getattr(sess, "inventory_df_variant", None))
    return inv_n > _MIN_STATUS_ROWS_FOR_LARGE_CATALOG


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
    if _status_too_sparse_for_catalog(df, sess):
        _log.warning(
            "Ignoring sparse sku_status_lead_df (%s rows vs %s inventory SKUs) — using global lead_time",
            len(df),
            _df_row_count(getattr(sess, "inventory_df_variant", None)),
        )
        return None
    # Re-canonicalize with the current SKU mapping so status rows align with inventory/sales
    # even when the mapping master was updated after the status sheet was uploaded.
    try:
        from .po_engine import canonical_oms_key

        sku_map = getattr(sess, "sku_mapping", None) or {}
        out = df.copy()
        out["OMS_SKU"] = out["OMS_SKU"].astype(str).map(lambda s: canonical_oms_key(s, sku_map))
        out = out[out["OMS_SKU"].astype(str).str.len() > 0]
        return out if not out.empty else None
    except Exception:
        _log.exception("re-canonicalize sku_status_lead_df failed")
        return df


_LARGE_WARM_FRAMES = frozenset({"sales_df", "daily_inventory_history_df"})


def _share_warm_frame_in_po_session_only(key: str) -> bool:
    try:
        from .shared_frames import should_skip_session_copy

        return should_skip_session_copy(key)
    except Exception:
        try:
            import backend.main as _main

            return _main.warm_cache_po_session_only() and key in _LARGE_WARM_FRAMES
        except Exception:
            return False


def _assign_frame(target: dict | object, key: str, val: Any, *, is_dict: bool) -> None:
    """Assign a dataframe without copying multi-million-row frames in local PO mode."""
    if val is None or not hasattr(val, "empty") or val.empty:
        return
    if _share_warm_frame_in_po_session_only(key):
        frame = val
    elif hasattr(val, "copy"):
        frame = val.copy()
    else:
        frame = val
    if is_dict:
        target[key] = frame
    else:
        setattr(target, key, frame)


def _should_replace_warm_frame(cur: Any, incoming: Any, *, key: str = "") -> bool:
    if incoming is None or not hasattr(incoming, "empty") or incoming.empty:
        return False
    if cur is None or not hasattr(cur, "empty") or cur.empty:
        return True
    if key == "daily_inventory_history_df":
        try:
            from .daily_inventory_history import inventory_history_is_newer_than

            return inventory_history_is_newer_than(incoming, cur)
        except Exception:
            pass
    if key == "existing_po_df":
        try:
            from .existing_po import existing_po_frame_is_newer_than

            return existing_po_frame_is_newer_than(incoming, cur)
        except Exception:
            pass
    return _df_row_count(incoming) > _df_row_count(cur)


def load_po_calc_essentials_from_disk() -> dict[str, Any]:
    """Read PO-calculate frames + metadata from WARM_CACHE_DIR (ignores manifest age).

    Previously this called _warm_cache_loose_parquets_from_dir which re-read ALL
    platform parquets (mtr_df, myntra_df …) from disk on every PO calculate run,
    even though they were already in the in-memory warm cache.  That caused the
    "stuck at 3%" symptom — 5-9 large parquet reads blocking the PO executor thread.
    Now: prefer the warm-cache copy; fall back to disk only when missing.
    """
    import backend.main as _main

    disk_dir = _warm_cache_dir()
    loaded: dict[str, Any] = {}

    wc = _main._warm_cache or {}
    for key in _PO_CALC_PARQUET_KEYS:
        # Use in-memory value when already loaded — avoids the disk read entirely.
        mem = wc.get(key)
        if mem is not None and hasattr(mem, "empty") and not mem.empty:
            loaded[key] = mem
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
        if _should_replace_warm_frame(cur, val, key=key):
            _assign_frame(_main._warm_cache, key, val, is_dict=True)
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
    """Load saved return overlay from warm cache or disk when the session is empty or stale."""
    cur = getattr(sess, "po_return_overlay_df", None)
    cur_n = _df_row_count(cur)
    cur_units = 0
    if cur_n > 0:
        try:
            cur_units = int(
                pd.to_numeric(cur.get("Return_Units"), errors="coerce").fillna(0).sum()
            )
        except Exception:
            cur_units = cur_n
    import backend.main as _main

    try:
        _main.restore_po_sidecars_from_warm(sess)
    except Exception:
        _log.exception("restore_po_sidecars_from_warm for return overlay failed")
    cur = getattr(sess, "po_return_overlay_df", None)
    cur_n = _df_row_count(cur)
    if cur_n > 0:
        try:
            cur_units = int(
                pd.to_numeric(cur.get("Return_Units"), errors="coerce").fillna(0).sum()
            )
        except Exception:
            cur_units = cur_n

    path = _warm_cache_dir() / "po_return_overlay_df.parquet"
    disk_n = 0
    disk_units = 0
    if path.is_file():
        try:
            disk_df = pd.read_parquet(path)
            disk_n = _df_row_count(disk_df)
            disk_units = int(
                pd.to_numeric(disk_df.get("Return_Units"), errors="coerce").fillna(0).sum()
            )
        except Exception:
            _log.exception("count po_return_overlay_df.parquet failed")
    meta_units = 0
    try:
        from .po_return_import import load_return_overlay_meta_from_disk

        meta_units = int(load_return_overlay_meta_from_disk().get("return_overlay_units") or 0)
    except Exception:
        meta_units = 0
    if cur_n > 0 and cur_units >= max(disk_units, meta_units) * 0.99:
        try:
            from .po_return_import import ensure_return_overlay_meta_hydrated

            ensure_return_overlay_meta_hydrated(sess)
        except Exception:
            _log.exception("return overlay meta hydrate failed (session already loaded)")
        return False
    if not path.is_file():
        return False
    if disk_units < meta_units * 0.9 and cur_units >= disk_units:
        try:
            from .po_return_import import ensure_return_overlay_meta_hydrated

            ensure_return_overlay_meta_hydrated(sess)
        except Exception:
            pass
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
    try:
        from .po_return_import import ensure_return_overlay_meta_hydrated

        ensure_return_overlay_meta_hydrated(sess)
        meta = {
            "return_overlay_uploaded_at": str(getattr(sess, "return_overlay_uploaded_at", "") or ""),
            "return_overlay_filename": str(getattr(sess, "return_overlay_filename", "") or ""),
            "return_overlay_skus": int(len(df)),
            "return_overlay_units": int(pd.to_numeric(df.get("Return_Units"), errors="coerce").fillna(0).sum()),
        }
        if meta.get("return_overlay_uploaded_at"):
            _main._warm_cache[_main._RETURN_OVERLAY_META_WARM_KEY] = meta
    except Exception:
        _log.exception("return overlay meta hydrate failed")
    _log.info("PO return overlay hydrated from disk (%s rows)", len(df))
    return True


_PO_PLATFORM_ATTRS = (
    "mtr_df",
    "myntra_df",
    "meesho_df",
    "flipkart_df",
    "snapdeal_df",
)


def _hydrate_platform_frames_from_disk_for_po(sess) -> None:
    """PO-session-only warm cache skips platform RAM; read parquets for ADS when empty."""
    import backend.main as _main

    if not _main.warm_cache_po_session_only():
        return
    if any(
        _df_row_count(getattr(sess, attr, None)) > 0 for attr in _PO_PLATFORM_ATTRS
    ):
        return
    disk = _warm_cache_dir()
    for attr in _PO_PLATFORM_ATTRS:
        path = disk / f"{attr}.parquet"
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            _log.exception("PO hydrate: failed reading %s", path)
            continue
        if df is not None and not df.empty:
            setattr(sess, attr, df)
            _log.info("PO hydrate: loaded %s from disk (%s rows)", attr, len(df))


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
            _assign_frame(sess, "sales_df", wc_sales, is_dict=False)

    try:
        from .inventory import sync_inventory_snapshot_from_warm

        sync_inventory_snapshot_from_warm(sess)
    except Exception:
        _log.exception("sync_inventory_snapshot_from_warm during PO hydrate failed")

    if _df_row_count(getattr(sess, "inventory_df_variant", None)) == 0:
        for key in _main._INVENTORY_WARM_KEYS:
            wc = _main._warm_cache.get(key)
            if wc is not None and not wc.empty:
                _assign_frame(sess, key, wc, is_dict=False)
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

    _hydrate_platform_frames_from_disk_for_po(sess)

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

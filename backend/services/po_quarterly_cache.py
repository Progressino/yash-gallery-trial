"""Server-wide PO quarterly cache — one build shared by all sessions."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_payloads: dict[tuple, dict[str, Any]] = {}
_building_key: Optional[tuple] = None
_progress: dict[str, Any] = {"progress": 0, "message": ""}

_DISK_CACHE_DIR = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")

_RECENT_METRIC_COLS = frozenset(
    {"Units_90d", "Units_30d", "Freq_30d", "ADS", "Status", "Avg_Monthly"}
)


def _disk_path(key: tuple) -> str:
    """Key is ``(schema_version, group_by_parent, n_quarters)`` from quarterly_cache_key."""
    if len(key) >= 3:
        schema, group_by_parent, n_quarters = key[0], key[1], key[2]
    else:
        schema, group_by_parent, n_quarters = 0, key[0], key[1]
    return os.path.join(
        _DISK_CACHE_DIR,
        f"quarterly_v{int(schema)}_{int(bool(group_by_parent))}_{int(n_quarters)}.json",
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _current_tier3_token() -> dict[str, str]:
    try:
        from .daily_store import get_tier3_sync_token

        return dict(get_tier3_sync_token() or {})
    except Exception:
        return {}


def stamp_quarterly_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach sync fingerprint so we can refresh incrementally after daily uploads."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out["tier3_sync_token"] = _current_tier3_token()
    out["built_at"] = datetime.now(timezone.utc).isoformat()
    return out


def quarterly_is_stale(payload: dict[str, Any] | None) -> bool:
    if not payload or not payload.get("loaded"):
        return True
    stored = payload.get("tier3_sync_token")
    if not isinstance(stored, dict):
        return True
    return dict(stored) != _current_tier3_token()


def merge_incremental_quarterly_payload(
    existing: dict[str, Any],
    incremental: dict[str, Any],
    *,
    recent_quarter_cols: list[str],
) -> dict[str, Any]:
    """Patch recent quarter columns + rolling metrics; keep deep history from *existing*."""
    if not existing.get("loaded") or not existing.get("rows"):
        return incremental
    if not incremental.get("loaded") or not incremental.get("rows"):
        return existing

    inc_by_sku = {
        str(r.get("OMS_SKU", "")).strip().upper(): r
        for r in incremental.get("rows") or []
        if r.get("OMS_SKU")
    }
    patch_cols = list(recent_quarter_cols) + [
        c for c in _RECENT_METRIC_COLS if c in (incremental.get("columns") or [])
    ]
    out_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in existing.get("rows") or []:
        nr = dict(row)
        sku = str(nr.get("OMS_SKU", "")).strip().upper()
        seen.add(sku)
        inc = inc_by_sku.get(sku)
        if inc:
            for c in patch_cols:
                if c in inc:
                    nr[c] = inc[c]
        out_rows.append(nr)
    for sku, inc in inc_by_sku.items():
        if sku not in seen:
            out_rows.append(dict(inc))
    cols = list(existing.get("columns") or [])
    for c in incremental.get("columns") or []:
        if c not in cols:
            cols.append(c)
    merged = {**existing, "columns": cols, "rows": out_rows}
    return stamp_quarterly_metadata(merged)


def save_shared_quarterly_to_disk(key: tuple, payload: dict[str, Any]) -> None:
    """Best-effort persist of a shared quarterly payload to /data/warm_cache/
    so it survives backend restarts and doesn't need a 30-180s rebuild."""
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        path = _disk_path(key)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        logger.exception("Failed to persist shared quarterly cache for %s", key)


def load_shared_quarterly_from_disk(key: tuple) -> Optional[dict[str, Any]]:
    try:
        path = _disk_path(key)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            payload = json.load(f)
        if payload.get("loaded") and payload.get("rows"):
            return payload
        return None
    except Exception:
        logger.exception("Failed to load shared quarterly cache for %s", key)
        return None


def invalidate_shared_quarterly() -> None:
    """Drop all shared quarterly payloads (memory + disk) — Tier-1 bulk / manual reset only."""
    with _lock:
        _payloads.clear()
    try:
        if os.path.isdir(_DISK_CACHE_DIR):
            for name in os.listdir(_DISK_CACHE_DIR):
                if name.startswith("quarterly_") and name.endswith(".json"):
                    try:
                        os.remove(os.path.join(_DISK_CACHE_DIR, name))
                    except OSError:
                        pass
    except Exception:
        logger.exception("Failed to clear persisted shared quarterly cache")


def get_shared_quarterly(key: tuple) -> Optional[dict[str, Any]]:
    with _lock:
        row = _payloads.get(key)
        if row:
            return dict(row)
    payload = load_shared_quarterly_from_disk(key)
    if payload:
        with _lock:
            _payloads[key] = payload
        return dict(payload)
    return None


def store_shared_quarterly(key: tuple, payload: dict[str, Any]) -> None:
    from .po_quarterly_warmup import normalize_quarterly_payload

    n_q = int(key[2]) if len(key) >= 3 else 8
    payload = normalize_quarterly_payload(payload, n_quarters=n_q)
    payload = stamp_quarterly_metadata(payload)
    with _lock:
        _payloads[key] = payload
    save_shared_quarterly_to_disk(key, payload)


def quarterly_build_status() -> dict[str, Any]:
    with _lock:
        return {
            "building": _building_key is not None,
            "progress": int(_progress.get("progress") or 0),
            "message": str(_progress.get("message") or ""),
        }


def _set_progress(pct: int, message: str) -> None:
    with _lock:
        _progress["progress"] = max(0, min(100, int(pct)))
        _progress["message"] = message


def start_shared_quarterly_build(
    key: tuple,
    build_fn: Callable[[Callable[[int, str], None]], dict[str, Any]],
) -> bool:
    """Run one server-wide build; returns False if already building or cached."""
    global _building_key
    with _lock:
        if _payloads.get(key, {}).get("loaded") and _payloads[key].get("rows"):
            if not quarterly_is_stale(_payloads[key]):
                return False
        if _building_key is not None:
            return False
        _building_key = key
        _progress["progress"] = 5
        _progress["message"] = "Starting quarterly history…"

    def _run() -> None:
        global _building_key
        try:
            result = build_fn(_set_progress)
            if result.get("loaded") and result.get("rows"):
                store_shared_quarterly(key, result)
                _set_progress(100, "")
                logger.info(
                    "Shared quarterly cache ready: %s rows",
                    len(result.get("rows") or []),
                )
            else:
                _set_progress(0, "No quarterly data in uploads.")
        except Exception:
            logger.exception("Shared quarterly build failed")
            _set_progress(0, "Quarterly build failed — try Reload.")
        finally:
            with _lock:
                _building_key = None

    threading.Thread(target=_run, daemon=True, name="po-qtr-shared").start()
    return True


def start_incremental_quarterly_refresh(
    key: tuple,
    build_fn: Callable[[Callable[[int, str], None]], dict[str, Any]],
    existing: dict[str, Any],
    *,
    recent_quarter_cols: list[str],
) -> bool:
    """Background refresh of recent quarters only — keeps deep history intact."""
    global _building_key
    with _lock:
        if _building_key is not None:
            return False
        _building_key = key
        _progress["progress"] = 5
        _progress["message"] = "Refreshing recent quarterly sales…"

    def _run() -> None:
        global _building_key
        try:
            inc = build_fn(_set_progress)
            if inc.get("loaded") and inc.get("rows"):
                merged = merge_incremental_quarterly_payload(
                    existing, inc, recent_quarter_cols=recent_quarter_cols
                )
                store_shared_quarterly(key, merged)
                _set_progress(100, "")
                logger.info(
                    "Incremental quarterly refresh: %s SKUs patched",
                    len(merged.get("rows") or []),
                )
            else:
                _set_progress(0, "Incremental quarterly refresh found no data.")
        except Exception:
            logger.exception("Incremental quarterly refresh failed")
            _set_progress(0, "Quarterly refresh failed.")
        finally:
            with _lock:
                _building_key = None

    threading.Thread(target=_run, daemon=True, name="po-qtr-incr").start()
    return True


def schedule_quarterly_refresh_if_stale(
    key: tuple,
    sess=None,
    *,
    force_full: bool = False,
) -> bool:
    """
    Non-blocking: full build when cache is missing; incremental when only Tier-3 dailies changed.
    Returns True if a background job was started.
    """
    from .po_quarterly_warmup import (
        build_incremental_quarterly_payload,
        build_quarterly_payload,
        expected_quarter_columns,
    )

    existing = get_shared_quarterly(key)
    n_q = int(key[2]) if len(key) >= 3 else 8

    if force_full or not existing or not existing.get("loaded"):
        if quarterly_build_status().get("building"):
            return False

        def _full(progress_cb):
            return build_quarterly_payload(
                sess,
                group_by_parent=bool(key[1]) if len(key) >= 2 else False,
                n_quarters=n_q,
                progress_cb=progress_cb,
            )

        return start_shared_quarterly_build(key, _full)

    if not quarterly_is_stale(existing):
        return False

    recent_n = 2
    try:
        recent_n = max(1, min(3, int(os.environ.get("PO_QUARTERLY_INCREMENTAL_QUARTERS", "2"))))
    except ValueError:
        recent_n = 2
    recent_cols = expected_quarter_columns(recent_n)

    def _incr(progress_cb):
        return build_incremental_quarterly_payload(
            sess,
            group_by_parent=bool(key[1]) if len(key) >= 2 else False,
            n_recent_quarters=recent_n,
            progress_cb=progress_cb,
        )

    return start_incremental_quarterly_refresh(
        key, _incr, existing, recent_quarter_cols=recent_cols
    )

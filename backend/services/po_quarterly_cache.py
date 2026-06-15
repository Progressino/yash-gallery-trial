"""Server-wide PO quarterly cache — one build shared by all sessions."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_payloads: dict[tuple, dict[str, Any]] = {}
_building_key: Optional[tuple] = None
_progress: dict[str, Any] = {"progress": 0, "message": ""}

_DISK_CACHE_DIR = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")


def _disk_path(key: tuple) -> str:
    group_by_parent, n_quarters = key[0], key[1]
    return os.path.join(
        _DISK_CACHE_DIR, f"quarterly_{int(bool(group_by_parent))}_{int(n_quarters)}.json"
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


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
    """Drop all shared quarterly payloads (memory + disk) — called when
    underlying sales/platform data actually changes."""
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
        return dict(row) if row else None


def store_shared_quarterly(key: tuple, payload: dict[str, Any]) -> None:
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

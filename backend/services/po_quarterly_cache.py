"""Server-wide PO quarterly cache — one build shared by all sessions."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_payloads: dict[tuple, dict[str, Any]] = {}
_building_key: Optional[tuple] = None
_progress: dict[str, Any] = {"progress": 0, "message": ""}


def get_shared_quarterly(key: tuple) -> Optional[dict[str, Any]]:
    with _lock:
        row = _payloads.get(key)
        return dict(row) if row else None


def store_shared_quarterly(key: tuple, payload: dict[str, Any]) -> None:
    with _lock:
        _payloads[key] = payload


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

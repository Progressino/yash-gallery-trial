"""HTTP request timing for FastAPI (endpoint perf visibility)."""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Request
from starlette.responses import Response

logger = logging.getLogger("perf")

# High-frequency polls — still timed, but omitted from logs unless slow.
_SKIP_UNLESS_SLOW = frozenset(
    {
        "/api/health",
        "/api/po/calculate/status",
        "/api/data/job-status",
        "/api/upload/inventory/upload-status",
    }
)


def timing_enabled() -> bool:
    raw = (os.environ.get("PERF_TIMING") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _min_log_seconds() -> float:
    raw = (os.environ.get("PERF_TIMING_MIN_SEC") or "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def register_timing_middleware(app: FastAPI) -> None:
    """Register outermost timing middleware (add after auth/session middleware)."""

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next) -> Response:
        if not timing_enabled():
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        min_sec = _min_log_seconds()
        skip_unless_slow = path in _SKIP_UNLESS_SLOW and elapsed < max(min_sec, 0.25)
        if skip_unless_slow or elapsed < min_sec:
            return response

        logger.info(
            "%s %s %s %.3fs",
            request.method,
            path,
            response.status_code,
            elapsed,
        )
        try:
            from ..services.perf_metrics import record_http

            record_http(request.method, path, response.status_code, elapsed)
        except Exception:
            pass
        return response

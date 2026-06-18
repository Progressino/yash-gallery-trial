"""
Thread pools — keep auth/health off the same queue as multi-minute uploads.

Starlette's default ``run_in_threadpool`` uses a small shared pool; one daily
upload can starve login and coverage for minutes.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# One heavy job at a time (warm-cache phase 2, scheduled sync).
HEAVY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="erp-heavy",
)

# Daily upload ingest runs on its own thread so it never queues behind warm-cache
# Phase 2 (which can hold HEAVY_EXECUTOR for 2-5 min on startup).
# A shared semaphore prevents them from running CONCURRENTLY (OOM risk on 7.5 GB).
_UPLOAD_MEMORY_LOCK = threading.Semaphore(1)

DAILY_UPLOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="erp-upload",
)

# Snapshot inventory parse — separate from daily sales ingest/rebuild so RAR uploads
# are not stuck at 5% behind a multi-minute sales rebuild on DAILY_UPLOAD_EXECUTOR.
INVENTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="erp-inventory",
)

# Session full-restore worker — its own queue so a backlog of per-session
# hydrate-warm / daily-upload jobs on DAILY_UPLOAD_EXECUTOR (e.g. many tabs
# reconnecting after a deploy) cannot leave a user's restore stuck at
# "Queued" for the entire backlog. The heavy memory-bound steps inside the
# restore still serialize via _UPLOAD_MEMORY_LOCK.
SESSION_RESTORE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="erp-restore",
)

# PO calculate runs separately so a daily-inventory parse does not queue behind PO math.
PO_CALC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="po-calc",
)

# Return overlay RAR/ZIP parse — must not block AUX (auth/session) or daily ingest queues.
RETURNS_IMPORT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="returns-import",
)

# Session warm-cache copy, SKU bundle, PG persist helpers.
AUX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="erp-aux",
)

# Auth only — never share with session warm-cache (aux pool can be busy for minutes).
AUTH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="erp-auth",
)


def upload_memory_lock_held() -> bool:
    """True when warm-cache load, restore, or a large upload holds the memory semaphore."""
    acquired = _UPLOAD_MEMORY_LOCK.acquire(blocking=False)
    if acquired:
        _UPLOAD_MEMORY_LOCK.release()
        return False
    return True


async def run_heavy(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        HEAVY_EXECUTOR,
        lambda: fn(*args, **kwargs),
    )


async def run_aux(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        AUX_EXECUTOR,
        lambda: fn(*args, **kwargs),
    )


async def run_po_calc(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        PO_CALC_EXECUTOR,
        lambda: fn(*args, **kwargs),
    )


async def run_auth(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        AUTH_EXECUTOR,
        lambda: fn(*args, **kwargs),
    )

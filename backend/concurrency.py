"""
Thread pools — keep auth/health off the same queue as multi-minute uploads.

Starlette's default ``run_in_threadpool`` uses a small shared pool; one daily
upload can starve login and coverage for minutes.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# One heavy job at a time (daily-auto ingest, sales rebuild, warm-cache phase 2).
HEAVY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="erp-heavy",
)

# Session warm-cache copy, SKU bundle, PG persist helpers.
AUX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="erp-aux",
)


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

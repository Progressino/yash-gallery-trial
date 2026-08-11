"""Async Inventory History wide-matrix CSV export (avoids Cloudflare 522).

Synchronous ``GET .../matrix.csv`` can exceed edge timeouts while pandas builds
a 20k×N day pivot. Jobs run on READ_API_EXECUTOR and stream the file when ready.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_MAX_JOBS = 24
_JOB_TTL_SEC = 3600


def _export_dir() -> Path:
    import os

    base = Path(os.environ.get("INVENTORY_EXPORT_DIR") or os.environ.get("WARM_CACHE_DIR") or "/tmp")
    d = base / "inventory_history_exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _purge_old_jobs() -> None:
    now = time.time()
    dead: list[str] = []
    for jid, job in list(_JOBS.items()):
        age = now - float(job.get("created_at") or now)
        if age > _JOB_TTL_SEC or job.get("status") in ("error", "ready") and age > 1800:
            dead.append(jid)
    for jid in dead:
        job = _JOBS.pop(jid, None) or {}
        path = job.get("path")
        if path:
            try:
                Path(path).unlink(missing_ok=True)  # type: ignore[call-arg]
            except TypeError:
                try:
                    p = Path(path)
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
            except Exception:
                pass
    # Cap map size
    if len(_JOBS) > _MAX_JOBS:
        for jid in sorted(_JOBS.keys(), key=lambda k: float(_JOBS[k].get("created_at") or 0))[
            : len(_JOBS) - _MAX_JOBS
        ]:
            _JOBS.pop(jid, None)


def get_matrix_export_job(job_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "progress": job.get("progress"),
            "message": job.get("message"),
            "filename": job.get("filename"),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "ready": job.get("status") == "ready",
            "bytes": job.get("bytes"),
        }


def read_matrix_export_file(job_id: str) -> tuple[bytes, str] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job or job.get("status") != "ready":
            return None
        path = job.get("path")
        filename = str(job.get("filename") or "inventory-matrix.csv")
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return p.read_bytes(), filename


def start_matrix_export_job(
    *,
    history_df,
    q: str = "",
    days: int = 30,
    end_date: str | None = None,
    channel: str = "combined",
    sku_mapping: dict | None = None,
) -> str:
    """Queue a full-matrix CSV build; returns job_id immediately."""
    from ..concurrency import READ_API_EXECUTOR
    from .daily_inventory_history import inventory_history_wide_matrix_csv

    with _LOCK:
        _purge_old_jobs()
        job_id = uuid.uuid4().hex[:16]
        _JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued inventory history export…",
            "created_at": time.time(),
            "filename": None,
            "path": None,
            "error": None,
            "bytes": 0,
        }

    def _run() -> None:
        try:
            with _LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "running"
                job["progress"] = 10
                job["message"] = "Building wide inventory matrix…"
            csv_bytes, filename = inventory_history_wide_matrix_csv(
                history_df,
                q=q,
                days=days,
                end_date=end_date,
                channel=channel,
                sku_mapping=sku_mapping,
            )
            out_path = _export_dir() / f"{job_id}-{filename}"
            out_path.write_bytes(csv_bytes)
            with _LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "ready"
                job["progress"] = 100
                job["message"] = "Export ready"
                job["filename"] = filename
                job["path"] = str(out_path)
                job["bytes"] = len(csv_bytes)
        except Exception as e:
            _log.exception("inventory history matrix export job failed")
            with _LOCK:
                job = _JOBS.get(job_id)
                if not job:
                    return
                job["status"] = "error"
                job["progress"] = 0
                job["message"] = "Export failed"
                job["error"] = str(e)

    READ_API_EXECUTOR.submit(_run)
    return job_id

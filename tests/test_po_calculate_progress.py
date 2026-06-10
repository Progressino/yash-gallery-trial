"""PO calculate progress field on status poll."""
import time

from backend.services.po_calculate_jobs import (
    clear_po_job,
    get_po_job,
    po_job_is_stale,
    set_po_job,
)


def test_set_po_job_stores_progress():
    set_po_job("sess-po-prog", status="running", progress=42, message="Running engine…")
    job = get_po_job("sess-po-prog")
    assert job["progress"] == 42
    assert job["message"] == "Running engine…"


def test_po_job_is_stale_after_idle():
    clear_po_job("sess-stale")
    set_po_job("sess-stale", status="running", progress=22, message="Preparing…")
    from backend.services import po_calculate_jobs as jobs

    with jobs._lock:
        jobs._jobs["sess-stale"]["updated_at"] = time.time() - 1300
    assert po_job_is_stale("sess-stale") is True
    clear_po_job("sess-stale")


def test_po_job_is_stale_sooner_at_low_progress():
    clear_po_job("sess-stuck-2")
    set_po_job("sess-stuck-2", status="running", progress=2, message="Calculating…")
    from backend.services import po_calculate_jobs as jobs

    with jobs._lock:
        jobs._jobs["sess-stuck-2"]["updated_at"] = time.time() - 200
    assert po_job_is_stale("sess-stuck-2") is True
    with jobs._lock:
        jobs._jobs["sess-stuck-2"]["updated_at"] = time.time() - 60
    assert po_job_is_stale("sess-stuck-2") is False
    clear_po_job("sess-stuck-2")

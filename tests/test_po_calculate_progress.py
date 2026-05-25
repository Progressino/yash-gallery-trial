"""PO calculate progress field on status poll."""
from backend.services.po_calculate_jobs import get_po_job, set_po_job


def test_set_po_job_stores_progress():
    set_po_job("sess-po-prog", status="running", progress=42, message="Running engine…")
    job = get_po_job("sess-po-prog")
    assert job["progress"] == 42
    assert job["message"] == "Running engine…"

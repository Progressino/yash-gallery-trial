"""Inventory upload progress fields and reset-stuck endpoint."""

from backend.session import AppSession
from backend.routers import upload as upload_router


def test_clear_stuck_inventory_upload_force():
    sess = AppSession()
    sess.inventory_upload_status = "running"
    sess.inventory_upload_started = 0.0
    sess.inventory_upload_progress = 55
    assert upload_router._clear_stuck_inventory_upload(sess, force=True) is True
    assert sess.inventory_upload_status == "error"
    assert sess.inventory_upload_progress == 0


def test_clear_stuck_inventory_upload_not_young_without_force():
    import time

    sess = AppSession()
    sess.inventory_upload_status = "running"
    sess.inventory_upload_started = time.time()
    assert upload_router._clear_stuck_inventory_upload(sess, force=False) is False
    assert sess.inventory_upload_status == "running"


def test_mark_inventory_upload_running_without_lock():
    from backend.session import AppSession

    sess = AppSession()
    upload_router._mark_inventory_upload_running(sess, "Upload received — starting parse…")
    assert sess.inventory_upload_status == "running"
    assert sess.inventory_upload_progress == 2
    assert sess.inventory_upload_message == "Upload received — starting parse…"
    assert sess.inventory_upload_result == {}


def test_set_inventory_upload_progress_clamps():
    sess = AppSession()
    upload_router._set_inventory_upload_progress(sess, 150, "Done")
    assert sess.inventory_upload_progress == 100
    upload_router._set_inventory_upload_progress(sess, -5, "Start")
    assert sess.inventory_upload_progress == 0

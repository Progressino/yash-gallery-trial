"""Daily inventory history upload stuck-state handling."""

import time

from backend.session import AppSession
from backend.services.daily_inventory_upload_run import clear_stuck_daily_inventory_upload


def test_clear_stuck_daily_inventory_upload_force():
    sess = AppSession()
    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_started = time.time()
    assert clear_stuck_daily_inventory_upload(sess, force=True) is True
    assert sess.daily_inventory_upload_status == "error"
    assert sess.daily_inventory_upload_started == 0.0


def test_clear_stuck_daily_inventory_upload_not_young_without_force():
    sess = AppSession()
    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_started = time.time()
    assert clear_stuck_daily_inventory_upload(sess, force=False) is False
    assert sess.daily_inventory_upload_status == "running"

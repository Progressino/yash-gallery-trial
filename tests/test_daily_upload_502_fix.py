"""Regression: daily-auto status must not block chunk/complete behind ingest lock."""

import time
import threading

from backend.session import AppSession
from backend.routers.upload import (
    _mark_daily_auto_ingest_running,
    _detect_platform,
)


def test_mark_daily_ingest_running_does_not_block_behind_restore_lock():
    sess = AppSession()
    sess._daily_restore_lock.acquire()
    try:
        done = threading.Event()

        def worker():
            _mark_daily_auto_ingest_running(sess, "test")
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert done.is_set()
        assert sess.daily_auto_ingest_status == "running"
    finally:
        sess._daily_restore_lock.release()


def test_detect_amazon_from_filename_with_spaces():
    raw = b'"Customer Shipment Date","Merchant SKU","Quantity"\n'
    plat = _detect_platform("973929020595_YG Amazon 20-21-5-26.csv", raw)
    assert plat == "amazon_b2c"

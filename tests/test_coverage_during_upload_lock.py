"""Coverage must not block when Tier-3 upload holds the daily restore lock."""

import time

from backend.session import AppSession


def test_coverage_skips_restore_when_upload_lock_held():
    from backend.routers.data import _restore_daily_if_needed

    sess = AppSession()
    assert sess._daily_restore_lock.acquire(blocking=False)

    t0 = time.monotonic()
    try:
        _restore_daily_if_needed(sess)
    finally:
        elapsed = time.monotonic() - t0
        sess._daily_restore_lock.release()

    assert elapsed < 1.0
    assert not sess.daily_restored

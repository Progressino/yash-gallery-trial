"""Restore progress fields update through phased steps."""
from backend.routers.data import _set_restore_step
from backend.session import AppSession


def test_set_restore_step_updates_progress():
    sess = AppSession()
    _set_restore_step(sess, "queued")
    assert sess.session_restore_progress == 1
    assert sess.session_restore_step == "queued"

    _set_restore_step(sess, "tier3", "Merging SQLite…")
    assert sess.session_restore_progress == 84
    assert "SQLite" in sess.session_restore_message

    _set_restore_step(sess, "github_amazon", "GitHub — Amazon (1,200,000 rows)…")
    assert sess.session_restore_progress == 32
    assert "Amazon" in sess.session_restore_message

    _set_restore_step(sess, "done", "All done")
    assert sess.session_restore_progress == 100

"""Background daily-auto stores parse outcome for coverage + UI notifications."""

from backend.routers.upload import _store_daily_auto_ingest_result
from backend.session import AppSession


def test_store_daily_auto_ingest_result_on_session():
    sess = AppSession()
    _store_daily_auto_ingest_result(
        sess,
        {
            "ok": True,
            "message": "Loaded 2 file(s): Amazon, Myntra.",
            "detected_platforms": ["Amazon (a.csv)", "Myntra (b.csv)"],
            "warnings": ["c.csv: unknown"],
            "processed_files": 3,
            "detected_files": 2,
            "unknown_files": 1,
        },
    )
    assert sess.daily_auto_ingest_result["detected_files"] == 2
    assert sess.daily_auto_ingest_result["unknown_files"] == 1
    assert len(sess.daily_auto_ingest_result["detected_platforms"]) == 2

"""POST /cache/load must not block on GitHub — queue async restore when warm cache is empty."""
from backend.session import AppSession, wipe_app_session


def test_cache_load_queues_async_restore_when_warm_empty(client, session_for_client, monkeypatch):
    import backend.main as main

    _, sess = session_for_client
    wipe_app_session(sess)
    main._warm_cache = {}
    main._warm_cache_generation = 0

    queued: list[str] = []

    def _fake_queue(s, session_id, *, reason=""):
        queued.append(session_id)
        s.session_restore_status = "running"
        return True

    monkeypatch.setattr(
        "backend.routers.data.queue_session_restore_if_needed",
        _fake_queue,
    )

    r = client.post("/api/cache/load")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "background" in body["message"].lower()
    assert len(queued) == 1


def test_bundled_sku_mapping_when_paused(monkeypatch):
    from backend.services.sku_mapping import restore_sku_mapping_to_session

    sess = AppSession()
    sess.pause_auto_data_restore = True
    monkeypatch.setattr(
        "backend.services.sku_mapping.load_bundled_sku_mapping",
        lambda: {"TEST-SKU": "TEST-SKU"},
    )
    assert restore_sku_mapping_to_session(sess) is True
    assert sess.sku_mapping == {"TEST-SKU": "TEST-SKU"}

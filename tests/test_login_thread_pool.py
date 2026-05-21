"""Login must stay responsive while warm-cache or session aux pools are busy."""

import time

from starlette.testclient import TestClient


def test_login_fast_while_heavy_pool_busy():
    """Simulate warm-cache monopolizing HEAVY — login must still finish quickly."""
    from backend.concurrency import HEAVY_EXECUTOR
    from backend.main import app

    gate = HEAVY_EXECUTOR.submit(time.sleep, 8)
    assert gate.running() or gate.done() is False

    client = TestClient(app)
    t0 = time.monotonic()
    r = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "wrong"},
    )
    elapsed = time.monotonic() - t0
    gate.result(timeout=15)

    assert elapsed < 5.0, f"login took {elapsed:.1f}s while heavy pool was busy"
    assert r.status_code == 401


def test_login_fast_while_aux_pool_busy():
    """Session warm-cache copy uses AUX — auth must not queue on that pool."""
    from backend.concurrency import AUX_EXECUTOR
    from backend.main import app

    gate = AUX_EXECUTOR.submit(time.sleep, 8)
    client = TestClient(app)
    t0 = time.monotonic()
    r = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "wrong"},
    )
    elapsed = time.monotonic() - t0
    gate.result(timeout=15)

    assert elapsed < 5.0, f"login took {elapsed:.1f}s while aux pool was busy"
    assert r.status_code == 401

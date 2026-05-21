"""Login must stay responsive while warm-cache runs on HEAVY_EXECUTOR."""

import time
from concurrent.futures import ThreadPoolExecutor

from starlette.testclient import TestClient


def test_login_fast_while_heavy_pool_busy(monkeypatch):
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

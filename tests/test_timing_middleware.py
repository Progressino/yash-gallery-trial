"""HTTP timing middleware logs endpoint duration."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware.timing import register_timing_middleware


def test_timing_middleware_logs_slow_request(monkeypatch, caplog):
    monkeypatch.setenv("PERF_TIMING", "1")
    monkeypatch.setenv("PERF_TIMING_MIN_SEC", "0")

    app = FastAPI()
    register_timing_middleware(app)

    @app.get("/api/po/calculate/status")
    def status():
        return {"status": "idle"}

    @app.post("/api/po/calculate")
    def calculate():
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="perf"):
        TestClient(app).post("/api/po/calculate")
        TestClient(app).get("/api/po/calculate/status")

    messages = [r.getMessage() for r in caplog.records if r.name == "perf"]
    assert any("POST /api/po/calculate" in m and "200" in m for m in messages)
    assert not any("/api/po/calculate/status" in m for m in messages)


def test_timing_middleware_logs_slow_poll(monkeypatch, caplog):
    monkeypatch.setenv("PERF_TIMING", "1")
    monkeypatch.setenv("PERF_TIMING_MIN_SEC", "0")

    app = FastAPI()
    register_timing_middleware(app)

    @app.get("/api/health")
    def health():
        import time

        time.sleep(0.3)
        return {"status": "ok"}

    with caplog.at_level(logging.INFO, logger="perf"):
        TestClient(app).get("/api/health")

    messages = [r.getMessage() for r in caplog.records if r.name == "perf"]
    assert any("GET /api/health" in m for m in messages)

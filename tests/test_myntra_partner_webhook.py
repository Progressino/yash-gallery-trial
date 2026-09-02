"""Myntra Partner outbound webhook receiver."""
from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from backend.db import myntra_partner_db
from backend.main import app


@pytest.fixture()
def partner_client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "myntra_partner_test.db")
    api_key = f"test-key-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("MYNTRA_PARTNER_DB_PATH", db_path)
    monkeypatch.setenv("MYNTRA_PARTNER_WEBHOOK_API_KEY", api_key)
    monkeypatch.setenv("MYNTRA_PARTNER_CODE", "PROGRESSINO")
    myntra_partner_db.init_db()
    client = TestClient(app)
    client.test_api_key = api_key  # type: ignore[attr-defined]
    return client


def test_webhook_health_public(partner_client):
    r = partner_client.get("/api/myntra/partner/webhook/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["partner_code"] == "PROGRESSINO"
    assert body["configured"] is True


def test_webhook_rejects_missing_api_key(partner_client):
    r = partner_client.post(
        "/api/myntra/partner/webhook",
        json={"sellerOrderId": "SO-1", "eventType": "create_order"},
    )
    assert r.status_code == 401


def test_webhook_accepts_create_order(partner_client):
    payload = {
        "sellerOrderId": "SO-1001",
        "eventType": "create_order",
        "orderLines": [{"orderLineId": "OL-1", "sku": "SKU1", "quantity": 1}],
    }
    r = partner_client.post(
        "/api/myntra/partner/webhook",
        json=payload,
        headers={
            "x-api-key": partner_client.test_api_key,
            "mocking-partner-name": "PROGRESSINO",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["received"] is True
    assert body["duplicate"] is False
    assert body["event_type"] == "create_order"
    assert body["seller_order_id"] == "SO-1001"


def test_webhook_idempotent_duplicate(partner_client):
    payload = {"sellerOrderId": "SO-DUP", "eventType": "shipped", "packetId": "PK-9"}
    headers = {"x-api-key": partner_client.test_api_key}
    r1 = partner_client.post("/api/myntra/partner/webhook", json=payload, headers=headers)
    r2 = partner_client.post("/api/myntra/partner/webhook", json=payload, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["duplicate"] is False
    assert r2.json()["duplicate"] is True
    assert r1.json()["event_id"] == r2.json()["event_id"]


def test_detect_event_type_from_header(partner_client):
    r = partner_client.post(
        "/api/myntra/partner/webhook",
        json={"packetId": "PK-55"},
        headers={
            "x-api-key": partner_client.test_api_key,
            "x-myntra-event-type": "delivered",
        },
    )
    assert r.status_code == 200
    assert r.json()["event_type"] == "delivered"

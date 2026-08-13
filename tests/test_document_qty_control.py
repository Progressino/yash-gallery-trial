"""Cutting receive tolerance is a configurable rule (currently unlimited)."""
from __future__ import annotations

from backend.services.document_qty_control import (
    cutting_receive_tolerance_pct,
    jo_should_auto_close,
    max_allowed_receive,
    validate_receive_qty,
)


def test_cutting_tolerance_default_unlimited(monkeypatch):
    monkeypatch.delenv("CUTTING_RECEIVE_TOLERANCE", raising=False)
    assert cutting_receive_tolerance_pct() is None
    assert jo_should_auto_close(1000, 1150, None) is True
    assert jo_should_auto_close(1000, 500, None) is False


def test_cutting_tolerance_env_restores_ten_percent(monkeypatch):
    monkeypatch.setenv("CUTTING_RECEIVE_TOLERANCE", "0.10")
    assert cutting_receive_tolerance_pct() == 0.10
    assert max_allowed_receive(40, 0.10) == 44.0
    assert jo_should_auto_close(40, 42, 0.10) is True
    assert jo_should_auto_close(40, 50, 0.10) is False


def test_po_grn_validate_receive_still_enforces_material_tolerance():
    validate_receive_qty(1000, 0, 1050, 0.05, doc_label="GRN")
    try:
        validate_receive_qty(1000, 0, 1100, 0.05, doc_label="GRN")
        raise AssertionError("expected over-tolerance GRN to fail")
    except ValueError as e:
        assert "tolerance" in str(e).lower()

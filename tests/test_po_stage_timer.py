"""PO calculate staged timing."""
from __future__ import annotations

import logging

from backend.services.po_stage_timer import PoStageTimer, po_stage_timing_enabled


def test_po_stage_timer_records_stages(monkeypatch):
    monkeypatch.setenv("PERF_PO_STAGES", "1")
    timer = PoStageTimer()
    timer.mark("sales load")
    timer.mark("inventory")
    timer.mark("forecast")
    summary = timer.as_dict()
    assert summary["sales load"] >= 0
    assert summary["inventory"] >= 0
    assert summary["forecast"] >= 0
    assert summary["TOTAL"] >= summary["sales load"]


def test_po_stage_timer_log_summary(monkeypatch, caplog):
    monkeypatch.setenv("PERF_PO_STAGES", "1")
    timer = PoStageTimer()
    timer.mark("sales load")
    timer.mark("po logic")

    with caplog.at_level(logging.INFO, logger="perf.po"):
        timer.log_summary(prefix="PO stages")

    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "perf.po")
    assert "sales load" in text
    assert "po logic" in text
    assert "TOTAL" in text


def test_po_stage_timer_disabled(monkeypatch):
    monkeypatch.setenv("PERF_PO_STAGES", "0")
    timer = PoStageTimer()
    timer.mark("sales load")
    assert timer.as_dict() == {"TOTAL": 0.0}
    assert po_stage_timing_enabled() is False

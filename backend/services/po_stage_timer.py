"""Staged timing for PO calculate (perf visibility)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("perf.po")


def po_stage_timing_enabled() -> bool:
    raw = (os.environ.get("PERF_PO_STAGES") or os.environ.get("PERF_TIMING") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


class PoStageTimer:
    """Record wall-clock duration per PO calculate stage."""

    __slots__ = ("_started", "_last", "_stages", "_enabled")

    def __init__(self, *, enabled: bool | None = None) -> None:
        self._enabled = po_stage_timing_enabled() if enabled is None else enabled
        now = time.perf_counter()
        self._started = now
        self._last = now
        self._stages: list[tuple[str, float]] = []

    def mark(self, name: str, *, since: float | None = None) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        start = self._last if since is None else since
        self._stages.append((name, now - start))
        self._last = now

    def as_dict(self) -> dict[str, float]:
        out = {name: round(sec, 3) for name, sec in self._stages}
        out["TOTAL"] = round(time.perf_counter() - self._started, 3)
        return out

    def log_summary(self, *, prefix: str = "PO stages") -> dict[str, float]:
        summary = self.as_dict()
        if not self._enabled or not self._stages:
            return summary
        lines = [f"{prefix}:"]
        width = max(len(name) for name, _ in self._stages)
        width = max(width, len("TOTAL"))
        for name, sec in self._stages:
            lines.append(f"  {name.ljust(width)}  {sec:6.3f}s")
        lines.append(f"  {'TOTAL'.ljust(width)}  {summary['TOTAL']:6.3f}s")
        logger.info("\n".join(lines))
        return summary


def merge_stage_timings(*parts: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in parts:
        if not part:
            continue
        for k, v in part.items():
            if k == "TOTAL":
                continue
            try:
                out[k] = out.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                pass
    if out:
        out["TOTAL"] = round(sum(v for k, v in out.items() if k != "TOTAL"), 3)
    return out

"""PO raise batch numbers — one PO number per Export & Confirm."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _counter_path() -> Path:
    base = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    return base / "po_raise_batches.json"


def allocate_po_number(raised_date: str | None = None) -> str:
    """Return the next PO number for the calendar day (PO-YYYYMMDD-NNNN)."""
    day = pd.Timestamp(raised_date or pd.Timestamp.now()).normalize()
    key = day.strftime("%Y%m%d")
    path = _counter_path()
    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    seq = int(data.get(key, 0) or 0) + 1
    data[key] = seq
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return f"PO-{key}-{seq:04d}"

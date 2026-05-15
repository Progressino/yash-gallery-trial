"""Server-side archive of PO CSV exports — enables auto-import without browser Downloads access."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .po_raise_import import (
    apply_ledger_import,
    ledger_has_positive_qty_on_day,
    parse_ledger_csv_text,
)

_IST = ZoneInfo("Asia/Kolkata")
_ARCHIVE_DIR = os.environ.get("PO_RAISE_ARCHIVE_DIR", "/data/po_raise_archive")
_DEV_FALLBACK_DIR = os.environ.get("PO_RAISE_ARCHIVE_DEV_DIR", "./po_raise_archive_dev")
_SESSION_ID_RE = re.compile(r"^[a-f0-9\-]{8,64}$", re.I)
_resolved_archive_root: Optional[Path] = None


def _safe_session_id(session_id: str) -> str:
    sid = (session_id or "").strip()
    if not sid or not _SESSION_ID_RE.match(sid):
        raise ValueError("Invalid session id for PO raise archive.")
    return sid


def archive_dir() -> Path:
    global _resolved_archive_root
    if _resolved_archive_root is not None:
        return _resolved_archive_root
    for candidate in (Path(_ARCHIVE_DIR), Path(_DEV_FALLBACK_DIR)):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            _resolved_archive_root = candidate
            return candidate
        except OSError:
            continue
    _resolved_archive_root = Path(_DEV_FALLBACK_DIR)
    return _resolved_archive_root


def archive_path(session_id: str, day: pd.Timestamp) -> Path:
    sid = _safe_session_id(session_id)
    d = pd.Timestamp(day).normalize()
    return archive_dir() / sid / f"{d.date()}.csv"


def save_archive(session_id: str, raised_date: pd.Timestamp, content: bytes) -> Path:
    path = archive_path(session_id, raised_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_archive(session_id: str, day: pd.Timestamp) -> Optional[bytes]:
    path = archive_path(session_id, day)
    if path.is_file():
        return path.read_bytes()
    return None


def ist_today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(_IST).date())


def resolve_planning_date(planning_date: Optional[str]) -> pd.Timestamp:
    if planning_date and str(planning_date).strip():
        return pd.Timestamp(pd.to_datetime(str(planning_date).strip()).normalize())
    return ist_today()


def yesterday_of(plan: pd.Timestamp) -> pd.Timestamp:
    return plan - timedelta(days=1)


def decode_csv_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def import_archive_into_session(
    sess,
    session_id: str,
    day: pd.Timestamp,
    *,
    group_by_parent: bool = False,
    replace_day: bool = True,
) -> Optional[dict]:
    raw = load_archive(session_id, day)
    if not raw:
        return None
    text = decode_csv_bytes(raw)
    accum, err = parse_ledger_csv_text(text)
    if err:
        return {"ok": False, "message": err}
    return apply_ledger_import(
        sess,
        accum,
        pd.Timestamp(day).normalize(),
        group_by_parent=group_by_parent,
        replace_day=replace_day,
    )


def try_auto_import_yesterday_ledger(
    sess,
    session_id: str,
    planning_date: Optional[str],
    *,
    group_by_parent: bool = False,
) -> Optional[dict]:
    """
  If yesterday's archived export exists and the ledger has no qty for that day yet,
  import it automatically. Returns result dict or None if skipped.
    """
    if not session_id:
        return None
    plan = resolve_planning_date(planning_date)
    yday = yesterday_of(plan)
    ledger = getattr(sess, "po_raise_ledger_df", pd.DataFrame())
    if ledger_has_positive_qty_on_day(ledger, yday):
        return None
    result = import_archive_into_session(
        sess,
        session_id,
        yday,
        group_by_parent=group_by_parent,
        replace_day=True,
    )
    if not result or not result.get("ok"):
        return result
    result["auto"] = True
    result["message"] = (
        f"Auto-imported archived export for {yday.date()} "
        f"({result.get('imported_skus', 0):,} SKUs, {result.get('total_units', 0):,} units)."
    )
    return result

"""Server-side archive of PO CSV exports — enables auto-import without browser Downloads access."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .po_raise_import import (
    apply_ledger_import,
    ledger_has_positive_qty_on_day,
    parse_ledger_csv_text,
    sync_ledger_to_durable_db,
)

_IST = ZoneInfo("Asia/Kolkata")
_ARCHIVE_DIR = os.environ.get("PO_RAISE_ARCHIVE_DIR", "/data/po_raise_archive")
_DEV_FALLBACK_DIR = os.environ.get("PO_RAISE_ARCHIVE_DEV_DIR", "./po_raise_archive_dev")
_GLOBAL_SUBDIR = "global"
_SESSION_ID_RE = re.compile(r"^[a-f0-9\-]{8,64}$", re.I)
_FILENAME_DATE_RE = re.compile(
    r"(?:po[_\s-]*recommendation|raise[_\s-]*po)?[^\d]*(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})",
    re.I,
)
_PO_SHEET_DATE_RE = re.compile(
    r"\bpo\s*(\d{1,2})[-\s]+([a-z]{3})[-\s]+(\d{2,4})\b",
    re.I,
)
_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
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


def global_archive_path(day: pd.Timestamp) -> Path:
    d = pd.Timestamp(day).normalize()
    root = archive_dir() / _GLOBAL_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{d.date()}.csv"


def archive_path(session_id: str, day: pd.Timestamp) -> Path:
    sid = _safe_session_id(session_id)
    d = pd.Timestamp(day).normalize()
    return archive_dir() / sid / f"{d.date()}.csv"


def save_archive(session_id: str, raised_date: pd.Timestamp, content: bytes) -> Path:
    """Persist export for all users (global) and this session (legacy path)."""
    day = pd.Timestamp(raised_date).normalize()
    gpath = global_archive_path(day)
    gpath.write_bytes(content)
    if session_id:
        try:
            spath = archive_path(session_id, day)
            spath.parent.mkdir(parents=True, exist_ok=True)
            spath.write_bytes(content)
        except ValueError:
            pass
    return gpath


def delete_archives_for_date(
    raised_date: str,
    session_id: str | None = None,
) -> bool:
    """Remove org-wide and session CSV archives so Calculate PO won't re-import this day."""
    try:
        day = pd.Timestamp(pd.to_datetime(str(raised_date).strip()[:10]).normalize())
    except Exception:
        return False
    removed = False
    gpath = global_archive_path(day)
    if gpath.is_file():
        gpath.unlink()
        removed = True
    if session_id:
        try:
            spath = archive_path(session_id, day)
            if spath.is_file():
                spath.unlink()
                removed = True
        except ValueError:
            pass
    return removed


def find_org_archive_bytes(day: pd.Timestamp) -> Optional[bytes]:
    """Return archived PO export bytes for ``day`` from global or any session folder."""
    d = pd.Timestamp(day).normalize()
    gpath = global_archive_path(d)
    if gpath.is_file():
        return gpath.read_bytes()
    stem = f"{d.date()}.csv"
    root = archive_dir()
    newest: tuple[float, Path] | None = None
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name == _GLOBAL_SUBDIR:
            continue
        candidate = sub / stem
        if not candidate.is_file():
            continue
        mtime = candidate.stat().st_mtime
        if newest is None or mtime > newest[0]:
            newest = (mtime, candidate)
    if newest is not None:
        return newest[1].read_bytes()
    return None


def load_archive(session_id: str, day: pd.Timestamp) -> Optional[bytes]:
    """Prefer org-wide global archive, then per-session file, then any session folder."""
    raw = find_org_archive_bytes(day)
    if raw is not None:
        return raw
    if session_id:
        try:
            path = archive_path(session_id, day)
            if path.is_file():
                return path.read_bytes()
        except ValueError:
            pass
    return None


def list_global_archive_dates(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> List[pd.Timestamp]:
    root = archive_dir() / _GLOBAL_SUBDIR
    if not root.is_dir():
        return []
    out: list[pd.Timestamp] = []
    ws = pd.Timestamp(window_start).normalize()
    we = pd.Timestamp(window_end).normalize()
    for p in root.glob("*.csv"):
        try:
            day = pd.Timestamp(pd.to_datetime(p.stem).normalize())
        except Exception:
            continue
        if ws <= day <= we:
            out.append(day)
    return sorted(set(out), reverse=True)


def _archive_csv_dates() -> set[str]:
    """Calendar days referenced by archived PO export filenames."""
    root = archive_dir()
    days: set[str] = set()
    paths: list[Path] = []
    gdir = root / _GLOBAL_SUBDIR
    if gdir.is_dir():
        paths.extend(gdir.glob("*.csv"))
    if root.is_dir():
        for sub in root.iterdir():
            if sub.is_dir() and sub.name != _GLOBAL_SUBDIR:
                paths.extend(sub.glob("*.csv"))
    for p in paths:
        try:
            days.add(str(pd.Timestamp(pd.to_datetime(p.stem).normalize().date())))
        except Exception:
            parsed = parse_raise_date_from_filename(p.name)
            if parsed is not None and pd.notna(parsed):
                days.add(str(pd.Timestamp(parsed).normalize().date()))
    return days


def delete_all_archives() -> int:
    """Remove every archived PO export so auto-import cannot resurrect deleted raises."""
    root = archive_dir()
    removed = 0
    gdir = root / _GLOBAL_SUBDIR
    if gdir.is_dir():
        for p in gdir.glob("*.csv"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    if root.is_dir():
        for sub in root.iterdir():
            if not sub.is_dir() or sub.name == _GLOBAL_SUBDIR:
                continue
            for p in sub.glob("*.csv"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def parse_raise_date_from_filename(filename: str) -> Optional[pd.Timestamp]:
    """e.g. ``po_recommendation 16-5-26.csv`` or ``Po 24-Jun-26.xlsx`` → 2026-06-24."""
    name = (filename or "").strip()
    if not name:
        return None
    m_po = _PO_SHEET_DATE_RE.search(name.replace("_", " "))
    if m_po:
        d = int(m_po.group(1))
        mo = _MONTH_MAP.get(m_po.group(2).lower()[:3])
        y = int(m_po.group(3))
        if y < 100:
            y += 2000
        if mo:
            try:
                return pd.Timestamp(year=y, month=mo, day=d).normalize()
            except Exception:
                pass
    m = _FILENAME_DATE_RE.search(name.replace("_", " "))
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    try:
        return pd.Timestamp(year=y, month=mo, day=d).normalize()
    except Exception:
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
    out = apply_ledger_import(
        sess,
        accum,
        pd.Timestamp(day).normalize(),
        group_by_parent=group_by_parent,
        replace_day=replace_day,
    )
    if out.get("ok"):
        sync_ledger_to_durable_db(sess, pd.Timestamp(day).normalize())
    return out


def try_auto_import_recent_ledgers(
    sess,
    session_id: str,
    planning_date: Optional[str],
    *,
    group_by_parent: bool = False,
    lookback_days: int = 14,
) -> Optional[dict]:
    """
    Import archived PO exports for any recent day missing from the session ledger.

    Scans yesterday back through ``lookback_days`` (covers weekend raises when the
    team returns Monday). Uses org-wide global archives so a new login session still
    sees Saturday's export.
    """
    if not session_id:
        return None
    plan = resolve_planning_date(planning_date)
    lb = max(1, int(lookback_days))
    window_start = plan - timedelta(days=lb)
    ledger = getattr(sess, "po_raise_ledger_df", pd.DataFrame())

    days_to_try: list[pd.Timestamp] = []
    from ..db.po_raised_db import is_raise_date_suppressed

    for i in range(1, lb + 1):
        d = (plan - timedelta(days=i)).normalize()
        if ledger_has_positive_qty_on_day(ledger, d):
            continue
        if is_raise_date_suppressed(str(d.date())):
            continue
        if load_archive(session_id, d) is not None:
            days_to_try.append(d)

    for d in list_global_archive_dates(window_start, plan - timedelta(days=1)):
        if ledger_has_positive_qty_on_day(ledger, d):
            continue
        if is_raise_date_suppressed(str(d.date())):
            continue
        if d not in days_to_try:
            days_to_try.append(d)

    days_to_try = sorted(set(days_to_try), reverse=True)
    if not days_to_try:
        return None

    imported_days: list[str] = []
    total_units = 0
    total_skus = 0
    last_err: Optional[str] = None

    for day in days_to_try:
        result = import_archive_into_session(
            sess,
            session_id,
            day,
            group_by_parent=group_by_parent,
            replace_day=True,
        )
        if not result:
            continue
        if not result.get("ok"):
            last_err = result.get("message")
            continue
        imported_days.append(str(day.date()))
        total_units += int(result.get("total_units") or 0)
        total_skus += int(result.get("imported_skus") or 0)
        ledger = getattr(sess, "po_raise_ledger_df", pd.DataFrame())

    if not imported_days:
        if last_err:
            return {"ok": False, "message": last_err}
        return None

    return {
        "ok": True,
        "auto": True,
        "imported_days": imported_days,
        "imported_skus": total_skus,
        "total_units": total_units,
        "message": (
            f"Auto-imported archived PO export(s) for {', '.join(imported_days)} "
            f"({total_skus:,} SKU lines, {total_units:,} units) — avoids repeating raises."
        ),
    }


def try_auto_import_yesterday_ledger(
    sess,
    session_id: str,
    planning_date: Optional[str],
    *,
    group_by_parent: bool = False,
) -> Optional[dict]:
    """Backward-compatible wrapper — prefer ``try_auto_import_recent_ledgers``."""
    return try_auto_import_recent_ledgers(
        sess,
        session_id,
        planning_date,
        group_by_parent=group_by_parent,
        lookback_days=14,
    )

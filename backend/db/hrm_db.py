"""HRM Module DB — Employees, Responsibilities, Task Tracking, Issues, Appraisal"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Performance rules cutover (IST calendar date). Rows before this stay on legacy formula.
PERFORMANCE_CUTOVER_DATE = os.environ.get("HRM_PERFORMANCE_CUTOVER", "2026-08-11").strip()[:10]


def _default_db_path() -> str:
    if os.path.isdir("/data"):
        return "/data/hrm.db"
    return os.path.join(os.path.dirname(__file__), "..", "hrm.db")


_DB = os.environ.get("HRM_DB_PATH", _default_db_path())

TASK_LOG_STATUSES = frozenset({"Done", "Partial", "Missed", "Blocked", "Leave", "N/A"})
NEUTRAL_TASK_STATUSES = frozenset({"Leave", "N/A"})
# Performance only after Approved / Self-complete (Done without Linked To) / Auto-Approved
PERF_CREDIT_STATUSES = frozenset({"Done", "Partial"})  # Done only if approval satisfied

# HR issue lifecycle (Resolved legacy maps to Resolve)
ISSUE_STATUSES = frozenset({"Open", "Resolve", "Hold", "Cancel"})
ISSUE_STATUS_ALIASES = {"Resolved": "Resolve", "Closed": "Resolve", "Cancelled": "Cancel"}

FREQUENCIES = (
    "Daily",
    "Weekly",
    "Fortnightly",
    "Monthly",
    "Quarterly",
    "Yearly",
    "Whenever Required",
)
PRIORITIES = ("High", "Medium", "Low", "Critical")
TIME_PERIODS = ("Morning", "Afternoon", "Evening", "Full Day", "Shift-A", "Shift-B", "Custom")
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
# HOD may change a locked status until end of the calendar day after first mark.
HOD_STATUS_EDIT_GRACE_DAYS = 1
# Employee may mark / Linked person may approve within task day + next 2 IST days.
TASK_WINDOW_EXTRA_DAYS = 2
MAX_TIMER_PAUSE_PER_DAY = 3
MAX_TIMER_RESUME_PER_DAY = 2
MAX_TIMER_COMPLETE_PER_DAY = 1


def today_ist() -> date:
    return datetime.now(IST).date()


def self_check_today_only(log_date: str | None) -> str:
    """Employee Check must use today's IST date — raises ValueError if not."""
    day = today_ist().isoformat()
    got = str(log_date or day)[:10]
    if got != day:
        raise ValueError("Employee Check is limited to today's tasks only")
    return day


def now_ist() -> datetime:
    return datetime.now(IST)


def _now_iso() -> str:
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")


def _parse_clock(value: str | None) -> str:
    """Normalize a datetime string to 'YYYY-MM-DD HH:MM:SS'. Empty stays empty."""
    raw = str(value or "").strip().replace("T", " ")
    if not raw:
        return ""
    raw = raw[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise ValueError("Invalid time. Use YYYY-MM-DD HH:MM")


BACKUP_ALLOCATION_UNITS = frozenset({"hours", "days"})


def timer_status(
    started_at: str | None,
    ended_at: str | None,
    paused_at: str | None = None,
) -> str:
    st = str(started_at or "").strip()
    en = str(ended_at or "").strip()
    pa = str(paused_at or "").strip()
    if st and en:
        return "Completed"
    if st and pa:
        return "Paused"
    if st:
        return "Active"
    return "Not Started"


def _seconds_between(start: str, end: str) -> int:
    if not start or not end:
        return 0
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        a = datetime.strptime(str(start)[:19], fmt)
        b = datetime.strptime(str(end)[:19], fmt)
        return max(0, int((b - a).total_seconds()))
    except ValueError:
        return 0


def _parse_backup_fields(data: dict, assignee_id: int, *, require: bool = True) -> tuple[int | None, float, str]:
    """Return (backup_employee_id, value, unit). Raises ValueError when require and missing/invalid."""
    raw = data.get("backup_employee_id")
    try:
        backup_id = int(raw) if raw not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        backup_id = None
    try:
        value = float(data.get("backup_allocation_value") or 0)
    except (TypeError, ValueError):
        value = 0.0
    unit = str(data.get("backup_allocation_unit") or "days").strip().lower()
    if unit in ("hour", "hr", "hrs"):
        unit = "hours"
    if unit in ("day", "d"):
        unit = "days"
    if unit not in BACKUP_ALLOCATION_UNITS:
        unit = "days"
    if require:
        if not backup_id:
            raise ValueError("Backup person is required")
        if int(backup_id) == int(assignee_id):
            raise ValueError("Backup person must be different from the assigned employee")
        if value <= 0:
            raise ValueError("Backup allocation duration must be greater than zero")
    return backup_id, value, unit


def _timer_payload(log: dict | None, *, events: list | None = None) -> dict:
    log = log or {}
    started = str(log.get("started_at") or "").strip()
    ended = str(log.get("ended_at") or "").strip()
    paused = str(log.get("paused_at") or "").strip()
    try:
        active_sec = int(log.get("active_seconds") or 0)
    except (TypeError, ValueError):
        active_sec = 0
    try:
        paused_sec = int(log.get("paused_seconds") or 0)
    except (TypeError, ValueError):
        paused_sec = 0
    ev = list(events or [])
    now = _now_iso()
    if started and not ended:
        if paused:
            paused_sec = paused_sec + _seconds_between(paused, now)
        else:
            seg = _last_active_segment_start(log, ev) if ev else started
            active_sec = active_sec + _seconds_between(seg, now)
    try:
        mins = int(log.get("duration_minutes") or 0)
    except (TypeError, ValueError):
        mins = 0
    if started and not ended:
        mins = active_sec // 60
    elif not mins and active_sec:
        mins = active_sec // 60
    elif not mins and started and ended:
        mins = _duration_minutes(started, ended)
    daily = _daily_active_breakdown(ev)
    if started and not ended and not paused:
        # include live open segment in daily breakdown
        seg = _last_active_segment_start(log, ev) if ev else started
        tmp: dict[str, int] = {d["date"]: d["active_seconds"] for d in daily}
        _add_span_to_days(tmp, seg, now)
        daily = [
            {"date": d, "active_seconds": sec, "active_minutes": sec // 60}
            for d, sec in sorted(tmp.items())
        ]
    return {
        "started_at": started,
        "ended_at": ended,
        "paused_at": paused,
        "duration_minutes": mins,
        "active_seconds": active_sec,
        "paused_seconds": paused_sec,
        "active_minutes": active_sec // 60,
        "paused_minutes": paused_sec // 60,
        "timer_status": timer_status(started, ended, paused),
        "timer_events": ev,
        "daily_time": daily,
        **_timer_limits_and_sessions(ev, log_date=str(log.get("log_date") or "")),
    }


def _count_timer_events(events: list, *types: str) -> int:
    want = {t.lower() for t in types}
    return sum(1 for e in events if str(e.get("event_type") or "").lower() in want)


def _work_sessions_for_day(events: list, log_date: str) -> list[dict]:
    """Completed work sessions (start/resume → pause/end) for one calendar day."""
    day = str(log_date)[:10]
    sessions: list[dict] = []
    open_at: str | None = None
    idx = 0
    for ev in events:
        et = str(ev.get("event_type") or "").lower()
        at = str(ev.get("event_at") or "").strip()
        if et in ("start", "resume"):
            open_at = at
        elif et in ("pause", "end") and open_at and at:
            if str(open_at)[:10] == day or str(at)[:10] == day:
                sec = _seconds_between(open_at, at)
                if sec > 0:
                    idx += 1
                    sessions.append(
                        {
                            "index": idx,
                            "started_at": open_at,
                            "ended_at": at,
                            "duration_seconds": sec,
                            "duration_minutes": sec // 60,
                            "duration_label": _format_duration_hm(sec),
                        }
                    )
            open_at = None
    return sessions


def _format_duration_hm(seconds: int) -> str:
    s = max(0, int(seconds or 0))
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    if h:
        return f"{h:02d}h {m:02d}m"
    return f"{m:02d}m"


def _timer_limits_and_sessions(events: list, *, log_date: str) -> dict:
    ev = list(events or [])
    pause_n = _count_timer_events(ev, "pause")
    resume_n = _count_timer_events(ev, "resume")
    complete_n = _count_timer_events(ev, "end")
    sessions = _work_sessions_for_day(ev, log_date) if log_date else []
    total_sec = sum(int(s.get("duration_seconds") or 0) for s in sessions)
    return {
        "work_sessions": sessions,
        "total_work_seconds": total_sec,
        "total_work_label": _format_duration_hm(total_sec),
        "pause_count": pause_n,
        "resume_count": resume_n,
        "complete_count": complete_n,
        "pause_limit": MAX_TIMER_PAUSE_PER_DAY,
        "resume_limit": MAX_TIMER_RESUME_PER_DAY,
        "complete_limit": MAX_TIMER_COMPLETE_PER_DAY,
        "can_pause": pause_n < MAX_TIMER_PAUSE_PER_DAY,
        "can_resume": resume_n < MAX_TIMER_RESUME_PER_DAY,
        "can_complete": complete_n < MAX_TIMER_COMPLETE_PER_DAY,
    }


def _daily_active_breakdown(events: list, live_extra_from_now: int | None = None) -> list[dict]:
    """Build per-day active seconds from start/resume→pause/end segments."""
    by_day: dict[str, int] = {}
    open_at: str | None = None
    for ev in events:
        et = str(ev.get("event_type") or "")
        at = str(ev.get("event_at") or "").strip()
        if et in ("start", "resume"):
            open_at = at
        elif et in ("pause", "end") and open_at and at:
            _add_span_to_days(by_day, open_at, at)
            open_at = None
    if open_at and live_extra_from_now is None:
        _add_span_to_days(by_day, open_at, _now_iso())
    rows = [
        {"date": d, "active_seconds": sec, "active_minutes": sec // 60}
        for d, sec in sorted(by_day.items())
    ]
    return rows


def _add_span_to_days(by_day: dict[str, int], start: str, end: str) -> None:
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        a = datetime.strptime(str(start)[:19], fmt)
        b = datetime.strptime(str(end)[:19], fmt)
    except ValueError:
        return
    if b <= a:
        return
    cur = a
    while cur.date() < b.date():
        day_end = datetime.combine(cur.date(), datetime.max.time()).replace(microsecond=0)
        # use next midnight
        next_mid = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time())
        sec = int((next_mid - cur).total_seconds())
        key = cur.date().isoformat()
        by_day[key] = by_day.get(key, 0) + max(0, sec)
        cur = next_mid
    sec = int((b - cur).total_seconds())
    key = cur.date().isoformat()
    by_day[key] = by_day.get(key, 0) + max(0, sec)


def in_task_action_window(task_date: str, *, as_of: date | None = None) -> bool:
    """True if as_of is within task_date .. task_date+2 (IST calendar days inclusive)."""
    as_of = as_of or today_ist()
    try:
        d0 = date.fromisoformat(str(task_date)[:10])
    except ValueError:
        return True
    return d0 <= as_of <= d0 + timedelta(days=TASK_WINDOW_EXTRA_DAYS)


def weekday_occurrence_in_month(d: date) -> int:
    """1-based count of this weekday in the month (1st Monday, 2nd Monday, …)."""
    n = 0
    for day in range(1, d.day + 1):
        if date(d.year, d.month, day).weekday() == d.weekday():
            n += 1
    return n


def quarterly_months(anchor_month: int) -> set[int]:
    """Months in the quarterly cycle starting at anchor (1-12). e.g. Jan → 1,4,7,10."""
    a = max(1, min(12, int(anchor_month or 1)))
    return {((a - 1 + i * 3) % 12) + 1 for i in range(4)}


def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS departments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE NOT NULL,
        description TEXT DEFAULT '',
        hod_name    TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS employees (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_code        TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        department_id   INTEGER REFERENCES departments(id),
        designation     TEXT DEFAULT '',
        phone           TEXT DEFAULT '',
        email           TEXT DEFAULT '',
        join_date       TEXT DEFAULT '',
        status          TEXT DEFAULT 'Active',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS responsibilities (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id     INTEGER NOT NULL REFERENCES employees(id),
        department_id   INTEGER REFERENCES departments(id),
        title           TEXT NOT NULL,
        description     TEXT DEFAULT '',
        frequency       TEXT DEFAULT 'Daily',
        category        TEXT DEFAULT 'General',
        added_by        TEXT DEFAULT '',
        active          INTEGER DEFAULT 1,
        created_at      TEXT DEFAULT (datetime('now')),
        updated_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS task_logs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        responsibility_id   INTEGER NOT NULL REFERENCES responsibilities(id),
        employee_id         INTEGER NOT NULL REFERENCES employees(id),
        log_date            TEXT NOT NULL,
        status              TEXT DEFAULT 'Pending',
        remarks             TEXT DEFAULT '',
        marked_by           TEXT DEFAULT '',
        marked_at           TEXT DEFAULT '',
        blocker_employee_id INTEGER REFERENCES employees(id),
        blocker_reason      TEXT DEFAULT '',
        created_at          TEXT DEFAULT (datetime('now')),
        UNIQUE(responsibility_id, log_date)
    );

    CREATE TABLE IF NOT EXISTS issue_logs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id         INTEGER NOT NULL REFERENCES employees(id),
        department_id       INTEGER REFERENCES departments(id),
        issue_date          TEXT NOT NULL,
        issue_type          TEXT DEFAULT 'General',
        severity            TEXT DEFAULT 'Minor',
        title               TEXT NOT NULL,
        description         TEXT DEFAULT '',
        recorded_by         TEXT DEFAULT '',
        caused_by_employee_id   INTEGER REFERENCES employees(id),
        caused_by_dept_id       INTEGER REFERENCES departments(id),
        task_log_id             INTEGER REFERENCES task_logs(id),
        resolution          TEXT DEFAULT '',
        status              TEXT DEFAULT 'Open',
        created_at          TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS issue_history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id        INTEGER NOT NULL REFERENCES issue_logs(id),
        action          TEXT NOT NULL,
        field_name      TEXT DEFAULT '',
        previous_value  TEXT DEFAULT '',
        new_value       TEXT DEFAULT '',
        user_id         INTEGER,
        user_name       TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS issue_comments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id        INTEGER NOT NULL REFERENCES issue_logs(id),
        comment_text    TEXT NOT NULL,
        user_id         INTEGER,
        user_name       TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS issue_attachments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id        INTEGER NOT NULL REFERENCES issue_logs(id),
        file_name       TEXT NOT NULL,
        file_url        TEXT DEFAULT '',
        content_type    TEXT DEFAULT '',
        file_size       INTEGER DEFAULT 0,
        uploaded_by     TEXT DEFAULT '',
        uploaded_by_user_id INTEGER,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS issue_voice_transcriptions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id        INTEGER,
        target_field    TEXT DEFAULT 'description',
        transcript      TEXT NOT NULL,
        status          TEXT DEFAULT 'success',
        error_message   TEXT DEFAULT '',
        user_id         INTEGER,
        user_name       TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS issue_notifications (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id        INTEGER,
        event_type      TEXT NOT NULL,
        recipient_user_id INTEGER,
        message         TEXT DEFAULT '',
        channel         TEXT DEFAULT 'in_app',
        is_read         INTEGER DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS one_time_tasks (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id         INTEGER NOT NULL REFERENCES employees(id),
        department_id       INTEGER REFERENCES departments(id),
        title               TEXT NOT NULL,
        description         TEXT DEFAULT '',
        due_date            TEXT DEFAULT '',
        assigned_by         TEXT DEFAULT '',
        status              TEXT DEFAULT 'Pending',
        started_at          TEXT DEFAULT '',
        completed_at        TEXT DEFAULT '',
        approved_at         TEXT DEFAULT '',
        approved_by         TEXT DEFAULT '',
        duration_minutes    INTEGER DEFAULT 0,
        completion_notes    TEXT DEFAULT '',
        approval_notes      TEXT DEFAULT '',
        active              INTEGER DEFAULT 1,
        created_at          TEXT DEFAULT (datetime('now')),
        updated_at          TEXT DEFAULT (datetime('now'))
    );
    """)

    for sql in (
        "ALTER TABLE task_logs ADD COLUMN blocker_employee_id INTEGER REFERENCES employees(id)",
        "ALTER TABLE task_logs ADD COLUMN blocker_reason TEXT DEFAULT ''",
        "ALTER TABLE issue_logs ADD COLUMN recorded_by_user_id INTEGER",
        "ALTER TABLE issue_logs ADD COLUMN subject_user_id INTEGER",
        "ALTER TABLE issue_logs ADD COLUMN caused_by_user_id INTEGER",
        "ALTER TABLE issue_logs ADD COLUMN subject_user_name TEXT DEFAULT ''",
        "ALTER TABLE issue_logs ADD COLUMN caused_by_user_name TEXT DEFAULT ''",
        "ALTER TABLE issue_logs ADD COLUMN updated_at TEXT DEFAULT ''",
        "ALTER TABLE issue_logs ADD COLUMN designation TEXT DEFAULT ''",
        # Responsibilities enhancements
        "ALTER TABLE responsibilities ADD COLUMN priority TEXT DEFAULT 'Medium'",
        "ALTER TABLE responsibilities ADD COLUMN mandatory INTEGER DEFAULT 0",
        "ALTER TABLE responsibilities ADD COLUMN schedule_weekday TEXT DEFAULT ''",
        "ALTER TABLE responsibilities ADD COLUMN schedule_month_day INTEGER DEFAULT 0",
        "ALTER TABLE responsibilities ADD COLUMN time_period TEXT DEFAULT ''",
        # One-time tasks enhancements
        "ALTER TABLE one_time_tasks ADD COLUMN priority TEXT DEFAULT 'Medium'",
        "ALTER TABLE one_time_tasks ADD COLUMN manual_duration_minutes INTEGER DEFAULT 0",
        # Org hierarchy
        "ALTER TABLE employees ADD COLUMN reports_to_employee_id INTEGER REFERENCES employees(id)",
        "ALTER TABLE departments ADD COLUMN parent_department_id INTEGER REFERENCES departments(id)",
        # Issue audio attachments (data URL / base64 payload stored in file_url when small)
        "ALTER TABLE issue_attachments ADD COLUMN attachment_kind TEXT DEFAULT 'file'",
        # Final confirmation: schedules, linked approval, reassignment
        "ALTER TABLE responsibilities ADD COLUMN schedule_month INTEGER DEFAULT 0",
        "ALTER TABLE responsibilities ADD COLUMN linked_to_employee_id INTEGER",
        "ALTER TABLE one_time_tasks ADD COLUMN linked_to_employee_id INTEGER",
        "ALTER TABLE task_logs ADD COLUMN approval_status TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN approved_by TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN approved_at TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN is_reassignment INTEGER DEFAULT 0",
        "ALTER TABLE task_logs ADD COLUMN reassigned_from_employee_id INTEGER DEFAULT 0",
        "ALTER TABLE task_logs ADD COLUMN reassignment_clone_id INTEGER DEFAULT 0",
        # Daily work report (DWR) time tracking per responsibility/day
        "ALTER TABLE task_logs ADD COLUMN started_at TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN ended_at TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN duration_minutes INTEGER DEFAULT 0",
        # Pause/resume + accumulated active/paused seconds
        "ALTER TABLE task_logs ADD COLUMN paused_at TEXT DEFAULT ''",
        "ALTER TABLE task_logs ADD COLUMN active_seconds INTEGER DEFAULT 0",
        "ALTER TABLE task_logs ADD COLUMN paused_seconds INTEGER DEFAULT 0",
        # Mandatory backup person + allocation duration on responsibilities / one-time tasks
        "ALTER TABLE responsibilities ADD COLUMN backup_employee_id INTEGER",
        "ALTER TABLE responsibilities ADD COLUMN backup_allocation_value REAL DEFAULT 0",
        "ALTER TABLE responsibilities ADD COLUMN backup_allocation_unit TEXT DEFAULT 'days'",
        "ALTER TABLE one_time_tasks ADD COLUMN backup_employee_id INTEGER",
        "ALTER TABLE one_time_tasks ADD COLUMN backup_allocation_value REAL DEFAULT 0",
        "ALTER TABLE one_time_tasks ADD COLUMN backup_allocation_unit TEXT DEFAULT 'days'",
    ):
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS hrm_task_audit (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type     TEXT NOT NULL,
            entity_id       INTEGER,
            action          TEXT NOT NULL,
            old_value       TEXT DEFAULT '',
            new_value       TEXT DEFAULT '',
            actor           TEXT DEFAULT '',
            notes           TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS day_reassignment_clones (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            original_responsibility_id  INTEGER NOT NULL REFERENCES responsibilities(id),
            original_employee_id        INTEGER NOT NULL REFERENCES employees(id),
            assignee_employee_id        INTEGER NOT NULL REFERENCES employees(id),
            reassignment_date           TEXT NOT NULL,
            title                       TEXT DEFAULT '',
            status                      TEXT DEFAULT 'Pending',
            remarks                     TEXT DEFAULT '',
            marked_by                   TEXT DEFAULT '',
            marked_at                   TEXT DEFAULT '',
            assigned_by                 TEXT DEFAULT '',
            created_at                  TEXT DEFAULT (datetime('now')),
            UNIQUE(original_responsibility_id, reassignment_date)
        );
        CREATE TABLE IF NOT EXISTS task_timer_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            task_log_id         INTEGER NOT NULL REFERENCES task_logs(id),
            responsibility_id   INTEGER NOT NULL,
            employee_id         INTEGER NOT NULL,
            log_date            TEXT NOT NULL,
            event_type          TEXT NOT NULL,
            event_at            TEXT NOT NULL,
            actor               TEXT DEFAULT '',
            notes               TEXT DEFAULT '',
            created_at          TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_timer_events_log ON task_timer_events(task_log_id, id);
        CREATE INDEX IF NOT EXISTS idx_timer_events_emp_date ON task_timer_events(employee_id, log_date);
        CREATE TABLE IF NOT EXISTS task_approval_notifications (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            task_log_id             INTEGER NOT NULL,
            responsibility_id       INTEGER NOT NULL,
            linked_to_employee_id   INTEGER NOT NULL,
            assignee_employee_id    INTEGER NOT NULL,
            log_date                TEXT NOT NULL,
            title                   TEXT DEFAULT '',
            assignee_name           TEXT DEFAULT '',
            message                 TEXT DEFAULT '',
            is_read                 INTEGER DEFAULT 0,
            created_at              TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_approval_notif_linked
            ON task_approval_notifications(linked_to_employee_id, is_read, id);
        """
    )

    # Lifecycle rename: Resolved → Resolve (keep legacy display mapping)
    try:
        conn.execute("UPDATE issue_logs SET status='Resolve' WHERE status='Resolved'")
    except sqlite3.OperationalError:
        pass

    conn.execute(
        "UPDATE one_time_tasks SET status='Done' WHERE status='Completed'"
    )
    for idx_sql in (
        "CREATE INDEX IF NOT EXISTS idx_issue_logs_status ON issue_logs(status)",
        "CREATE INDEX IF NOT EXISTS idx_issue_logs_emp ON issue_logs(employee_id)",
        "CREATE INDEX IF NOT EXISTS idx_issue_logs_date ON issue_logs(issue_date)",
        "CREATE INDEX IF NOT EXISTS idx_issue_history_issue ON issue_history(issue_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_issue_comments_issue ON issue_comments(issue_id)",
        "CREATE INDEX IF NOT EXISTS idx_resp_freq ON responsibilities(frequency)",
        "CREATE INDEX IF NOT EXISTS idx_resp_priority ON responsibilities(priority)",
        "CREATE INDEX IF NOT EXISTS idx_ott_priority ON one_time_tasks(priority)",
        "CREATE INDEX IF NOT EXISTS idx_emp_reports ON employees(reports_to_employee_id)",
        "CREATE INDEX IF NOT EXISTS idx_task_logs_approval ON task_logs(approval_status)",
        "CREATE INDEX IF NOT EXISTS idx_reassign_day ON day_reassignment_clones(assignee_employee_id, reassignment_date)",
    ):
        try:
            conn.execute(idx_sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def write_task_audit(
    entity_type: str,
    entity_id: int | None,
    action: str,
    *,
    old_value: str = "",
    new_value: str = "",
    actor: str = "",
    notes: str = "",
    conn=None,
) -> None:
    owns = conn is None
    if owns:
        conn = _connect()
    conn.execute(
        """INSERT INTO hrm_task_audit(entity_type, entity_id, action, old_value, new_value, actor, notes, created_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            entity_type,
            entity_id,
            action,
            old_value or "",
            new_value or "",
            actor or "",
            notes or "",
            _now_iso(),
        ),
    )
    if owns:
        conn.commit()
        conn.close()


def parse_duration_to_minutes(value) -> int:
    """Parse minutes (int) or HH:MM / H:MM time strings into minutes. Raises ValueError."""
    if value is None or value == "":
        raise ValueError("Duration is required")
    if isinstance(value, (int, float)):
        mins = int(value)
        if mins < 0:
            raise ValueError("Duration cannot be negative")
        return mins
    s = str(value).strip()
    if not s:
        raise ValueError("Duration is required")
    if s.isdigit():
        return max(0, int(s))
    # HH:MM or H:MM
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError("Time must be HH:MM")
        h_s, m_s = parts[0].strip(), parts[1].strip()
        if not h_s.isdigit() or not m_s.isdigit():
            raise ValueError("Time must be HH:MM")
        h, m = int(h_s), int(m_s)
        if h < 0 or m < 0 or m > 59:
            raise ValueError("Invalid time (minutes 0-59)")
        return h * 60 + m
    raise ValueError("Invalid duration format (use minutes or HH:MM)")


def is_schedule_due(
    frequency: str,
    check_date: str,
    weekday: str = "",
    month_day: int = 0,
    schedule_month: int = 0,
) -> bool:
    """True if a responsibility should appear on check_date for its frequency/schedule.

    Fortnightly: 2nd and 4th occurrence of the selected weekday each month.
    Quarterly: anchor month (1–12) then every 3 months; no day selection (due all days in cycle months).
    Whenever Required: always available (UI lists at bottom of Employee Check).
    """
    freq = (frequency or "Daily").strip()
    if not freq or freq == "Whenever Required":
        return True
    try:
        d = date.fromisoformat(check_date[:10])
    except ValueError:
        return True
    wd_name = WEEKDAYS[d.weekday()] if d.weekday() < len(WEEKDAYS) else ""
    if freq == "Daily":
        return True
    if freq == "Weekly":
        wanted = (weekday or "").strip()
        if not wanted:
            return True  # unscheduled weekly still shows (backward compatible)
        return wanted.lower() == wd_name.lower()
    if freq == "Fortnightly":
        wanted = (weekday or "").strip()
        if not wanted:
            return False
        if wanted.lower() != wd_name.lower():
            return False
        return weekday_occurrence_in_month(d) in (2, 4)
    if freq == "Monthly":
        md = int(month_day or 0)
        if md <= 0:
            return True
        return d.day == min(md, 28) or d.day == md
    if freq == "Quarterly":
        anchor = int(schedule_month or 0) or int(month_day or 0) or 1
        # Month-based cycle only — show every day of the anchor quarter months
        return d.month in quarterly_months(anchor)
    if freq == "Yearly":
        md = int(month_day or 1)
        sm = int(schedule_month or 1)
        return d.month == sm and d.day == min(max(md, 1), 28)
    return True


def task_log_counts_for_performance(
    status: str,
    *,
    approval_status: str = "",
    log_date: str = "",
    linked_to_employee_id: int | None = None,
) -> str | None:
    """Return performance bucket or None if neutral / not counted.

    Pre-cutover: Done/Partial credit without linked approval (legacy freeze).
    Post-cutover: N/A has no impact; Done only after Self/Approved/Auto-Approved.
    """
    st = (status or "").strip()
    if st in NEUTRAL_TASK_STATUSES:
        return None
    ld = (log_date or "")[:10]
    legacy = bool(ld and ld < PERFORMANCE_CUTOVER_DATE)
    appr = (approval_status or "").strip()
    linked = bool(linked_to_employee_id)

    if st == "Done":
        if legacy:
            return "done"
        if not linked or appr in ("Approved", "Auto-Approved", "Self"):
            return "done"
        return None  # awaiting linked approval
    if st == "Partial":
        if legacy:
            return "partial"
        if not linked or appr in ("Approved", "Auto-Approved", "Self"):
            return "partial"
        return None
    if st == "Missed":
        return "missed"
    if st == "Blocked":
        return "blocked"
    return None


def hod_status_editable(marked_at: str | None, *, now: datetime | None = None) -> bool:
    """Status remains editable until end of next calendar day after first mark."""
    if not marked_at:
        return True
    now = now or datetime.now()
    try:
        raw = str(marked_at).replace("T", " ").strip()
        marked_dt = datetime.fromisoformat(raw[:19] if len(raw) >= 19 else raw[:10])
    except ValueError:
        try:
            marked_dt = datetime.strptime(str(marked_at)[:10], "%Y-%m-%d")
        except ValueError:
            return True
    deadline = (marked_dt.date() + timedelta(days=HOD_STATUS_EDIT_GRACE_DAYS + 1))
    # end of next day after assignment/mark day ⇒ deadline is exclusive date after next day
    # marked Mon → editable through end of Tue → now.date() <= Tue
    last_editable = marked_dt.date() + timedelta(days=HOD_STATUS_EDIT_GRACE_DAYS)
    return now.date() <= last_editable


def find_employees_by_name_prefix(query: str, *, limit: int = 15) -> list[dict]:
    """Autocomplete: matching active employees by name."""
    q = (query or "").strip()
    if len(q) < 1:
        return []
    conn = _connect()
    rows = conn.execute(
        """
        SELECT e.id, e.emp_code, e.name, e.department_id, e.designation, d.name as department_name
        FROM employees e
        LEFT JOIN departments d ON d.id=e.department_id
        WHERE e.status='Active' AND LOWER(e.name) LIKE LOWER(?)
        ORDER BY e.name
        LIMIT ?
        """,
        (f"%{q}%", int(limit)),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def employee_name_exists(name: str, *, exclude_id: int | None = None) -> bool:
    conn = _connect()
    if exclude_id:
        row = conn.execute(
            "SELECT id FROM employees WHERE LOWER(TRIM(name))=LOWER(TRIM(?)) AND status='Active' AND id!=? LIMIT 1",
            (name, exclude_id),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM employees WHERE LOWER(TRIM(name))=LOWER(TRIM(?)) AND status='Active' LIMIT 1",
            (name,),
        ).fetchone()
    conn.close()
    return row is not None


def emp_code_exists(emp_code: str, *, exclude_id: int | None = None) -> bool:
    conn = _connect()
    if exclude_id:
        row = conn.execute(
            "SELECT id FROM employees WHERE UPPER(TRIM(emp_code))=UPPER(TRIM(?)) AND id!=? LIMIT 1",
            (emp_code, exclude_id),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM employees WHERE UPPER(TRIM(emp_code))=UPPER(TRIM(?)) LIMIT 1",
            (emp_code,),
        ).fetchone()
    conn.close()
    return row is not None


def get_dashboard_stats(department_id: int | None = None, employee_id: int | None = None) -> dict:
    conn = _connect()
    emp_conds = ["e.status='Active'"]
    emp_params: list = []
    if department_id:
        emp_conds.append("e.department_id=?")
        emp_params.append(department_id)
    if employee_id:
        emp_conds.append("e.id=?")
        emp_params.append(employee_id)
    emp_where = " AND ".join(emp_conds)
    total_employees = conn.execute(
        f"SELECT COUNT(*) FROM employees e WHERE {emp_where}", emp_params
    ).fetchone()[0]
    if department_id:
        dept_count = 1
        hod_rows = conn.execute(
            "SELECT id, name, hod_name FROM departments WHERE id=?",
            (department_id,),
        ).fetchall()
    else:
        dept_count = conn.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
        hod_rows = conn.execute(
            "SELECT id, name, hod_name FROM departments ORDER BY name"
        ).fetchall()
    open_issues = conn.execute(
        "SELECT COUNT(*) FROM issue_logs WHERE status='Open'"
    ).fetchone()[0]
    pending_tasks = conn.execute(
        "SELECT COUNT(*) FROM one_time_tasks WHERE active=1 AND status IN ('Pending','In Progress','Done')"
    ).fetchone()[0]
    conn.close()
    return {
        "department_count": int(dept_count or 0),
        "total_employees": int(total_employees or 0),
        "open_issues": int(open_issues or 0),
        "pending_tasks": int(pending_tasks or 0),
        "hods": [
            {"department_id": r["id"], "department": r["name"], "hod_name": r["hod_name"] or ""}
            for r in hod_rows
        ],
    }


def get_org_hierarchy() -> dict:
    """Departments tree + employee reporting tree."""
    conn = _connect()
    depts = [dict(r) for r in conn.execute("SELECT * FROM departments ORDER BY name").fetchall()]
    emps = [
        dict(r)
        for r in conn.execute(
            """
            SELECT e.id, e.emp_code, e.name, e.department_id, e.designation,
                   e.reports_to_employee_id, d.name as department_name, d.hod_name
            FROM employees e
            LEFT JOIN departments d ON d.id=e.department_id
            WHERE e.status='Active'
            ORDER BY e.name
            """
        ).fetchall()
    ]
    conn.close()

    dept_by_id = {d["id"]: {**d, "children": []} for d in depts}
    dept_roots = []
    for d in depts:
        node = dept_by_id[d["id"]]
        pid = d.get("parent_department_id")
        if pid and pid in dept_by_id:
            dept_by_id[pid]["children"].append(node)
        else:
            dept_roots.append(node)

    emp_by_id = {e["id"]: {**e, "reports": []} for e in emps}
    emp_roots = []
    for e in emps:
        node = emp_by_id[e["id"]]
        mid = e.get("reports_to_employee_id")
        if mid and mid in emp_by_id:
            emp_by_id[mid]["reports"].append(node)
        else:
            emp_roots.append(node)

    return {
        "departments": dept_roots,
        "reporting": emp_roots,
        "employees_flat": emps,
        "departments_flat": depts,
    }


def get_task_based_report(
    *,
    department_id: int | None = None,
    employee_id: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    priority: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Task-oriented report combining responsibilities (period activity) + one-time tasks."""
    today = date.today().isoformat()
    fd = from_date or today
    td = to_date or today
    conn = _connect()
    rows: list[dict] = []

    # One-time tasks as primary rows
    ot_conds = ["t.active=1"]
    ot_params: list = []
    if department_id:
        ot_conds.append("t.department_id=?")
        ot_params.append(department_id)
    if employee_id:
        ot_conds.append("t.employee_id=?")
        ot_params.append(employee_id)
    if priority:
        ot_conds.append("COALESCE(t.priority,'Medium')=?")
        ot_params.append(priority)
    if status:
        ot_conds.append("t.status=?")
        ot_params.append(status)
    ot_where = " AND ".join(ot_conds)
    for r in conn.execute(
        f"""
        SELECT t.*, e.name as employee_name, d.name as department_name
        FROM one_time_tasks t
        LEFT JOIN employees e ON e.id=t.employee_id
        LEFT JOIN departments d ON d.id=t.department_id
        WHERE {ot_where}
        ORDER BY t.due_date, t.title
        """,
        ot_params,
    ).fetchall():
        d = dict(r)
        st = d.get("status") or "Pending"
        pct_map = {
            "Pending": 0,
            "In Progress": 40,
            "Done": 80,
            "Completed": 80,
            "Approved": 100,
            "Rejected": 20,
        }
        rows.append(
            {
                "kind": "task",
                "task": d.get("title"),
                "assigned_to": d.get("employee_name") or "",
                "assigned_by": d.get("assigned_by") or "",
                "department": d.get("department_name") or "",
                "due_date": d.get("due_date") or "",
                "status": st,
                "completion_pct": pct_map.get(st, 0),
                "priority": d.get("priority") or "Medium",
                "frequency": "One-time",
                "mandatory": False,
                "task_id": d.get("id"),
                "employee_id": d.get("employee_id"),
            }
        )

    # Responsibility summary in period
    resp_conds = ["r.active=1"]
    resp_params: list = []
    if department_id:
        resp_conds.append("r.department_id=?")
        resp_params.append(department_id)
    if employee_id:
        resp_conds.append("r.employee_id=?")
        resp_params.append(employee_id)
    if priority:
        resp_conds.append("COALESCE(r.priority,'Medium')=?")
        resp_params.append(priority)
    resp_where = " AND ".join(resp_conds)
    for r in conn.execute(
        f"""
        SELECT r.*, e.name as employee_name, d.name as department_name,
          (SELECT COUNT(*) FROM task_logs tl
             WHERE tl.responsibility_id=r.id AND tl.log_date BETWEEN ? AND ?
               AND tl.status NOT IN ('Leave','N/A')) AS log_total,
          (SELECT COUNT(*) FROM task_logs tl
             WHERE tl.responsibility_id=r.id AND tl.log_date BETWEEN ? AND ?
               AND tl.status='Done') AS log_done,
          (SELECT COUNT(*) FROM task_logs tl
             WHERE tl.responsibility_id=r.id AND tl.log_date BETWEEN ? AND ?
               AND tl.status='Partial') AS log_partial
        FROM responsibilities r
        LEFT JOIN employees e ON e.id=r.employee_id
        LEFT JOIN departments d ON d.id=r.department_id
        WHERE {resp_where}
        ORDER BY r.title
        """,
        [fd, td, fd, td, fd, td, *resp_params],
    ).fetchall():
        d = dict(r)
        total = int(d.get("log_total") or 0)
        done = int(d.get("log_done") or 0)
        partial = int(d.get("log_partial") or 0)
        pct = round((done + partial * 0.5) / total * 100, 1) if total else 0
        st = "Done" if pct >= 100 else ("In Progress" if total else "Pending")
        if status and st != status and (status not in (st, "Pending") or total):
            if status == "Pending" and total:
                continue
            if status not in (st,):
                continue
        rows.append(
            {
                "kind": "responsibility",
                "task": d.get("title"),
                "assigned_to": d.get("employee_name") or "",
                "assigned_by": d.get("added_by") or "",
                "department": d.get("department_name") or "",
                "due_date": "",
                "status": st,
                "completion_pct": pct,
                "priority": d.get("priority") or "Medium",
                "frequency": d.get("frequency") or "Daily",
                "mandatory": bool(int(d.get("mandatory") or 0)),
                "responsibility_id": d.get("id"),
                "employee_id": d.get("employee_id"),
            }
        )
    conn.close()
    return rows


def _next_emp_code(conn):
    row = conn.execute("SELECT emp_code FROM employees ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try:
            n = int(row[0].replace("EMP-", "")) + 1
        except ValueError:
            pass
    return f"EMP-{n:03d}"


def employee_department_id(employee_id: int) -> int | None:
    conn = _connect()
    row = conn.execute("SELECT department_id FROM employees WHERE id=?", (employee_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return row["department_id"]


def get_responsibility_owner(responsibility_id: int) -> int | None:
    conn = _connect()
    row = conn.execute(
        "SELECT employee_id FROM responsibilities WHERE id=? AND active=1",
        (responsibility_id,),
    ).fetchone()
    conn.close()
    return int(row["employee_id"]) if row else None


def list_departments(department_id: int | None = None):
    conn = _connect()
    if department_id is not None and int(department_id) < 0:
        conn.close()
        return []
    if department_id is not None:
        rows = conn.execute(
            "SELECT * FROM departments WHERE id=? ORDER BY name",
            (int(department_id),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM departments ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_department(data: dict):
    conn = _connect()
    conn.execute(
        "INSERT INTO departments(name,description,hod_name) VALUES(?,?,?)",
        (data["name"], data.get("description", ""), data.get("hod_name", "")),
    )
    conn.commit()
    conn.close()


def update_department(did: int, data: dict):
    conn = _connect()
    allowed = ["name", "description", "hod_name", "parent_department_id"]
    sets = ", ".join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [did]
    if sets:
        conn.execute(f"UPDATE departments SET {sets} WHERE id=?", vals)
        conn.commit()
    conn.close()


def list_employees(department_id=None, status="Active", employee_id: int | None = None):
    conn = _connect()
    if employee_id is not None and int(employee_id) < 0:
        conn.close()
        return []
    if employee_id is not None:
        rows = conn.execute(
            """
            SELECT e.*, d.name as department_name
            FROM employees e LEFT JOIN departments d ON d.id=e.department_id
            WHERE e.id=? AND e.status=? ORDER BY e.name
        """,
            (int(employee_id), status),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    if department_id is not None and int(department_id) < 0:
        conn.close()
        return []
    if department_id:
        rows = conn.execute(
            """
            SELECT e.*, d.name as department_name
            FROM employees e LEFT JOIN departments d ON d.id=e.department_id
            WHERE e.department_id=? AND e.status=? ORDER BY e.name
        """,
            (department_id, status),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT e.*, d.name as department_name
            FROM employees e LEFT JOIN departments d ON d.id=e.department_id
            WHERE e.status=? ORDER BY d.name, e.name
        """,
            (status,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_employee(data: dict):
    conn = _connect()
    name = (data.get("name") or "").strip()
    if not name:
        conn.close()
        raise ValueError("Employee name is required")
    # Prevent duplicate active employee by name
    existing = conn.execute(
        "SELECT id, emp_code FROM employees WHERE LOWER(TRIM(name))=LOWER(?) AND status='Active' LIMIT 1",
        (name,),
    ).fetchone()
    if existing:
        conn.close()
        raise ValueError(f"Employee already exists: {name} ({existing['emp_code']})")
    code = (data.get("emp_code") or "").strip() or _next_emp_code(conn)
    if conn.execute(
        "SELECT id FROM employees WHERE UPPER(TRIM(emp_code))=UPPER(?) LIMIT 1",
        (code,),
    ).fetchone():
        conn.close()
        raise ValueError(f"Employee ID already exists: {code}")
    conn.execute(
        """INSERT INTO employees(emp_code,name,department_id,designation,phone,email,join_date,status,reports_to_employee_id)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            code,
            name,
            data.get("department_id"),
            data.get("designation", ""),
            data.get("phone", ""),
            data.get("email", ""),
            data.get("join_date", ""),
            "Active",
            data.get("reports_to_employee_id"),
        ),
    )
    conn.commit()
    conn.close()
    return code


def update_employee(eid: int, data: dict):
    conn = _connect()
    allowed = [
        "name",
        "department_id",
        "designation",
        "phone",
        "email",
        "join_date",
        "status",
        "emp_code",
        "reports_to_employee_id",
    ]
    payload = {k: data[k] for k in data if k in allowed}
    if "name" in payload and payload["name"]:
        dup = conn.execute(
            "SELECT id FROM employees WHERE LOWER(TRIM(name))=LOWER(TRIM(?)) AND status='Active' AND id!=? LIMIT 1",
            (payload["name"], eid),
        ).fetchone()
        if dup:
            conn.close()
            raise ValueError(f"Employee name already exists: {payload['name']}")
    if "emp_code" in payload and payload["emp_code"]:
        code = str(payload["emp_code"]).strip()
        if not code:
            conn.close()
            raise ValueError("Employee ID cannot be empty")
        dup = conn.execute(
            "SELECT id FROM employees WHERE UPPER(TRIM(emp_code))=UPPER(?) AND id!=? LIMIT 1",
            (code, eid),
        ).fetchone()
        if dup:
            conn.close()
            raise ValueError(f"Employee ID already exists: {code}")
        payload["emp_code"] = code
    sets = ", ".join(f"{k}=?" for k in payload)
    vals = list(payload.values()) + [eid]
    if sets:
        conn.execute(f"UPDATE employees SET {sets} WHERE id=?", vals)
        conn.commit()
    conn.close()


def delete_employee(eid: int) -> bool:
    conn = _connect()
    row = conn.execute("SELECT id FROM employees WHERE id=?", (eid,)).fetchone()
    if not row:
        conn.close()
        return False
    conn.execute("UPDATE employees SET status='Inactive' WHERE id=?", (eid,))
    conn.execute("UPDATE responsibilities SET active=0 WHERE employee_id=?", (eid,))
    conn.commit()
    conn.close()
    return True


def _normalize_import_row(row: dict) -> dict:
    out: dict = {}
    for k, v in row.items():
        key = str(k).strip().lower().replace(" ", "_")
        out[key] = "" if v is None else str(v).strip()
    return out


def _resolve_employee_id(conn, row: dict) -> int | None:
    code = row.get("employee_code") or row.get("emp_code") or ""
    name = row.get("employee_name") or row.get("employee") or row.get("name") or ""
    if code:
        found = conn.execute(
            "SELECT id FROM employees WHERE emp_code=? AND status='Active'",
            (code,),
        ).fetchone()
        if found:
            return int(found["id"])
    if name:
        found = conn.execute(
            "SELECT id FROM employees WHERE LOWER(name)=LOWER(?) AND status='Active'",
            (name,),
        ).fetchone()
        if found:
            return int(found["id"])
    return None


def import_responsibilities(rows: list[dict]) -> dict:
    created = 0
    errors: list[str] = []
    conn = _connect()
    try:
        for idx, raw in enumerate(rows, start=1):
            row = _normalize_import_row(raw)
            title = row.get("title") or row.get("task") or row.get("responsibility") or ""
            if not title:
                errors.append(f"Row {idx}: missing title")
                continue
            emp_id = _resolve_employee_id(conn, row)
            if not emp_id:
                errors.append(f"Row {idx}: employee not found ({row.get('employee_name') or row.get('emp_code') or '?'})")
                continue
            create_responsibility(
                {
                    "employee_id": emp_id,
                    "title": title,
                    "description": row.get("description", ""),
                    "frequency": row.get("frequency") or "Daily",
                    "category": row.get("category") or "General",
                    "added_by": row.get("added_by") or row.get("assigned_by") or "",
                }
            )
            created += 1
    finally:
        conn.close()
    return {"created": created, "errors": errors}


def import_one_time_tasks(rows: list[dict]) -> dict:
    created = 0
    errors: list[str] = []
    conn = _connect()
    try:
        for idx, raw in enumerate(rows, start=1):
            row = _normalize_import_row(raw)
            title = row.get("title") or row.get("task") or ""
            if not title:
                errors.append(f"Row {idx}: missing title")
                continue
            emp_id = _resolve_employee_id(conn, row)
            if not emp_id:
                errors.append(f"Row {idx}: employee not found ({row.get('employee_name') or row.get('emp_code') or '?'})")
                continue
            create_one_time_task(
                {
                    "employee_id": emp_id,
                    "title": title,
                    "description": row.get("description", ""),
                    "due_date": row.get("due_date") or row.get("due") or "",
                    "assigned_by": row.get("assigned_by") or row.get("added_by") or "",
                }
            )
            created += 1
    finally:
        conn.close()
    return {"created": created, "errors": errors}


def list_responsibilities(employee_id=None, department_id=None, active_only=True):
    conn = _connect()
    conditions = []
    params = []
    if active_only:
        conditions.append("r.active=1")
    if employee_id:
        conditions.append("r.employee_id=?")
        params.append(employee_id)
    if department_id:
        conditions.append("r.department_id=?")
        params.append(department_id)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(
        f"""
        SELECT r.*, e.name as employee_name, d.name as department_name,
               le.name as linked_to_employee_name,
               be.name as backup_employee_name
        FROM responsibilities r
        LEFT JOIN employees e ON e.id=r.employee_id
        LEFT JOIN departments d ON d.id=r.department_id
        LEFT JOIN employees le ON le.id=r.linked_to_employee_id
        LEFT JOIN employees be ON be.id=r.backup_employee_id
        {where}
        ORDER BY d.name, e.name, r.frequency, r.title
    """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _validate_responsibility_schedule(freq: str, weekday: str, month_day: int, schedule_month: int):
    if freq == "Weekly" and not (weekday or "").strip():
        raise ValueError("Weekly responsibilities require a weekday")
    if freq == "Fortnightly" and not (weekday or "").strip():
        raise ValueError("Fortnightly responsibilities require a weekday (Mon–Sun)")
    if freq == "Monthly" and month_day <= 0:
        raise ValueError("Monthly responsibilities require a calendar day (1-31)")
    if freq == "Quarterly" and not (1 <= int(schedule_month or 0) <= 12):
        raise ValueError("Quarterly responsibilities require an anchor month (1–12)")


def create_responsibility(data: dict):
    conn = _connect()
    dept_id = data.get("department_id")
    if not dept_id and data.get("employee_id"):
        row = conn.execute(
            "SELECT department_id FROM employees WHERE id=?", (data["employee_id"],)
        ).fetchone()
        if row:
            dept_id = row["department_id"]
    freq = data.get("frequency", "Daily") or "Daily"
    if freq not in FREQUENCIES:
        # accept legacy free-text frequencies
        pass
    weekday = (data.get("schedule_weekday") or "").strip()
    month_day = int(data.get("schedule_month_day") or 0)
    schedule_month = int(data.get("schedule_month") or 0)
    if freq == "Quarterly" and schedule_month <= 0 and month_day > 0 and month_day <= 12:
        schedule_month = month_day  # tolerate older clients
    try:
        _validate_responsibility_schedule(freq, weekday, month_day, schedule_month)
    except ValueError:
        conn.close()
        raise
    priority = data.get("priority") or "Medium"
    if priority not in PRIORITIES:
        priority = "Medium"
    linked = data.get("linked_to_employee_id")
    try:
        linked_id = int(linked) if linked not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        linked_id = None
    if linked_id and int(data.get("employee_id") or 0) and int(linked_id) == int(data["employee_id"]):
        conn.close()
        raise ValueError("Linked person must be different from the assigned employee")
    require_backup = bool(data.get("require_backup", False))
    try:
        backup_id, backup_val, backup_unit = _parse_backup_fields(
            data, int(data["employee_id"]), require=require_backup
        )
    except ValueError:
        conn.close()
        raise
    conn.execute(
        """INSERT INTO responsibilities(
            employee_id,department_id,title,description,frequency,category,added_by,active,
            priority,mandatory,schedule_weekday,schedule_month_day,time_period,
            schedule_month,linked_to_employee_id,
            backup_employee_id,backup_allocation_value,backup_allocation_unit
        )
        VALUES(?,?,?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?)""",
        (
            data["employee_id"],
            dept_id,
            data["title"],
            data.get("description", ""),
            freq,
            data.get("category", "General"),
            data.get("added_by", ""),
            priority,
            1 if data.get("mandatory") else 0,
            weekday,
            month_day,
            data.get("time_period", "") or "",
            schedule_month or 0,
            linked_id,
            backup_id,
            backup_val,
            backup_unit,
        ),
    )
    conn.commit()
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return rid


def update_responsibility(rid: int, data: dict):
    conn = _connect()
    allowed = [
        "title",
        "description",
        "frequency",
        "category",
        "employee_id",
        "active",
        "added_by",
        "priority",
        "mandatory",
        "schedule_weekday",
        "schedule_month_day",
        "time_period",
        "department_id",
        "schedule_month",
        "linked_to_employee_id",
        "backup_employee_id",
        "backup_allocation_value",
        "backup_allocation_unit",
    ]
    payload = {k: data[k] for k in data if k in allowed}
    if "mandatory" in payload:
        payload["mandatory"] = 1 if payload["mandatory"] else 0
    if "linked_to_employee_id" in payload:
        v = payload["linked_to_employee_id"]
        try:
            payload["linked_to_employee_id"] = int(v) if v not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            payload["linked_to_employee_id"] = None
    # Backup is mandatory on create and when assignee/backup fields change.
    # Sending an unchanged employee_id (common on full-form edits) must not
    # force backup validation on legacy rows that predate the backup requirement.
    backup_touched = any(
        k in payload
        for k in (
            "backup_employee_id",
            "backup_allocation_value",
            "backup_allocation_unit",
        )
    )
    emp_changed = False
    row0 = None
    if "employee_id" in payload or backup_touched:
        row0 = conn.execute("SELECT * FROM responsibilities WHERE id=?", (rid,)).fetchone()
        if row0 and "employee_id" in payload:
            try:
                emp_changed = int(payload["employee_id"]) != int(row0["employee_id"])
            except (TypeError, ValueError):
                emp_changed = True
    if backup_touched or emp_changed:
        if row0 is None:
            row0 = conn.execute("SELECT * FROM responsibilities WHERE id=?", (rid,)).fetchone()
        if row0:
            merged = dict(row0)
            merged.update(payload)
            try:
                bid, bval, bunit = _parse_backup_fields(
                    merged, int(merged["employee_id"]), require=True
                )
            except ValueError:
                conn.close()
                raise
            payload["backup_employee_id"] = bid
            payload["backup_allocation_value"] = bval
            payload["backup_allocation_unit"] = bunit
    if "backup_employee_id" in payload:
        v = payload["backup_employee_id"]
        try:
            payload["backup_employee_id"] = int(v) if v not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            payload["backup_employee_id"] = None
    if "linked_to_employee_id" in payload or "employee_id" in payload:
        row_link = conn.execute(
            "SELECT employee_id, linked_to_employee_id FROM responsibilities WHERE id=?",
            (rid,),
        ).fetchone()
        if row_link:
            try:
                emp_id = int(payload.get("employee_id", row_link["employee_id"]))
            except (TypeError, ValueError):
                emp_id = int(row_link["employee_id"])
            linked_raw = payload.get("linked_to_employee_id", row_link["linked_to_employee_id"])
            try:
                linked_id = (
                    int(linked_raw) if linked_raw not in (None, "", 0, "0") else None
                )
            except (TypeError, ValueError):
                linked_id = None
            if linked_id and linked_id == emp_id:
                conn.close()
                raise ValueError(
                    "Linked person must be different from the assigned employee"
                )
    if any(
        k in payload
        for k in (
            "frequency",
            "schedule_weekday",
            "schedule_month_day",
            "schedule_month",
        )
    ):
        row = conn.execute("SELECT * FROM responsibilities WHERE id=?", (rid,)).fetchone()
        if row:
            freq = payload.get("frequency", row["frequency"])
            weekday = payload.get(
                "schedule_weekday",
                row["schedule_weekday"] if "schedule_weekday" in row.keys() else "",
            )
            month_day = int(
                payload.get(
                    "schedule_month_day",
                    row["schedule_month_day"] if "schedule_month_day" in row.keys() else 0,
                )
                or 0
            )
            schedule_month = int(
                payload.get(
                    "schedule_month",
                    row["schedule_month"] if "schedule_month" in row.keys() else 0,
                )
                or 0
            )
            try:
                _validate_responsibility_schedule(freq, weekday, month_day, schedule_month)
            except ValueError:
                conn.close()
                raise
    sets = ", ".join(f"{k}=?" for k in payload)
    vals = list(payload.values())
    if sets:
        vals.append(now_ist().strftime("%Y-%m-%d %H:%M:%S"))
        vals.append(rid)
        conn.execute(f"UPDATE responsibilities SET {sets}, updated_at=? WHERE id=?", vals)
        conn.commit()
    conn.close()


def delete_responsibility(rid: int):
    conn = _connect()
    conn.execute("UPDATE responsibilities SET active=0 WHERE id=?", (rid,))
    conn.commit()
    conn.close()


def mark_task(
    responsibility_id: int,
    log_date: str,
    status: str,
    marked_by: str = "",
    remarks: str = "",
    blocker_employee_id: int = None,
    blocker_reason: str = "",
    *,
    allow_override: bool = False,
):
    if status not in TASK_LOG_STATUSES:
        return "invalid_status"

    # Employees may only mark within task date + next 2 IST calendar days
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"

    conn = _connect()
    resp = conn.execute(
        """SELECT employee_id, department_id, linked_to_employee_id
           FROM responsibilities WHERE id=?""",
        (responsibility_id,),
    ).fetchone()
    if not resp:
        conn.close()
        return False

    linked_id = None
    if "linked_to_employee_id" in resp.keys() and resp["linked_to_employee_id"]:
        try:
            linked_id = int(resp["linked_to_employee_id"])
        except (TypeError, ValueError):
            linked_id = None

    # Self-complete when no Linked To; else Done/Partial need linked approval
    if status in ("Done", "Partial"):
        if linked_id:
            approval_status = "Pending"
            approved_by = ""
            approved_at = ""
        else:
            approval_status = "Self"
            approved_by = marked_by or "self"
            approved_at = _now_iso()
    elif status in NEUTRAL_TASK_STATUSES:
        approval_status = "N/A" if status == "N/A" else ""
        approved_by = ""
        approved_at = ""
    else:
        approval_status = ""
        approved_by = ""
        approved_at = ""

    existing = conn.execute(
        "SELECT id, marked_at, approval_status, status FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if existing:
        existing_status = str(existing["status"] or "Pending").strip() or "Pending"
        # A timer-only Pending DWR row is not a committed quality mark — employee can still mark Done.
        timer_only = existing_status in ("Pending",)
        if not allow_override and not timer_only:
            conn.close()
            return "locked"
        # HOD status editing only until end of next day after first assignment/mark
        if allow_override and not timer_only and not hod_status_editable(
            existing["marked_at"] if "marked_at" in existing.keys() else None
        ):
            conn.close()
            return "window_closed"
        conn.execute(
            """UPDATE task_logs
               SET status=?, remarks=?, marked_by=?, marked_at=?,
                   blocker_employee_id=?, blocker_reason=?,
                   approval_status=?, approved_by=?, approved_at=?
               WHERE responsibility_id=? AND log_date=?""",
            (
                status,
                remarks,
                marked_by,
                _now_iso(),
                blocker_employee_id,
                blocker_reason,
                approval_status,
                approved_by,
                approved_at,
                responsibility_id,
                log_date,
            ),
        )
        task_log_id = existing
    else:
        conn.execute(
            """INSERT INTO task_logs(
                responsibility_id,employee_id,log_date,status,remarks,marked_by,marked_at,
                blocker_employee_id,blocker_reason,approval_status,approved_by,approved_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(responsibility_id,log_date) DO NOTHING
        """,
            (
                responsibility_id,
                resp["employee_id"],
                log_date,
                status,
                remarks,
                marked_by,
                _now_iso(),
                blocker_employee_id,
                blocker_reason,
                approval_status,
                approved_by,
                approved_at,
            ),
        )

        task_log_id = conn.execute(
            "SELECT id FROM task_logs WHERE responsibility_id=? AND log_date=?",
            (responsibility_id, log_date),
        ).fetchone()

    if status == "Blocked" and blocker_employee_id:
        blocker = conn.execute(
            "SELECT name, department_id FROM employees WHERE id=?",
            (blocker_employee_id,),
        ).fetchone()
        resp_row = conn.execute(
            """
            SELECT r.title, e.name as emp_name
            FROM responsibilities r JOIN employees e ON e.id=r.employee_id
            WHERE r.id=?""",
            (responsibility_id,),
        ).fetchone()

        if blocker and resp_row and task_log_id:
            dup = conn.execute(
                "SELECT id FROM issue_logs WHERE task_log_id=? LIMIT 1",
                (task_log_id["id"],),
            ).fetchone()
            if not dup:
                conn.execute(
                """INSERT INTO issue_logs(
                employee_id, department_id, issue_date, issue_type, severity,
                title, description, recorded_by,
                caused_by_employee_id, caused_by_dept_id, task_log_id, status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    blocker_employee_id,
                    blocker["department_id"],
                    log_date,
                    "Dependency Missed",
                    "Moderate",
                    f"Dependency not met — {resp_row['emp_name']}'s work was blocked",
                    f"Task '{resp_row['title']}' was blocked. Reason: {blocker_reason}",
                    marked_by,
                    resp["employee_id"],
                    resp["department_id"],
                    task_log_id["id"] if task_log_id else None,
                    "Open",
                ),
            )

    # Notify Linked To person that approval is required (in-app inbox)
    if approval_status == "Pending" and linked_id and task_log_id:
        log_id = int(task_log_id["id"] if hasattr(task_log_id, "keys") else task_log_id["id"])
        title_row = conn.execute(
            """SELECT r.title, e.name as assignee_name
               FROM responsibilities r JOIN employees e ON e.id=r.employee_id
               WHERE r.id=?""",
            (responsibility_id,),
        ).fetchone()
        title = (title_row["title"] if title_row else "") or ""
        assignee_name = (title_row["assignee_name"] if title_row else "") or ""
        msg = (
            f"{assignee_name} marked '{title}' as {status} on {log_date}. "
            f"Your approval is required (Linked To)."
        )
        conn.execute(
            """DELETE FROM task_approval_notifications
               WHERE task_log_id=? AND linked_to_employee_id=? AND is_read=0""",
            (log_id, linked_id),
        )
        conn.execute(
            """INSERT INTO task_approval_notifications(
                task_log_id, responsibility_id, linked_to_employee_id, assignee_employee_id,
                log_date, title, assignee_name, message, is_read, created_at
            ) VALUES (?,?,?,?,?,?,?,?,0,?)""",
            (
                log_id,
                responsibility_id,
                linked_id,
                int(resp["employee_id"]),
                log_date,
                title,
                assignee_name,
                msg,
                _now_iso(),
            ),
        )

    conn.commit()
    conn.close()
    return True


def list_pending_linked_approvals(linked_employee_id: int) -> list[dict]:
    """Tasks awaiting Approve/Cancel for this Linked To employee."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT tl.id as task_log_id, tl.responsibility_id, tl.employee_id as assignee_employee_id,
               tl.log_date, tl.status, tl.approval_status, tl.remarks, tl.marked_by, tl.marked_at,
               r.title, r.priority, r.frequency, r.backup_employee_id,
               r.backup_allocation_value, r.backup_allocation_unit,
               e.name as assignee_name, d.name as department_name,
               be.name as backup_employee_name,
               le.name as linked_to_employee_name,
               r.linked_to_employee_id
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        JOIN employees e ON e.id=tl.employee_id
        LEFT JOIN departments d ON d.id=e.department_id
        LEFT JOIN employees be ON be.id=r.backup_employee_id
        LEFT JOIN employees le ON le.id=r.linked_to_employee_id
        WHERE r.linked_to_employee_id=?
          AND tl.approval_status='Pending'
          AND tl.status IN ('Done','Partial')
          AND r.active=1
          AND r.linked_to_employee_id != tl.employee_id
        ORDER BY tl.log_date DESC, tl.id DESC
        """,
        (int(linked_employee_id),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_approval_notifications(
    linked_employee_id: int, *, unread_only: bool = False, limit: int = 50
) -> list[dict]:
    conn = _connect()
    q = """SELECT * FROM task_approval_notifications
           WHERE linked_to_employee_id=?"""
    params: list = [int(linked_employee_id)]
    if unread_only:
        q += " AND is_read=0"
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_approval_notifications_read(
    linked_employee_id: int, *, task_log_id: int | None = None
) -> int:
    conn = _connect()
    if task_log_id is not None:
        cur = conn.execute(
            """UPDATE task_approval_notifications SET is_read=1
               WHERE linked_to_employee_id=? AND task_log_id=? AND is_read=0""",
            (int(linked_employee_id), int(task_log_id)),
        )
    else:
        cur = conn.execute(
            """UPDATE task_approval_notifications SET is_read=1
               WHERE linked_to_employee_id=? AND is_read=0""",
            (int(linked_employee_id),),
        )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return int(n or 0)


def _responsibility_owner(conn, responsibility_id: int):
    return conn.execute(
        "SELECT employee_id FROM responsibilities WHERE id=? AND active=1",
        (responsibility_id,),
    ).fetchone()


def _ensure_task_log_row(conn, responsibility_id: int, employee_id: int, log_date: str) -> dict:
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if row:
        return dict(row)
    conn.execute(
        """INSERT INTO task_logs(responsibility_id, employee_id, log_date, status)
           VALUES(?,?,?,'Pending')""",
        (responsibility_id, employee_id, log_date),
    )
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    return dict(row)


def _list_timer_events(conn, task_log_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT id, task_log_id, responsibility_id, employee_id, log_date,
                  event_type, event_at, actor, notes
           FROM task_timer_events WHERE task_log_id=? ORDER BY id""",
        (task_log_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _append_timer_event(
    conn,
    *,
    task_log_id: int,
    responsibility_id: int,
    employee_id: int,
    log_date: str,
    event_type: str,
    event_at: str,
    actor: str = "",
) -> None:
    conn.execute(
        """INSERT INTO task_timer_events(
            task_log_id, responsibility_id, employee_id, log_date, event_type, event_at, actor
        ) VALUES(?,?,?,?,?,?,?)""",
        (task_log_id, responsibility_id, employee_id, log_date, event_type, event_at, actor or ""),
    )


def _employee_has_active_timer(conn, employee_id: int, *, exclude_log_id: int | None = None) -> bool:
    """True if this employee already has an Active (started, not ended, not paused) timer."""
    q = """
        SELECT id FROM task_logs
        WHERE employee_id=?
          AND IFNULL(started_at,'') != ''
          AND IFNULL(ended_at,'') = ''
          AND IFNULL(paused_at,'') = ''
    """
    params: list = [employee_id]
    if exclude_log_id:
        q += " AND id!=?"
        params.append(exclude_log_id)
    return conn.execute(q, params).fetchone() is not None


def _last_active_segment_start(log: dict, events: list[dict]) -> str:
    for ev in reversed(events):
        if ev.get("event_type") in ("start", "resume"):
            return str(ev.get("event_at") or "")
    return str(log.get("started_at") or "")


def start_responsibility_timer(
    responsibility_id: int,
    log_date: str,
    *,
    allow_override: bool = False,
    actor: str = "",
):
    """Record start time on the day's DWR row. Does not overwrite other responsibilities."""
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"
    conn = _connect()
    resp = _responsibility_owner(conn, responsibility_id)
    if not resp:
        conn.close()
        return "not_found"
    emp_id = int(resp["employee_id"])
    row = _ensure_task_log_row(conn, responsibility_id, emp_id, log_date)
    started = str(row.get("started_at") or "").strip()
    ended = str(row.get("ended_at") or "").strip()
    paused = str(row.get("paused_at") or "").strip()
    if ended:
        conn.close()
        return "already_ended"
    if started and not paused:
        conn.commit()
        conn.close()
        return True  # already Active — idempotent
    if started and paused:
        conn.close()
        return "paused"  # must resume, not start again
    if _employee_has_active_timer(conn, emp_id, exclude_log_id=int(row["id"])):
        conn.close()
        return "already_active"
    now = _now_iso()
    conn.execute(
        """UPDATE task_logs SET started_at=?, paused_at='', active_seconds=0, paused_seconds=0,
               duration_minutes=0
           WHERE responsibility_id=? AND log_date=?""",
        (now, responsibility_id, log_date),
    )
    _append_timer_event(
        conn,
        task_log_id=int(row["id"]),
        responsibility_id=responsibility_id,
        employee_id=emp_id,
        log_date=log_date,
        event_type="start",
        event_at=now,
        actor=actor,
    )
    conn.commit()
    conn.close()
    return True


def pause_responsibility_timer(
    responsibility_id: int,
    log_date: str,
    *,
    allow_override: bool = False,
    actor: str = "",
):
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"
    conn = _connect()
    resp = _responsibility_owner(conn, responsibility_id)
    if not resp:
        conn.close()
        return "not_found"
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if not row:
        conn.close()
        return "not_started"
    row = dict(row)
    started = str(row.get("started_at") or "").strip()
    ended = str(row.get("ended_at") or "").strip()
    paused = str(row.get("paused_at") or "").strip()
    if not started:
        conn.close()
        return "not_started"
    if ended:
        conn.close()
        return "already_ended"
    if paused:
        conn.close()
        return "already_paused"
    events = _list_timer_events(conn, int(row["id"]))
    if not allow_override and _count_timer_events(events, "pause") >= MAX_TIMER_PAUSE_PER_DAY:
        conn.close()
        return "pause_limit"
    now = _now_iso()
    seg = _last_active_segment_start(row, events)
    add = _seconds_between(seg, now)
    active = int(row.get("active_seconds") or 0) + add
    conn.execute(
        """UPDATE task_logs SET paused_at=?, active_seconds=?, duration_minutes=?
           WHERE id=?""",
        (now, active, active // 60, int(row["id"])),
    )
    _append_timer_event(
        conn,
        task_log_id=int(row["id"]),
        responsibility_id=responsibility_id,
        employee_id=int(resp["employee_id"]),
        log_date=log_date,
        event_type="pause",
        event_at=now,
        actor=actor,
    )
    conn.commit()
    conn.close()
    return True


def resume_responsibility_timer(
    responsibility_id: int,
    log_date: str,
    *,
    allow_override: bool = False,
    actor: str = "",
):
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"
    conn = _connect()
    resp = _responsibility_owner(conn, responsibility_id)
    if not resp:
        conn.close()
        return "not_found"
    emp_id = int(resp["employee_id"])
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if not row:
        conn.close()
        return "not_started"
    row = dict(row)
    started = str(row.get("started_at") or "").strip()
    ended = str(row.get("ended_at") or "").strip()
    paused = str(row.get("paused_at") or "").strip()
    if not started:
        conn.close()
        return "not_started"
    if ended:
        conn.close()
        return "already_ended"
    if not paused:
        conn.close()
        return "already_active"
    events = _list_timer_events(conn, int(row["id"]))
    if not allow_override and _count_timer_events(events, "resume") >= MAX_TIMER_RESUME_PER_DAY:
        conn.close()
        return "resume_limit"
    if _employee_has_active_timer(conn, emp_id, exclude_log_id=int(row["id"])):
        conn.close()
        return "already_active"
    now = _now_iso()
    add_pause = _seconds_between(paused, now)
    paused_sec = int(row.get("paused_seconds") or 0) + add_pause
    conn.execute(
        """UPDATE task_logs SET paused_at='', paused_seconds=? WHERE id=?""",
        (paused_sec, int(row["id"])),
    )
    _append_timer_event(
        conn,
        task_log_id=int(row["id"]),
        responsibility_id=responsibility_id,
        employee_id=emp_id,
        log_date=log_date,
        event_type="resume",
        event_at=now,
        actor=actor,
    )
    conn.commit()
    conn.close()
    return True


def end_responsibility_timer(
    responsibility_id: int,
    log_date: str,
    *,
    allow_override: bool = False,
    actor: str = "",
):
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"
    conn = _connect()
    resp = _responsibility_owner(conn, responsibility_id)
    if not resp:
        conn.close()
        return "not_found"
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if not row:
        conn.close()
        return "not_started"
    row = dict(row)
    started = str(row.get("started_at") or "").strip()
    ended = str(row.get("ended_at") or "").strip()
    paused = str(row.get("paused_at") or "").strip()
    if not started:
        conn.close()
        return "not_started"
    if ended:
        conn.commit()
        conn.close()
        return "already_ended"
    events = _list_timer_events(conn, int(row["id"]))
    if _count_timer_events(events, "end") >= MAX_TIMER_COMPLETE_PER_DAY:
        conn.close()
        return "complete_limit"
    now = _now_iso()
    active = int(row.get("active_seconds") or 0)
    paused_sec = int(row.get("paused_seconds") or 0)
    if paused:
        paused_sec += _seconds_between(paused, now)
    else:
        seg = _last_active_segment_start(row, events)
        active += _seconds_between(seg, now)
    mins = active // 60
    conn.execute(
        """UPDATE task_logs SET ended_at=?, paused_at='', active_seconds=?, paused_seconds=?,
               duration_minutes=?
           WHERE id=?""",
        (now, active, paused_sec, mins, int(row["id"])),
    )
    _append_timer_event(
        conn,
        task_log_id=int(row["id"]),
        responsibility_id=responsibility_id,
        employee_id=int(resp["employee_id"]),
        log_date=log_date,
        event_type="end",
        event_at=now,
        actor=actor,
    )
    conn.commit()
    conn.close()
    return True


def set_responsibility_manual_time(
    responsibility_id: int,
    log_date: str,
    started_at: str | None,
    ended_at: str | None,
    *,
    allow_override: bool = False,
    actor: str = "",
):
    """Set or edit DWR start/end times. End cannot be earlier than start. Clears pause state."""
    if not allow_override and not in_task_action_window(log_date):
        return "window_closed"
    try:
        start = _parse_clock(started_at)
        end = _parse_clock(ended_at)
    except ValueError:
        return "invalid_time"
    if end and not start:
        return "missing_start"
    if start and end:
        fmt = "%Y-%m-%d %H:%M:%S"
        if datetime.strptime(end, fmt) < datetime.strptime(start, fmt):
            return "invalid_range"
    conn = _connect()
    resp = _responsibility_owner(conn, responsibility_id)
    if not resp:
        conn.close()
        return "not_found"
    row = _ensure_task_log_row(conn, responsibility_id, int(resp["employee_id"]), log_date)
    mins = _duration_minutes(start, end) if start and end else 0
    active = mins * 60
    conn.execute(
        """UPDATE task_logs SET started_at=?, ended_at=?, paused_at='',
               active_seconds=?, paused_seconds=0, duration_minutes=?
           WHERE responsibility_id=? AND log_date=?""",
        (start, end, active, mins, responsibility_id, log_date),
    )
    # Manual edit: replace event history with a clean start/(end) pair
    conn.execute("DELETE FROM task_timer_events WHERE task_log_id=?", (int(row["id"]),))
    if start:
        _append_timer_event(
            conn,
            task_log_id=int(row["id"]),
            responsibility_id=responsibility_id,
            employee_id=int(resp["employee_id"]),
            log_date=log_date,
            event_type="start",
            event_at=start,
            actor=actor or "manual",
        )
    if end:
        _append_timer_event(
            conn,
            task_log_id=int(row["id"]),
            responsibility_id=responsibility_id,
            employee_id=int(resp["employee_id"]),
            log_date=log_date,
            event_type="end",
            event_at=end,
            actor=actor or "manual",
        )
    conn.commit()
    conn.close()
    return True


def get_responsibility_timer_detail(responsibility_id: int, log_date: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (responsibility_id, log_date),
    ).fetchone()
    if not row:
        conn.close()
        return None
    row = dict(row)
    events = _list_timer_events(conn, int(row["id"]))
    conn.close()
    return {"task_log_id": row["id"], **_timer_payload(row, events=events)}


def list_timer_events_for_log(task_log_id: int) -> list[dict]:
    conn = _connect()
    ev = _list_timer_events(conn, task_log_id)
    conn.close()
    return ev


def approve_task_log(
    task_log_id: int,
    *,
    actor: str = "",
    linked_employee_id: int | None = None,
    action: str = "Approved",
    notes: str = "",
    allow_override: bool = False,
) -> str | bool:
    """Linked-person Approve/Cancel a Done/Partial task. action: Approved | Cancelled."""
    if action not in ("Approved", "Cancelled"):
        return "invalid_action"
    conn = _connect()
    row = conn.execute(
        """
        SELECT tl.*, r.linked_to_employee_id, r.title
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        WHERE tl.id=?
        """,
        (task_log_id,),
    ).fetchone()
    if not row:
        conn.close()
        return False
    linked = row["linked_to_employee_id"] if "linked_to_employee_id" in row.keys() else None
    assignee_id = int(row["employee_id"])
    # Assigned employee cannot approve/cancel their own responsibility mark
    if (
        linked_employee_id is not None
        and int(linked_employee_id) == assignee_id
        and not allow_override
    ):
        conn.close()
        return "self_forbidden"
    if linked and linked_employee_id is not None and int(linked) != int(linked_employee_id) and not allow_override:
        conn.close()
        return "forbidden"
    if (row["status"] or "") not in ("Done", "Partial"):
        conn.close()
        return "not_pending"
    appr = (row["approval_status"] if "approval_status" in row.keys() else "") or ""
    if appr not in ("Pending",):
        conn.close()
        return "not_pending"
    if not allow_override and not in_task_action_window(row["log_date"]):
        # Outside window — auto-process will handle; still allow HOD override
        conn.close()
        return "window_closed"

    new_status = row["status"]
    if action == "Cancelled":
        new_status = "Missed"
        approval = "Cancelled"
    else:
        approval = "Approved"

    conn.execute(
        """UPDATE task_logs SET status=?, approval_status=?, approved_by=?, approved_at=?, remarks=?
           WHERE id=?""",
        (
            new_status,
            approval,
            actor,
            _now_iso(),
            (notes or row["remarks"] or "") if notes else (row["remarks"] or ""),
            task_log_id,
        ),
    )
    write_task_audit(
        "task_log",
        task_log_id,
        f"linked_{action.lower()}",
        old_value=appr,
        new_value=approval,
        actor=actor,
        notes=notes,
        conn=conn,
    )
    # Clear inbox notifications for this approval
    try:
        linked_clear = int(row["linked_to_employee_id"]) if row["linked_to_employee_id"] else None
    except (TypeError, ValueError, KeyError):
        linked_clear = None
    if linked_clear:
        conn.execute(
            """UPDATE task_approval_notifications SET is_read=1
               WHERE task_log_id=? AND linked_to_employee_id=?""",
            (task_log_id, linked_clear),
        )
    if action == "Cancelled":
        assignee_id = int(row["employee_id"])
        title = str(row["title"] if "title" in row.keys() and row["title"] else "Task")
        msg = (
            f"Your mark on '{title}' ({row['log_date']}) was rejected by {actor or 'linked approver'}. "
            f"Please rework and update the task."
        )
        conn.execute(
            """INSERT INTO task_approval_notifications(
                task_log_id, responsibility_id, linked_to_employee_id, assignee_employee_id,
                log_date, title, assignee_name, message, is_read, created_at
            ) VALUES (?,?,?,?,?,?,?,?,0,?)""",
            (
                task_log_id,
                int(row["responsibility_id"]),
                assignee_id,
                assignee_id,
                str(row["log_date"]),
                title,
                "",
                msg,
                _now_iso(),
            ),
        )
    conn.commit()
    conn.close()
    return True


def reassign_mandatory_for_day(
    *,
    original_responsibility_id: int,
    to_employee_id: int,
    reassignment_date: str,
    assigned_by: str = "",
) -> int:
    """One-day clone only — original responsibility is NOT permanently transferred."""
    conn = _connect()
    orig = conn.execute(
        """SELECT id, employee_id, title, mandatory FROM responsibilities WHERE id=? AND active=1""",
        (original_responsibility_id,),
    ).fetchone()
    if not orig:
        conn.close()
        raise ValueError("Responsibility not found")
    if int(orig["employee_id"]) == int(to_employee_id):
        conn.close()
        raise ValueError("Cannot reassign to the same employee")
    day = reassignment_date[:10]
    title = f"{orig['title']} (HOD reassignment)"
    existing = conn.execute(
        """SELECT id FROM day_reassignment_clones
           WHERE original_responsibility_id=? AND reassignment_date=?""",
        (original_responsibility_id, day),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE day_reassignment_clones
               SET assignee_employee_id=?, assigned_by=?, title=?, status='Pending'
               WHERE id=?""",
            (to_employee_id, assigned_by, title, existing["id"]),
        )
        cid = int(existing["id"])
    else:
        conn.execute(
            """INSERT INTO day_reassignment_clones(
                original_responsibility_id, original_employee_id, assignee_employee_id,
                reassignment_date, title, status, assigned_by
            ) VALUES(?,?,?,?,?,'Pending',?)""",
            (
                original_responsibility_id,
                orig["employee_id"],
                to_employee_id,
                day,
                title,
                assigned_by,
            ),
        )
        cid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    write_task_audit(
        "reassignment",
        cid,
        "created",
        old_value=str(orig["employee_id"]),
        new_value=str(to_employee_id),
        actor=assigned_by,
        notes=f"day={day} resp={original_responsibility_id}",
        conn=conn,
    )
    conn.commit()
    conn.close()
    return cid


def reassign_mandatory_for_range(
    *,
    original_responsibility_id: int,
    to_employee_id: int,
    date_from: str,
    date_to: str,
    assigned_by: str = "",
) -> list[int]:
    """One-day clones for each date in [date_from, date_to] inclusive."""
    d0 = date.fromisoformat(str(date_from)[:10])
    d1 = date.fromisoformat(str(date_to)[:10])
    if d1 < d0:
        raise ValueError("date_to must be on or after date_from")
    if (d1 - d0).days > 366:
        raise ValueError("Reassignment range cannot exceed 366 days")
    clone_ids: list[int] = []
    cur = d0
    while cur <= d1:
        clone_ids.append(
            reassign_mandatory_for_day(
                original_responsibility_id=original_responsibility_id,
                to_employee_id=to_employee_id,
                reassignment_date=cur.isoformat(),
                assigned_by=assigned_by,
            )
        )
        cur += timedelta(days=1)
    return clone_ids


def mark_reassignment_clone(
    clone_id: int,
    status: str,
    marked_by: str = "",
    remarks: str = "",
) -> str | bool:
    if status not in TASK_LOG_STATUSES:
        return "invalid_status"
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM day_reassignment_clones WHERE id=?", (clone_id,)
    ).fetchone()
    if not row:
        conn.close()
        return False
    if not in_task_action_window(row["reassignment_date"]):
        conn.close()
        return "window_closed"
    conn.execute(
        """UPDATE day_reassignment_clones
           SET status=?, remarks=?, marked_by=?, marked_at=?
           WHERE id=?""",
        (status, remarks, marked_by, _now_iso(), clone_id),
    )
    # Also mirror onto original responsibility log for the day if empty
    log_exists = conn.execute(
        """SELECT id FROM task_logs WHERE responsibility_id=? AND log_date=?""",
        (row["original_responsibility_id"], row["reassignment_date"]),
    ).fetchone()
    if not log_exists and status in ("Done", "Partial", "Missed", "Leave", "N/A"):
        conn.execute(
            """INSERT INTO task_logs(
                responsibility_id, employee_id, log_date, status, remarks, marked_by, marked_at,
                approval_status, approved_by, approved_at, is_reassignment,
                reassigned_from_employee_id, reassignment_clone_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?)""",
            (
                row["original_responsibility_id"],
                row["original_employee_id"],
                row["reassignment_date"],
                status,
                remarks or f"Covered by reassignment assignee (clone {clone_id})",
                marked_by,
                _now_iso(),
                "Self" if status in ("Done", "Partial") else "",
                marked_by if status in ("Done", "Partial") else "",
                _now_iso() if status in ("Done", "Partial") else "",
                row["original_employee_id"],
                clone_id,
            ),
        )
    conn.commit()
    conn.close()
    return True


def process_auto_closures_ist(*, as_of: date | None = None, actor: str = "system-auto") -> dict:
    """Miss unmarked after task day + 2 IST days; auto-approve Pending linked after window."""
    as_of = as_of or today_ist()
    cutoff_task_date = (as_of - timedelta(days=TASK_WINDOW_EXTRA_DAYS + 0)).isoformat()
    # Window is task_date .. task_date+2 inclusive → after as_of > task_date+2
    # i.e. task_date < as_of - 2 days → task_date <= as_of - 3 days? Let's check:
    # Window closed when NOT (d0 <= as_of <= d0+2) for as_of > d0+2, i.e. d0 <= as_of-3
    # Actually: last day of window is d0+2. On as_of = d0+3, window closed.
    # So process when as_of > d0 + 2 → d0 <= as_of - 3... wait
    # d0 <= as_of, closed when as_of > d0+2 i.e. d0 < as_of - 2 → d0 <= as_of - 3 for integer dates
    # Example: task 01-Aug. Window: 1,2,3. On 4-Aug closed. as_of=4, d0 <= 4-3=1 → d0<=1. Yes.
    limit_date = (as_of - timedelta(days=TASK_WINDOW_EXTRA_DAYS + 1)).isoformat()

    conn = _connect()
    missed_n = 0
    approved_n = 0

    # Auto-approve Pending Done/Partial
    pending = conn.execute(
        """
        SELECT tl.id, tl.log_date, tl.status, tl.approval_status
        FROM task_logs tl
        WHERE tl.approval_status='Pending'
          AND tl.status IN ('Done', 'Partial')
          AND tl.log_date <= ?
        """,
        (limit_date,),
    ).fetchall()
    for p in pending:
        if in_task_action_window(p["log_date"], as_of=as_of):
            continue
        conn.execute(
            """UPDATE task_logs SET approval_status=?, approved_by=?, approved_at=? WHERE id=?""",
            ("Auto-Approved", actor, _now_iso(), p["id"]),
        )
        write_task_audit(
            "task_log",
            p["id"],
            "auto_approved",
            old_value="Pending",
            new_value="Auto-Approved",
            actor=actor,
            notes=f"linked approval timeout (window ended, log_date={p['log_date']})",
            conn=conn,
        )
        approved_n += 1

    # Auto-Missed: due scheduled responsibilities with no log after window
    # Only scan last 14 days of candidate due dates that already expired
    scan_from = (as_of - timedelta(days=14)).isoformat()
    resps = conn.execute(
        """
        SELECT r.id, r.employee_id, r.frequency, r.schedule_weekday, r.schedule_month_day,
               COALESCE(r.schedule_month, 0) as schedule_month
        FROM responsibilities r
        WHERE r.active=1
        """
    ).fetchall()
    cur = date.fromisoformat(scan_from)
    end = date.fromisoformat(limit_date)
    while cur <= end:
        dstr = cur.isoformat()
        if in_task_action_window(dstr, as_of=as_of):
            cur += timedelta(days=1)
            continue
        for r in resps:
            if not is_schedule_due(
                r["frequency"] or "Daily",
                dstr,
                r["schedule_weekday"] or "",
                int(r["schedule_month_day"] or 0),
                int(r["schedule_month"] or 0),
            ):
                continue
            # Skip Whenever Required for auto-missed
            if (r["frequency"] or "") == "Whenever Required":
                continue
            exists = conn.execute(
                "SELECT id FROM task_logs WHERE responsibility_id=? AND log_date=?",
                (r["id"], dstr),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                """INSERT INTO task_logs(
                    responsibility_id, employee_id, log_date, status, remarks, marked_by, marked_at, approval_status)
                   VALUES(?,?,?,'Missed',?,?,?,'')""",
                (
                    r["id"],
                    r["employee_id"],
                    dstr,
                    "Auto-marked: not updated within 2 days after task date (IST)",
                    actor,
                    _now_iso(),
                ),
            )
            write_task_audit(
                "task_log",
                None,
                "auto_missed",
                old_value="Pending",
                new_value="Missed",
                actor=actor,
                notes=f"resp={r['id']} date={dstr}",
                conn=conn,
            )
            missed_n += 1
        cur += timedelta(days=1)

    # Reassignment clones
    clones = conn.execute(
        """SELECT id, reassignment_date FROM day_reassignment_clones
           WHERE status='Pending' AND reassignment_date <= ?""",
        (limit_date,),
    ).fetchall()
    for c in clones:
        if not in_task_action_window(c["reassignment_date"], as_of=as_of):
            conn.execute(
                "UPDATE day_reassignment_clones SET status='Missed', marked_by=?, marked_at=? WHERE id=?",
                (actor, _now_iso(), c["id"]),
            )
            missed_n += 1

    conn.commit()
    conn.close()
    return {"missed": missed_n, "auto_approved": approved_n, "as_of": as_of.isoformat()}

def get_task_logs(department_id=None, employee_id=None, from_date=None, to_date=None):
    conn = _connect()
    today = date.today().isoformat()
    fd = from_date or today
    td = to_date or today
    conditions = ["r.active=1", "tl.log_date BETWEEN ? AND ?"]
    params = [fd, td]
    if department_id:
        conditions.append("r.department_id=?")
        params.append(department_id)
    if employee_id:
        conditions.append("r.employee_id=?")
        params.append(employee_id)
    where = "WHERE " + " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT tl.*, r.title, r.frequency, r.category,
               e.name as employee_name, d.name as department_name,
               be.name as blocker_name
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        JOIN employees e ON e.id=tl.employee_id
        LEFT JOIN departments d ON d.id=r.department_id
        LEFT JOIN employees be ON be.id=tl.blocker_employee_id
        {where}
        ORDER BY tl.log_date DESC, e.name
    """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_issue_status(status: str | None) -> str:
    s = (status or "Open").strip()
    s = ISSUE_STATUS_ALIASES.get(s, s)
    if s not in ISSUE_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Allowed: {', '.join(sorted(ISSUE_STATUSES))}")
    return s


def _append_issue_history(
    conn: sqlite3.Connection,
    issue_id: int,
    *,
    action: str,
    field_name: str = "",
    previous_value: str = "",
    new_value: str = "",
    user_id: int | None = None,
    user_name: str = "",
) -> None:
    conn.execute(
        """INSERT INTO issue_history(
            issue_id, action, field_name, previous_value, new_value, user_id, user_name, created_at
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            issue_id,
            action,
            field_name or "",
            previous_value if previous_value is not None else "",
            new_value if new_value is not None else "",
            user_id,
            user_name or "",
            _now(),
        ),
    )


def _notify_issue(
    conn: sqlite3.Connection,
    issue_id: int | None,
    event_type: str,
    message: str,
    recipient_user_id: int | None = None,
) -> None:
    conn.execute(
        """INSERT INTO issue_notifications(issue_id, event_type, recipient_user_id, message, channel, is_read, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (issue_id, event_type, recipient_user_id, message, "in_app", 0, _now()),
    )


def list_issues(
    employee_id=None,
    department_id=None,
    from_date=None,
    to_date=None,
    *,
    status: str | None = None,
    caused_by_employee_id: int | None = None,
    caused_by_user_id: int | None = None,
    recorded_by_user_id: int | None = None,
    recorded_by: str | None = None,
    subject_user_id: int | None = None,
    designation: str | None = None,
    q: str | None = None,
):
    conn = _connect()
    conditions = []
    params: list = []
    if employee_id:
        conditions.append("il.employee_id=?")
        params.append(employee_id)
    if department_id:
        conditions.append("il.department_id=?")
        params.append(department_id)
    if from_date:
        conditions.append("il.issue_date >= ?")
        params.append(from_date)
    if to_date:
        conditions.append("il.issue_date <= ?")
        params.append(to_date)
    if status:
        st = _normalize_issue_status(status) if status not in ("", None) else None
        if st:
            conditions.append("il.status=?")
            params.append(st)
    if caused_by_employee_id:
        conditions.append("il.caused_by_employee_id=?")
        params.append(caused_by_employee_id)
    if caused_by_user_id:
        conditions.append("il.caused_by_user_id=?")
        params.append(caused_by_user_id)
    if recorded_by_user_id:
        conditions.append("il.recorded_by_user_id=?")
        params.append(recorded_by_user_id)
    if recorded_by:
        conditions.append("LOWER(il.recorded_by) LIKE ?")
        params.append(f"%{recorded_by.strip().lower()}%")
    if subject_user_id:
        conditions.append("il.subject_user_id=?")
        params.append(subject_user_id)
    if designation:
        conditions.append(
            "(LOWER(COALESCE(il.designation,'')) LIKE ? OR LOWER(COALESCE(e.designation,'')) LIKE ?)"
        )
        d = f"%{designation.strip().lower()}%"
        params.extend([d, d])
    if q and str(q).strip():
        like = f"%{str(q).strip().lower()}%"
        conditions.append(
            """(
            LOWER(e.name) LIKE ? OR LOWER(il.title) LIKE ? OR LOWER(COALESCE(il.description,'')) LIKE ?
            OR LOWER(COALESCE(il.recorded_by,'')) LIKE ? OR LOWER(COALESCE(il.subject_user_name,'')) LIKE ?
            OR LOWER(COALESCE(il.caused_by_user_name,'')) LIKE ? OR LOWER(COALESCE(e.emp_code,'')) LIKE ?
            OR LOWER(COALESCE(ce.name,'')) LIKE ?
            )"""
        )
        params.extend([like] * 8)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(
        f"""
        SELECT il.*,
               e.name as employee_name, e.emp_code as employee_code,
               e.designation as employee_designation, e.email as employee_email,
               e.phone as employee_phone,
               d.name as department_name,
               ce.name as caused_by_name, cd.name as caused_by_dept_name
        FROM issue_logs il
        JOIN employees e ON e.id=il.employee_id
        LEFT JOIN departments d ON d.id=il.department_id
        LEFT JOIN employees ce ON ce.id=il.caused_by_employee_id
        LEFT JOIN departments cd ON cd.id=il.caused_by_dept_id
        {where}
        ORDER BY COALESCE(NULLIF(il.updated_at,''), il.created_at) DESC, il.id DESC
    """,
        params,
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        # Display helpers: prefer linked ERP user names when set
        d["display_employee"] = (d.get("subject_user_name") or d.get("employee_name") or "").strip()
        d["display_caused_by"] = (d.get("caused_by_user_name") or d.get("caused_by_name") or "").strip()
        d["display_recorded_by"] = (d.get("recorded_by") or "").strip()
        emp_disp = d["display_employee"].lower()
        rec_disp = d["display_recorded_by"].lower()
        d["show_recorded_by"] = bool(rec_disp and emp_disp and rec_disp != emp_disp)
        # Normalize legacy status for clients
        if d.get("status") == "Resolved":
            d["status"] = "Resolve"
        out.append(d)
    return out


def get_issue(issue_id: int) -> dict | None:
    rows = list_issues()
    for r in rows:
        if int(r["id"]) == int(issue_id):
            return r
    conn = _connect()
    row = conn.execute(
        """SELECT il.*, e.name as employee_name, d.name as department_name,
                  ce.name as caused_by_name
           FROM issue_logs il
           JOIN employees e ON e.id=il.employee_id
           LEFT JOIN departments d ON d.id=il.department_id
           LEFT JOIN employees ce ON ce.id=il.caused_by_employee_id
           WHERE il.id=?""",
        (issue_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_issue(data: dict) -> int:
    conn = _connect()
    dept_id = data.get("department_id")
    if not dept_id and data.get("employee_id"):
        row = conn.execute(
            "SELECT department_id FROM employees WHERE id=?", (data["employee_id"],)
        ).fetchone()
        if row:
            dept_id = row["department_id"]
    title = (data.get("title") or "").strip()
    if not title:
        conn.close()
        raise ValueError("title is required")
    if not data.get("employee_id"):
        conn.close()
        raise ValueError("employee is required")
    status = _normalize_issue_status(data.get("status") or "Open")
    # recorded_by must come from auth layer — store what is provided but never allow override later
    recorded_by = (data.get("recorded_by") or "").strip()
    recorded_by_user_id = data.get("recorded_by_user_id")
    now = _now()
    emp = conn.execute(
        "SELECT name, designation FROM employees WHERE id=?", (data["employee_id"],)
    ).fetchone()
    designation = (data.get("designation") or (emp["designation"] if emp else "") or "").strip()
    cur = conn.execute(
        """INSERT INTO issue_logs(
        employee_id, department_id, issue_date, issue_type, severity,
        title, description, recorded_by, recorded_by_user_id,
        subject_user_id, subject_user_name, caused_by_user_id, caused_by_user_name,
        caused_by_employee_id, caused_by_dept_id, status, designation, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data["employee_id"],
            dept_id,
            data.get("issue_date") or date.today().isoformat(),
            data.get("issue_type", "General"),
            data.get("severity", "Minor"),
            title,
            data.get("description", "") or "",
            recorded_by,
            recorded_by_user_id,
            data.get("subject_user_id"),
            (data.get("subject_user_name") or "").strip(),
            data.get("caused_by_user_id"),
            (data.get("caused_by_user_name") or "").strip(),
            data.get("caused_by_employee_id"),
            data.get("caused_by_dept_id"),
            status,
            designation,
            now,
        ),
    )
    issue_id = int(cur.lastrowid)
    _append_issue_history(
        conn,
        issue_id,
        action="Issue Created",
        field_name="status",
        new_value=status,
        user_id=recorded_by_user_id,
        user_name=recorded_by,
    )
    _notify_issue(
        conn,
        issue_id,
        "created",
        f"New issue recorded: {title}",
        recipient_user_id=data.get("subject_user_id"),
    )
    conn.commit()
    conn.close()
    return issue_id


def get_issue_employee_id(issue_id: int) -> int | None:
    conn = _connect()
    row = conn.execute("SELECT employee_id FROM issue_logs WHERE id=?", (issue_id,)).fetchone()
    conn.close()
    return int(row["employee_id"]) if row else None


def get_issue_raw(issue_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM issue_logs WHERE id=?", (issue_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def resolve_issue(issue_id: int, resolution: str, *, user_id: int | None = None, user_name: str = ""):
    """Backward-compatible resolve → status Resolve."""
    return update_issue_status(
        issue_id,
        "Resolve",
        resolution=resolution,
        user_id=user_id,
        user_name=user_name,
    )


def update_issue_status(
    issue_id: int,
    status: str,
    *,
    resolution: str = "",
    user_id: int | None = None,
    user_name: str = "",
) -> dict:
    status = _normalize_issue_status(status)
    conn = _connect()
    row = conn.execute("SELECT * FROM issue_logs WHERE id=?", (issue_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Issue not found")
    old = dict(row)
    old_status = old.get("status") or ""
    if old_status == "Resolved":
        old_status = "Resolve"
    sets = ["status=?", "updated_at=?"]
    params: list = [status, _now()]
    if resolution:
        sets.append("resolution=?")
        params.append(resolution)
    params.append(issue_id)
    conn.execute(f"UPDATE issue_logs SET {', '.join(sets)} WHERE id=?", params)
    _append_issue_history(
        conn,
        issue_id,
        action="Status Changed",
        field_name="status",
        previous_value=old_status,
        new_value=status,
        user_id=user_id,
        user_name=user_name,
    )
    _notify_issue(
        conn,
        issue_id,
        f"status_{status.lower()}",
        f"Issue #{issue_id} status → {status}",
        recipient_user_id=old.get("subject_user_id") or old.get("recorded_by_user_id"),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "id": issue_id, "status": status}


def update_issue(
    issue_id: int,
    data: dict,
    *,
    user_id: int | None = None,
    user_name: str = "",
) -> dict:
    """Edit mutable fields only — never recorded_by / created_at / id."""
    # recorded_by, created_at, id must never be editable
    forbidden = {"recorded_by", "recorded_by_user_id", "created_at", "id"}
    data = {k: v for k, v in (data or {}).items() if k not in forbidden and v is not None}

    conn = _connect()
    row = conn.execute("SELECT * FROM issue_logs WHERE id=?", (issue_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Issue not found")
    old = dict(row)

    field_map = {
        "title": "title",
        "description": "description",
        "employee_id": "employee_id",
        "department_id": "department_id",
        "issue_type": "issue_type",
        "severity": "severity",
        "caused_by_employee_id": "caused_by_employee_id",
        "caused_by_dept_id": "caused_by_dept_id",
        "caused_by_user_id": "caused_by_user_id",
        "caused_by_user_name": "caused_by_user_name",
        "subject_user_id": "subject_user_id",
        "subject_user_name": "subject_user_name",
        "issue_date": "issue_date",
        "designation": "designation",
        "resolution": "resolution",
    }
    updates = []
    params: list = []
    changes: list[tuple[str, str, str, str]] = []  # action, field, old, new

    if "status" in data and data["status"] is not None:
        new_st = _normalize_issue_status(data["status"])
        old_st = ISSUE_STATUS_ALIASES.get(old.get("status") or "", old.get("status") or "")
        if new_st != old_st:
            updates.append("status=?")
            params.append(new_st)
            changes.append(("Status Changed", "status", old_st, new_st))

    for src, col in field_map.items():
        if src not in data or data[src] is None:
            continue
        new_val = data[src]
        old_val = old.get(col)
        if str(new_val if new_val is not None else "") == str(old_val if old_val is not None else ""):
            continue
        updates.append(f"{col}=?")
        params.append(new_val)
        action = "Issue Updated"
        if col == "employee_id":
            action = "Employee Changed"
        elif col in ("caused_by_employee_id", "caused_by_user_id", "caused_by_user_name"):
            action = "Caused By Changed"
        changes.append((action, col, str(old_val if old_val is not None else ""), str(new_val if new_val is not None else "")))

    if "employee_id" in data and data["employee_id"] and not data.get("department_id"):
        er = conn.execute(
            "SELECT department_id FROM employees WHERE id=?", (data["employee_id"],)
        ).fetchone()
        if er and er["department_id"] != old.get("department_id"):
            updates.append("department_id=?")
            params.append(er["department_id"])
            changes.append(
                (
                    "Issue Updated",
                    "department_id",
                    str(old.get("department_id") or ""),
                    str(er["department_id"] or ""),
                )
            )

    if not updates:
        conn.close()
        return {"ok": True, "id": issue_id, "changed": False}

    updates.append("updated_at=?")
    params.append(_now())
    params.append(issue_id)
    conn.execute(f"UPDATE issue_logs SET {', '.join(updates)} WHERE id=?", params)
    for action, field, prev, new in changes:
        _append_issue_history(
            conn,
            issue_id,
            action=action,
            field_name=field,
            previous_value=prev,
            new_value=new,
            user_id=user_id,
            user_name=user_name,
        )
    _notify_issue(
        conn,
        issue_id,
        "updated",
        f"Issue #{issue_id} updated",
        recipient_user_id=old.get("subject_user_id"),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "id": issue_id, "changed": True, "changes": len(changes)}


def add_issue_comment(
    issue_id: int,
    comment_text: str,
    *,
    user_id: int | None = None,
    user_name: str = "",
) -> int:
    text = (comment_text or "").strip()
    if not text:
        raise ValueError("comment is required")
    conn = _connect()
    if not conn.execute("SELECT id FROM issue_logs WHERE id=?", (issue_id,)).fetchone():
        conn.close()
        raise ValueError("Issue not found")
    cur = conn.execute(
        """INSERT INTO issue_comments(issue_id, comment_text, user_id, user_name, created_at)
           VALUES (?,?,?,?,?)""",
        (issue_id, text, user_id, user_name, _now()),
    )
    cid = int(cur.lastrowid)
    _append_issue_history(
        conn,
        issue_id,
        action="Comments Added",
        field_name="comment",
        new_value=text[:500],
        user_id=user_id,
        user_name=user_name,
    )
    _notify_issue(conn, issue_id, "comment", f"Comment on issue #{issue_id}", recipient_user_id=None)
    conn.execute("UPDATE issue_logs SET updated_at=? WHERE id=?", (_now(), issue_id))
    conn.commit()
    conn.close()
    return cid


def list_issue_comments(issue_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM issue_comments WHERE issue_id=? ORDER BY id ASC", (issue_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_issue_attachment(
    issue_id: int,
    data: dict,
    *,
    user_id: int | None = None,
    user_name: str = "",
) -> int:
    file_name = (data.get("file_name") or "").strip()
    if not file_name:
        raise ValueError("file_name is required")
    conn = _connect()
    if not conn.execute("SELECT id FROM issue_logs WHERE id=?", (issue_id,)).fetchone():
        conn.close()
        raise ValueError("Issue not found")
    cur = conn.execute(
        """INSERT INTO issue_attachments(
            issue_id, file_name, file_url, content_type, file_size,
            uploaded_by, uploaded_by_user_id, created_at
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            issue_id,
            file_name,
            data.get("file_url") or "",
            data.get("content_type") or "",
            int(data.get("file_size") or 0),
            user_name,
            user_id,
            _now(),
        ),
    )
    aid = int(cur.lastrowid)
    _append_issue_history(
        conn,
        issue_id,
        action="Attachments Added",
        field_name="attachment",
        new_value=file_name,
        user_id=user_id,
        user_name=user_name,
    )
    conn.execute("UPDATE issue_logs SET updated_at=? WHERE id=?", (_now(), issue_id))
    conn.commit()
    conn.close()
    return aid


def list_issue_attachments(issue_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM issue_attachments WHERE issue_id=? ORDER BY id DESC", (issue_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_issue_history(issue_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM issue_history WHERE issue_id=? ORDER BY id ASC", (issue_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_voice_transcription(data: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO issue_voice_transcriptions(
            issue_id, target_field, transcript, status, error_message, user_id, user_name, created_at
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            data.get("issue_id"),
            data.get("target_field") or "description",
            data.get("transcript") or "",
            data.get("status") or "success",
            data.get("error_message") or "",
            data.get("user_id"),
            data.get("user_name") or "",
            _now(),
        ),
    )
    tid = int(cur.lastrowid)
    if data.get("issue_id") and (data.get("status") or "success") == "success":
        _append_issue_history(
            conn,
            int(data["issue_id"]),
            action="Voice Transcription Created",
            field_name=data.get("target_field") or "description",
            new_value=(data.get("transcript") or "")[:500],
            user_id=data.get("user_id"),
            user_name=data.get("user_name") or "",
        )
    conn.commit()
    conn.close()
    return tid


def list_issue_notifications(*, limit: int = 50, unread_only: bool = False) -> list[dict]:
    conn = _connect()
    q = "SELECT * FROM issue_notifications WHERE 1=1"
    if unread_only:
        q += " AND is_read=0"
    q += " ORDER BY id DESC LIMIT ?"
    rows = conn.execute(q, (int(limit),)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_issue(issue_id: int, *, user_id: int | None = None, user_name: str = "") -> bool:
    """Soft-cancel via Cancel status rather than hard delete (audit preserved)."""
    update_issue_status(issue_id, "Cancel", user_id=user_id, user_name=user_name)
    return True


def get_hod_dashboard(
    department_id: int,
    from_date: str = None,
    to_date: str = None,
    employee_id: int | None = None,
):
    today = date.today().isoformat()
    fd = from_date or today
    td = to_date or today
    start = date.fromisoformat(fd)
    end = date.fromisoformat(td)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.isoformat())
        cur += timedelta(days=1)

    conn = _connect()
    resp_sql = """
        SELECT r.*, e.name as employee_name,
               le.name as linked_to_employee_name
        FROM responsibilities r
        JOIN employees e ON e.id=r.employee_id
        LEFT JOIN employees le ON le.id=r.linked_to_employee_id
        WHERE r.department_id=? AND r.active=1
    """
    resp_params: list = [department_id]
    if employee_id:
        resp_sql += " AND r.employee_id=?"
        resp_params.append(employee_id)
    resp_sql += " ORDER BY e.name, r.frequency, r.title"
    resps = conn.execute(resp_sql, resp_params).fetchall()

    logs = conn.execute(
        """
        SELECT tl.responsibility_id, tl.log_date, tl.status, tl.remarks,
               tl.marked_by, tl.marked_at, tl.blocker_employee_id, tl.blocker_reason,
               be.name as blocker_name,
               COALESCE(tl.started_at,'') as started_at,
               COALESCE(tl.ended_at,'') as ended_at,
               COALESCE(tl.duration_minutes,0) as duration_minutes
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        LEFT JOIN employees be ON be.id=tl.blocker_employee_id
        WHERE r.department_id=? AND tl.log_date BETWEEN ? AND ?
    """,
        (department_id, fd, td),
    ).fetchall()
    conn.close()

    log_map = {}
    for l in logs:
        key = (l["responsibility_id"], l["log_date"])
        mat = l["marked_at"] if "marked_at" in l.keys() else ""
        ld = dict(l)
        status = l["status"]
        quality_marked = str(status or "Pending").strip() not in ("Pending", "")
        log_map[key] = {
            "status": l["status"],
            "remarks": l["remarks"],
            "marked_by": l["marked_by"],
            "marked_at": mat or "",
            "blocker_name": l["blocker_name"] or "",
            "blocker_reason": l["blocker_reason"] or "",
            "marked": quality_marked,
            "editable": hod_status_editable(mat),
            **_timer_payload(ld),
        }

    result = []
    for r in resps:
        rd = dict(r)
        rd["dates"] = {}
        for d in dates:
            # Hide weekly/monthly etc. when not scheduled for that calendar day
            if not is_schedule_due(
                rd.get("frequency") or "Daily",
                d,
                rd.get("schedule_weekday") or "",
                int(rd.get("schedule_month_day") or 0),
                int(rd.get("schedule_month") or 0),
            ):
                continue
            key = (r["id"], d)
            rd["dates"][d] = log_map.get(
                key,
                {
                    "status": "Pending",
                    "remarks": "",
                    "marked_by": "",
                    "marked_at": "",
                    "blocker_name": "",
                    "blocker_reason": "",
                    "marked": False,
                    "editable": True,
                    "started_at": "",
                    "ended_at": "",
                    "duration_minutes": 0,
                    "timer_status": "Not Started",
                },
            )
        if not rd["dates"] and dates:
            # Keep row but empty dates if none due in range — skip empty optional
            pass
        result.append(rd)

    return {"responsibilities": result, "dates": dates}


def get_appraisal(employee_id: int, from_date: str = None, to_date: str = None):
    today = date.today().isoformat()
    fd = from_date or date.today().replace(month=1, day=1).isoformat()
    td = to_date or today

    conn = _connect()
    emp = conn.execute(
        """
        SELECT e.*, d.name as department_name
        FROM employees e LEFT JOIN departments d ON d.id=e.department_id
        WHERE e.id=?""",
        (employee_id,),
    ).fetchone()
    if not emp:
        conn.close()
        return None
    emp = dict(emp)

    task_logs = conn.execute(
        """
        SELECT tl.*, r.title, r.frequency, r.linked_to_employee_id
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        WHERE tl.employee_id=? AND tl.log_date BETWEEN ? AND ?
        ORDER BY tl.log_date DESC
    """,
        (employee_id, fd, td),
    ).fetchall()

    issues = conn.execute(
        """
        SELECT il.*, ce.name as caused_by_name
        FROM issue_logs il
        LEFT JOIN employees ce ON ce.id=il.caused_by_employee_id
        WHERE il.employee_id=? AND il.issue_date BETWEEN ? AND ?
        ORDER BY il.issue_date DESC
    """,
        (employee_id, fd, td),
    ).fetchall()

    blockers_caused = conn.execute(
        """
        SELECT tl.log_date, tl.blocker_reason, r.title as task_title,
               e.name as affected_employee
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        JOIN employees e ON e.id=tl.employee_id
        WHERE tl.blocker_employee_id=? AND tl.status='Blocked'
        AND tl.log_date BETWEEN ? AND ?
        ORDER BY tl.log_date DESC
    """,
        (employee_id, fd, td),
    ).fetchall()

    ot_rows = conn.execute(
        """
        SELECT * FROM one_time_tasks
        WHERE employee_id=? AND active=1
        ORDER BY created_at DESC
        """,
        (employee_id,),
    ).fetchall()

    conn.close()

    done = partial = missed = blocked = leave = na = 0
    credit_done = credit_partial = 0
    total_counted = 0
    for t in task_logs:
        st = t["status"]
        if st == "Leave":
            leave += 1
        if st == "N/A":
            na += 1
        bucket = task_log_counts_for_performance(
            st,
            approval_status=(t["approval_status"] if "approval_status" in t.keys() else "") or "",
            log_date=t["log_date"] or "",
            linked_to_employee_id=t["linked_to_employee_id"]
            if "linked_to_employee_id" in t.keys()
            else None,
        )
        if bucket is None:
            continue
        total_counted += 1
        if bucket == "done":
            done += 1
            credit_done += 1
        elif bucket == "partial":
            partial += 1
            credit_partial += 1
        elif bucket == "missed":
            missed += 1
        elif bucket == "blocked":
            blocked += 1
    # Also count legacy raw statuses for display (exclude N/A from denominator)
    for t in task_logs:
        st = t["status"]
        if st == "Done" and credit_done == 0 and total_counted == 0:
            pass  # already handled
    resp_pct = (
        round((credit_done + credit_partial * 0.5) / total_counted * 100, 1)
        if total_counted > 0
        else 0
    )

    ot_summary, ot_period = _summarize_one_time_tasks(
        [_one_time_task_row(r) for r in ot_rows], fd, td, today
    )
    combined_pct = _combined_performance_pct(resp_pct, ot_summary)

    return {
        "employee": emp,
        "period": {"from": fd, "to": td},
        "task_summary": {
            "total": total_counted,
            "done": done,
            "partial": partial,
            "missed": missed,
            "blocked": blocked,
            "leave": leave,
            "na": na,
            "responsibility_performance_pct": resp_pct,
            "one_time_performance_pct": ot_summary["performance_pct"],
            "performance_pct": combined_pct,
            "cutover_date": PERFORMANCE_CUTOVER_DATE,
        },
        "one_time_summary": ot_summary,
        "one_time_tasks": ot_period,
        "issues": [dict(i) for i in issues],
        "blockers_caused": [dict(b) for b in blockers_caused],
        "task_logs": [dict(t) for t in task_logs],
    }


def get_employee_day_check(employee_id: int, check_date: str | None = None) -> dict | None:
    """
    Snapshot of what an employee worked on vs did not for a given day.
    Unmarked daily responsibilities appear under not_worked as Pending.
    Whenever Required items are listed last under whenever_required.
    Additional Work holds one-day HOD reassignment clones.
    """
    day = check_date or today_ist().isoformat()
    conn = _connect()
    emp = conn.execute(
        """
        SELECT e.*, d.name as department_name
        FROM employees e LEFT JOIN departments d ON d.id=e.department_id
        WHERE e.id=?
        """,
        (employee_id,),
    ).fetchone()
    if not emp:
        conn.close()
        return None

    resps = conn.execute(
        """
        SELECT r.id, r.title, r.description, r.frequency, r.category,
               r.priority, r.mandatory, r.schedule_weekday, r.schedule_month_day, r.time_period,
               COALESCE(r.schedule_month, 0) as schedule_month,
               r.linked_to_employee_id, le.name as linked_to_employee_name,
               r.backup_employee_id, be.name as backup_employee_name,
               COALESCE(r.backup_allocation_value, 0) as backup_allocation_value,
               COALESCE(r.backup_allocation_unit, 'days') as backup_allocation_unit
        FROM responsibilities r
        LEFT JOIN employees le ON le.id=r.linked_to_employee_id
        LEFT JOIN employees be ON be.id=r.backup_employee_id
        WHERE r.employee_id=? AND r.active=1
        ORDER BY
          CASE r.frequency
            WHEN 'Daily' THEN 0
            WHEN 'Weekly' THEN 1
            WHEN 'Fortnightly' THEN 2
            WHEN 'Monthly' THEN 3
            WHEN 'Quarterly' THEN 4
            WHEN 'Yearly' THEN 5
            WHEN 'Whenever Required' THEN 9
            ELSE 6
          END,
          r.title
        """,
        (employee_id,),
    ).fetchall()

    logs = conn.execute(
        """
        SELECT id, responsibility_id, status, remarks, marked_by, marked_at,
               blocker_employee_id, blocker_reason,
               COALESCE(approval_status,'') as approval_status,
               COALESCE(approved_by,'') as approved_by,
               COALESCE(started_at,'') as started_at,
               COALESCE(ended_at,'') as ended_at,
               COALESCE(paused_at,'') as paused_at,
               COALESCE(duration_minutes,0) as duration_minutes,
               COALESCE(active_seconds,0) as active_seconds,
               COALESCE(paused_seconds,0) as paused_seconds
        FROM task_logs
        WHERE employee_id=? AND log_date=?
        """,
        (employee_id, day),
    ).fetchall()
    log_map = {int(l["responsibility_id"]): dict(l) for l in logs}

    # Who took over original's mandatory tasks today
    reassigned_out = conn.execute(
        """
        SELECT original_responsibility_id, assignee_employee_id, e.name as assignee_name
        FROM day_reassignment_clones c
        JOIN employees e ON e.id=c.assignee_employee_id
        WHERE c.original_employee_id=? AND c.reassignment_date=?
        """,
        (employee_id, day),
    ).fetchall()
    reassign_out_map = {
        int(r["original_responsibility_id"]): dict(r) for r in reassigned_out
    }

    # Additional work assigned to this employee for the day
    additional = conn.execute(
        """
        SELECT c.*, e.name as original_employee_name, r.frequency, r.priority
        FROM day_reassignment_clones c
        JOIN employees e ON e.id=c.original_employee_id
        JOIN responsibilities r ON r.id=c.original_responsibility_id
        WHERE c.assignee_employee_id=? AND c.reassignment_date=?
        """,
        (employee_id, day),
    ).fetchall()

    ot_rows = conn.execute(
        """
        SELECT id, title, description, due_date, status, started_at, completed_at, priority
        FROM one_time_tasks
        WHERE employee_id=? AND active=1
          AND status IN ('Pending', 'In Progress', 'Done', 'Rejected')
        ORDER BY
          CASE status
            WHEN 'In Progress' THEN 0
            WHEN 'Pending' THEN 1
            WHEN 'Done' THEN 2
            ELSE 3
          END,
          due_date
        """,
        (employee_id,),
    ).fetchall()

    # Preload timer events while connection is open (conn closed before item build below).
    timer_events_by_log: dict[int, list[dict]] = {}
    for log in log_map.values():
        lid = log.get("id")
        if lid:
            timer_events_by_log[int(lid)] = _list_timer_events(conn, int(lid))
    conn.close()

    worked_on: list[dict] = []
    not_worked: list[dict] = []
    other: list[dict] = []
    whenever_required: list[dict] = []
    skipped_schedule: list[dict] = []
    submitted_for_approval: list[dict] = []

    def _bucket(item: dict, status: str):
        if item.get("frequency") == "Whenever Required":
            whenever_required.append(item)
            return
        if status in ("Done", "Partial"):
            if item.get("approval_status") == "Pending":
                submitted_for_approval.append(item)
            worked_on.append(item)
        elif status in ("Leave", "N/A"):
            other.append(item)
        else:
            not_worked.append(item)

    for r in resps:
        rid = int(r["id"])
        rdict = dict(r)
        if not is_schedule_due(
            rdict.get("frequency") or "Daily",
            day,
            rdict.get("schedule_weekday") or "",
            int(rdict.get("schedule_month_day") or 0),
            int(rdict.get("schedule_month") or 0),
        ):
            skipped_schedule.append(
                {
                    "responsibility_id": rid,
                    "title": r["title"],
                    "frequency": r["frequency"],
                    "schedule_weekday": rdict.get("schedule_weekday") or "",
                    "schedule_month_day": rdict.get("schedule_month_day") or 0,
                    "schedule_month": rdict.get("schedule_month") or 0,
                }
            )
            continue
        log = log_map.get(rid)
        status = (log or {}).get("status") or "Pending"
        re_out = reassign_out_map.get(rid)
        quality_marked = status not in ("Pending", "")
        lid = int(log["id"]) if log and log.get("id") else None
        events = timer_events_by_log.get(lid, []) if lid else []
        item = {
            "responsibility_id": rid,
            "task_log_id": (log or {}).get("id"),
            "title": r["title"],
            "description": r["description"] or "",
            "frequency": r["frequency"],
            "category": r["category"],
            "priority": rdict.get("priority") or "Medium",
            "mandatory": bool(int(rdict.get("mandatory") or 0)),
            "time_period": rdict.get("time_period") or "",
            "linked_to_employee_id": rdict.get("linked_to_employee_id"),
            "linked_to_employee_name": rdict.get("linked_to_employee_name") or "",
            "backup_employee_id": rdict.get("backup_employee_id"),
            "backup_employee_name": rdict.get("backup_employee_name") or "",
            "backup_allocation_value": float(rdict.get("backup_allocation_value") or 0),
            "backup_allocation_unit": rdict.get("backup_allocation_unit") or "days",
            "approval_status": (log or {}).get("approval_status") or "",
            "status": status,
            "marked": quality_marked,
            "quality_marked": quality_marked,
            "remarks": (log or {}).get("remarks") or "",
            "marked_by": (log or {}).get("marked_by") or "",
            "blocker_reason": (log or {}).get("blocker_reason") or "",
            "editable": hod_status_editable((log or {}).get("marked_at")),
            "in_action_window": in_task_action_window(day),
            "reassigned_out": bool(re_out),
            "reassigned_to_name": (re_out or {}).get("assignee_name") or "",
            **_timer_payload({**(log or {}), "log_date": day}, events=events),
        }
        # Original still holds master — if reassigned for the day, don't force pending scoreboard
        if re_out and not log:
            item["status"] = "Reassigned"
            other.append(item)
            continue
        _bucket(item, status)

    additional_work = []
    for c in additional:
        cd = dict(c)
        additional_work.append(
            {
                "clone_id": cd["id"],
                "responsibility_id": cd["original_responsibility_id"],
                "title": cd.get("title") or "",
                "original_employee_name": cd.get("original_employee_name") or "",
                "assigned_by": cd.get("assigned_by") or "",
                "status": cd.get("status") or "Pending",
                "remarks": cd.get("remarks") or "",
                "frequency": cd.get("frequency") or "Daily",
                "priority": cd.get("priority") or "Medium",
                "section": "Additional Work (Assigned by HOD)",
                "in_action_window": in_task_action_window(day),
            }
        )

    one_time = [_one_time_task_row(r) for r in ot_rows]
    working_tasks = [t for t in one_time if t.get("status") == "In Progress"]
    pending_tasks = [t for t in one_time if t.get("status") in ("Pending", "Rejected")]
    awaiting_approval = [t for t in one_time if t.get("status") == "Done"]

    expected_daily = [
        i
        for i in worked_on + not_worked + other
        if i["frequency"] == "Daily" or i.get("mandatory")
    ]
    done_daily = sum(1 for i in expected_daily if i["status"] == "Done")
    partial_daily = sum(1 for i in expected_daily if i["status"] == "Partial")
    pending_daily = sum(1 for i in expected_daily if i["status"] == "Pending")
    missed_daily = sum(1 for i in expected_daily if i["status"] == "Missed")

    return {
        "employee": dict(emp),
        "check_date": day,
        "worked_on": worked_on,
        "submitted_for_approval": submitted_for_approval,
        "not_worked": not_worked,
        "other": other,
        "whenever_required": whenever_required,
        "additional_work": additional_work,
        "not_scheduled_today": skipped_schedule,
        "one_time_working": working_tasks,
        "one_time_pending": pending_tasks,
        "one_time_awaiting_approval": awaiting_approval,
        "summary": {
            "responsibilities_total": len(worked_on)
            + len(not_worked)
            + len(other)
            + len(whenever_required),
            "worked_on": len(worked_on),
            "not_worked": len(not_worked),
            "other": len(other),
            "whenever_required": len(whenever_required),
            "additional_work": len(additional_work),
            "daily_expected": len(expected_daily),
            "daily_done": done_daily,
            "daily_partial": partial_daily,
            "daily_pending": pending_daily,
            "daily_missed": missed_daily,
            "completion_pct": round(
                (done_daily + partial_daily * 0.5) / len(expected_daily) * 100, 1
            )
            if expected_daily
            else 0,
            "unmarked_daily": pending_daily,
        },
        "time_period_filter": TIME_PERIODS,
        "performance_cutover": PERFORMANCE_CUTOVER_DATE,
    }


def list_dwr_rows(
    *,
    employee_id: int | None = None,
    department_id: int | None = None,
    check_date: str | None = None,
) -> dict:
    """Flat Daily Work Report rows for Admin/HOD (one row per responsibility that day)."""
    day = check_date or today_ist().isoformat()
    emp_ids: list[int] = []
    if employee_id:
        emp_ids = [int(employee_id)]
    else:
        conn = _connect()
        q = "SELECT id FROM employees WHERE IFNULL(status,'Active') != 'Inactive'"
        params: list = []
        if department_id:
            q += " AND department_id=?"
            params.append(int(department_id))
        emp_ids = [int(r["id"]) for r in conn.execute(q, params).fetchall()]
        conn.close()

    rows: list[dict] = []
    for eid in emp_ids:
        snap = get_employee_day_check(eid, day)
        if not snap:
            continue
        emp = snap.get("employee") or {}
        for bucket in ("worked_on", "not_worked", "other", "whenever_required"):
            for item in snap.get(bucket) or []:
                linked_name = item.get("linked_to_employee_name") or ""
                rows.append(
                    {
                        "employee_id": eid,
                        "employee_name": emp.get("name") or "",
                        "department_name": emp.get("department_name") or "",
                        "check_date": day,
                        "responsibility_id": item.get("responsibility_id"),
                        "title": item.get("title"),
                        "frequency": item.get("frequency"),
                        "status": item.get("status"),
                        "timer_status": item.get("timer_status") or "Not Started",
                        "started_at": item.get("started_at") or "",
                        "ended_at": item.get("ended_at") or "",
                        "duration_minutes": int(item.get("duration_minutes") or 0),
                        "linked_to_employee_id": item.get("linked_to_employee_id"),
                        "linked_to_employee_name": linked_name,
                        "linked_person": linked_name or "Self-complete",
                    }
                )
    return {"check_date": day, "rows": rows}


def mark_unmarked_daily_as_missed(
    employee_id: int,
    check_date: str | None = None,
    *,
    marked_by: str = "system",
) -> dict:
    """Auto-close unmarked Daily responsibilities for a day as Missed."""
    day = check_date or date.today().isoformat()
    snap = get_employee_day_check(employee_id, day)
    if not snap:
        return {"ok": False, "marked": 0, "error": "Employee not found"}

    marked = 0
    for item in snap["not_worked"]:
        if item["frequency"] != "Daily":
            continue
        if item["status"] != "Pending" or item["marked"]:
            continue
        mark_task(
            item["responsibility_id"],
            day,
            "Missed",
            marked_by=marked_by,
            remarks="Auto-marked: not updated by end of day",
        )
        marked += 1
    return {"ok": True, "marked": marked, "check_date": day, "employee_id": employee_id}


def process_end_of_day_missed_ist(*, as_of: date | None = None, actor: str = "system-eod") -> dict:
    """
    At IST day rollover, mark yesterday's due unmarked tasks as Missed.
    Does not overwrite Done/Partial/Missed/Leave/N/A/Blocked or Whenever Required.
    """
    as_of = as_of or today_ist()
    target_day = (as_of - timedelta(days=1)).isoformat()
    missed_n = 0
    conn = _connect()
    try:
        resps = conn.execute(
            """
            SELECT id, employee_id, frequency, schedule_weekday, schedule_month_day,
                   COALESCE(schedule_month, 0) as schedule_month
            FROM responsibilities WHERE active=1
            """
        ).fetchall()
        for r in resps:
            freq = r["frequency"] or "Daily"
            if freq == "Whenever Required":
                continue
            if not is_schedule_due(
                freq,
                target_day,
                r["schedule_weekday"] or "",
                int(r["schedule_month_day"] or 0),
                int(r["schedule_month"] or 0),
            ):
                continue
            row = conn.execute(
                "SELECT id, status FROM task_logs WHERE responsibility_id=? AND log_date=?",
                (r["id"], target_day),
            ).fetchone()
            if row:
                st = str(row["status"] or "Pending").strip()
                if st not in ("Pending",):
                    continue
            ok = mark_task(
                int(r["id"]),
                target_day,
                "Missed",
                marked_by=actor,
                remarks="Auto-missed: not updated by end of day (IST)",
                allow_override=True,
            )
            if ok is True:
                missed_n += 1
    finally:
        conn.close()
    return {"ok": True, "target_day": target_day, "marked_missed": missed_n, "as_of": as_of.isoformat()}


def get_performance(department_id=None, from_date=None, to_date=None):
    today = date.today().isoformat()
    fd = from_date or date.today().replace(day=1).isoformat()
    td = to_date or today
    start = date.fromisoformat(fd)
    end = date.fromisoformat(td)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.isoformat())
        cur += timedelta(days=1)
    total_days = len(dates)

    conn = _connect()
    cond = "WHERE r.active=1"
    params = []
    if department_id:
        cond += " AND r.department_id=?"
        params.append(department_id)

    resps = conn.execute(
        f"""
        SELECT r.id, r.employee_id, r.frequency, e.name as employee_name,
               d.name as department_name
        FROM responsibilities r
        JOIN employees e ON e.id=r.employee_id
        LEFT JOIN departments d ON d.id=r.department_id
        {cond}
    """,
        params,
    ).fetchall()

    logs = conn.execute(
        f"""
        SELECT tl.responsibility_id, tl.log_date, tl.status, tl.blocker_employee_id,
               COALESCE(tl.approval_status,'') as approval_status,
               r.linked_to_employee_id
        FROM task_logs tl
        JOIN responsibilities r ON r.id=tl.responsibility_id
        {cond} AND tl.log_date BETWEEN ? AND ?
    """,
        params + [fd, td],
    ).fetchall()

    issue_cond = "WHERE 1=1"
    issue_params = []
    if department_id:
        issue_cond += " AND department_id=?"
        issue_params.append(department_id)
    issue_counts = conn.execute(
        f"""
        SELECT employee_id, COUNT(*) as cnt, severity
        FROM issue_logs
        {issue_cond} AND issue_date BETWEEN ? AND ?
        GROUP BY employee_id, severity
    """,
        issue_params + [fd, td],
    ).fetchall()

    blocker_counts = conn.execute(
        """
        SELECT blocker_employee_id, COUNT(*) as cnt
        FROM task_logs
        WHERE status='Blocked' AND blocker_employee_id IS NOT NULL
        AND log_date BETWEEN ? AND ?
        GROUP BY blocker_employee_id
    """,
        [fd, td],
    ).fetchall()

    ot_cond = "WHERE t.active=1"
    ot_params: list = []
    if department_id:
        ot_cond += " AND t.department_id=?"
        ot_params.append(department_id)
    ot_rows = conn.execute(
        f"""
        SELECT t.*, e.name as employee_name, d.name as department_name
        FROM one_time_tasks t
        LEFT JOIN employees e ON e.id=t.employee_id
        LEFT JOIN departments d ON d.id=t.department_id
        {ot_cond}
        """,
        ot_params,
    ).fetchall()

    conn.close()

    log_map = {}
    for l in logs:
        log_map[(l["responsibility_id"], l["log_date"])] = {
            "status": l["status"],
            "approval_status": l["approval_status"] if "approval_status" in l.keys() else "",
            "linked_to_employee_id": l["linked_to_employee_id"]
            if "linked_to_employee_id" in l.keys()
            else None,
        }
    issue_map: dict = {}
    for i in issue_counts:
        eid = i["employee_id"]
        if eid not in issue_map:
            issue_map[eid] = {"Minor": 0, "Moderate": 0, "Major": 0, "total": 0}
        issue_map[eid][i["severity"]] = i["cnt"]
        issue_map[eid]["total"] += i["cnt"]

    blocker_map = {b["blocker_employee_id"]: b["cnt"] for b in blocker_counts}

    ot_by_emp: dict[int, list] = {}
    for row in ot_rows:
        task = _one_time_task_row(row)
        ot_by_emp.setdefault(int(task["employee_id"]), []).append(task)

    emp_stats: dict = {}
    for r in resps:
        eid = r["employee_id"]
        if eid not in emp_stats:
            emp_stats[eid] = {
                "employee_name": r["employee_name"],
                "department_name": r["department_name"],
                "total_tasks": 0,
                "done_tasks": 0,
                "missed_tasks": 0,
                "blocked_tasks": 0,
            }
        expected = (
            total_days
            if r["frequency"] == "Daily"
            else (total_days // 7 if r["frequency"] == "Weekly" else 1)
        )
        done_weight = 0.0
        missed = 0
        blocked = 0
        for d in dates:
            entry = log_map.get((r["id"], d))
            if not entry:
                continue
            bucket = task_log_counts_for_performance(
                entry["status"],
                approval_status=entry.get("approval_status") or "",
                log_date=d,
                linked_to_employee_id=entry.get("linked_to_employee_id"),
            )
            if bucket == "done":
                done_weight += 1
            elif bucket == "partial":
                done_weight += 0.5
            elif bucket == "missed":
                missed += 1
            elif bucket == "blocked":
                blocked += 1
        emp_stats[eid]["total_tasks"] += expected
        emp_stats[eid]["done_tasks"] += done_weight
        emp_stats[eid]["missed_tasks"] += missed
        emp_stats[eid]["blocked_tasks"] += blocked

    all_eids = set(emp_stats) | set(ot_by_emp)
    result = []
    for eid in all_eids:
        stats = emp_stats.get(
            eid,
            {
                "employee_name": ot_by_emp.get(eid, [{}])[0].get("employee_name", ""),
                "department_name": ot_by_emp.get(eid, [{}])[0].get("department_name", ""),
                "total_tasks": 0,
                "done_tasks": 0,
                "missed_tasks": 0,
                "blocked_tasks": 0,
            },
        )
        total = stats["total_tasks"]
        done = stats["done_tasks"]
        resp_pct = round((done / total * 100) if total > 0 else 0, 1)
        ot_summary, _ = _summarize_one_time_tasks(
            ot_by_emp.get(eid, []), fd, td, today
        )
        combined_pct = _combined_performance_pct(resp_pct, ot_summary)
        issues = issue_map.get(eid, {"Minor": 0, "Moderate": 0, "Major": 0, "total": 0})
        result.append(
            {
                "employee_id": eid,
                "employee_name": stats["employee_name"],
                "department_name": stats["department_name"],
                "total_tasks": total,
                "done_tasks": done,
                "missed_tasks": stats["missed_tasks"],
                "blocked_tasks": stats["blocked_tasks"],
                "pending_tasks": total - done,
                "responsibility_performance_pct": resp_pct,
                "one_time_performance_pct": ot_summary["performance_pct"],
                "one_time_summary": ot_summary,
                "performance_pct": combined_pct,
                "issues_total": issues["total"],
                "issues_minor": issues.get("Minor", 0),
                "issues_moderate": issues.get("Moderate", 0),
                "issues_major": issues.get("Major", 0),
                "blockers_caused": blocker_map.get(eid, 0),
            }
        )

    return sorted(result, key=lambda x: -x["performance_pct"])


# ── One-time Tasks (distinct from recurring responsibilities) ─────────────────


def _normalize_one_time_status(status: str) -> str:
    return "Done" if status == "Completed" else status


def _task_in_appraisal_period(task: dict, fd: str, td: str, today: str) -> bool:
    created = (task.get("created_at") or "")[:10]
    due = task.get("due_date") or ""
    completed = (task.get("completed_at") or "")[:10]
    approved = (task.get("approved_at") or "")[:10]
    status = _normalize_one_time_status(task.get("status") or "")
    if created and fd <= created <= td:
        return True
    if due and fd <= due <= td:
        return True
    if completed and fd <= completed <= td:
        return True
    if approved and fd <= approved <= td:
        return True
    if status in ("Pending", "In Progress", "Rejected") and due and due < today:
        return True
    if status == "Done" and completed and fd <= completed <= td:
        return True
    return False


def _task_completed_on_time(task: dict) -> bool:
    due = task.get("due_date") or ""
    completed = (task.get("completed_at") or "")[:10]
    if not due:
        return True
    return bool(completed and completed <= due)


def _summarize_one_time_tasks(tasks: list, fd: str, td: str, today: str):
    period_tasks = [t for t in tasks if _task_in_appraisal_period(t, fd, td, today)]
    total = len(period_tasks)
    # Post-cutover: awaiting approval does not inflate performance (counts only after Approved)
    # Pre-cutover tasks (created before cutover) keep partial credit for Done awaiting HOD.
    period_pre = []
    period_post = []
    for t in period_tasks:
        created = (t.get("created_at") or "")[:10]
        if created and created < PERFORMANCE_CUTOVER_DATE:
            period_pre.append(t)
        else:
            period_post.append(t)

    def _score(tasks_subset, credit_awaiting: bool):
        aot = alate = aw = pend = ove = ip = rej = 0
        for task in tasks_subset:
            status = _normalize_one_time_status(task.get("status") or "")
            due = task.get("due_date") or ""
            is_overdue = bool(due and due < today and status in ("Pending", "In Progress"))
            if status == "Approved":
                if _task_completed_on_time(task):
                    aot += 1
                else:
                    alate += 1
            elif status == "Done":
                aw += 1
            elif status == "Rejected":
                rej += 1
            elif status == "In Progress":
                ip += 1
                if is_overdue:
                    ove += 1
            elif status == "Pending":
                pend += 1
                if is_overdue:
                    ove += 1
        pos = aot + alate * 0.6 + (aw * 0.25 if credit_awaiting else 0)
        neg = rej + ove + pend * 0.35 + ip * 0.15 + (0 if credit_awaiting else aw * 0.4)
        return pos, neg, aot, alate, aw, pend, ove, ip, rej

    pos1, neg1, aot1, alate1, aw1, pend1, ove1, ip1, rej1 = _score(period_pre, True)
    pos2, neg2, aot2, alate2, aw2, pend2, ove2, ip2, rej2 = _score(period_post, False)
    approved_on_time = aot1 + aot2
    approved_late = alate1 + alate2
    awaiting_approval = aw1 + aw2
    pending = pend1 + pend2
    overdue = ove1 + ove2
    in_progress = ip1 + ip2
    rejected = rej1 + rej2
    positive = pos1 + pos2
    negative = neg1 + neg2
    if total == 0:
        ot_pct = None
    else:
        ot_pct = round(max(0, min(100, (positive / max(positive + negative, 0.1)) * 100)), 1)

    summary = {
        "total": total,
        "approved_on_time": approved_on_time,
        "approved_late": approved_late,
        "awaiting_approval": awaiting_approval,
        "pending": pending,
        "overdue": overdue,
        "in_progress": in_progress,
        "rejected": rejected,
        "performance_pct": ot_pct,
    }
    return summary, period_tasks


def _combined_performance_pct(resp_pct: float, ot_summary: dict) -> float:
    ot_pct = ot_summary.get("performance_pct")
    if ot_summary.get("total", 0) == 0 or ot_pct is None:
        return resp_pct
    if resp_pct == 0 and ot_summary.get("total", 0) > 0:
        return ot_pct
    return round(resp_pct * 0.7 + ot_pct * 0.3, 1)


def _now_iso() -> str:
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")


def _duration_minutes(started_at: str, completed_at: str) -> int:
    if not started_at or not completed_at:
        return 0
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        t0 = datetime.strptime(started_at[:19], fmt)
        t1 = datetime.strptime(completed_at[:19], fmt)
        return max(0, int((t1 - t0).total_seconds() // 60))
    except ValueError:
        return 0


def _one_time_task_row(row) -> dict:
    d = dict(row)
    d["status"] = _normalize_one_time_status(d.get("status") or "")
    if not d.get("duration_minutes") and d.get("started_at") and d.get("completed_at"):
        d["duration_minutes"] = _duration_minutes(d["started_at"], d["completed_at"])
    return d


def get_one_time_task_owner(task_id: int) -> int | None:
    conn = _connect()
    row = conn.execute(
        "SELECT employee_id FROM one_time_tasks WHERE id=? AND active=1",
        (task_id,),
    ).fetchone()
    conn.close()
    return int(row["employee_id"]) if row else None


def list_one_time_tasks(
    employee_id=None,
    department_id=None,
    status: str | None = None,
    active_only=True,
):
    conn = _connect()
    conditions = []
    params: list = []
    if active_only:
        conditions.append("t.active=1")
    if employee_id:
        conditions.append("t.employee_id=?")
        params.append(int(employee_id))
    if department_id:
        conditions.append("t.department_id=?")
        params.append(int(department_id))
    if status:
        conditions.append("t.status=?")
        params.append(status)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(
        f"""
        SELECT t.*, e.name as employee_name, d.name as department_name,
               be.name as backup_employee_name
        FROM one_time_tasks t
        LEFT JOIN employees e ON e.id=t.employee_id
        LEFT JOIN departments d ON d.id=t.department_id
        LEFT JOIN employees be ON be.id=t.backup_employee_id
        {where}
        ORDER BY
            CASE t.status
                WHEN 'Pending' THEN 1
                WHEN 'In Progress' THEN 2
                WHEN 'Done' THEN 3
                WHEN 'Completed' THEN 3
                WHEN 'Rejected' THEN 4
                WHEN 'Approved' THEN 5
                ELSE 6
            END,
            t.due_date ASC,
            t.created_at DESC
        """,
        params,
    ).fetchall()
    conn.close()
    return [_one_time_task_row(r) for r in rows]


def create_one_time_task(data: dict) -> int:
    conn = _connect()
    dept_id = data.get("department_id")
    if not dept_id and data.get("employee_id"):
        row = conn.execute(
            "SELECT department_id FROM employees WHERE id=?",
            (data["employee_id"],),
        ).fetchone()
        if row:
            dept_id = row["department_id"]
    priority = data.get("priority") or "Medium"
    if priority not in PRIORITIES:
        priority = "Medium"
    require_backup = bool(data.get("require_backup", False))
    try:
        backup_id, backup_val, backup_unit = _parse_backup_fields(
            data, int(data["employee_id"]), require=require_backup
        )
    except ValueError:
        conn.close()
        raise
    linked = data.get("linked_to_employee_id")
    try:
        linked_id = int(linked) if linked not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        linked_id = None
    cur = conn.execute(
        """INSERT INTO one_time_tasks(
            employee_id, department_id, title, description, due_date, assigned_by, status, active, priority,
            linked_to_employee_id, backup_employee_id, backup_allocation_value, backup_allocation_unit
        ) VALUES(?,?,?,?,?,?,?,1,?,?,?,?,?)""",
        (
            data["employee_id"],
            dept_id,
            data["title"],
            data.get("description", ""),
            data.get("due_date", ""),
            data.get("assigned_by", ""),
            "Pending",
            priority,
            linked_id,
            backup_id,
            backup_val,
            backup_unit,
        ),
    )
    tid = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return tid


def update_one_time_task(task_id: int, data: dict):
    conn = _connect()
    allowed = [
        "title",
        "description",
        "due_date",
        "employee_id",
        "department_id",
        "active",
        "assigned_by",
        "priority",
        "duration_minutes",
        "manual_duration_minutes",
        "linked_to_employee_id",
        "backup_employee_id",
        "backup_allocation_value",
        "backup_allocation_unit",
    ]
    payload = {k: data[k] for k in data if k in allowed}
    if "manual_duration_minutes" in payload and "duration_minutes" not in payload:
        try:
            payload["duration_minutes"] = parse_duration_to_minutes(payload["manual_duration_minutes"])
        except ValueError:
            pass
    backup_touched = any(
        k in payload
        for k in (
            "backup_employee_id",
            "backup_allocation_value",
            "backup_allocation_unit",
        )
    )
    emp_changed = False
    row0 = None
    if "employee_id" in payload or backup_touched:
        row0 = conn.execute("SELECT * FROM one_time_tasks WHERE id=?", (task_id,)).fetchone()
        if row0 and "employee_id" in payload:
            try:
                emp_changed = int(payload["employee_id"]) != int(row0["employee_id"])
            except (TypeError, ValueError):
                emp_changed = True
    if backup_touched or emp_changed:
        if row0 is None:
            row0 = conn.execute("SELECT * FROM one_time_tasks WHERE id=?", (task_id,)).fetchone()
        if row0:
            merged = dict(row0)
            merged.update(payload)
            try:
                bid, bval, bunit = _parse_backup_fields(
                    merged, int(merged["employee_id"]), require=True
                )
            except ValueError:
                conn.close()
                raise
            payload["backup_employee_id"] = bid
            payload["backup_allocation_value"] = bval
            payload["backup_allocation_unit"] = bunit
    sets = ", ".join(f"{k}=?" for k in payload)
    vals = list(payload.values())
    if sets:
        vals.append(_now_iso())
        vals.append(task_id)
        conn.execute(f"UPDATE one_time_tasks SET {sets}, updated_at=? WHERE id=?", vals)
        conn.commit()
    conn.close()


def set_manual_task_duration(task_id: int, duration_value) -> int:
    """Set manual duration on a one-time task; returns minutes stored."""
    mins = parse_duration_to_minutes(duration_value)
    conn = _connect()
    row = conn.execute(
        "SELECT id FROM one_time_tasks WHERE id=? AND active=1", (task_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise ValueError("Task not found")
    now = _now_iso()
    conn.execute(
        """UPDATE one_time_tasks
           SET duration_minutes=?, manual_duration_minutes=?, updated_at=?
           WHERE id=?""",
        (mins, mins, now, task_id),
    )
    conn.commit()
    conn.close()
    return mins


def cancel_one_time_task(task_id: int):
    update_one_time_task(task_id, {"active": 0})


def start_one_time_task(task_id: int) -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT status FROM one_time_tasks WHERE id=? AND active=1",
        (task_id,),
    ).fetchone()
    if not row or row["status"] not in ("Pending", "Rejected"):
        conn.close()
        return False
    now = _now_iso()
    conn.execute(
        """UPDATE one_time_tasks
           SET status='In Progress', started_at=?, completed_at='', approved_at='',
               duration_minutes=0, updated_at=?
           WHERE id=?""",
        (now, now, task_id),
    )
    conn.commit()
    conn.close()
    return True


def complete_one_time_task(task_id: int, completion_notes: str = "") -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT status, started_at FROM one_time_tasks WHERE id=? AND active=1",
        (task_id,),
    ).fetchone()
    if not row or row["status"] != "In Progress":
        conn.close()
        return False
    now = _now_iso()
    started = row["started_at"] or now
    mins = _duration_minutes(started, now)
    conn.execute(
        """UPDATE one_time_tasks
           SET status='Done', completed_at=?, duration_minutes=?,
               completion_notes=?, updated_at=?
           WHERE id=?""",
        (now, mins, completion_notes or "", now, task_id),
    )
    conn.commit()
    conn.close()
    return True


def approve_one_time_task(task_id: int, approved_by: str = "", approval_notes: str = "") -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT status FROM one_time_tasks WHERE id=? AND active=1",
        (task_id,),
    ).fetchone()
    if not row or _normalize_one_time_status(row["status"]) != "Done":
        conn.close()
        return False
    now = _now_iso()
    conn.execute(
        """UPDATE one_time_tasks
           SET status='Approved', approved_at=?, approved_by=?, approval_notes=?, updated_at=?
           WHERE id=?""",
        (now, approved_by or "", approval_notes or "", now, task_id),
    )
    conn.commit()
    conn.close()
    return True


def reject_one_time_task(task_id: int, approved_by: str = "", approval_notes: str = "") -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT status FROM one_time_tasks WHERE id=? AND active=1",
        (task_id,),
    ).fetchone()
    if not row or _normalize_one_time_status(row["status"]) != "Done":
        conn.close()
        return False
    now = _now_iso()
    conn.execute(
        """UPDATE one_time_tasks
           SET status='Rejected', approved_by=?, approval_notes=?, updated_at=?
           WHERE id=?""",
        (approved_by or "", approval_notes or "", now, task_id),
    )
    conn.commit()
    conn.close()
    return True

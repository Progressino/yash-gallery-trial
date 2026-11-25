"""Attendance & work monitoring dashboard (biometric + HRM office + work start).

Biometric punches live in ``hrm_biometric_punches`` (mapped by ``employees.biometric_code``
or emp_code). Until punches are synced, Attendance is never forced to Absent.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..db import hrm_db as H
from ..db import hrm_worktime as WT

_log = logging.getLogger(__name__)

ABSENT_CUTOFF_HHMM = "11:00"  # IST — after this, no punch/leave/holiday → Absent


def ensure_attendance_tables(conn=None) -> None:
    owns = conn is None
    if owns:
        conn = H._connect()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(employees)").fetchall()}
        if "biometric_code" not in cols:
            conn.execute("ALTER TABLE employees ADD COLUMN biometric_code TEXT DEFAULT ''")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hrm_biometric_punches (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id     INTEGER,
                biometric_code  TEXT NOT NULL,
                punch_at        TEXT NOT NULL,
                punch_type      TEXT DEFAULT '',
                source          TEXT DEFAULT 'import',
                raw_payload     TEXT DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_bio_punch_emp_day
                ON hrm_biometric_punches(employee_id, punch_at);
            CREATE INDEX IF NOT EXISTS idx_bio_punch_code_day
                ON hrm_biometric_punches(biometric_code, punch_at);
            CREATE TABLE IF NOT EXISTS hrm_attendance_reminders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id     INTEGER NOT NULL,
                work_date       TEXT NOT NULL,
                reminder_kind   TEXT NOT NULL,
                message         TEXT DEFAULT '',
                sent_by         TEXT DEFAULT '',
                sent_by_employee_id INTEGER,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_att_remind_emp_day
                ON hrm_attendance_reminders(employee_id, work_date, id);
            CREATE TABLE IF NOT EXISTS hrm_biometric_sync_state (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                last_sync_at    TEXT DEFAULT '',
                last_error      TEXT DEFAULT '',
                last_ok_at      TEXT DEFAULT '',
                note            TEXT DEFAULT ''
            );
            INSERT OR IGNORE INTO hrm_biometric_sync_state(id, note)
            VALUES (1, 'Waiting for biometric device / import configuration');
            """
        )
        if owns:
            conn.commit()
    finally:
        if owns:
            conn.close()


def _sync_state(conn) -> dict:
    ensure_attendance_tables(conn)
    row = conn.execute("SELECT * FROM hrm_biometric_sync_state WHERE id=1").fetchone()
    return dict(row) if row else {"last_sync_at": "", "last_error": "", "last_ok_at": "", "note": ""}


def _punches_for_day(conn, employee_id: int, day: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT punch_at, punch_type FROM hrm_biometric_punches
        WHERE employee_id=? AND date(punch_at)=?
        ORDER BY punch_at
        """,
        (int(employee_id), day),
    ).fetchall()
    return [dict(r) for r in rows]


def _first_work_slot(conn, employee_id: int, day: str) -> dict | None:
    row = conn.execute(
        """
        SELECT entity_type, entity_id, started_at, ended_at, notes
        FROM hrm_time_slots
        WHERE employee_id=? AND log_date=?
        ORDER BY started_at LIMIT 1
        """,
        (int(employee_id), day),
    ).fetchone()
    return dict(row) if row else None


def _running_work(conn, employee_id: int, day: str) -> dict | None:
    row = conn.execute(
        """
        SELECT entity_type, entity_id, started_at, ended_at
        FROM hrm_time_slots
        WHERE employee_id=? AND log_date=? AND TRIM(COALESCE(ended_at,''))=''
        ORDER BY id DESC LIMIT 1
        """,
        (int(employee_id), day),
    ).fetchone()
    return dict(row) if row else None


def _work_title(conn, entity_type: str, entity_id: int) -> str:
    table = "responsibilities" if entity_type == "responsibility" else "one_time_tasks"
    try:
        row = conn.execute(f"SELECT title FROM {table} WHERE id=?", (int(entity_id),)).fetchone()
        return str((row["title"] if row else "") or "")
    except Exception:
        return ""


def _last_reminder(conn, employee_id: int, day: str) -> dict | None:
    row = conn.execute(
        """
        SELECT reminder_kind, created_at, sent_by FROM hrm_attendance_reminders
        WHERE employee_id=? AND work_date=?
        ORDER BY id DESC LIMIT 1
        """,
        (int(employee_id), day),
    ).fetchone()
    return dict(row) if row else None


def _is_holiday(conn, day: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM hrm_holidays WHERE holiday_date=?", (day,)
    ).fetchone()
    return bool(row)


def _attendance_status(
    *,
    day: str,
    now: str,
    on_leave: bool,
    holiday: bool,
    punches: list[dict],
    sync_has_data: bool,
) -> str:
    if on_leave:
        return "Leave"
    if holiday:
        return "Holiday"
    # Sunday weekly off (simple default; per-employee weekly off can be added later)
    try:
        if H._ts(f"{day} 12:00:00").weekday() == 6:
            return "Weekly Off"
    except Exception:
        pass
    if punches:
        return "Present"
    if not sync_has_data:
        return "Not synced"
    # After cutoff with sync data but no punch → Absent
    cutoff = f"{day} {ABSENT_CUTOFF_HHMM}:00"
    if now >= cutoff:
        return "Absent"
    return "Pending"


def attendance_monitor_dashboard(
    *,
    work_date: str = "",
    department_id: int | None = None,
    employee_id: int | None = None,
    reporting_hod_id: int | None = None,
    attendance_status: str = "",
    hrm_started: str = "",
    work_started: str = "",
) -> dict:
    day = (work_date or H.today_ist().isoformat())[:10]
    now = H._now_iso()
    conn = H._connect()
    try:
        ensure_attendance_tables(conn)
        state = _sync_state(conn)
        emps = H.list_employees(department_id, "Active", employee_id=employee_id)
        if reporting_hod_id:
            emps = [
                e for e in emps
                if int(e.get("reports_to_employee_id") or 0) == int(reporting_hod_id)
            ]
        punch_any = conn.execute(
            "SELECT 1 FROM hrm_biometric_punches WHERE date(punch_at)=? LIMIT 1", (day,)
        ).fetchone()
        sync_has_data = bool(punch_any) or bool(str(state.get("last_ok_at") or "").strip())

        rows: list[dict] = []
        for e in emps:
            eid = int(e["id"])
            leave = H.employee_leave_on(eid, day, conn)
            holiday = _is_holiday(conn, day)
            punches = _punches_for_day(conn, eid, day)
            office = WT.get_office_session(eid, day, conn) or {}
            o_start = str(office.get("office_start") or "")
            first_slot = _first_work_slot(conn, eid, day)
            running = _running_work(conn, eid, day)
            summary = WT.day_time_summary(eid, day, conn=conn, now=now)
            rem = _last_reminder(conn, eid, day)
            att = _attendance_status(
                day=day,
                now=now,
                on_leave=bool(leave),
                holiday=holiday,
                punches=punches,
                sync_has_data=sync_has_data,
            )
            hrm_yes = bool(o_start)
            work_yes = bool(first_slot)
            current = ""
            current_state = "No Active Work"
            if running:
                current = _work_title(conn, str(running["entity_type"]), int(running["entity_id"]))
                current_state = "Running"
            elif first_slot and first_slot.get("ended_at"):
                current = _work_title(conn, str(first_slot["entity_type"]), int(first_slot["entity_id"]))
                current_state = "Paused"

            bio_in = punches[0]["punch_at"] if punches else ""
            bio_out = punches[-1]["punch_at"] if len(punches) > 1 else (punches[0]["punch_at"] if punches else "")

            row = {
                "employee_id": eid,
                "employee_name": e.get("name"),
                "department_id": e.get("department_id"),
                "department_name": e.get("department_name") or "",
                "attendance_status": att,
                "biometric_in": bio_in,
                "biometric_out": bio_out if bio_out != bio_in else "",
                "hrm_started": hrm_yes,
                "hrm_start_at": o_start,
                "work_started": work_yes,
                "work_start_at": (first_slot or {}).get("started_at") or "",
                "current_work": current or current_state,
                "current_work_state": current_state,
                "working_seconds": summary.get("working_seconds") or 0,
                "working_label": summary.get("working_label") or "0m",
                "last_reminder_at": (rem or {}).get("created_at") or "",
                "last_reminder_kind": (rem or {}).get("reminder_kind") or "",
                "can_remind_hrm": att == "Present" and not hrm_yes,
                "can_remind_work": att == "Present" and hrm_yes and not work_yes,
            }
            if attendance_status and row["attendance_status"] != attendance_status:
                continue
            if hrm_started == "yes" and not row["hrm_started"]:
                continue
            if hrm_started == "no" and row["hrm_started"]:
                continue
            if work_started == "yes" and not row["work_started"]:
                continue
            if work_started == "no" and row["work_started"]:
                continue
            rows.append(row)

        def _count(pred) -> int:
            return sum(1 for r in rows if pred(r))

        summary = {
            "total_active": len(rows),
            "present": _count(lambda r: r["attendance_status"] == "Present"),
            "absent": _count(lambda r: r["attendance_status"] == "Absent"),
            "present_hrm_not_started": _count(
                lambda r: r["attendance_status"] == "Present" and not r["hrm_started"]
            ),
            "hrm_started_work_not_started": _count(
                lambda r: r["hrm_started"] and not r["work_started"]
            ),
            "leave": _count(lambda r: r["attendance_status"] == "Leave"),
            "not_synced": _count(lambda r: r["attendance_status"] == "Not synced"),
        }
        return {
            "work_date": day,
            "rows": rows,
            "summary": summary,
            "sync": state,
            "absent_cutoff": ABSENT_CUTOFF_HHMM,
        }
    finally:
        conn.close()


def send_attendance_reminder(
    employee_id: int,
    *,
    work_date: str,
    kind: str,
    actor: str = "",
    actor_employee_id: int | None = None,
) -> dict | str:
    if kind not in ("start_hrm", "start_work"):
        return "bad_kind"
    day = (work_date or H.today_ist().isoformat())[:10]
    conn = H._connect()
    try:
        ensure_attendance_tables(conn)
        emp = conn.execute(
            "SELECT id, name FROM employees WHERE id=? AND status='Active'", (int(employee_id),)
        ).fetchone()
        if not emp:
            return "not_found"
        if kind == "start_hrm":
            title = "Reminder: Start HRM Office Time"
            message = (
                f"You are marked Present for {day}. Please start your Office Time in HRM."
            )
        else:
            title = "Reminder: Start Work"
            message = (
                f"Your HRM Office Time has started for {day}, but no Task/Responsibility is running. "
                "Please start your work."
            )
        conn.execute(
            """INSERT INTO hrm_attendance_reminders(
                employee_id, work_date, reminder_kind, message, sent_by, sent_by_employee_id)
               VALUES (?,?,?,?,?,?)""",
            (int(employee_id), day, kind, message, actor, actor_employee_id),
        )
        # Reuse assignment notifications inbox when available
        try:
            conn.execute(
                """INSERT INTO hrm_assignment_notifications(
                    to_employee_id, entity_type, entity_id, title, message, event_type, is_read)
                   VALUES (?,?,?,?,?,?,0)""",
                (int(employee_id), "attendance", 0, title, message, kind),
            )
        except Exception:
            pass
        conn.commit()
        return {"ok": True, "title": title, "message": message}
    finally:
        conn.close()


def import_biometric_punches(rows: list[dict], *, source: str = "import", actor: str = "") -> dict:
    """Import punch rows: each needs biometric_code or employee_id + punch_at."""
    conn = H._connect()
    try:
        ensure_attendance_tables(conn)
        inserted = 0
        mapped = 0
        for r in rows:
            code = str(r.get("biometric_code") or r.get("emp_code") or "").strip()
            eid = r.get("employee_id")
            punch_at = str(r.get("punch_at") or "").strip()
            if not punch_at:
                continue
            if not eid and code:
                er = conn.execute(
                    """SELECT id FROM employees
                       WHERE status='Active' AND (
                         LOWER(TRIM(COALESCE(biometric_code,'')))=LOWER(TRIM(?))
                         OR LOWER(TRIM(emp_code))=LOWER(TRIM(?))
                       ) LIMIT 1""",
                    (code, code),
                ).fetchone()
                if er:
                    eid = int(er["id"])
                    mapped += 1
            conn.execute(
                """INSERT INTO hrm_biometric_punches(
                    employee_id, biometric_code, punch_at, punch_type, source)
                   VALUES (?,?,?,?,?)""",
                (
                    int(eid) if eid else None,
                    code or str(eid or ""),
                    punch_at,
                    str(r.get("punch_type") or ""),
                    source,
                ),
            )
            inserted += 1
        stamp = H._now_iso()
        conn.execute(
            """UPDATE hrm_biometric_sync_state
               SET last_sync_at=?, last_ok_at=?, last_error='', note=?
               WHERE id=1""",
            (stamp, stamp, f"Imported {inserted} punches by {actor or 'system'}"),
        )
        conn.commit()
        return {"ok": True, "inserted": inserted, "mapped_employees": mapped}
    except Exception as e:
        _log.exception("biometric import failed")
        try:
            conn.execute(
                "UPDATE hrm_biometric_sync_state SET last_sync_at=?, last_error=? WHERE id=1",
                (H._now_iso(), str(e)[:500]),
            )
            conn.commit()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()

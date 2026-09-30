"""HRM work time — office sessions, time-slot edits, working hours, free time, DWR range.

All working-time figures come from ``hrm_time_slots`` (see hrm_db time-slot helpers).
Office time is tracked separately in ``hrm_office_sessions`` and never counted as work.
Timestamps are IST-local strings ("YYYY-MM-DD HH:MM:SS") like the rest of HRM.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from . import hrm_db as H

DWR_MAX_DAYS = 31
REPORT_MAX_DAYS = 62


class OfficeCloseBlocked(Exception):
    def __init__(self, pending: list[dict]):
        self.pending = pending
        names = ", ".join(p["title"] for p in pending[:8])
        more = f" (+{len(pending) - 8} more)" if len(pending) > 8 else ""
        super().__init__(
            f"Update the status of {len(pending)} pending responsibilit"
            f"{'y' if len(pending) == 1 else 'ies'} before Office Close: {names}{more}"
        )


def _day_bounds(day: str) -> tuple[str, str]:
    d = date.fromisoformat(str(day)[:10])
    return f"{d.isoformat()} 00:00:00", f"{(d + timedelta(days=1)).isoformat()} 00:00:00"


def _date_range(from_date: str, to_date: str, max_days: int) -> list[str]:
    d0 = date.fromisoformat(str(from_date)[:10])
    d1 = date.fromisoformat(str(to_date)[:10])
    if d1 < d0:
        raise ValueError("To date must be on or after From date")
    if (d1 - d0).days + 1 > max_days:
        raise ValueError(f"Date range cannot exceed {max_days} days")
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


# ── Office sessions ──────────────────────────────────────────────────────────


def get_office_session(employee_id: int, day: str, conn=None) -> dict | None:
    owns = conn is None
    if owns:
        conn = H._connect()
    row = conn.execute(
        "SELECT * FROM hrm_office_sessions WHERE employee_id=? AND work_date=?",
        (int(employee_id), str(day)[:10]),
    ).fetchone()
    if owns:
        conn.close()
    return dict(row) if row else None


def office_start(employee_id: int, *, actor: str = "") -> dict:
    day = H.today_ist().isoformat()
    conn = H._connect()
    try:
        if H.employee_leave_on(employee_id, day, conn):
            raise ValueError("You are on approved leave today")
        existing = get_office_session(employee_id, day, conn)
        if existing and existing.get("office_start"):
            if existing.get("office_close"):
                raise ValueError("Office time is already closed for today")
            raise ValueError("Office time already started")
        now = H._now_iso()
        if existing:
            conn.execute(
                "UPDATE hrm_office_sessions SET office_start=?, started_by=?, updated_at=? WHERE id=?",
                (now, actor or "", now, int(existing["id"])),
            )
        else:
            conn.execute(
                """INSERT INTO hrm_office_sessions(employee_id, work_date, office_start, started_by, updated_at)
                   VALUES(?,?,?,?,?)""",
                (int(employee_id), day, now, actor or "", now),
            )
        H.write_task_audit("office", int(employee_id), "office_start", new_value=now, actor=actor, conn=conn)
        conn.commit()
    finally:
        conn.close()
    return day_time_summary(employee_id, day)


def office_close_pending(employee_id: int, day: str) -> list[dict]:
    """Responsibilities visible in Employee Check that still have no status."""
    snap = H.get_employee_day_check(employee_id, day)
    if not snap:
        return []
    pending: list[dict] = []
    for bucket in ("not_worked", "whenever_required"):
        for i in snap.get(bucket) or []:
            if str(i.get("status") or "Pending") == "Pending":
                pending.append(
                    {"responsibility_id": i.get("responsibility_id"), "title": i.get("title") or "",
                     "frequency": i.get("frequency") or "", "section": bucket}
                )
    for c in snap.get("additional_work") or []:
        if str(c.get("status") or "Pending") == "Pending":
            pending.append(
                {"responsibility_id": c.get("responsibility_id"), "title": c.get("title") or "",
                 "frequency": c.get("frequency") or "", "section": "additional_work"}
            )
    return pending


def office_close(employee_id: int, *, actor: str = "") -> dict:
    day = H.today_ist().isoformat()
    session = get_office_session(employee_id, day)
    if not session or not session.get("office_start"):
        raise ValueError("Start office time before closing it")
    if session.get("office_close"):
        raise ValueError("Office time is already closed for today")
    pending = office_close_pending(employee_id, day)
    if pending:
        raise OfficeCloseBlocked(pending)
    conn = H._connect()
    try:
        close_ts = H._now_iso()
        paused_resp = 0
        for r in conn.execute(
            """SELECT * FROM task_logs WHERE employee_id=?
               AND IFNULL(started_at,'')!='' AND IFNULL(ended_at,'')='' AND IFNULL(paused_at,'')=''""",
            (int(employee_id),),
        ).fetchall():
            H._pause_task_log_row(
                conn, dict(r), actor=actor or "office-close",
                notes="Auto-paused at Office Close", reason="office_close", at=close_ts,
            )
            paused_resp += 1
        paused_tasks = 0
        for r in conn.execute(
            """SELECT * FROM one_time_tasks WHERE employee_id=? AND active=1
               AND status='In Progress' AND IFNULL(paused_at,'')=''""",
            (int(employee_id),),
        ).fetchall():
            H._pause_one_time_row(
                conn, dict(r), actor=actor or "office-close", reason="office_close",
                notes="Auto-paused at Office Close", at=close_ts,
            )
            paused_tasks += 1
        conn.execute(
            "UPDATE hrm_office_sessions SET office_close=?, closed_by=?, updated_at=? WHERE id=?",
            (close_ts, actor or "", close_ts, int(session["id"])),
        )
        H.write_task_audit(
            "office", int(employee_id), "office_close", new_value=close_ts, actor=actor,
            notes=f"Auto-paused {paused_tasks} one-time task(s), {paused_resp} responsibility timer(s)",
            conn=conn,
        )
        conn.commit()
    finally:
        conn.close()
    out = day_time_summary(employee_id, day)
    out["auto_paused_tasks"] = paused_tasks
    out["auto_paused_responsibilities"] = paused_resp
    return out


# ── Slot math ────────────────────────────────────────────────────────────────


def _slot_net_in_window(slot: dict, w0: str, w1: str, now: str) -> int:
    """Net working seconds of a slot inside [w0, w1) — clipped, break-deducted, legacy-scaled."""
    start = str(slot["started_at"])
    end = str(slot.get("ended_at") or "").strip() or now
    lo, hi = max(start, w0), min(end, w1)
    if hi <= lo:
        return 0
    clipped = H._seconds_between(lo, hi)
    span = H._seconds_between(start, end)
    if slot.get("ended_at") and span > 0:
        stored = int(slot.get("duration_seconds") or 0)
        if 0 <= stored < span:
            clipped = int(clipped * stored / span)
    if str(slot.get("break_mode") or "") == "deduct" and slot.get("ended_at"):
        clipped -= sum(o["overlap_seconds"] for o in H.break_overlaps(lo, hi))
    return max(0, clipped)


def _employee_slots_between(conn, employee_id: int, w0: str, w1: str) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM hrm_time_slots
           WHERE employee_id=? AND started_at < ?
             AND (IFNULL(ended_at,'')='' OR ended_at > ?)
           ORDER BY started_at, id""",
        (int(employee_id), w1, w0),
    ).fetchall()
    return [dict(r) for r in rows]


def _merge_intervals(intervals: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[list[str]] = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if merged and a <= merged[-1][1]:
            if b > merged[-1][1]:
                merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def free_time_gaps(window_start: str, window_end: str, busy: list[tuple[str, str]]) -> list[dict]:
    """Office window minus the union of busy intervals (overlaps never double-counted)."""
    if not window_start or not window_end or window_end <= window_start:
        return []
    clipped = [(max(a, window_start), min(b, window_end)) for a, b in busy]
    gaps: list[dict] = []
    cursor = window_start
    for a, b in _merge_intervals(clipped):
        if a > cursor:
            gaps.append({"start": cursor, "end": a, "seconds": H._seconds_between(cursor, a)})
        cursor = max(cursor, b)
    if cursor < window_end:
        gaps.append({"start": cursor, "end": window_end, "seconds": H._seconds_between(cursor, window_end)})
    for g in gaps:
        g["minutes"] = g["seconds"] // 60
        g["label"] = H._format_duration_hm(g["seconds"])
    return [g for g in gaps if g["seconds"] > 0]


def day_time_summary(employee_id: int, day: str, *, conn=None, now: str | None = None) -> dict:
    owns = conn is None
    if owns:
        conn = H._connect()
    try:
        day = str(day)[:10]
        now = now or H._now_iso()
        d0, d1 = _day_bounds(day)
        session = get_office_session(employee_id, day, conn) or {}
        slots = _employee_slots_between(conn, employee_id, d0, d1)
        leave = H.employee_leave_on(employee_id, day, conn)
    finally:
        if owns:
            conn.close()
    working = sum(_slot_net_in_window(s, d0, d1, now) for s in slots)
    o_start = str(session.get("office_start") or "")
    o_close = str(session.get("office_close") or "")
    office_open = bool(o_start and not o_close)
    if o_start:
        if o_close:
            o_end = o_close
        elif day == H.today_ist().isoformat():
            o_end = max(o_start, min(now, d1))
        else:
            ends = [str(s.get("ended_at") or "") for s in slots if s.get("ended_at")]
            o_end = max([o_start, *[e for e in ends if e <= d1]])
    else:
        o_end = ""
    office_seconds = H._seconds_between(o_start, o_end) if o_start and o_end else 0
    busy = [(str(s["started_at"]), str(s.get("ended_at") or "") or now) for s in slots]
    gaps = free_time_gaps(o_start, o_end, busy) if office_seconds else []
    free_seconds = sum(g["seconds"] for g in gaps)
    return {
        "employee_id": int(employee_id),
        "work_date": day,
        "office_start": o_start,
        "office_close": o_close,
        "office_open": office_open,
        "office_seconds": office_seconds,
        "office_label": H._format_duration_hm(office_seconds),
        "working_seconds": working,
        "working_label": H._format_duration_hm(working),
        "free_seconds": free_seconds,
        "free_label": H._format_duration_hm(free_seconds),
        "free_gaps": gaps,
        "slot_count": len(slots),
        "on_leave": bool(leave),
        "leave_id": (leave or {}).get("id"),
    }


def working_hours_report(employee_ids: list[int], from_date: str, to_date: str) -> dict:
    days = _date_range(from_date, to_date, REPORT_MAX_DAYS)
    conn = H._connect()
    try:
        names = {
            int(r["id"]): dict(r)
            for r in conn.execute(
                f"""SELECT e.id, e.name, d.name AS department_name FROM employees e
                    LEFT JOIN departments d ON d.id=e.department_id
                    WHERE e.id IN ({','.join('?' * len(employee_ids)) or 'NULL'})""",
                [int(e) for e in employee_ids],
            ).fetchall()
        }
        now = H._now_iso()
        rows: list[dict] = []
        per_emp: dict[int, dict] = {}
        for eid in employee_ids:
            for day in days:
                s = day_time_summary(eid, day, conn=conn, now=now)
                if not (s["office_start"] or s["slot_count"] or s["on_leave"]):
                    continue
                emp = names.get(int(eid), {})
                s["employee_name"] = emp.get("name") or ""
                s["department_name"] = emp.get("department_name") or ""
                rows.append(s)
                t = per_emp.setdefault(
                    int(eid),
                    {"employee_id": int(eid), "employee_name": s["employee_name"],
                     "department_name": s["department_name"], "office_seconds": 0,
                     "working_seconds": 0, "free_seconds": 0, "days": 0, "leave_days": 0},
                )
                t["office_seconds"] += s["office_seconds"]
                t["working_seconds"] += s["working_seconds"]
                t["free_seconds"] += s["free_seconds"]
                t["days"] += 1
                t["leave_days"] += 1 if s["on_leave"] else 0
    finally:
        conn.close()
    totals = {
        "office_seconds": sum(t["office_seconds"] for t in per_emp.values()),
        "working_seconds": sum(t["working_seconds"] for t in per_emp.values()),
        "free_seconds": sum(t["free_seconds"] for t in per_emp.values()),
    }
    for t in [*per_emp.values(), totals]:
        for k in ("office", "working", "free"):
            t[f"{k}_label"] = H._format_duration_hm(t[f"{k}_seconds"])
    return {
        "from_date": days[0],
        "to_date": days[-1],
        "rows": rows,
        "employees": sorted(per_emp.values(), key=lambda t: -t["working_seconds"]),
        "totals": totals,
    }


# ── Time-slot manual edit / notes / add ──────────────────────────────────────


def get_slot(slot_id: int) -> dict | None:
    conn = H._connect()
    row = conn.execute("SELECT * FROM hrm_time_slots WHERE id=?", (int(slot_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


def _entity_time_locked(conn, entity_type: str, entity_id: int) -> bool:
    if entity_type == H.SLOT_RESP:
        row = conn.execute("SELECT status FROM task_logs WHERE id=?", (int(entity_id),)).fetchone()
        return bool(row) and str(row["status"] or "Pending").strip() not in ("Pending", "")
    row = conn.execute("SELECT status FROM one_time_tasks WHERE id=?", (int(entity_id),)).fetchone()
    return bool(row) and str(row["status"] or "") == "Approved"


def _sync_parent_after_slot_change(conn, entity_type: str, entity_id: int) -> None:
    H._recompute_entity_totals(conn, entity_type, entity_id)
    if entity_type != H.SLOT_RESP:
        return
    agg = conn.execute(
        """SELECT MIN(started_at) AS s, MAX(CASE WHEN IFNULL(ended_at,'')='' THEN NULL ELSE ended_at END) AS e,
                  SUM(CASE WHEN IFNULL(ended_at,'')='' THEN 1 ELSE 0 END) AS open_n
           FROM hrm_time_slots WHERE entity_type=? AND entity_id=?""",
        (entity_type, int(entity_id)),
    ).fetchone()
    log = conn.execute("SELECT ended_at FROM task_logs WHERE id=?", (int(entity_id),)).fetchone()
    if not agg or not log:
        return
    if agg["s"]:
        conn.execute("UPDATE task_logs SET started_at=? WHERE id=?", (agg["s"], int(entity_id)))
    if str(log["ended_at"] or "").strip() and agg["e"] and not int(agg["open_n"] or 0):
        conn.execute("UPDATE task_logs SET ended_at=? WHERE id=?", (agg["e"], int(entity_id)))


def _clock_on(value: str, base: str) -> str:
    """Accept 'HH:MM' (anchored to base's date) or a full 'YYYY-MM-DD HH:MM[:SS]'."""
    raw = str(value or "").strip()
    if len(raw) <= 8 and ":" in raw:
        raw = f"{str(base)[:10]} {raw}"
    return H._parse_clock(raw)


def edit_time_slot(
    slot_id: int,
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
    notes: str | None = None,
    actor: str = "",
    allow_override: bool = False,
) -> dict | str:
    conn = H._connect()
    try:
        row = conn.execute("SELECT * FROM hrm_time_slots WHERE id=?", (int(slot_id),)).fetchone()
        if not row:
            return "not_found"
        slot = dict(row)
        is_open = not str(slot.get("ended_at") or "").strip()
        try:
            new_start = (
                _clock_on(started_at, slot["started_at"]) if started_at not in (None, "") else str(slot["started_at"])
            )
            new_end = (
                _clock_on(ended_at, slot.get("ended_at") or new_start)
                if ended_at not in (None, "") else str(slot.get("ended_at") or "")
            )
        except ValueError:
            return "invalid_time"
        times_changed = new_start != str(slot["started_at"]) or new_end != str(slot.get("ended_at") or "")
        now = H._now_iso()
        if times_changed:
            if is_open and new_end:
                return "slot_open"
            if not allow_override and _entity_time_locked(conn, slot["entity_type"], int(slot["entity_id"])):
                return "status_locked"
            if new_end and new_end < new_start:
                return "invalid_range"
            limit = (H._ts(now) + timedelta(minutes=1)).strftime(H._TS_FMT)
            if new_start > limit or (new_end and new_end > limit):
                return "future_time"
            first_edit = not int(slot.get("manual_edited") or 0)
            dur = H._seconds_between(new_start, new_end) if new_end else 0
            ded = int(slot.get("break_deduct_seconds") or 0)
            if str(slot.get("break_mode") or "") == "deduct" and new_end:
                ded = min(dur, sum(o["overlap_seconds"] for o in H.break_overlaps(new_start, new_end)))
            conn.execute(
                """UPDATE hrm_time_slots
                   SET started_at=?, ended_at=?, duration_seconds=?, break_deduct_seconds=?,
                       manual_edited=1, edited_by=?, edited_at=?,
                       original_started_at=CASE WHEN ?=1 THEN started_at ELSE original_started_at END,
                       original_ended_at=CASE WHEN ?=1 THEN IFNULL(ended_at,'') ELSE original_ended_at END
                   WHERE id=?""",
                (new_start, new_end, dur, ded, actor or "", now,
                 1 if first_edit else 0, 1 if first_edit else 0, int(slot_id)),
            )
            H.write_task_audit(
                "time_slot",
                int(slot_id),
                "manual_edit",
                old_value=f"{slot['started_at']} → {slot.get('ended_at') or ''}",
                new_value=f"{new_start} → {new_end}",
                actor=actor,
                notes=f"{slot['entity_type']}#{slot['entity_id']}",
                conn=conn,
            )
            _sync_parent_after_slot_change(conn, slot["entity_type"], int(slot["entity_id"]))
        if notes is not None and str(notes) != str(slot.get("notes") or ""):
            conn.execute("UPDATE hrm_time_slots SET notes=? WHERE id=?", (str(notes)[:2000], int(slot_id)))
            H.write_task_audit(
                "time_slot",
                int(slot_id),
                "notes",
                old_value=str(slot.get("notes") or ""),
                new_value=str(notes)[:2000],
                actor=actor,
                conn=conn,
            )
        conn.commit()
        out = conn.execute("SELECT * FROM hrm_time_slots WHERE id=?", (int(slot_id),)).fetchone()
        return H._slot_view(dict(out))
    finally:
        conn.close()


def add_manual_slot(
    *,
    entity_type: str,
    entity_id: int | None = None,
    responsibility_id: int | None = None,
    log_date: str | None = None,
    started_at: str,
    ended_at: str,
    notes: str = "",
    actor: str = "",
    allow_override: bool = False,
) -> dict | str:
    if entity_type not in H.SLOT_ENTITY_TYPES:
        return "invalid_entity"
    try:
        start = H._parse_clock(started_at)
        end = H._parse_clock(ended_at)
    except ValueError:
        return "invalid_time"
    if not start or not end:
        return "missing_time"
    if end < start:
        return "invalid_range"
    now = H._now_iso()
    if end > (H._ts(now) + timedelta(minutes=1)).strftime(H._TS_FMT):
        return "future_time"
    conn = H._connect()
    try:
        if entity_type == H.SLOT_RESP:
            owner = H._responsibility_owner(conn, int(responsibility_id or 0))
            if not owner:
                return "not_found"
            day = str(log_date or start[:10])[:10]
            log = H._ensure_task_log_row(conn, int(responsibility_id), int(owner["employee_id"]), day)
            eid, emp_id, rid = int(log["id"]), int(owner["employee_id"]), int(responsibility_id)
        else:
            t = conn.execute(
                "SELECT id, employee_id FROM one_time_tasks WHERE id=? AND active=1", (int(entity_id or 0),)
            ).fetchone()
            if not t:
                return "not_found"
            eid, emp_id, rid, day = int(t["id"]), int(t["employee_id"]), None, start[:10]
        if not allow_override and _entity_time_locked(conn, entity_type, eid):
            return "status_locked"
        sid = H._insert_closed_slot(
            conn, entity_type=entity_type, entity_id=eid, employee_id=emp_id, log_date=day,
            started_at=start, ended_at=end, responsibility_id=rid, source="manual",
            manual=True, notes=notes, actor=actor,
        )
        conn.execute(
            "UPDATE hrm_time_slots SET original_started_at='', original_ended_at='' WHERE id=?", (sid,)
        )
        if entity_type == H.SLOT_RESP:
            log_row = conn.execute("SELECT started_at FROM task_logs WHERE id=?", (eid,)).fetchone()
            if log_row and not str(log_row["started_at"] or "").strip():
                conn.execute(
                    "UPDATE task_logs SET started_at=?, ended_at=?, paused_at='' WHERE id=?",
                    (start, end, eid),
                )
        H.write_task_audit(
            "time_slot", sid, "manual_add", new_value=f"{start} → {end}", actor=actor,
            notes=f"{entity_type}#{eid}", conn=conn,
        )
        _sync_parent_after_slot_change(conn, entity_type, eid)
        conn.commit()
        out = conn.execute("SELECT * FROM hrm_time_slots WHERE id=?", (sid,)).fetchone()
        return H._slot_view(dict(out))
    finally:
        conn.close()


def slot_audit(slot_id: int) -> list[dict]:
    conn = H._connect()
    rows = conn.execute(
        "SELECT * FROM hrm_task_audit WHERE entity_type='time_slot' AND entity_id=? ORDER BY id",
        (int(slot_id),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Daily Working Report (date / range) ──────────────────────────────────────


def _one_time_slots_for_day(employee_id: int, d0: str, d1: str, now: str) -> list[tuple[dict, list[dict]]]:
    conn = H._connect()
    try:
        slots = conn.execute(
            """SELECT * FROM hrm_time_slots
               WHERE employee_id=? AND entity_type=? AND started_at>=? AND started_at<?
               ORDER BY started_at, id""",
            (int(employee_id), H.SLOT_ONE_TIME, d0, d1),
        ).fetchall()
        by_task: dict[int, list[dict]] = {}
        for s in slots:
            by_task.setdefault(int(s["entity_id"]), []).append(H._slot_view(dict(s), now))
        if not by_task:
            return []
        ids = list(by_task)
        tasks = conn.execute(
            f"""SELECT t.*, le.name AS linked_to_employee_name FROM one_time_tasks t
                LEFT JOIN employees le ON le.id=t.linked_to_employee_id
                WHERE t.id IN ({','.join('?' * len(ids))})""",
            ids,
        ).fetchall()
    finally:
        conn.close()
    out = []
    for t in tasks:
        td = H._one_time_task_row(t)
        td["linked_to_employee_name"] = t["linked_to_employee_name"] or ""
        out.append((td, by_task.get(int(t["id"]), [])))
    return out


def _user_updated(item: dict) -> bool:
    by = str(item.get("marked_by") or "").lower()
    marked = str(item.get("status") or "Pending") not in ("Pending", "", "Reassigned") and not by.startswith("system")
    return marked or bool(item.get("time_slots"))


def list_dwr_report(*, employee_ids: list[int], from_date: str, to_date: str) -> dict:
    days = _date_range(from_date, to_date, DWR_MAX_DAYS)
    now = H._now_iso()
    rows: list[dict] = []
    leave_rows: list[dict] = []
    for eid in employee_ids:
        for day in days:
            leave = H.employee_leave_on(eid, day)
            if leave:
                conn = H._connect()
                emp = conn.execute(
                    """SELECT e.name, d.name AS department_name FROM employees e
                       LEFT JOIN departments d ON d.id=e.department_id WHERE e.id=?""",
                    (int(eid),),
                ).fetchone()
                conn.close()
                leave_rows.append(
                    {
                        "row_type": "leave",
                        "employee_id": int(eid),
                        "employee_name": (emp["name"] if emp else "") or "",
                        "department_name": (emp["department_name"] if emp else "") or "",
                        "check_date": day,
                        "title": "Leave",
                        "status": "Leave",
                        "leave_id": leave["id"],
                        "leave_from": leave["from_date"],
                        "leave_to": leave["to_date"],
                        "duration_seconds": 0,
                        "duration_minutes": 0,
                        "slots": [],
                    }
                )
                continue
            snap = H.get_employee_day_check(eid, day)
            if not snap:
                continue
            emp = snap.get("employee") or {}
            resp_items = [
                *(snap.get("worked_on") or []),
                *(snap.get("not_worked") or []),
                *(snap.get("other") or []),
                *[i for i in (snap.get("whenever_required") or []) if _user_updated(i)],
            ]
            d0, d1 = _day_bounds(day)
            ot_items = _one_time_slots_for_day(int(eid), d0, d1, now)
            clone_items = [
                c for c in (snap.get("additional_work") or [])
                if str(c.get("status") or "Pending") != "Pending"
            ]
            updated = any(_user_updated(i) for i in resp_items) or bool(ot_items) or bool(clone_items)
            if not updated:
                continue
            base = {
                "employee_id": int(eid),
                "employee_name": emp.get("name") or "",
                "department_name": emp.get("department_name") or "",
                "check_date": day,
            }
            for i in resp_items:
                slots = i.get("time_slots") or []
                secs = sum(int(s.get("net_seconds") or 0) for s in slots)
                if not slots:
                    secs = int(i.get("active_seconds") or 0)
                linked = i.get("linked_to_employee_name") or ""
                rows.append(
                    {
                        **base,
                        "row_type": "responsibility",
                        "responsibility_id": i.get("responsibility_id"),
                        "task_log_id": i.get("task_log_id"),
                        "title": i.get("title"),
                        "frequency": i.get("frequency"),
                        "status": i.get("status"),
                        "approval_status": i.get("approval_status") or "",
                        "auto_approved": (i.get("approval_status") or "") == "Auto-Approved",
                        "timer_status": i.get("timer_status") or "Not Started",
                        "started_at": slots[0]["started_at"] if slots else i.get("started_at") or "",
                        "ended_at": (slots[-1].get("ended_at") or "") if slots else i.get("ended_at") or "",
                        "duration_seconds": secs,
                        "duration_minutes": secs // 60,
                        "duration_label": H._format_duration_hm(secs),
                        "linked_to_employee_id": i.get("linked_to_employee_id"),
                        "linked_to_employee_name": linked,
                        "linked_person": linked or "Self-complete",
                        "remarks": i.get("remarks") or "",
                        "slots": slots,
                        "has_manual_slots": any(int(s.get("manual_edited") or 0) for s in slots),
                    }
                )
            for c in clone_items:
                rows.append(
                    {
                        **base,
                        "row_type": "backup_cover",
                        "responsibility_id": c.get("responsibility_id"),
                        "title": c.get("title"),
                        "frequency": c.get("frequency"),
                        "status": c.get("status"),
                        "approval_status": "",
                        "auto_approved": False,
                        "timer_status": "",
                        "started_at": "",
                        "ended_at": "",
                        "duration_seconds": 0,
                        "duration_minutes": 0,
                        "duration_label": H._format_duration_hm(0),
                        "linked_person": f"Cover for {c.get('original_employee_name') or ''}",
                        "remarks": c.get("remarks") or "",
                        "slots": [],
                        "has_manual_slots": False,
                    }
                )
            for t, day_slots in ot_items:
                secs = sum(_slot_net_in_window(s, d0, d1, now) for s in day_slots)
                linked = t.get("linked_to_employee_name") or ""
                rows.append(
                    {
                        **base,
                        "row_type": "one_time",
                        "task_id": t.get("id"),
                        "title": t.get("title"),
                        "frequency": "One-time",
                        "status": t.get("status"),
                        "approval_status": "Auto-Approved" if int(t.get("auto_approved") or 0) else (
                            "Approved" if t.get("status") == "Approved" else ""
                        ),
                        "auto_approved": bool(int(t.get("auto_approved") or 0)),
                        "timer_status": t.get("timer_status") or "",
                        "started_at": day_slots[0]["started_at"],
                        "ended_at": day_slots[-1].get("ended_at") or "",
                        "duration_seconds": secs,
                        "duration_minutes": secs // 60,
                        "duration_label": H._format_duration_hm(secs),
                        "linked_to_employee_id": t.get("linked_to_employee_id"),
                        "linked_to_employee_name": linked,
                        "linked_person": linked or (t.get("assigned_by") or ""),
                        "remarks": t.get("completion_notes") or "",
                        "slots": day_slots,
                        "has_manual_slots": any(int(s.get("manual_edited") or 0) for s in day_slots),
                    }
                )
    rows.sort(key=lambda r: (-int(r.get("duration_seconds") or 0), r.get("check_date") or "",
                             r.get("employee_name") or "", str(r.get("title") or "")))
    return {
        "from_date": days[0],
        "to_date": days[-1],
        "check_date": days[0],
        "rows": rows,
        "leave_rows": sorted(leave_rows, key=lambda r: (r["check_date"], r["employee_name"])),
        "total_seconds": sum(int(r.get("duration_seconds") or 0) for r in rows),
    }

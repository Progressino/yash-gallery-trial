"""HRM cross-department assignment workflow.

Anyone with ``can_assign_work`` may assign a Responsibility or a one-time Task to any active
employee. The work then passes through (as needed):

    pending_hod_review -> pending_employee_acceptance -> accepted
                     \\-> rejected_by_hod
    clarification_required  (HOD or employee asked the assigner a question)

Work that is not yet ``accepted`` is stored with ``active=0`` so every existing screen, timer,
daily check and report ignores it until the assignee accepts. Rows created before this feature
have no ``workflow_status`` and behave as accepted work.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable

from fastapi import HTTPException

from ..db import hrm_db as H

ENTITY_RESPONSIBILITY = "responsibility"
ENTITY_TASK = "task"
_TABLES = {ENTITY_RESPONSIBILITY: "responsibilities", ENTITY_TASK: "one_time_tasks"}

PENDING_HOD_REVIEW = "pending_hod_review"
CLARIFICATION_REQUIRED = "clarification_required"
PENDING_EMPLOYEE_ACCEPTANCE = "pending_employee_acceptance"
ACCEPTED = "accepted"
ON_HOLD = "on_hold"
CANCELLED = "cancelled"
REJECTED_BY_HOD = "rejected_by_hod"

WORKFLOW_STATUSES = (
    PENDING_HOD_REVIEW,
    CLARIFICATION_REQUIRED,
    PENDING_EMPLOYEE_ACCEPTANCE,
    ACCEPTED,
    ON_HOLD,
    CANCELLED,
    REJECTED_BY_HOD,
)
# Statuses where the work is not actionable yet (stored with active=0).
NOT_ACTIONABLE = (
    PENDING_HOD_REVIEW,
    CLARIFICATION_REQUIRED,
    PENDING_EMPLOYEE_ACCEPTANCE,
    REJECTED_BY_HOD,
    CANCELLED,
)
# Statuses that are still visible in tracker even though active=0.
_VISIBLE_INACTIVE = NOT_ACTIONABLE + (ON_HOLD,)

ACCEPTED_WITHOUT_CLARIFICATION = "accepted_without_clarification"
ACCEPTED_WITH_CLARIFICATION = "accepted_with_clarification"

VIEW_ASSIGNED_BY_ME = "assigned_by_me"
VIEW_ASSIGNED_TO_ME = "assigned_to_me"
VIEW_HOD_INBOX = "hod_inbox"

REMINDER_GAP_MINUTES = 60

_NEW_COLUMNS = (
    ("assigned_by_employee_id", "INTEGER"),
    ("assignee_hod_employee_id", "INTEGER"),
    ("workflow_status", "TEXT"),
    ("accepted_at", "TEXT DEFAULT ''"),
    ("acceptance_mode", "TEXT DEFAULT ''"),
    ("assign_remarks", "TEXT DEFAULT ''"),
    ("attachment_links", "TEXT DEFAULT ''"),
    ("voice_note_url", "TEXT DEFAULT ''"),
    ("hold_reason", "TEXT DEFAULT ''"),
    ("cancel_reason", "TEXT DEFAULT ''"),
    ("clarification_from", "TEXT DEFAULT ''"),
    ("workflow_updated_at", "TEXT DEFAULT ''"),
)


# ── Schema ───────────────────────────────────────────────────────────────────


def _columns(conn, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_schema(conn) -> None:
    """ALTER-safe migration. Called from ``hrm_db.init_db`` and idempotent."""
    for table in _TABLES.values():
        have = _columns(conn, table)
        for name, decl in _NEW_COLUMNS:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS hrm_assignment_events (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type       TEXT NOT NULL,
            entity_id         INTEGER NOT NULL,
            event_type        TEXT NOT NULL,
            actor_employee_id INTEGER,
            actor_name        TEXT DEFAULT '',
            remarks           TEXT DEFAULT '',
            created_at        TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_hrm_assign_events_entity
            ON hrm_assignment_events(entity_type, entity_id, id);
        CREATE TABLE IF NOT EXISTS hrm_assignment_notifications (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            to_employee_id INTEGER NOT NULL,
            entity_type    TEXT NOT NULL,
            entity_id      INTEGER NOT NULL,
            title          TEXT DEFAULT '',
            message        TEXT DEFAULT '',
            event_type     TEXT DEFAULT '',
            is_read        INTEGER DEFAULT 0,
            created_at     TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_hrm_assign_notif_to
            ON hrm_assignment_notifications(to_employee_id, is_read, id);
        """
    )


# ── Small helpers ────────────────────────────────────────────────────────────


def normalize_entity_type(value: str | None) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    if v in ("responsibility", "responsibilities", "resp", "r"):
        return ENTITY_RESPONSIBILITY
    if v in ("task", "tasks", "one_time", "one_time_task", "one_time_tasks", "t"):
        return ENTITY_TASK
    raise HTTPException(400, "entity_type must be 'responsibility' or 'task'")


def clean_attachment_links(links: Iterable[str] | str | None) -> list[str]:
    """Keep only http(s) URLs; raise ValueError for anything else the user typed."""
    if links is None:
        return []
    if isinstance(links, str):
        raw = [p for p in re.split(r"[\s,]+", links.strip()) if p]
    else:
        raw = [str(p).strip() for p in links if str(p or "").strip()]
    out: list[str] = []
    for url in raw:
        if not re.match(r"^https?://\S+$", url, re.IGNORECASE):
            raise ValueError(f"Attachment link must start with http:// or https:// — got '{url[:60]}'")
        if url not in out:
            out.append(url)
    return out[:10]


def parse_links(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    s = str(raw or "").strip()
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x) for x in val if x]
    except (TypeError, ValueError):
        pass
    return [s] if s.lower().startswith("http") else []


def _clean_voice_url(value: str | None) -> str:
    v = str(value or "").strip()
    if v and not re.match(r"^https?://\S+$", v, re.IGNORECASE):
        raise ValueError("Voice note link must start with http:// or https://")
    return v


def effective_status(row: dict) -> str:
    """Legacy rows (no workflow_status) count as accepted work."""
    return str(row.get("workflow_status") or "").strip() or ACCEPTED


def _employee_name(conn, employee_id: int | None) -> str:
    if not employee_id:
        return ""
    row = conn.execute("SELECT name FROM employees WHERE id=?", (int(employee_id),)).fetchone()
    return str(row["name"]) if row else ""


def _hod_users_for_department(department_id: int) -> list[int]:
    """employee_id of every active HOD user linked to this HRM department."""
    try:
        from ..db import users_db

        conn = users_db._connect()
        try:
            rows = conn.execute(
                """SELECT u.employee_id FROM erp_users u
                   JOIN roles r ON r.id=u.role_id
                   WHERE r.role_name='HOD' AND u.active=1
                     AND u.hrm_department_id=? AND u.employee_id IS NOT NULL
                   ORDER BY u.id""",
                (int(department_id),),
            ).fetchall()
        finally:
            conn.close()
        return [int(r["employee_id"]) for r in rows if r["employee_id"]]
    except Exception:
        return []


def resolve_assignee_hod_employee_id(employee_id: int) -> int | None:
    """HOD of the assignee's department.

    1. ERP user with role HOD and the same ``hrm_department_id`` -> its employee_id
    2. otherwise the department's ``hod_name`` matched to an active employee
    3. otherwise None
    """
    conn = H._connect()
    try:
        row = conn.execute("SELECT department_id FROM employees WHERE id=?", (int(employee_id),)).fetchone()
        dept_id = row["department_id"] if row else None
        if not dept_id:
            return None
        hods = _hod_users_for_department(int(dept_id))
        if hods:
            return hods[0]
        d = conn.execute("SELECT hod_name FROM departments WHERE id=?", (int(dept_id),)).fetchone()
        name = str((d["hod_name"] if d else "") or "").strip().lower()
        if not name:
            return None
        cands = conn.execute(
            "SELECT id, department_id FROM employees WHERE status='Active' AND LOWER(TRIM(name))=? ORDER BY id",
            (name,),
        ).fetchall()
        for c in cands:
            if c["department_id"] and int(c["department_id"]) == int(dept_id):
                return int(c["id"])
        return int(cands[0]["id"]) if cands else None
    finally:
        conn.close()


def _log_event(
    conn,
    et: str,
    eid: int,
    event_type: str,
    *,
    actor_employee_id: int | None,
    actor_name: str = "",
    remarks: str = "",
) -> None:
    conn.execute(
        """INSERT INTO hrm_assignment_events(
            entity_type, entity_id, event_type, actor_employee_id, actor_name, remarks, created_at
        ) VALUES (?,?,?,?,?,?,?)""",
        (
            et,
            int(eid),
            event_type,
            int(actor_employee_id) if actor_employee_id else None,
            actor_name or _employee_name(conn, actor_employee_id),
            remarks or "",
            H._now_iso(),
        ),
    )


def _notify(
    conn,
    to_employee_id: int | None,
    et: str,
    eid: int,
    *,
    title: str,
    message: str,
    event_type: str,
) -> None:
    if not to_employee_id:
        return
    conn.execute(
        """INSERT INTO hrm_assignment_notifications(
            to_employee_id, entity_type, entity_id, title, message, event_type, is_read, created_at
        ) VALUES (?,?,?,?,?,?,0,?)""",
        (int(to_employee_id), et, int(eid), title, message, event_type, H._now_iso()),
    )


def _load(conn, et: str, eid: int) -> dict:
    table = _TABLES[et]
    row = conn.execute(
        f"""SELECT x.*, e.name AS employee_name, e.department_id AS employee_department_id
            FROM {table} x LEFT JOIN employees e ON e.id=x.employee_id WHERE x.id=?""",
        (int(eid),),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Assignment not found")
    d = dict(row)
    d["entity_type"] = et
    return d


def _set_status(conn, et: str, eid: int, status: str, **cols: Any) -> None:
    """Update workflow_status and keep ``active`` in step (only accepted work is active)."""
    table = _TABLES[et]
    sets = ["workflow_status=?", "workflow_updated_at=?", "active=?"]
    params: list[Any] = [status, H._now_iso(), 0 if status in NOT_ACTIONABLE else 1]
    for k, v in cols.items():
        sets.append(f"{k}=?")
        params.append(v)
    params.append(int(eid))
    conn.execute(f"UPDATE {table} SET {', '.join(sets)} WHERE id=?", params)


def _has_event(conn, et: str, eid: int, event_type: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM hrm_assignment_events WHERE entity_type=? AND entity_id=? AND event_type=? LIMIT 1",
            (et, int(eid), event_type),
        ).fetchone()
        is not None
    )


def _require_remarks(remarks: str | None, what: str) -> str:
    r = str(remarks or "").strip()
    if not r:
        raise HTTPException(400, f"{what} is required")
    return r


def _state_error(row: dict, expected: str) -> HTTPException:
    return HTTPException(
        409, f"This assignment is '{effective_status(row)}' — expected '{expected}'"
    )


def _title(row: dict) -> str:
    return str(row.get("title") or "Assignment")


def _kind_label(et: str) -> str:
    return "responsibility" if et == ENTITY_RESPONSIBILITY else "task"


# ── Workflow start ───────────────────────────────────────────────────────────


def resolve_assigned_by(
    *,
    assigned_by_employee_id: int | None = None,
    assigned_by_name: str = "",
    fallback_employee_id: int | None = None,
    fallback_name: str = "",
) -> tuple[int | None, str]:
    """Resolve the person shown as Assigned By (may differ from the login that created the row).

    Order: explicit employee id → exact active employee name match → login fallback.
    """
    name = (assigned_by_name or "").strip()
    if assigned_by_employee_id:
        try:
            eid = int(assigned_by_employee_id)
        except (TypeError, ValueError):
            eid = 0
        if eid > 0:
            conn = H._connect()
            try:
                row = conn.execute(
                    "SELECT id, name FROM employees WHERE id=? AND status='Active'", (eid,)
                ).fetchone()
            finally:
                conn.close()
            if row:
                return int(row["id"]), str(row["name"] or name or "").strip()
    if name:
        conn = H._connect()
        try:
            row = conn.execute(
                """SELECT id, name FROM employees
                   WHERE status='Active' AND LOWER(TRIM(name))=LOWER(TRIM(?))
                   LIMIT 1""",
                (name,),
            ).fetchone()
        finally:
            conn.close()
        if row:
            return int(row["id"]), str(row["name"] or name).strip()
    fb_name = (fallback_name or name or "").strip()
    if fallback_employee_id:
        try:
            return int(fallback_employee_id), fb_name
        except (TypeError, ValueError):
            pass
    return None, fb_name


def start_assignment_workflow(
    entity_type: str,
    entity_id: int,
    *,
    assigner_employee_id: int | None,
    assigner_name: str = "",
    is_org_manager: bool = False,
    remarks: str = "",
    attachment_links: Iterable[str] | None = None,
    voice_note_url: str = "",
    created_by_name: str = "",
    created_by_employee_id: int | None = None,
) -> dict:
    """Call right after a responsibility/task is created.

    Org managers (Admin / Super Admin / Sir) outrank HODs, so their work is accepted straight
    away (still logged and attributed). Otherwise: assigner is the assignee's HOD, assignee is the
    HOD, or the department has no HOD -> assignee acceptance; else the HOD reviews first.
    """
    et = normalize_entity_type(entity_type)
    links = clean_attachment_links(attachment_links)
    voice = _clean_voice_url(voice_note_url)
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        assignee_id = int(row["employee_id"])
        hod_id = resolve_assignee_hod_employee_id(assignee_id)
        if assigner_employee_id and not assigner_name:
            assigner_name = _employee_name(conn, assigner_employee_id)

        if is_org_manager:
            status = ACCEPTED
        elif hod_id is None or hod_id == assignee_id or (
            assigner_employee_id and int(assigner_employee_id) == int(hod_id)
        ):
            status = PENDING_EMPLOYEE_ACCEPTANCE
        else:
            status = PENDING_HOD_REVIEW

        _set_status(
            conn,
            et,
            entity_id,
            status,
            assigned_by_employee_id=int(assigner_employee_id) if assigner_employee_id else None,
            assignee_hod_employee_id=hod_id,
            assign_remarks=str(remarks or "").strip(),
            attachment_links=json.dumps(links) if links else "",
            voice_note_url=voice,
            accepted_at=H._now_iso() if status == ACCEPTED else "",
        )
        _log_event(
            conn,
            et,
            entity_id,
            "created",
            actor_employee_id=assigner_employee_id,
            actor_name=assigner_name,
            remarks=str(remarks or "").strip(),
        )
        creator = (created_by_name or "").strip()
        if creator and (
            not assigner_name
            or creator.lower() != assigner_name.lower()
            or (
                created_by_employee_id
                and assigner_employee_id
                and int(created_by_employee_id) != int(assigner_employee_id)
            )
            or (created_by_employee_id and not assigner_employee_id)
        ):
            _log_event(
                conn,
                et,
                entity_id,
                "created_by_login",
                actor_employee_id=created_by_employee_id,
                actor_name=creator,
                remarks=(
                    f"Created by {creator}"
                    + (f" (Assigned By: {assigner_name})" if assigner_name else "")
                ),
            )
        title = _title(row)
        kind = _kind_label(et)
        who = assigner_name or "Management"
        if status == PENDING_HOD_REVIEW:
            _notify(
                conn, hod_id, et, entity_id,
                title="Assignment needs your review",
                message=f"{who} assigned the {kind} '{title}' to {row['employee_name']}. Please review and forward.",
                event_type="hod_review_requested",
            )
        elif status == PENDING_EMPLOYEE_ACCEPTANCE:
            _notify(
                conn, assignee_id, et, entity_id,
                title="New work assigned to you",
                message=f"{who} assigned you the {kind} '{title}'. Please accept or ask for an explanation.",
                event_type="acceptance_requested",
            )
        else:
            _notify(
                conn, assignee_id, et, entity_id,
                title="New work assigned to you",
                message=f"{who} assigned you the {kind} '{title}'.",
                event_type="assigned",
            )
        conn.commit()
    finally:
        conn.close()
    if status != ACCEPTED and et == ENTITY_RESPONSIBILITY:
        _resync_leave_backups(entity_id)
    return {"workflow_status": status, "assignee_hod_employee_id": hod_id}


def _resync_leave_backups(rid: int) -> None:
    try:
        H.sync_leave_backups_for_responsibility(int(rid))
    except Exception:
        pass


# ── Transitions ──────────────────────────────────────────────────────────────


def _is_hod_actor(row: dict, actor_employee_id: int | None, is_org_manager: bool) -> bool:
    if is_org_manager:
        return True
    hod = row.get("assignee_hod_employee_id")
    return bool(actor_employee_id and hod and int(hod) == int(actor_employee_id))


def _is_assigner(row: dict, actor_employee_id: int | None, is_org_manager: bool) -> bool:
    if is_org_manager:
        return True
    by = row.get("assigned_by_employee_id")
    return bool(actor_employee_id and by and int(by) == int(actor_employee_id))


def hod_forward(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    et = normalize_entity_type(entity_type)
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        if effective_status(row) != PENDING_HOD_REVIEW:
            raise _state_error(row, PENDING_HOD_REVIEW)
        if not _is_hod_actor(row, actor_employee_id, is_org_manager):
            raise HTTPException(403, "Only the assignee's HOD can forward this assignment")
        _set_status(conn, et, entity_id, PENDING_EMPLOYEE_ACCEPTANCE, clarification_from="")
        _log_event(conn, et, entity_id, "hod_forwarded", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=remarks)
        title, kind = _title(row), _kind_label(et)
        _notify(conn, row["employee_id"], et, entity_id,
                title="New work assigned to you",
                message=f"Your HOD forwarded the {kind} '{title}' to you. Please accept or ask for an explanation.",
                event_type="acceptance_requested")
        _notify(conn, row.get("assigned_by_employee_id"), et, entity_id,
                title="Assignment forwarded by HOD",
                message=f"HOD forwarded your {kind} '{title}' to {row['employee_name']}.",
                event_type="hod_forwarded")
        conn.commit()
    finally:
        conn.close()
    return {"workflow_status": PENDING_EMPLOYEE_ACCEPTANCE}


def hod_reject(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    et = normalize_entity_type(entity_type)
    reason = _require_remarks(remarks, "Reason for rejection")
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        if effective_status(row) != PENDING_HOD_REVIEW:
            raise _state_error(row, PENDING_HOD_REVIEW)
        if not _is_hod_actor(row, actor_employee_id, is_org_manager):
            raise HTTPException(403, "Only the assignee's HOD can reject this assignment")
        _set_status(conn, et, entity_id, REJECTED_BY_HOD, cancel_reason=reason)
        _log_event(conn, et, entity_id, "hod_rejected", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=reason)
        _notify(conn, row.get("assigned_by_employee_id"), et, entity_id,
                title="Assignment rejected by HOD",
                message=f"HOD rejected your {_kind_label(et)} '{_title(row)}': {reason}",
                event_type="hod_rejected")
        conn.commit()
    finally:
        conn.close()
    return {"workflow_status": REJECTED_BY_HOD}


def employee_accept(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    remarks: str = "",
) -> dict:
    et = normalize_entity_type(entity_type)
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        if effective_status(row) != PENDING_EMPLOYEE_ACCEPTANCE:
            raise _state_error(row, PENDING_EMPLOYEE_ACCEPTANCE)
        if not actor_employee_id or int(actor_employee_id) != int(row["employee_id"]):
            raise HTTPException(403, "Only the assigned employee can accept this work")
        mode = (
            ACCEPTED_WITH_CLARIFICATION
            if _has_event(conn, et, entity_id, "clarification_requested")
            else ACCEPTED_WITHOUT_CLARIFICATION
        )
        _set_status(conn, et, entity_id, ACCEPTED, accepted_at=H._now_iso(), acceptance_mode=mode,
                    clarification_from="")
        _log_event(conn, et, entity_id, "accepted", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=remarks)
        msg = f"{row['employee_name']} accepted your {_kind_label(et)} '{_title(row)}'."
        for target in {row.get("assigned_by_employee_id"), row.get("assignee_hod_employee_id")}:
            if target and int(target) != int(actor_employee_id):
                _notify(conn, target, et, entity_id, title="Assignment accepted", message=msg,
                        event_type="accepted")
        conn.commit()
    finally:
        conn.close()
    if et == ENTITY_RESPONSIBILITY:
        _resync_leave_backups(entity_id)
    return {"workflow_status": ACCEPTED, "acceptance_mode": mode}


def request_clarification(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    """HOD (while reviewing) or the assignee (while accepting) asks the assigner to explain."""
    et = normalize_entity_type(entity_type)
    question = _require_remarks(remarks, "Your question / explanation request")
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        status = effective_status(row)
        if status == PENDING_HOD_REVIEW:
            if not _is_hod_actor(row, actor_employee_id, is_org_manager):
                raise HTTPException(403, "Only the assignee's HOD can ask for clarification here")
            source = "hod"
        elif status == PENDING_EMPLOYEE_ACCEPTANCE:
            if not actor_employee_id or int(actor_employee_id) != int(row["employee_id"]):
                raise HTTPException(403, "Only the assigned employee can ask for clarification here")
            source = "employee"
        else:
            raise HTTPException(409, f"Cannot request clarification while '{status}'")
        _set_status(conn, et, entity_id, CLARIFICATION_REQUIRED, clarification_from=source)
        _log_event(conn, et, entity_id, "clarification_requested", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=question)
        who = "HOD" if source == "hod" else str(row["employee_name"])
        _notify(conn, row.get("assigned_by_employee_id"), et, entity_id,
                title="Explanation requested",
                message=f"{who} needs an explanation for the {_kind_label(et)} '{_title(row)}': {question}",
                event_type="clarification_requested")
        conn.commit()
    finally:
        conn.close()
    return {"workflow_status": CLARIFICATION_REQUIRED, "clarification_from": source}


def employee_request_clarification(entity_type: str, entity_id: int, **kw: Any) -> dict:
    return request_clarification(entity_type, entity_id, **kw)


def assigner_reply_clarification(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    et = normalize_entity_type(entity_type)
    answer = _require_remarks(remarks, "Explanation")
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        if effective_status(row) != CLARIFICATION_REQUIRED:
            raise _state_error(row, CLARIFICATION_REQUIRED)
        if not _is_assigner(row, actor_employee_id, is_org_manager):
            raise HTTPException(403, "Only the person who assigned this work can reply")
        back_to_hod = str(row.get("clarification_from") or "") == "hod" and row.get("assignee_hod_employee_id")
        status = PENDING_HOD_REVIEW if back_to_hod else PENDING_EMPLOYEE_ACCEPTANCE
        _set_status(conn, et, entity_id, status, clarification_from="")
        _log_event(conn, et, entity_id, "clarification_replied", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=answer)
        target = row.get("assignee_hod_employee_id") if status == PENDING_HOD_REVIEW else row["employee_id"]
        _notify(conn, target, et, entity_id,
                title="Explanation received",
                message=f"{actor_name or 'The assigner'} replied on the {_kind_label(et)} '{_title(row)}': {answer}",
                event_type="clarification_replied")
        conn.commit()
    finally:
        conn.close()
    return {"workflow_status": status}


def remind_hod(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    """Assigner nudges whoever the assignment is waiting on (HOD review or employee acceptance)."""
    et = normalize_entity_type(entity_type)
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        status = effective_status(row)
        if status not in (PENDING_HOD_REVIEW, PENDING_EMPLOYEE_ACCEPTANCE):
            raise HTTPException(409, "Reminder is only for assignments waiting on HOD review or acceptance")
        if not _is_assigner(row, actor_employee_id, is_org_manager):
            raise HTTPException(403, "Only the person who assigned this work can send a reminder")
        last = conn.execute(
            """SELECT created_at FROM hrm_assignment_events
               WHERE entity_type=? AND entity_id=? AND event_type='reminder' ORDER BY id DESC LIMIT 1""",
            (et, int(entity_id)),
        ).fetchone()
        if last and last["created_at"]:
            try:
                gap = (
                    datetime.strptime(H._now_iso()[:19], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(str(last["created_at"])[:19], "%Y-%m-%d %H:%M:%S")
                ).total_seconds() / 60
                if gap < REMINDER_GAP_MINUTES:
                    raise HTTPException(429, "A reminder was sent recently — try again later")
            except ValueError:
                pass
        target = row.get("assignee_hod_employee_id") if status == PENDING_HOD_REVIEW else row["employee_id"]
        if not target:
            raise HTTPException(409, "No one to remind for this assignment")
        _log_event(conn, et, entity_id, "reminder", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=remarks)
        _notify(conn, target, et, entity_id,
                title="Reminder: assignment pending",
                message=f"{actor_name or 'The assigner'} is waiting on you for the {_kind_label(et)} "
                        f"'{_title(row)}'.",
                event_type="reminder")
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "reminded_employee_id": int(target)}


def cancel_assignment(
    entity_type: str, entity_id: int, *, actor_employee_id: int | None, actor_name: str = "",
    is_org_manager: bool = False, remarks: str = "",
) -> dict:
    """Assigner withdraws work that has not been accepted yet."""
    et = normalize_entity_type(entity_type)
    reason = _require_remarks(remarks, "Cancel reason")
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        status = effective_status(row)
        if status not in (PENDING_HOD_REVIEW, CLARIFICATION_REQUIRED, PENDING_EMPLOYEE_ACCEPTANCE):
            raise HTTPException(409, f"Cannot cancel an assignment that is '{status}'")
        if not _is_assigner(row, actor_employee_id, is_org_manager):
            raise HTTPException(403, "Only the person who assigned this work can cancel it")
        _set_status(conn, et, entity_id, CANCELLED, cancel_reason=reason)
        _log_event(conn, et, entity_id, "cancelled", actor_employee_id=actor_employee_id,
                   actor_name=actor_name, remarks=reason)
        for target in {row["employee_id"], row.get("assignee_hod_employee_id")}:
            if target and int(target) != int(actor_employee_id or 0):
                _notify(conn, target, et, entity_id, title="Assignment cancelled",
                        message=f"The {_kind_label(et)} '{_title(row)}' was cancelled: {reason}",
                        event_type="cancelled")
        conn.commit()
    finally:
        conn.close()
    return {"workflow_status": CANCELLED}


def mark_cancelled_if_open(entity_type: str, entity_id: int) -> None:
    """Called when an Admin deletes a record that was still waiting in the workflow."""
    et = normalize_entity_type(entity_type)
    conn = H._connect()
    try:
        row = conn.execute(
            f"SELECT workflow_status FROM {_TABLES[et]} WHERE id=?", (int(entity_id),)
        ).fetchone()
        if row and str(row["workflow_status"] or "") in (
            PENDING_HOD_REVIEW, CLARIFICATION_REQUIRED, PENDING_EMPLOYEE_ACCEPTANCE
        ):
            _set_status(conn, et, entity_id, CANCELLED, cancel_reason="Deleted by Admin")
            conn.commit()
    finally:
        conn.close()


# ── Tracker / lists ──────────────────────────────────────────────────────────

_OPEN_TASK_STATUSES = ("Pending", "In Progress", "Rejected")


def _pending_with(row: dict) -> tuple[str, int | None]:
    """(role label, employee_id) of whoever must act next."""
    st = effective_status(row)
    if st == PENDING_HOD_REVIEW:
        return "HOD", row.get("assignee_hod_employee_id")
    if st == PENDING_EMPLOYEE_ACCEPTANCE:
        return "Assignee", row.get("employee_id")
    if st == CLARIFICATION_REQUIRED:
        return "Assigner", row.get("assigned_by_employee_id")
    if st == ACCEPTED and row.get("entity_type") == ENTITY_TASK:
        ts = str(row.get("task_status") or "")
        if ts == "Done":
            return "Approver", row.get("linked_to_employee_id") or row.get("assigned_by_employee_id")
        if ts in _OPEN_TASK_STATUSES:
            return "Assignee", row.get("employee_id")
    return "", None


def _due_label(row: dict, today: str) -> str:
    if effective_status(row) != ACCEPTED or row.get("entity_type") != ENTITY_TASK:
        return ""
    due = str(row.get("deadline") or "")[:10]
    if not due or str(row.get("task_status") or "") not in _OPEN_TASK_STATUSES:
        return ""
    if due < today:
        return "Overdue"
    if due == today:
        return "Due Today"
    return ""


def _actions(row: dict, viewer_id: int | None, org: bool) -> dict:
    st = effective_status(row)
    vid = int(viewer_id) if viewer_id else None
    is_hod = _is_hod_actor(row, vid, org)
    is_assignee = bool(vid and int(row["employee_id"]) == vid)
    is_assigner = _is_assigner(row, vid, org)
    return {
        "can_forward": st == PENDING_HOD_REVIEW and is_hod,
        "can_reject": st == PENDING_HOD_REVIEW and is_hod,
        "can_hod_clarify": st == PENDING_HOD_REVIEW and is_hod,
        "can_accept": st == PENDING_EMPLOYEE_ACCEPTANCE and is_assignee,
        "can_clarify": st == PENDING_EMPLOYEE_ACCEPTANCE and is_assignee,
        "can_reply": st == CLARIFICATION_REQUIRED and is_assigner,
        "can_remind": st in (PENDING_HOD_REVIEW, PENDING_EMPLOYEE_ACCEPTANCE) and is_assigner,
        "can_cancel": st in (PENDING_HOD_REVIEW, CLARIFICATION_REQUIRED, PENDING_EMPLOYEE_ACCEPTANCE)
        and is_assigner,
    }


_BASE_SELECT = """
    SELECT x.id, x.employee_id, x.title, x.description, x.priority, x.active, x.created_at,
           x.assigned_by_employee_id, x.assignee_hod_employee_id, x.workflow_status,
           x.accepted_at, x.acceptance_mode, x.assign_remarks, x.attachment_links, x.voice_note_url,
           x.hold_reason, x.cancel_reason, x.clarification_from, x.linked_to_employee_id,
           {extra},
           e.name AS employee_name, COALESCE(x.department_id, e.department_id) AS department_id,
           d.name AS department_name, ab.name AS assigned_by_name, hh.name AS hod_name,
           le.name AS linked_to_name
    FROM {table} x
    LEFT JOIN employees e ON e.id=x.employee_id
    LEFT JOIN departments d ON d.id=COALESCE(x.department_id, e.department_id)
    LEFT JOIN employees ab ON ab.id=x.assigned_by_employee_id
    LEFT JOIN employees hh ON hh.id=x.assignee_hod_employee_id
    LEFT JOIN employees le ON le.id=x.linked_to_employee_id
"""

_VISIBLE_SQL = (
    "(x.active=1 OR COALESCE(x.workflow_status,'') IN ("
    + ",".join(f"'{s}'" for s in _VISIBLE_INACTIVE)
    + "))"
)


def _fetch_rows(conn, *, side: str, viewer_id: int | None, org: bool) -> list[dict]:
    """Raw unified rows for the view (side = by | to | hod)."""
    out: list[dict] = []
    for et, table in _TABLES.items():
        if et == ENTITY_RESPONSIBILITY:
            extra = "x.frequency AS frequency, '' AS deadline, '' AS task_status"
        else:
            extra = "'' AS frequency, x.due_date AS deadline, x.status AS task_status"
        where = [_VISIBLE_SQL]
        params: list[Any] = []
        if side == "by":
            where.append("x.assigned_by_employee_id=?")
            params.append(int(viewer_id or -1))
        elif side == "to":
            where.append("x.employee_id=?")
            params.append(int(viewer_id or -1))
        else:  # hod inbox
            if org and not viewer_id:
                where.append("COALESCE(x.workflow_status,'') IN ('pending_hod_review','clarification_required')")
            else:
                where.append("x.assignee_hod_employee_id=?")
                params.append(int(viewer_id or -1))
        rows = conn.execute(
            _BASE_SELECT.format(extra=extra, table=table) + " WHERE " + " AND ".join(where), params
        ).fetchall()
        for r in rows:
            d = dict(r)
            d["entity_type"] = et
            out.append(d)
    return out


def _fmt_work_seconds(sec: int) -> str:
    s = max(0, int(sec or 0))
    h, rem = divmod(s, 3600)
    m = rem // 60
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _work_metrics(conn, row: dict) -> dict:
    """Working time / notes / approval status for accepted tracker rows."""
    et = row.get("entity_type")
    eid = int(row.get("id") or 0)
    out = {
        "total_working_seconds": 0,
        "total_working_time_label": "—",
        "time_slot_count": 0,
        "notes": "",
        "approval_status": "",
    }
    try:
        if et == ENTITY_TASK:
            r = conn.execute(
                """SELECT COALESCE(active_seconds,0) AS sec,
                          COALESCE(NULLIF(completion_notes,''), approval_notes, '') AS notes,
                          status
                   FROM one_time_tasks WHERE id=?""",
                (eid,),
            ).fetchone()
            if r:
                out["total_working_seconds"] = int(r["sec"] or 0)
                out["notes"] = str(r["notes"] or "")
                out["approval_status"] = str(r["status"] or "")
            try:
                n = conn.execute(
                    "SELECT COUNT(*) FROM hrm_time_slots WHERE entity_type='one_time' AND entity_id=?",
                    (eid,),
                ).fetchone()
                out["time_slot_count"] = int((n[0] if n else 0) or 0)
            except Exception:
                out["time_slot_count"] = 0
        else:
            r = conn.execute(
                """SELECT COALESCE(SUM(duration_seconds),0) FROM hrm_time_slots
                   WHERE responsibility_id=? OR (entity_type='responsibility' AND entity_id=?)""",
                (eid, eid),
            ).fetchone()
            out["total_working_seconds"] = int((r[0] if r else 0) or 0)
            n = conn.execute(
                """SELECT COUNT(*) FROM hrm_time_slots
                   WHERE responsibility_id=? OR (entity_type='responsibility' AND entity_id=?)""",
                (eid, eid),
            ).fetchone()
            out["time_slot_count"] = int((n[0] if n else 0) or 0)
            last = conn.execute(
                """SELECT remarks, approval_status, status FROM task_logs
                   WHERE responsibility_id=? ORDER BY log_date DESC, id DESC LIMIT 1""",
                (eid,),
            ).fetchone()
            if last:
                out["notes"] = str(last["remarks"] or "")
                out["approval_status"] = str(last["approval_status"] or last["status"] or "")
    except Exception:
        pass
    if out["total_working_seconds"]:
        out["total_working_time_label"] = _fmt_work_seconds(out["total_working_seconds"])
    elif out["time_slot_count"]:
        out["total_working_time_label"] = f"{out['time_slot_count']} slots"
    return out


def _shape(row: dict, viewer_id: int | None, org: bool, today: str, names: dict[int, str], work: dict | None = None) -> dict:
    label, pw_id = _pending_with(row)
    freq = str(row.get("frequency") or "")
    work = work or {}
    return {
        "entity_type": row["entity_type"],
        "id": int(row["id"]),
        "employee_id": row["employee_id"],
        "employee_name": row.get("employee_name") or "",
        "department_id": row.get("department_id"),
        "department_name": row.get("department_name") or "",
        "title": row.get("title") or "",
        "description": row.get("description") or "",
        "type": "Responsibility" if row["entity_type"] == ENTITY_RESPONSIBILITY else "Task",
        "frequency": freq,
        "assigned_by_employee_id": row.get("assigned_by_employee_id"),
        "assigned_by_name": row.get("assigned_by_name") or "",
        "assign_date": str(row.get("created_at") or "")[:10],
        "deadline": str(row.get("deadline") or "")[:10],
        "priority": row.get("priority") or "Medium",
        "workflow_status": effective_status(row),
        "task_status": row.get("task_status") or "",
        "status": effective_status(row)
        if effective_status(row) != ACCEPTED or not row.get("task_status")
        else row["task_status"],
        "pending_with_role": label,
        "pending_with_employee_id": pw_id,
        "pending_with": names.get(int(pw_id), "") if pw_id else "",
        "due_state": _due_label(row, today),
        "hod_employee_id": row.get("assignee_hod_employee_id"),
        "hod_name": row.get("hod_name") or "",
        "linked_to_employee_id": row.get("linked_to_employee_id"),
        "linked_to_name": row.get("linked_to_name") or "",
        "assign_remarks": row.get("assign_remarks") or "",
        "attachment_links": parse_links(row.get("attachment_links")),
        "voice_note_url": row.get("voice_note_url") or "",
        "acceptance_mode": row.get("acceptance_mode") or "",
        "accepted_at": row.get("accepted_at") or "",
        "cancel_reason": row.get("cancel_reason") or "",
        "clarification_from": row.get("clarification_from") or "",
        "total_working_seconds": work.get("total_working_seconds") or 0,
        "total_working_time_label": work.get("total_working_time_label") or "—",
        "time_slot_count": work.get("time_slot_count") or 0,
        "notes": work.get("notes") or "",
        "approval_status": work.get("approval_status") or "",
        "actions": _actions(row, viewer_id, org),
    }


def _load_names(conn, rows: list[dict]) -> dict[int, str]:
    ids = {int(_pending_with(r)[1]) for r in rows if _pending_with(r)[1]}
    names: dict[int, str] = {}
    if not ids:
        return names
    marks = ",".join("?" * len(ids))
    for r in conn.execute(f"SELECT id, name FROM employees WHERE id IN ({marks})", list(ids)).fetchall():
        names[int(r["id"])] = str(r["name"])
    return names


def list_tracker(
    *,
    view: str,
    viewer_employee_id: int | None,
    is_org_manager: bool = False,
    status: str | None = None,
    entity_type: str | None = None,
    priority: str | None = None,
    department_id: int | None = None,
    employee_id: int | None = None,
    overdue: bool = False,
    q: str | None = None,
) -> dict:
    """Unified tracker rows + summary counts for a view.

    Only the viewer's own side is returned: Assigned By Me shows what *they* assigned (any
    department) and nothing else about those employees.
    """
    side = {VIEW_ASSIGNED_BY_ME: "by", VIEW_ASSIGNED_TO_ME: "to", VIEW_HOD_INBOX: "hod"}.get(view)
    if side is None:
        raise HTTPException(400, "view must be assigned_by_me, assigned_to_me or hod_inbox")
    if not viewer_employee_id and not (side == "hod" and is_org_manager):
        return {"view": view, "rows": [], "summary": _empty_summary()}
    today = H.today_ist().isoformat()
    conn = H._connect()
    try:
        raw = _fetch_rows(conn, side=side, viewer_id=viewer_employee_id, org=is_org_manager)
        # legacy rows only matter as the assignee's own accepted work
        if side != "to":
            raw = [r for r in raw if str(r.get("workflow_status") or "").strip()]
        names = _load_names(conn, raw)
        shaped = []
        for r in raw:
            work = _work_metrics(conn, r) if effective_status(r) == ACCEPTED else {}
            shaped.append(_shape(r, viewer_employee_id, is_org_manager, today, names, work))
    finally:
        conn.close()
    summary = _summarize(shaped)
    et = normalize_entity_type(entity_type) if entity_type else None
    needle = (q or "").strip().lower()
    rows = []
    for r in shaped:
        if et and r["entity_type"] != et:
            continue
        if status and status != "all":
            if status == "overdue":
                if r["due_state"] != "Overdue":
                    continue
            elif r["workflow_status"] != status:
                continue
        if overdue and r["due_state"] != "Overdue":
            continue
        if priority and r["priority"] != priority:
            continue
        if department_id and int(r["department_id"] or 0) != int(department_id):
            continue
        if employee_id and int(r["employee_id"] or 0) != int(employee_id):
            continue
        if needle and needle not in " ".join(
            str(r[k]).lower() for k in ("title", "employee_name", "assigned_by_name", "department_name")
        ):
            continue
        rows.append(r)
    rows.sort(key=lambda r: (r["assign_date"], r["id"]), reverse=True)
    return {"view": view, "rows": rows, "summary": summary}


def _empty_summary() -> dict:
    s = {k: 0 for k in WORKFLOW_STATUSES}
    s.update(total=0, overdue=0, needs_my_action=0)
    return s


def _summarize(rows: list[dict]) -> dict:
    s = _empty_summary()
    for r in rows:
        s["total"] += 1
        s[r["workflow_status"]] = s.get(r["workflow_status"], 0) + 1
        if r["due_state"] == "Overdue":
            s["overdue"] += 1
        if any(r["actions"][k] for k in ("can_forward", "can_accept", "can_reply")):
            s["needs_my_action"] += 1
    return s


def summary(*, viewer_employee_id: int | None, is_org_manager: bool = False) -> dict:
    """Counts for all three views plus unread notifications (for tab badges)."""
    out: dict[str, Any] = {}
    for view in (VIEW_ASSIGNED_TO_ME, VIEW_ASSIGNED_BY_ME, VIEW_HOD_INBOX):
        out[view] = list_tracker(
            view=view, viewer_employee_id=viewer_employee_id, is_org_manager=is_org_manager
        )["summary"]
    out["unread_notifications"] = unread_count(viewer_employee_id)
    return out


def list_assigned_by_me(entity_type: str, employee_id: int | None) -> list[dict]:
    """Same row shape as the normal list endpoints, restricted to what this person assigned."""
    et = normalize_entity_type(entity_type)
    if not employee_id:
        return []
    if et == ENTITY_RESPONSIBILITY:
        rows = H.list_responsibilities(active_only=False)
    else:
        rows = H.list_one_time_tasks(active_only=False, include_hold=True)
    out = []
    for r in rows:
        if not r.get("assigned_by_employee_id") or int(r["assigned_by_employee_id"]) != int(employee_id):
            continue
        ws = str(r.get("workflow_status") or "")
        if not int(r.get("active") or 0) and ws not in _VISIBLE_INACTIVE:
            continue
        r["workflow_status"] = effective_status(r)
        r["attachment_links_list"] = parse_links(r.get("attachment_links"))
        out.append(r)
    return out


# ── Notifications & history ──────────────────────────────────────────────────


def unread_count(employee_id: int | None) -> int:
    if not employee_id:
        return 0
    conn = H._connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM hrm_assignment_notifications WHERE to_employee_id=? AND is_read=0",
            (int(employee_id),),
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()


def list_notifications(employee_id: int | None, *, unread_only: bool = False, limit: int = 50) -> list[dict]:
    if not employee_id:
        return []
    conn = H._connect()
    try:
        sql = "SELECT * FROM hrm_assignment_notifications WHERE to_employee_id=?"
        params: list[Any] = [int(employee_id)]
        if unread_only:
            sql += " AND is_read=0"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def mark_notifications_read(employee_id: int | None, notification_id: int | None = None) -> int:
    if not employee_id:
        return 0
    conn = H._connect()
    try:
        if notification_id:
            cur = conn.execute(
                "UPDATE hrm_assignment_notifications SET is_read=1 WHERE to_employee_id=? AND id=?",
                (int(employee_id), int(notification_id)),
            )
        else:
            cur = conn.execute(
                "UPDATE hrm_assignment_notifications SET is_read=1 WHERE to_employee_id=? AND is_read=0",
                (int(employee_id),),
            )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def list_history(
    entity_type: str, entity_id: int, *, viewer_employee_id: int | None, is_org_manager: bool = False
) -> list[dict]:
    et = normalize_entity_type(entity_type)
    conn = H._connect()
    try:
        row = _load(conn, et, entity_id)
        involved = {
            row.get("employee_id"),
            row.get("assigned_by_employee_id"),
            row.get("assignee_hod_employee_id"),
            row.get("linked_to_employee_id"),
        }
        if not is_org_manager and not (viewer_employee_id and int(viewer_employee_id) in {int(x) for x in involved if x}):
            raise HTTPException(403, "Access denied for this assignment")
        rows = conn.execute(
            "SELECT * FROM hrm_assignment_events WHERE entity_type=? AND entity_id=? ORDER BY id",
            (et, int(entity_id)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

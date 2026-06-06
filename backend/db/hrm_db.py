"""HRM Module DB — Employees, Responsibilities, Task Tracking, Issues, Appraisal"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Optional


def _default_db_path() -> str:
    if os.path.isdir("/data"):
        return "/data/hrm.db"
    return os.path.join(os.path.dirname(__file__), "..", "hrm.db")


_DB = os.environ.get("HRM_DB_PATH", _default_db_path())


def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
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
    ):
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass

    conn.execute(
        "UPDATE one_time_tasks SET status='Done' WHERE status='Completed'"
    )
    conn.commit()
    conn.close()


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
    allowed = ["name", "description", "hod_name"]
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
    code = _next_emp_code(conn)
    conn.execute(
        """INSERT INTO employees(emp_code,name,department_id,designation,phone,email,join_date,status)
        VALUES(?,?,?,?,?,?,?,?)""",
        (
            code,
            data["name"],
            data.get("department_id"),
            data.get("designation", ""),
            data.get("phone", ""),
            data.get("email", ""),
            data.get("join_date", ""),
            "Active",
        ),
    )
    conn.commit()
    conn.close()
    return code


def update_employee(eid: int, data: dict):
    conn = _connect()
    allowed = ["name", "department_id", "designation", "phone", "email", "join_date", "status"]
    sets = ", ".join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [eid]
    if sets:
        conn.execute(f"UPDATE employees SET {sets} WHERE id=?", vals)
        conn.commit()
    conn.close()


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
        SELECT r.*, e.name as employee_name, d.name as department_name
        FROM responsibilities r
        LEFT JOIN employees e ON e.id=r.employee_id
        LEFT JOIN departments d ON d.id=r.department_id
        {where}
        ORDER BY d.name, e.name, r.frequency, r.title
    """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_responsibility(data: dict):
    conn = _connect()
    dept_id = data.get("department_id")
    if not dept_id and data.get("employee_id"):
        row = conn.execute(
            "SELECT department_id FROM employees WHERE id=?", (data["employee_id"],)
        ).fetchone()
        if row:
            dept_id = row["department_id"]
    conn.execute(
        """INSERT INTO responsibilities(employee_id,department_id,title,description,frequency,category,added_by,active)
        VALUES(?,?,?,?,?,?,?,1)""",
        (
            data["employee_id"],
            dept_id,
            data["title"],
            data.get("description", ""),
            data.get("frequency", "Daily"),
            data.get("category", "General"),
            data.get("added_by", ""),
        ),
    )
    conn.commit()
    conn.close()


def update_responsibility(rid: int, data: dict):
    conn = _connect()
    allowed = ["title", "description", "frequency", "category", "employee_id", "active"]
    sets = ", ".join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if sets:
        vals.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
):
    conn = _connect()
    resp = conn.execute(
        "SELECT employee_id, department_id FROM responsibilities WHERE id=?",
        (responsibility_id,),
    ).fetchone()
    if not resp:
        conn.close()
        return False

    conn.execute(
        """INSERT INTO task_logs(responsibility_id,employee_id,log_date,status,remarks,marked_by,marked_at,blocker_employee_id,blocker_reason)
        VALUES(?,?,?,?,?,?,datetime('now'),?,?)
        ON CONFLICT(responsibility_id,log_date) DO UPDATE SET
        status=excluded.status, remarks=excluded.remarks,
        marked_by=excluded.marked_by, marked_at=datetime('now'),
        blocker_employee_id=excluded.blocker_employee_id,
        blocker_reason=excluded.blocker_reason
    """,
        (
            responsibility_id,
            resp["employee_id"],
            log_date,
            status,
            remarks,
            marked_by,
            blocker_employee_id,
            blocker_reason,
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

        if blocker and resp_row:
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

    conn.commit()
    conn.close()
    return True


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


def list_issues(employee_id=None, department_id=None, from_date=None, to_date=None):
    conn = _connect()
    conditions = []
    params = []
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
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(
        f"""
        SELECT il.*,
               e.name as employee_name, d.name as department_name,
               ce.name as caused_by_name, cd.name as caused_by_dept_name
        FROM issue_logs il
        JOIN employees e ON e.id=il.employee_id
        LEFT JOIN departments d ON d.id=il.department_id
        LEFT JOIN employees ce ON ce.id=il.caused_by_employee_id
        LEFT JOIN departments cd ON cd.id=il.caused_by_dept_id
        {where}
        ORDER BY il.issue_date DESC
    """,
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_issue(data: dict):
    conn = _connect()
    dept_id = data.get("department_id")
    if not dept_id and data.get("employee_id"):
        row = conn.execute(
            "SELECT department_id FROM employees WHERE id=?", (data["employee_id"],)
        ).fetchone()
        if row:
            dept_id = row["department_id"]
    conn.execute(
        """INSERT INTO issue_logs(
        employee_id, department_id, issue_date, issue_type, severity,
        title, description, recorded_by,
        caused_by_employee_id, caused_by_dept_id, status)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data["employee_id"],
            dept_id,
            data.get("issue_date") or date.today().isoformat(),
            data.get("issue_type", "General"),
            data.get("severity", "Minor"),
            data["title"],
            data.get("description", ""),
            data.get("recorded_by", ""),
            data.get("caused_by_employee_id"),
            data.get("caused_by_dept_id"),
            "Open",
        ),
    )
    conn.commit()
    conn.close()


def get_issue_employee_id(issue_id: int) -> int | None:
    conn = _connect()
    row = conn.execute("SELECT employee_id FROM issue_logs WHERE id=?", (issue_id,)).fetchone()
    conn.close()
    return int(row["employee_id"]) if row else None


def resolve_issue(issue_id: int, resolution: str):
    conn = _connect()
    conn.execute(
        "UPDATE issue_logs SET status='Resolved', resolution=? WHERE id=?",
        (resolution, issue_id),
    )
    conn.commit()
    conn.close()


def get_hod_dashboard(department_id: int, from_date: str = None, to_date: str = None):
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
    resps = conn.execute(
        """
        SELECT r.*, e.name as employee_name
        FROM responsibilities r
        JOIN employees e ON e.id=r.employee_id
        WHERE r.department_id=? AND r.active=1
        ORDER BY e.name, r.frequency, r.title
    """,
        (department_id,),
    ).fetchall()

    logs = conn.execute(
        """
        SELECT tl.responsibility_id, tl.log_date, tl.status, tl.remarks,
               tl.marked_by, tl.blocker_employee_id, tl.blocker_reason,
               be.name as blocker_name
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
        log_map[key] = {
            "status": l["status"],
            "remarks": l["remarks"],
            "marked_by": l["marked_by"],
            "blocker_name": l["blocker_name"] or "",
            "blocker_reason": l["blocker_reason"] or "",
        }

    result = []
    for r in resps:
        rd = dict(r)
        rd["dates"] = {}
        for d in dates:
            key = (r["id"], d)
            rd["dates"][d] = log_map.get(
                key,
                {
                    "status": "Pending",
                    "remarks": "",
                    "marked_by": "",
                    "blocker_name": "",
                    "blocker_reason": "",
                },
            )
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
        SELECT tl.*, r.title, r.frequency
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

    total = len(task_logs)
    done = sum(1 for t in task_logs if t["status"] == "Done")
    partial = sum(1 for t in task_logs if t["status"] == "Partial")
    missed = sum(1 for t in task_logs if t["status"] == "Missed")
    blocked = sum(1 for t in task_logs if t["status"] == "Blocked")
    resp_pct = round((done + partial * 0.5) / total * 100, 1) if total > 0 else 0

    ot_summary, ot_period = _summarize_one_time_tasks(
        [_one_time_task_row(r) for r in ot_rows], fd, td, today
    )
    combined_pct = _combined_performance_pct(resp_pct, ot_summary)

    return {
        "employee": emp,
        "period": {"from": fd, "to": td},
        "task_summary": {
            "total": total,
            "done": done,
            "partial": partial,
            "missed": missed,
            "blocked": blocked,
            "responsibility_performance_pct": resp_pct,
            "one_time_performance_pct": ot_summary["performance_pct"],
            "performance_pct": combined_pct,
        },
        "one_time_summary": ot_summary,
        "one_time_tasks": ot_period,
        "issues": [dict(i) for i in issues],
        "blockers_caused": [dict(b) for b in blockers_caused],
        "task_logs": [dict(t) for t in task_logs],
    }


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
        SELECT tl.responsibility_id, tl.log_date, tl.status, tl.blocker_employee_id
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

    log_map = {(l["responsibility_id"], l["log_date"]): l["status"] for l in logs}
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
        done = sum(1 for d in dates if log_map.get((r["id"], d), "") == "Done")
        missed = sum(1 for d in dates if log_map.get((r["id"], d), "") == "Missed")
        blocked = sum(1 for d in dates if log_map.get((r["id"], d), "") == "Blocked")
        emp_stats[eid]["total_tasks"] += expected
        emp_stats[eid]["done_tasks"] += done
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
    approved_on_time = 0
    approved_late = 0
    awaiting_approval = 0
    pending = 0
    overdue = 0
    in_progress = 0
    rejected = 0

    for task in period_tasks:
        status = _normalize_one_time_status(task.get("status") or "")
        due = task.get("due_date") or ""
        is_overdue = bool(due and due < today and status in ("Pending", "In Progress"))
        if status == "Approved":
            if _task_completed_on_time(task):
                approved_on_time += 1
            else:
                approved_late += 1
        elif status == "Done":
            awaiting_approval += 1
        elif status == "Rejected":
            rejected += 1
        elif status == "In Progress":
            in_progress += 1
            if is_overdue:
                overdue += 1
        elif status == "Pending":
            pending += 1
            if is_overdue:
                overdue += 1

    total = len(period_tasks)
    positive = approved_on_time + approved_late * 0.6 + awaiting_approval * 0.25
    negative = rejected + overdue + pending * 0.35 + in_progress * 0.15
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
        SELECT t.*, e.name as employee_name, d.name as department_name
        FROM one_time_tasks t
        LEFT JOIN employees e ON e.id=t.employee_id
        LEFT JOIN departments d ON d.id=t.department_id
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
    cur = conn.execute(
        """INSERT INTO one_time_tasks(
            employee_id, department_id, title, description, due_date, assigned_by, status, active
        ) VALUES(?,?,?,?,?,?,?,1)""",
        (
            data["employee_id"],
            dept_id,
            data["title"],
            data.get("description", ""),
            data.get("due_date", ""),
            data.get("assigned_by", ""),
            "Pending",
        ),
    )
    tid = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return tid


def update_one_time_task(task_id: int, data: dict):
    conn = _connect()
    allowed = ["title", "description", "due_date", "employee_id", "department_id", "active"]
    sets = ", ".join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if sets:
        vals.append(_now_iso())
        vals.append(task_id)
        conn.execute(f"UPDATE one_time_tasks SET {sets}, updated_at=? WHERE id=?", vals)
        conn.commit()
    conn.close()


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

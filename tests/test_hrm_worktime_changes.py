"""Employee Check / Reports / Task change request — slots, breaks, office time, leave,
schedule rules, hold, auto-approve, free time and DWR range."""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from backend.db import hrm_db, hrm_worktime
from backend.db.hrm_db import (
    create_department,
    create_employee,
    create_responsibility,
    list_employees,
)

DAY = "2026-08-25"  # Tuesday


class Clock:
    def __init__(self, t: str):
        self.t = t

    def __call__(self) -> str:
        return self.t


@pytest.fixture()
def hrm(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_worktime.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    return hrm_db


@pytest.fixture()
def clock(monkeypatch):
    c = Clock(f"{DAY} 09:00:00")
    monkeypatch.setattr(hrm_db, "_now_iso", c)
    monkeypatch.setattr(hrm_db, "today_ist", lambda: date.fromisoformat(c.t[:10]))
    return c


def _emps(hrm):
    create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm.list_departments()[0]["id"]
    for n in ("Worker A", "Backup B", "Boss C"):
        create_employee({"name": n, "department_id": did})
    ids = {e["name"]: e["id"] for e in list_employees(did)}
    return ids["Worker A"], ids["Backup B"], ids["Boss C"]


def _resp(emp, **kw):
    data = {"employee_id": emp, "title": f"Task-{uuid.uuid4().hex[:4]}", "frequency": "Daily"}
    data.update(kw)
    return create_responsibility(data)


def _resp_row(rid):
    conn = hrm_db._connect()
    r = dict(conn.execute("SELECT * FROM responsibilities WHERE id=?", (rid,)).fetchone())
    conn.close()
    return r


def _due(rid, day):
    conn = hrm_db._connect()
    ctx = hrm_db.DueContext(conn)
    out = hrm_db.responsibility_due_on(_resp_row(rid), day, ctx)
    conn.close()
    return out


# ── Breaks ───────────────────────────────────────────────────────────────────


def test_break_overlaps_lunch_tea_and_midnight():
    lunch = hrm_db.break_overlaps(f"{DAY} 12:50:00", f"{DAY} 13:40:00")
    assert [(o["name"], o["overlap_seconds"]) for o in lunch] == [("Lunch Break", 30 * 60)]
    tea = hrm_db.break_overlaps(f"{DAY} 16:10:00", f"{DAY} 16:30:00")
    assert [(o["name"], o["overlap_seconds"]) for o in tea] == [("Tea Break", 5 * 60)]
    assert hrm_db.break_overlaps(f"{DAY} 23:00:00", "2026-08-26 01:00:00") == []
    assert hrm_db.break_overlaps(f"{DAY} 09:00:00", f"{DAY} 12:00:00") == []


def test_responsibility_complete_break_confirm_deduct(hrm, clock):
    a, _, _ = _emps(hrm)
    rid = _resp(a)
    clock.t = f"{DAY} 12:45:00"
    assert hrm.start_responsibility_timer(rid, DAY) is True
    clock.t = f"{DAY} 13:45:00"
    res = hrm.end_responsibility_timer(rid, DAY)
    assert isinstance(res, dict) and res["status"] == "break_confirm"
    assert res["breaks"][0]["name"] == "Lunch Break"
    assert hrm.end_responsibility_timer(rid, DAY, break_decision="deduct") is True
    detail = hrm.get_responsibility_timer_detail(rid, DAY)
    assert detail["total_work_seconds"] == 30 * 60


def test_one_time_complete_break_count(hrm, clock):
    a, _, _ = _emps(hrm)
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Tea overlap"})
    clock.t = f"{DAY} 15:50:00"
    assert hrm.start_one_time_task(tid) is True
    clock.t = f"{DAY} 16:30:00"
    res = hrm.complete_one_time_task(tid)
    assert isinstance(res, dict) and res["breaks"][0]["name"] == "Tea Break"
    assert hrm.complete_one_time_task(tid, break_decision="count") is True
    t = hrm.list_one_time_tasks(employee_id=a, status="Done")[0]
    assert t["total_work_seconds"] == 40 * 60


# ── Slots: pause/resume, manual edit, notes ──────────────────────────────────


def test_one_time_slots_and_manual_edit_audit(hrm, clock):
    a, _, _ = _emps(hrm)
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Slots"})
    for start, stop in (("09:00", "09:30"), ("10:00", "10:20"), ("11:00", "11:10")):
        clock.t = f"{DAY} {start}:00"
        if start == "09:00":
            assert hrm.start_one_time_task(tid) is True
        else:
            assert hrm.resume_one_time_task(tid) is True
        clock.t = f"{DAY} {stop}:00"
        assert hrm.pause_one_time_task(tid) is True
    t = hrm.list_one_time_tasks(employee_id=a)[0]
    assert len(t["time_slots"]) == 3
    assert t["total_work_seconds"] == 60 * 60

    slot = t["time_slots"][1]
    clock.t = f"{DAY} 12:00:00"
    out = hrm_worktime.edit_time_slot(
        slot["id"], started_at="10:00", ended_at="10:40", notes="Call with vendor", actor="Worker A"
    )
    assert out["manual_edited"] == 1
    assert out["original_ended_at"] == f"{DAY} 10:20:00"
    assert out["notes"] == "Call with vendor"
    t = hrm.list_one_time_tasks(employee_id=a)[0]
    assert t["total_work_seconds"] == 80 * 60
    assert t["has_manual_slots"] is True
    actions = [r["action"] for r in hrm_worktime.slot_audit(slot["id"])]
    assert "manual_edit" in actions and "notes" in actions
    assert hrm_worktime.edit_time_slot(slot["id"], started_at="10:50", ended_at="10:40") == "invalid_range"


def test_manual_slot_add_for_responsibility(hrm, clock):
    a, _, _ = _emps(hrm)
    rid = _resp(a)
    clock.t = f"{DAY} 18:00:00"
    out = hrm_worktime.add_manual_slot(
        entity_type="responsibility", responsibility_id=rid, log_date=DAY,
        started_at=f"{DAY} 10:00", ended_at=f"{DAY} 10:45", notes="forgot timer", actor="A",
    )
    assert out["manual_edited"] == 1 and out["net_seconds"] == 45 * 60
    detail = hrm.get_responsibility_timer_detail(rid, DAY)
    assert detail["total_work_seconds"] == 45 * 60


# ── Office time, close validation, free time ─────────────────────────────────


def test_office_close_blocked_then_auto_pauses(hrm, clock):
    a, _, _ = _emps(hrm)
    rid = _resp(a, title="Daily stock report")
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Running"})
    clock.t = f"{DAY} 09:00:00"
    hrm_worktime.office_start(a, actor="A")
    clock.t = f"{DAY} 10:00:00"
    hrm.start_one_time_task(tid)
    clock.t = f"{DAY} 18:00:00"
    with pytest.raises(hrm_worktime.OfficeCloseBlocked) as exc:
        hrm_worktime.office_close(a, actor="A")
    assert "Daily stock report" in str(exc.value)
    assert exc.value.pending[0]["responsibility_id"] == rid

    hrm.mark_task(rid, DAY, "Done", "Worker A", allow_override=True)
    out = hrm_worktime.office_close(a, actor="A")
    assert out["auto_paused_tasks"] == 1
    assert out["office_seconds"] == 9 * 3600
    assert out["working_seconds"] == 8 * 3600
    assert out["free_seconds"] == 3600
    assert out["free_gaps"][0]["start"] == f"{DAY} 09:00:00"
    slots = hrm.list_one_time_tasks(employee_id=a)[0]["time_slots"]
    assert slots[-1]["ended_at"] == f"{DAY} 18:00:00"
    with pytest.raises(ValueError):
        hrm_worktime.office_close(a, actor="A")


def test_free_time_union_no_double_count():
    busy = [
        (f"{DAY} 10:00:00", f"{DAY} 11:00:00"),
        (f"{DAY} 10:30:00", f"{DAY} 11:30:00"),  # overlaps the first
        (f"{DAY} 14:00:00", f"{DAY} 15:00:00"),
    ]
    gaps = hrm_worktime.free_time_gaps(f"{DAY} 09:00:00", f"{DAY} 18:00:00", busy)
    assert [(g["start"][11:16], g["end"][11:16]) for g in gaps] == [
        ("09:00", "10:00"), ("11:30", "14:00"), ("15:00", "18:00"),
    ]
    assert sum(g["seconds"] for g in gaps) == (60 + 150 + 180) * 60


def test_working_hours_report(hrm, clock):
    a, _, _ = _emps(hrm)
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Work"})
    clock.t = f"{DAY} 09:00:00"
    hrm_worktime.office_start(a)
    clock.t = f"{DAY} 09:30:00"
    hrm.start_one_time_task(tid)
    clock.t = f"{DAY} 11:30:00"
    hrm.pause_one_time_task(tid)
    clock.t = f"{DAY} 17:00:00"
    hrm_worktime.office_close(a)
    rep = hrm_worktime.working_hours_report([a], DAY, DAY)
    assert rep["totals"]["office_seconds"] == 8 * 3600
    assert rep["totals"]["working_seconds"] == 2 * 3600
    assert rep["totals"]["free_seconds"] == 6 * 3600


# ── Approval auto-closure ────────────────────────────────────────────────────


def test_auto_approve_after_two_days_keeps_manual(hrm, clock):
    a, _, boss = _emps(hrm)
    r1 = _resp(a, linked_to_employee_id=boss)
    r2 = _resp(a, linked_to_employee_id=boss)
    hrm.mark_task(r1, DAY, "Done", "Worker A", allow_override=True)
    hrm.mark_task(r2, DAY, "Done", "Worker A", allow_override=True)
    conn = hrm._connect()
    log2 = conn.execute("SELECT id FROM task_logs WHERE responsibility_id=?", (r2,)).fetchone()["id"]
    conn.close()
    assert hrm.approve_task_log(log2, actor="Boss C", linked_employee_id=boss, action="Cancelled") is True

    assert hrm.process_approval_auto_closures(now=f"{DAY} 12:00:00")["auto_approved"] == 0
    out = hrm.process_approval_auto_closures(now="2026-08-27 10:00:00")
    assert out["task_logs"] == 1
    conn = hrm._connect()
    s1 = conn.execute("SELECT approval_status FROM task_logs WHERE responsibility_id=?", (r1,)).fetchone()[0]
    s2 = conn.execute("SELECT approval_status FROM task_logs WHERE responsibility_id=?", (r2,)).fetchone()[0]
    audit = conn.execute(
        "SELECT COUNT(*) FROM hrm_task_audit WHERE action LIKE 'auto_approve%'"
    ).fetchone()[0]
    conn.close()
    assert s1 == "Auto-Approved"
    assert s2 == "Cancelled"
    assert audit >= 1


# ── Schedule rules + Sunday/holiday/leave shift ──────────────────────────────


def test_schedule_rule_parse_normalize():
    assert hrm_db.normalize_schedule_rule("1st monday") == "1st Monday"
    assert hrm_db.normalize_schedule_rule("2nd & 4th saturday") == "2nd & 4th Saturday"
    assert hrm_db.normalize_schedule_rule("lwd") == "Last Working Day"
    with pytest.raises(ValueError):
        hrm_db.parse_schedule_rule("every blue moon")


def test_dynamic_monthly_and_fortnightly(hrm):
    a, _, _ = _emps(hrm)
    m = _resp(a, frequency="Monthly", schedule_rule="1st Monday")
    assert _due(m, "2026-09-07") and not _due(m, "2026-09-14")
    f = _resp(a, frequency="Fortnightly", schedule_rule="1st Saturday")
    assert _due(f, "2026-09-05") and _due(f, "2026-09-19") and not _due(f, "2026-09-12")
    lwd = _resp(a, frequency="Monthly", schedule_rule="Last Working Day")
    assert _due(lwd, "2026-09-30") and not _due(lwd, "2026-09-29")


def test_sunday_holiday_leave_shift_next_working_day(hrm, clock):
    a, _, _ = _emps(hrm)
    rid = _resp(a, frequency="Monthly", schedule_month_day=6)  # 6 Sep 2026 = Sunday
    assert not _due(rid, "2026-09-06") and _due(rid, "2026-09-07")
    hrm.upsert_holiday("2026-09-07", "Festival")
    assert not _due(rid, "2026-09-07") and _due(rid, "2026-09-08")
    clock.t = "2026-09-01 09:00:00"
    hrm.create_leave(a, "2026-09-08", "2026-09-09")
    assert not _due(rid, "2026-09-08") and _due(rid, "2026-09-10")


# ── Leave with backup cover ──────────────────────────────────────────────────


def test_leave_backup_cover_and_restore(hrm, clock):
    a, b, _ = _emps(hrm)
    rid = _resp(
        a, title="Mandatory daily", mandatory=True, require_backup=True, backup_employee_id=b,
        backup_allocation_value=1, backup_allocation_unit="days",
    )
    clock.t = "2026-09-01 09:00:00"
    out = hrm.create_leave(a, "2026-09-05", "2026-09-07")  # Sat–Mon incl. Sunday
    assert out["days"] == 3 and out["sundays_included"] == 1
    assert out["backup_assignments"] == 2  # Sat + Mon (Sunday not a working day)
    assert _resp_row(rid)["employee_id"] == a  # no permanent ownership change
    with pytest.raises(ValueError, match="overlaps"):
        hrm.create_leave(a, "2026-09-07", "2026-09-08")

    snap = hrm.get_employee_day_check(a, "2026-09-05")
    assert snap["on_leave"]
    assert not [i for i in snap["not_worked"] if i["responsibility_id"] == rid]
    covers = hrm.get_employee_day_check(b, "2026-09-05")["additional_work"]
    assert any(c.get("responsibility_id") == rid for c in covers)
    assert not hrm.get_employee_day_check(b, "2026-09-08")["additional_work"]

    dwr = hrm_worktime.list_dwr_report(employee_ids=[a], from_date="2026-09-05", to_date="2026-09-07")
    assert [r["check_date"] for r in dwr["leave_rows"]] == ["2026-09-05", "2026-09-06", "2026-09-07"]

    res = hrm.process_leave_restorations(as_of=date(2026, 9, 8))
    assert res["restored"] == 1
    assert hrm.process_leave_restorations(as_of=date(2026, 9, 9))["restored"] == 0


def test_leave_cancel_removes_future_cover(hrm, clock):
    a, b, _ = _emps(hrm)
    _resp(a, mandatory=True, require_backup=True, backup_employee_id=b,
          backup_allocation_value=1, backup_allocation_unit="days")
    clock.t = "2026-09-01 09:00:00"
    lv = hrm.create_leave(a, "2026-09-08", "2026-09-10")
    assert hrm.cancel_leave(lv["id"], actor="A") is True
    conn = hrm._connect()
    n = conn.execute(
        "SELECT COUNT(*) FROM day_reassignment_clones WHERE assigned_by=?", (f"Leave #{lv['id']}",)
    ).fetchone()[0]
    conn.close()
    assert n == 0
    assert hrm.employee_leave_on(a, "2026-09-09") is None


# ── Hold / auto-resume ───────────────────────────────────────────────────────


def test_hold_hides_and_auto_resumes(hrm, clock):
    a, _, _ = _emps(hrm)
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Hold me"})
    clock.t = f"{DAY} 10:00:00"
    hrm.start_one_time_task(tid)
    clock.t = f"{DAY} 11:00:00"
    assert hrm.hold_one_time_task(tid, DAY) == "resume_in_past"
    assert hrm.hold_one_time_task(tid, "2026-08-28", reason="Waiting on vendor", actor="HOD") is True
    assert not hrm.list_one_time_tasks(employee_id=a)
    assert hrm.list_one_time_tasks(employee_id=a, status="On Hold")[0]["hold_until"] == "2026-08-28"
    snap = hrm.get_employee_day_check(a, DAY)
    assert not [t for t in snap.get("one_time_tasks") or [] if t["id"] == tid]

    clock.t = "2026-08-27 18:00:00"
    held = hrm.list_one_time_tasks(employee_id=a, status="On Hold")[0]
    assert held["total_work_seconds"] == 3600  # no accumulation while held

    assert hrm.process_task_hold_resume(as_of=date(2026, 8, 28))["resumed"] == 1
    t = hrm.list_one_time_tasks(employee_id=a)[0]
    assert t["status"] == "In Progress" and t["timer_status"] == "Paused"
    assert t["total_work_seconds"] == 3600


# ── DWR ──────────────────────────────────────────────────────────────────────


def test_dwr_range_sorted_with_slots_and_notes(hrm, clock):
    a, _, _ = _emps(hrm)
    r_short = _resp(a, title="Short")
    r_long = _resp(a, title="Long")
    clock.t = f"{DAY} 09:00:00"
    hrm.start_responsibility_timer(r_short, DAY)
    clock.t = f"{DAY} 09:10:00"
    hrm.end_responsibility_timer(r_short, DAY, break_decision="count")
    hrm.start_responsibility_timer(r_long, DAY)
    clock.t = f"{DAY} 11:10:00"
    hrm.end_responsibility_timer(r_long, DAY, break_decision="count")
    hrm.mark_task(r_short, DAY, "Done", "Worker A", allow_override=True)
    hrm.mark_task(r_long, DAY, "Done", "Worker A", allow_override=True)
    slot_id = hrm.get_responsibility_timer_detail(r_long, DAY)["time_slots"][0]["id"]
    hrm_worktime.edit_time_slot(slot_id, notes="Reconciled GRN", actor="A", allow_override=True)

    dwr = hrm_worktime.list_dwr_report(employee_ids=[a], from_date=DAY, to_date=DAY)
    titles = [r["title"] for r in dwr["rows"] if r["row_type"] == "responsibility"]
    assert titles[:2] == ["Long", "Short"]
    long_row = dwr["rows"][0]
    assert long_row["duration_seconds"] == 2 * 3600
    assert long_row["slots"][0]["notes"] == "Reconciled GRN"

    # A user with no status update that day is excluded
    assert hrm_worktime.list_dwr_report(employee_ids=[a], from_date="2026-08-26", to_date="2026-08-26")["rows"] == []
    with pytest.raises(ValueError):
        hrm_worktime.list_dwr_report(employee_ids=[a], from_date=DAY, to_date="2026-08-24")


# ── Backfill of pre-existing timer data (non-destructive migration) ──────────


def test_backfill_slots_from_legacy_timers(hrm):
    a, _, _ = _emps(hrm)
    r_events = _resp(a)
    r_legacy = _resp(a)
    tid = hrm.create_one_time_task({"employee_id": a, "title": "Old task"})
    conn = hrm._connect()
    cur = conn.execute(
        """INSERT INTO task_logs(responsibility_id, employee_id, log_date, status, started_at, ended_at)
           VALUES(?,?,?,?,?,?)""",
        (r_events, a, DAY, "Done", f"{DAY} 09:00:00", f"{DAY} 10:30:00"),
    )
    lid = cur.lastrowid
    for et, at in (("start", "09:00"), ("pause", "09:30"), ("resume", "10:00"), ("end", "10:30")):
        conn.execute(
            """INSERT INTO task_timer_events(task_log_id, responsibility_id, employee_id, log_date, event_type, event_at)
               VALUES(?,?,?,?,?,?)""",
            (lid, r_events, a, DAY, et, f"{DAY} {at}:00"),
        )
    conn.execute(
        """INSERT INTO task_logs(responsibility_id, employee_id, log_date, status, started_at, ended_at)
           VALUES(?,?,?,?,?,?)""",
        (r_legacy, a, DAY, "Done", f"{DAY} 11:00:00", f"{DAY} 11:45:00"),
    )
    conn.execute(
        """UPDATE one_time_tasks SET status='Done', started_at=?, completed_at=?, active_seconds=?
           WHERE id=?""",
        (f"{DAY} 14:00:00", f"{DAY} 16:00:00", 3600, tid),
    )
    conn.execute("DELETE FROM hrm_meta WHERE key='slots_backfill_v1'")
    conn.commit()
    hrm._backfill_time_slots(conn)
    conn.commit()
    conn.close()

    assert hrm.get_responsibility_timer_detail(r_events, DAY)["total_work_seconds"] == 60 * 60
    assert len(hrm.get_responsibility_timer_detail(r_events, DAY)["time_slots"]) == 2
    assert hrm.get_responsibility_timer_detail(r_legacy, DAY)["total_work_seconds"] == 45 * 60
    task = hrm.list_one_time_tasks(employee_id=a, status="Done")[0]
    assert task["total_work_seconds"] == 3600  # legacy active_seconds preserved, not wall-clock 2h

    conn = hrm._connect()
    hrm._backfill_time_slots(conn)  # guarded: second run is a no-op
    n = conn.execute("SELECT COUNT(*) FROM hrm_time_slots").fetchone()[0]
    conn.close()
    assert n == 4


# ── Import linked person ─────────────────────────────────────────────────────


def test_import_linked_person_validated(hrm):
    a, _, boss = _emps(hrm)
    res = hrm.import_responsibilities(
        [
            {"title": "Imp ok", "employee_name": "Worker A", "frequency": "Monthly",
             "schedule_rule": "2nd saturday", "linked_person_name": "Boss C"},
            {"title": "Imp bad", "employee_name": "Worker A", "linked_person_name": "Nobody"},
        ]
    )
    assert res["created"] == 1
    assert any("linked person not found" in e for e in res["errors"])
    conn = hrm._connect()
    row = conn.execute("SELECT * FROM responsibilities WHERE title='Imp ok'").fetchone()
    conn.close()
    assert row["linked_to_employee_id"] == boss
    assert row["schedule_rule"] == "2nd Saturday"
